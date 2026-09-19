"""
Selective extraction of individual images from the GenImage split zip on Harvard Dataverse.

GenImage is published as one ~654 GB Zip64 archive cut into 500 fixed-size parts (``GenImage.z000`` ... ``GenImage.z499``). 
```
Dataverse
│
├── GenImage.z000
├── GenImage.z001
├── GenImage.z002
├── ...
└── GenImage.z499
```

This program reconstructs a combined virtual ZIP without physically joining the individual file parts. 

The file parts are plain byte slices of a single archive and every offset in the zip's central directory is a *global* offset into the concatenation of the parts i.e.:

    image_A.jpg
    global ZIP offset = 128,450,000,000
    compressed size   = 183,421 bytes

This lets us treat the parts as one virtual file and fetch only the bytes of the members we want using HTTP Range requests.
```
Virtual 654 GB ZIP
│
├───────────────────────┬───────────────────────┬───────
│ GenImage.z000         │ GenImage.z001         │ ...
│ bytes 0 ... N         │ bytes N+1 ... 2N      │
└───────────────────────┴───────────────────────┴───────
```

So in short:
```
global offset
    ↓
which .z file?
    ↓
offset inside that .z file
    ↓
HTTP Range request
```

Dataverse answers each datafile with a redirect to a short-lived signed S3 URL (cloud platform where the data is actually stored), and the Range header has to be sent to that S3 URL.

Typical usage:
    zf = RemoteSplitZip(cache_dir=Path("data/raw/genimage/_cache"))
    members = zf.read_directory(wanted=set(paths))
    zf.extract_members(members, dest=Path("data/raw/genimage"))

The conceptual architecture (roughly):
```
Harvard Dataverse
       │
       │ Dataset DOI
       ▼
┌─────────────────────────────┐
│ GenImage dataset            │
│                             │
│ GenImage.z000               │
│ GenImage.z001               │
│ GenImage.z002               │
│ ...                         │
│ GenImage.z499               │
└──────────────┬──────────────┘
               │
               │ Dataverse API
               ▼
     metadata + file IDs
               │
               ▼
      temporary signed URLs
               │
               ▼
          Amazon S3
               │
               │ HTTP Range:
               │ bytes=123456-234567
               ▼
      only requested bytes
               │
               ▼
      Bytes (zip file parts) combination
               │
               ▼
     virtual 654 GB ZIP
               │
               ▼
       ZIP central directory
               │
               ▼
        desired filename
               │
               ▼
       exact byte offset
               │
               ▼
       download that image
```
"""

from __future__ import annotations

import contextlib
import mmap
import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from requests.adapters import HTTPAdapter

DATAVERSE = "https://dataverse.harvard.edu"
DOI = "doi:10.7910/DVN/AKDIHF"
HEADERS = {"User-Agent": "Mozilla/5.0"}
URL_TTL_SECONDS = 45 * 60  # Signed URLs last 1 hour; refresh early
LOCAL_HEADER_SLACK = 30 + 1024  # Fixed local header + generous room for name and extra field
CD_CHUNK = 16 << 20  # Range size of each parallel central-directory fetch
INDEX_BATCH = 250_000  # Directory entries buffered before a parquet row group is flushed
INDEX_NAME = "directory_index.parquet"

# Central-directory file header, bytes 10..46 (the 4-byte signature and the version fields are skipped)
_CD_ENTRY = struct.Struct("<HHHIIIHHHHHII")
_EXTRA_HDR = struct.Struct("<HH")
_CD_SIG = b"PK\x01\x02"
_U32 = 0xFFFFFFFF


def _zip64(extra: bytes, size: int, csize: int, offset: int) -> tuple[int, int, int]:
    """Resolve the 32-bit slots that overflowed. The Zip64 extra field (id 0x0001) carries only the fields whose 32-bit slot is 0xFFFFFFFF, in fixed order."""
    q, n = 0, len(extra)
    while q + 4 <= n:
        tag, sz = _EXTRA_HDR.unpack_from(extra, q)
        if tag == 1:
            vals = iter(struct.unpack_from(f"<{sz // 8}Q", extra, q + 4))
            if size == _U32:
                size = next(vals)
            if csize == _U32:
                csize = next(vals)
            if offset == _U32:
                offset = next(vals)
            break
        q += 4 + sz
    return size, csize, offset


def _index_schema():
    """Parquet schema of the cached directory index. Names stay ``binary`` so no entry is ever decoded to ``str`` unless it is one we asked for."""
    import pyarrow as pa

    return pa.schema([("name", pa.binary()), ("method", pa.uint16()), ("crc32", pa.uint32()),
                      ("compressed_size", pa.uint64()), ("size", pa.uint64()), ("offset", pa.uint64())])


@dataclass(frozen=True)
class Member:
    name: str
    method: int  # 0 = stored, 8 = deflate
    compressed_size: int
    size: int
    crc32: int
    offset: int  # Global offset of the local header in the virtual concatenated archive


class RemoteSplitZip:
    """Random access to members of the split GenImage zip without downloading the whole archive."""

    def __init__(self, cache_dir: Path, doi: str = DOI, workers: int = 32) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self._lock = threading.Lock()
        self._urls: dict[int, tuple[str, float]] = {}
        self._local = threading.local()

        resp = requests.get(f"{DATAVERSE}/api/datasets/:persistentId/versions/:latest",
                            params={"persistentId": doi}, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        parts = [f["dataFile"] for f in resp.json()["data"]["files"] if f["label"].startswith("GenImage.z")]
        parts.sort(key=lambda p: p["filename"])
        self.part_ids = [p["id"] for p in parts]
        self.part_size = parts[0]["filesize"]  # All parts except the last share this size
        self.total_size = sum(p["filesize"] for p in parts)

    # ── Low-level range reads ──────────────────────────────────────────
    def _session(self) -> requests.Session:
        if not hasattr(self._local, "s"):
            s = requests.Session()
            # Keep enough pooled connections for every in-flight request, or urllib3 discards
            # and re-opens sockets (a fresh TLS handshake per image) once workers is raised.
            adapter = HTTPAdapter(pool_connections=8, pool_maxsize=max(self.workers, 10))
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            self._local.s = s
        return self._local.s

    def _signed_url(self, part: int, refresh: bool = False) -> str:
        with self._lock:
            cached = self._urls.get(part)
            if cached and not refresh and time.time() - cached[1] < URL_TTL_SECONDS:
                return cached[0]
        r = requests.get(f"{DATAVERSE}/api/access/datafile/{self.part_ids[part]}",
                         headers=HEADERS, allow_redirects=False, timeout=60)
        r.raise_for_status()
        url = r.headers["Location"]
        with self._lock:
            self._urls[part] = (url, time.time())
        return url

    def _get_range(self, part: int, start: int, end: int, stream: bool = False) -> requests.Response:
        """GET bytes [start, end] (inclusive) of one part, refreshing the signed URL once if it has expired."""
        for attempt in range(4):
            url = self._signed_url(part, refresh=attempt > 0)
            try:
                r = self._session().get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120, stream=stream)
                if r.status_code == 206:
                    return r
                if r.status_code not in (403, 500, 503):
                    r.raise_for_status()
            except requests.RequestException:
                if attempt == 3:
                    raise
            time.sleep(2 ** attempt)
        raise RuntimeError(f"Range request failed for part {part} bytes {start}-{end}")

    def _locate(self, offset: int) -> tuple[int, int]:
        """Map a global offset to (part index, offset within that part). The last part is slightly larger than the rest."""
        part = min(offset // self.part_size, len(self.part_ids) - 1)
        return part, offset - part * self.part_size

    def _part_end(self, part: int) -> int:
        """Size of `part` in bytes."""
        return self.total_size - part * self.part_size if part == len(self.part_ids) - 1 else self.part_size

    def read(self, offset: int, length: int) -> bytes:
        """Read `length` bytes at a global offset, stitching across part boundaries."""
        length = min(length, self.total_size - offset)
        chunks: list[bytes] = []
        while length > 0:
            part, local = self._locate(offset)
            n = min(length, self._part_end(part) - local)
            chunks.append(self._get_range(part, local, local + n - 1).content)
            offset += n
            length -= n
        return b"".join(chunks)

    # ── Central directory ──────────────────────────────────────────────
    def _locate_directory(self) -> tuple[int, int, int]:
        """Return (global offset, size, entry count) of the central directory via the Zip64 end-of-directory record."""
        tail_len = 1 << 20
        tail = self.read(self.total_size - tail_len, tail_len)
        i = tail.rfind(b"PK\x06\x06")
        if i < 0:
            raise RuntimeError("Zip64 end-of-central-directory record not found in the last part")
        _, _, _, _, _, _, _, entries, cd_size, cd_offset = struct.unpack("<4sQHHIIQQQQ", tail[i:i + 56])
        return cd_offset, cd_size, entries

    def _download_directory(self) -> Path:
        """Fetch the ~380 MB central directory once and cache it on disk, pulling the byte ranges in parallel."""
        cd_offset, cd_size, _ = self._locate_directory()
        path = self.cache_dir / "central_directory.bin"
        if path.exists() and path.stat().st_size == cd_size:
            return path
        tmp = path.with_suffix(".part")
        with tmp.open("wb") as f:
            f.truncate(cd_size)  # Preallocate so each worker can seek straight to its own slice
        spans = [(rel, min(CD_CHUNK, cd_size - rel)) for rel in range(0, cd_size, CD_CHUNK)]
        done = 0

        def pull(span: tuple[int, int]) -> int:
            rel, n = span
            data = self.read(cd_offset + rel, n)
            with tmp.open("r+b") as f:  # One handle per worker; the spans never overlap
                f.seek(rel)
                f.write(data)
            return len(data)

        with ThreadPoolExecutor(max(1, min(self.workers, len(spans)))) as pool:
            for fut in as_completed([pool.submit(pull, span) for span in spans]):
                done += fut.result()
                print(f"\r  central directory: {done / 1e6:7.1f} / {cd_size / 1e6:.1f} MB", end="", flush=True)
        print()
        tmp.replace(path)
        return path

    # ── Directory index ────────────────────────────────────────────────
    def _build_index(self, cd_path: Path, index: Path, wanted: set[str]) -> dict[str, Member]:
        """
        Stream the central directory once: write every entry to a compact parquet index *and* return the members in `wanted`.

        The blob is mmap'd and unpacked in place, names are compared as raw bytes against the pre-encoded `wanted` set, and the Zip64 extra field is only walked for entries that actually overflowed a 32-bit slot -- so nothing is copied or decoded that we do not keep.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = _index_schema()
        raw_wanted = {w.encode("utf-8") for w in wanted}
        found: dict[str, Member] = {}
        cols: tuple[list, ...] = ([], [], [], [], [], [])
        tmp = index.with_suffix(".part")
        unpack, n = _CD_ENTRY.unpack_from, 0

        def flush(writer) -> None:
            writer.write_batch(pa.RecordBatch.from_arrays([pa.array(c, type=f.type)
                                                           for c, f in zip(cols, schema, strict=True)], schema=schema))
            for c in cols:
                c.clear()

        try:
            with pq.ParquetWriter(tmp, schema, compression="zstd") as writer, cd_path.open("rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    p, end = 0, len(mm)
                    while p < end:
                        if mm[p:p + 4] != _CD_SIG:
                            raise RuntimeError(f"Corrupt central directory at byte {p}")
                        (method, _, _, crc, csize, size, nl, el, cl, _, _, _, offset) = unpack(mm, p + 10)
                        q = p + 46
                        name = mm[q:q + nl]
                        if size == _U32 or csize == _U32 or offset == _U32:
                            size, csize, offset = _zip64(mm[q + nl:q + nl + el], size, csize, offset)
                        cols[0].append(name)
                        cols[1].append(method)
                        cols[2].append(crc)
                        cols[3].append(csize)
                        cols[4].append(size)
                        cols[5].append(offset)
                        if name in raw_wanted:
                            text = name.decode("utf-8", "replace")
                            found[text] = Member(text, method, csize, size, crc, offset)
                        p = q + nl + el + cl
                        n += 1
                        if len(cols[0]) >= INDEX_BATCH:
                            flush(writer)
                            print(f"\r  indexing directory: {n:,} entries", end="", flush=True)
                    if cols[0]:
                        flush(writer)
                finally:
                    mm.close()
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        print(f"\r  indexing directory: {n:,} entries -> {index.name}")
        tmp.replace(index)
        return found

    def _lookup_index(self, index: Path, wanted: set[str]) -> dict[str, Member]:
        """Resolve `wanted` against the cached index, one row group at a time so a 2.7 M-entry directory never lands in memory whole."""
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        value_set = pa.array([w.encode("utf-8") for w in wanted], type=pa.binary())
        options = pc.SetLookupOptions(value_set=value_set)  # `pc.is_in` is registered dynamically, so type checkers can't see it
        found: dict[str, Member] = {}
        for batch in pq.ParquetFile(index).iter_batches(batch_size=INDEX_BATCH):
            hits = batch.filter(pc.call_function("is_in", [batch.column("name")], options))
            if hits.num_rows:
                for name, method, crc, csize, size, offset in zip(*(c.to_pylist() for c in hits.columns), strict=True):
                    text = name.decode("utf-8", "replace")
                    found[text] = Member(text, method, csize, size, crc, offset)
                if len(found) == len(wanted):
                    break
        return found

    def read_directory(self, wanted: set[str]) -> dict[str, Member]:
        """
        Resolve archive paths to :class:`Member` records.

        The first call downloads and indexes the central directory; the raw 380 MB blob is discarded afterwards because the index supersedes it. Every later call -- a different selection, or a retry after failures -- is served from the index alone.
        """
        index = self.cache_dir / INDEX_NAME
        if index.exists():
            return self._lookup_index(index, wanted)
        cd_path = self._download_directory()
        found = self._build_index(cd_path, index, wanted)
        cd_path.unlink(missing_ok=True)
        return found

    # ── Member extraction ──────────────────────────────────────────────
    def _fetch_member(self, m: Member) -> bytes:
        raw = self.read(m.offset, LOCAL_HEADER_SLACK + m.compressed_size)
        if raw[:4] != b"PK\x03\x04":
            raise RuntimeError(f"Bad local header for {m.name} at {m.offset}")
        nl, el = struct.unpack("<HH", raw[26:30])
        payload = raw[30 + nl + el:30 + nl + el + m.compressed_size]
        data = payload if m.method == 0 else zlib.decompressobj(-15).decompress(payload)
        if zlib.crc32(data) != m.crc32 or len(data) != m.size:
            raise RuntimeError(f"CRC/size mismatch for {m.name}")
        return data

    def _prewarm_urls(self, parts: set[int]) -> None:
        """Sign every part we are about to read, in parallel, so the first range request against each of the 500 parts doesn't also pay a Dataverse round trip."""
        now = time.time()
        with self._lock:
            fresh = {p for p, (_, at) in self._urls.items() if now - at < URL_TTL_SECONDS}
        todo = sorted(parts - fresh)
        if not todo:
            return

        def sign(part: int) -> None:
            with contextlib.suppress(Exception):  # Non-fatal: _get_range signs lazily and retries anyway
                self._signed_url(part)

        with ThreadPoolExecutor(max(1, min(self.workers, len(todo)))) as pool:
            list(pool.map(sign, todo))

    def _touched_parts(self, members: Iterable[Member]) -> set[int]:
        """Part indices covered by the members' byte ranges."""
        parts: set[int] = set()
        for m in members:
            last = min(m.offset + LOCAL_HEADER_SLACK + m.compressed_size, self.total_size) - 1
            parts.update(range(self._locate(m.offset)[0], self._locate(max(last, m.offset))[0] + 1))
        return parts

    def extract_members(self, members: Iterable[Member], dest: Path) -> tuple[list[str], list[tuple[str, str]]]:
        """Download and write members under `dest`, skipping files that already exist. Returns (written, failed)."""
        todo = []
        for m in members:
            target = dest / m.name
            if target.exists() and target.stat().st_size == m.size:
                continue
            todo.append(m)
        written: list[str] = []
        failed: list[tuple[str, str]] = []
        if not todo:
            return written, failed
        self._prewarm_urls(self._touched_parts(todo))

        def work(m: Member) -> str:
            target = dest / m.name
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".part")
            tmp.write_bytes(self._fetch_member(m))
            tmp.replace(target)
            return m.name

        with ThreadPoolExecutor(self.workers) as pool:
            futures = {pool.submit(work, m): m for m in sorted(todo, key=lambda m: m.offset)}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    written.append(fut.result())
                except Exception as exc:  # Keep going; failures are reported and can be retried by re-running
                    failed.append((futures[fut].name, str(exc)))
                if i % 100 == 0 or i == len(futures):
                    print(f"\r  extracted {i}/{len(futures)}", end="", flush=True)
        print()
        return written, failed
