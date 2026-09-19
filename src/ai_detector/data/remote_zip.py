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

The parts are plain byte slices of a single archive and every offset in the zip's central directory is a *global* offset into the concatenation of the parts.
That lets us treat the parts as one virtual file and fetch only the bytes of the members we want using HTTP Range requests. Dataverse answers each datafile with a redirect to a short-lived signed S3 URL (cloud platform where the data is actually stored), and the Range header has to be sent to that S3 URL.

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

import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

DATAVERSE = "https://dataverse.harvard.edu"
DOI = "doi:10.7910/DVN/AKDIHF"
HEADERS = {"User-Agent": "Mozilla/5.0"}
URL_TTL_SECONDS = 45 * 60  # Signed URLs last 1 hour; refresh early
LOCAL_HEADER_SLACK = 30 + 1024  # Fixed local header + generous room for name and extra field


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

    def __init__(self, cache_dir: Path, doi: str = DOI, workers: int = 8) -> None:
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
            self._local.s = requests.Session()
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
        """Fetch the ~380 MB central directory once and cache it on disk."""
        cd_offset, cd_size, _ = self._locate_directory()
        path = self.cache_dir / "central_directory.bin"
        if path.exists() and path.stat().st_size == cd_size:
            return path
        tmp = path.with_suffix(".part")
        done = 0
        with tmp.open("wb") as f:
            while done < cd_size:
                n = min(cd_size - done, 8 << 20)
                data = self.read(cd_offset + done, n)
                f.write(data)
                done += len(data)
                print(f"\r  central directory: {done / 1e6:7.1f} / {cd_size / 1e6:.1f} MB", end="", flush=True)
        print()
        tmp.replace(path)
        return path

    def read_directory(self, wanted: set[str]) -> dict[str, Member]:
        """Parse the central directory, keeping only the entries whose path is in `wanted`."""
        blob = self._download_directory().read_bytes()
        found: dict[str, Member] = {}
        p, end = 0, len(blob)
        while p < end and len(found) < len(wanted):
            if blob[p:p + 4] != b"PK\x01\x02":
                raise RuntimeError(f"Corrupt central directory at byte {p}")
            (_, _, _, _, method, _, _, crc, csize, size, nl, el, cl, _, _, _, offset) = struct.unpack(
                "<4sHHHHHHIIIHHHHHII", blob[p:p + 46])
            name = blob[p + 46:p + 46 + nl].decode("utf-8", "replace")
            if name in wanted:
                extra = blob[p + 46 + nl:p + 46 + nl + el]
                # Zip64 extra field (id 0x0001) holds only the fields whose 32-bit slot is 0xFFFFFFFF, in fixed order
                q = 0
                while q + 4 <= len(extra):
                    tag, sz = struct.unpack("<HH", extra[q:q + 4])
                    if tag == 1:
                        vals = iter(struct.unpack(f"<{sz // 8}Q", extra[q + 4:q + 4 + sz]))
                        if size == 0xFFFFFFFF:
                            size = next(vals)
                        if csize == 0xFFFFFFFF:
                            csize = next(vals)
                        if offset == 0xFFFFFFFF:
                            offset = next(vals)
                    q += 4 + sz
                found[name] = Member(name, method, csize, size, crc, offset)
            p += 46 + nl + el + cl
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
