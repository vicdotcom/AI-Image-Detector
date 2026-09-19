"""
Tests for src/ai_detector/data/remote_zip.py

No network access: the archive lives in memory and the HTTP layer is replaced by fakes.

Run with: ``pytest -q tests/test_remote_zip.py``
"""
from __future__ import annotations

import io
import struct
import sys
import threading
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_detector.data import remote_zip
from ai_detector.data.remote_zip import Member, RemoteSplitZip

FILES = {
    "a/one.jpg": b"\x00\x01\x02" * 500,
    "a/two.jpg": bytes(range(256)) * 10,
    "b/three.png": b"hello world",
}


def _build_archive(files: dict[str, bytes], method: int = zipfile.ZIP_DEFLATED) -> bytes:
    """Zip `files`, then append a Zip64 end-of-central-directory record (as GenImage has)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", method) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    raw = buf.getvalue()
    eocd = raw.rfind(b"PK\x05\x06")
    cd_size, cd_offset = struct.unpack("<II", raw[eocd + 12:eocd + 20])
    zip64 = struct.pack("<4sQHHIIQQQQ", b"PK\x06\x06", 44, 45, 45, 0, 0,
                        len(files), len(files), cd_size, cd_offset)
    return raw + zip64


class FakeZip(RemoteSplitZip):
    """RemoteSplitZip backed by an in-memory archive instead of Dataverse."""

    def __init__(self, cache_dir: Path, archive: bytes) -> None:
        # Deliberately skip the network-bound super().__init__
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.workers = 2
        self._lock = threading.Lock()
        self._urls = {}
        self._local = threading.local()
        self.archive = archive
        self.total_size = len(archive)
        self.part_size = len(archive)
        self.part_ids = [0]

    def read(self, offset: int, length: int) -> bytes:
        return self.archive[max(offset, 0):offset + length]


@pytest.fixture
def archive() -> bytes:
    return _build_archive(FILES)


@pytest.fixture
def rz(tmp_path: Path, archive: bytes) -> FakeZip:
    return FakeZip(tmp_path / "cache", archive)


# ── Range reads over split parts ───────────────────────────────────────
@pytest.fixture
def split(tmp_path: Path):
    """Virtual archive of 3 parts: 10 + 10 + 13 bytes (last part is larger)."""
    data = bytes(range(33))
    parts = [data[:10], data[10:20], data[20:]]
    z = RemoteSplitZip.__new__(RemoteSplitZip)
    z.part_ids = [0, 1, 2]
    z.part_size = 10
    z.total_size = len(data)
    calls = []

    def fake_get_range(part, start, end, stream=False):
        calls.append((part, start, end))
        return SimpleNamespace(content=parts[part][start:end + 1])

    z._get_range = fake_get_range
    return z, data, calls


def test_locate_maps_global_offsets(split):
    z, _, _ = split
    assert z._locate(0) == (0, 0)
    assert z._locate(15) == (1, 5)
    assert z._locate(20) == (2, 0)
    assert z._locate(32) == (2, 12)  # Oversized last part absorbs the remainder


def test_part_end(split):
    z, _, _ = split
    assert [z._part_end(i) for i in range(3)] == [10, 10, 13]


def test_read_within_one_part(split):
    z, data, calls = split
    assert z.read(2, 5) == data[2:7]
    assert calls == [(0, 2, 6)]


def test_read_stitches_across_parts(split):
    z, data, calls = split
    assert z.read(8, 15) == data[8:23]
    assert [c[0] for c in calls] == [0, 1, 2]


def test_read_clamps_to_archive_end(split):
    z, data, _ = split
    assert z.read(30, 100) == data[30:]


# ── _get_range retry behaviour ─────────────────────────────────────────
def _range_zip(responses: list) -> tuple[RemoteSplitZip, list]:
    z = RemoteSplitZip.__new__(RemoteSplitZip)
    refreshes = []
    z._signed_url = lambda part, refresh=False: refreshes.append(refresh) or "http://signed"
    it = iter(responses)

    class Session:
        def get(self, *a, **kw):
            item = next(it)
            if isinstance(item, Exception):
                raise item
            return item

    z._session = lambda: Session()
    return z, refreshes


def test_get_range_returns_on_206(monkeypatch):
    monkeypatch.setattr(remote_zip.time, "sleep", lambda s: None)
    ok = SimpleNamespace(status_code=206)
    z, refreshes = _range_zip([ok])
    assert z._get_range(0, 0, 9) is ok
    assert refreshes == [False]


def test_get_range_refreshes_url_after_403(monkeypatch):
    monkeypatch.setattr(remote_zip.time, "sleep", lambda s: None)
    ok = SimpleNamespace(status_code=206)
    z, refreshes = _range_zip([SimpleNamespace(status_code=403), ok])
    assert z._get_range(0, 0, 9) is ok
    assert refreshes == [False, True]


def test_get_range_gives_up_after_repeated_failures(monkeypatch):
    monkeypatch.setattr(remote_zip.time, "sleep", lambda s: None)
    z, _ = _range_zip([SimpleNamespace(status_code=503)] * 4)
    with pytest.raises(RuntimeError, match="Range request failed"):
        z._get_range(0, 0, 9)


def test_get_range_reraises_network_error_on_last_attempt(monkeypatch):
    monkeypatch.setattr(remote_zip.time, "sleep", lambda s: None)
    z, _ = _range_zip([requests.ConnectionError("boom")] * 4)
    with pytest.raises(requests.ConnectionError):
        z._get_range(0, 0, 9)


# ── Central directory ──────────────────────────────────────────────────
def test_locate_directory_reads_zip64_record(rz: FakeZip):
    cd_offset, cd_size, entries = rz._locate_directory()
    assert entries == len(FILES)
    assert rz.archive[cd_offset:cd_offset + 4] == b"PK\x01\x02"
    assert cd_offset + cd_size <= rz.total_size


def test_locate_directory_missing_record(tmp_path: Path):
    z = FakeZip(tmp_path / "c", b"not a zip at all")
    with pytest.raises(RuntimeError, match="end-of-central-directory"):
        z._locate_directory()


def test_download_directory_caches_to_disk(rz: FakeZip):
    path = rz._download_directory()
    _, cd_size, _ = rz._locate_directory()
    assert path.stat().st_size == cd_size

    reads = []
    original = rz.read
    rz.read = lambda offset, length: reads.append(length) or original(offset, length)
    assert rz._download_directory() == path
    assert reads == [1 << 20]  # Only the tail probe; the directory itself is not re-downloaded


def test_read_directory_filters_to_wanted(rz: FakeZip):
    members = rz.read_directory({"a/two.jpg", "b/three.png", "missing.jpg"})
    assert set(members) == {"a/two.jpg", "b/three.png"}
    m = members["b/three.png"]
    assert m.size == len(FILES["b/three.png"])
    assert m.crc32 == zlib.crc32(FILES["b/three.png"])
    assert m.method == zipfile.ZIP_DEFLATED


def test_read_directory_parses_zip64_extra_field(rz: FakeZip, tmp_path: Path):
    name = b"big/file.jpg"
    extra = struct.pack("<HHQQQ", 1, 24, 5_000_000_000, 4_000_000_000, 7_000_000_000)
    entry = struct.pack("<4sHHHHHHIIIHHHHHII", b"PK\x01\x02", 45, 45, 0, 8, 0, 0, 123,
                        0xFFFFFFFF, 0xFFFFFFFF, len(name), len(extra), 0, 0, 0, 0, 0xFFFFFFFF)
    blob = entry + name + extra
    path = tmp_path / "cd.bin"
    path.write_bytes(blob)
    rz._download_directory = lambda: path

    m = rz.read_directory({"big/file.jpg"})["big/file.jpg"]
    assert (m.size, m.compressed_size, m.offset) == (5_000_000_000, 4_000_000_000, 7_000_000_000)


def test_read_directory_detects_corruption(rz: FakeZip, tmp_path: Path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"garbage" * 20)
    rz._download_directory = lambda: path
    with pytest.raises(RuntimeError, match="Corrupt central directory"):
        rz.read_directory({"a/one.jpg"})


# ── Member extraction ──────────────────────────────────────────────────
@pytest.mark.parametrize("method", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_fetch_member_roundtrip(tmp_path: Path, method: int):
    z = FakeZip(tmp_path / "c", _build_archive(FILES, method))
    members = z.read_directory(set(FILES))
    for name, data in FILES.items():
        assert z._fetch_member(members[name]) == data


def test_fetch_member_bad_local_header(rz: FakeZip):
    m = rz.read_directory({"a/one.jpg"})["a/one.jpg"]
    bad = Member(m.name, m.method, m.compressed_size, m.size, m.crc32, m.offset + 1)
    with pytest.raises(RuntimeError, match="Bad local header"):
        rz._fetch_member(bad)


def test_fetch_member_crc_mismatch(rz: FakeZip):
    m = rz.read_directory({"a/one.jpg"})["a/one.jpg"]
    bad = Member(m.name, m.method, m.compressed_size, m.size, m.crc32 ^ 1, m.offset)
    with pytest.raises(RuntimeError, match="CRC/size mismatch"):
        rz._fetch_member(bad)


def test_extract_members_writes_files(rz: FakeZip, tmp_path: Path):
    dest = tmp_path / "out"
    members = rz.read_directory(set(FILES)).values()
    written, failed = rz.extract_members(members, dest)
    assert sorted(written) == sorted(FILES)
    assert failed == []
    for name, data in FILES.items():
        assert (dest / name).read_bytes() == data
    assert not list(dest.rglob("*.part"))


def test_extract_members_skips_existing(rz: FakeZip, tmp_path: Path):
    dest = tmp_path / "out"
    members = list(rz.read_directory(set(FILES)).values())
    rz.extract_members(members, dest)
    written, failed = rz.extract_members(members, dest)
    assert written == [] and failed == []


def test_extract_members_reports_failures(rz: FakeZip, tmp_path: Path):
    good = rz.read_directory({"a/one.jpg"})["a/one.jpg"]
    bad = Member("b/broken.jpg", 0, 5, 5, 0, 0)  # Offset 0 is a real local header, but the CRC is wrong
    written, failed = rz.extract_members([good, bad], tmp_path / "out")
    assert written == ["a/one.jpg"]
    assert [n for n, _ in failed] == ["b/broken.jpg"]
