"""
This module provides restartable/resumable, idempotent download handlers for each data source used
in this project. Every handler follows the same contract:
1. Accept a destination ``Path`` plus source-specific options.
2. Download files (skipping anything already present and correct).
3. Extract archives where needed.
4. Return a list of :class:`ProvenanceRecord` so the caller can persist an auditable ``provenance.json``.


Its architecture is as follows:
```
    CLI  (scripts/download_images.py)
            │
            ▼
    download.py  (this module)
            │
┌────┼────────┬──────────┬────────┐
▼    ▼        ▼          ▼        ▼
coco genimage ntire    raise   (future)
            │
            ▼
    data/raw/<source>/
            │
            ▼
    provenance.json
```

Every handler checks whether the target already exists and, where possible, whether its size matches the expected value *before* opening a network connection. Running the same command twice should print ``[skip]`` lines and complete in seconds (Idempotency). In short, it prevents duplicate downloads or in case image download was halted, it can simply resume where it left off.

It's main dependencies for image data access are as follows:
- ``requests``  (HTTP downloads — already a project dependency)
- ``gdown``     (Google Drive — **optional**, only for GenImage)
- ``huggingface_hub``  (HuggingFace — **optional**, only for NTIRE)

Download info is recorded in a `provenance` file
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple
import requests


## Constants -------------------------------------
USER_AGENT = (
    "AI-Image-Detector-Research/0.1 "
    "(educational project; contact: vradeny@gmail.com)"
    )

HEADERS = {"User-Agent": USER_AGENT}


## Provenance Record -----------------------------------------
@dataclass
class ProvenanceRecord:
    """
    One downloaded artifact's provenance metadata.

    Recorded so that any future run can verify:
    - *What* was downloaded (filename, sha256)
    - *Where* it came from (source_url)
    - *When* it was fetched (downloaded_utc)
    - *How big* it is (bytes)
    """

    filename: str
    source_url: str
    bytes: int
    sha256: str
    downloaded_utc: str


def write_provenance(dest: Path, 
                     records: list[ProvenanceRecord], 
                     source: str, 
                     extra: dict | None =None)-> Path:
    """
    Writes or merges into a ``provenance.json`` inside `dest`.

    If the file already exists, new records are **merged** by filename,
    so incremental downloads extend the provenance rather than replacing
    it.  This makes the script safe to run repeatedly as you add
    generators or shards over multiple sessions.

    Params:
        dest (Path): Directory that ``provenance.json`` will be written to.
        source (str): Human-readable name of the data source (``"coco"``, ``"genimage"``, etc.).
        extra (dict | None): Additional top-level keys to include in the provenance file (e.g. ``{"gdrive_folder_id": "..."}``).

    Returns:
        Path (Path):  The path to the written ``provenance.json``.
    """

    prov_path= dest/"provenance.json"

    # Load any existing provenance (from a previous partial run)
    if prov_path.exists():
        existing = json.loads(prov_path.read_text())
        existing_files = {r["filename"]: r for r in existing.get("files", [])}
        created = existing.get("created_utc")
    else:
        existing_files = {}
        created = None

    # Merge: new records overwrite by filename
    for rec in records:
        existing_files[rec.filename] = asdict(rec)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    provenance: dict = {
        "source": source,
        "created_utc": created or now,
        "last_updated_utc": now,
        "files": list(existing_files.values()),
    }
    if extra:
        provenance.update(extra)

    prov_path.write_text(json.dumps(provenance, indent=2))
    return prov_path


## SHA-256 -------------------------------------------
# A duplicate of the same funciton contained in integrity.py as the downloader may run before the porject itself is installed
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-256, returning a 64-char hex digest. Prevents file/record duplicates"""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


## Download utilities --------------------------------------------------------
def stream_download(url: str, dest: Path, 
                    expected_size: int | None = None, 
                    chunk_size: int = 1 << 20, 
                    timeout: int = 600, 
                    headers: dict | None = None) -> Path:
    """
    Download a URL to *dest* with a streaming progress indicator.

    Idempotent
        If *dest* already exists and *expected_size* is given, the
        download is skipped when the file sizes match.

    Atomic
        Data is written to a ``.part`` temp file first.  Only when the
        full response has been received is the temp file renamed to
        *dest*.  An interrupted download therefore never leaves a
        truncated file masquerading as a complete one.
    
    Params:
        url (str): The HTTP(S) URL to fetch.
        dest (Path): Where to write the file on disk
        expected_size (int | None): If known, the expected Content-Length. Used for the skip check and for a percentage progress bar.
        chunk_size (int): Bytes per read.  Default 1 MB (``1 << 20``).
        timeout (int): Seconds before the connection times out.
        headers (dict | None): Additional HTTP headers (merged with the default User-Agent).

    Returns:
        Path: The *dest* path (useful for chaining).
    """
    # Skip check
    if dest.exists():
        if expected_size and dest.stat().st_size == expected_size:
            print(f"[skip] {dest.name} already present"
                  f"({expected_size / 1e6:.1f} MB)")
            return dest
        if not expected_size:
            # File exists but we can't verify size — assume good
            print(f"  [skip] {dest.name} already present "
                  f"({dest.stat().st_size / 1e6:.1f} MB)")
            return dest

    # Download
    hdrs= dict(HEADERS)
    if headers:
        hdrs.update(headers)

    dest.parent.mkdir(parents=True, exist_ok=True) # In case the image soruce parent directory doesn't exist
    tmp = dest.with_suffix(dest.suffix + ".part") # For incomplete files

    with requests.get(url, stream= True, timeout= timeout, headers= hdrs) as r:
        r.raise_for_status() # Raises unsuccessful HTTP response codes.
        content_length = int(r.headers.get("content-length", 0)) 
            # Expected https response size (for progress tracking)
        written = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                written += len(chunk) # For byte download progress tracking
                if content_length:
                    pct = written / content_length * 100
                    print(
                        f"\r  {dest.name}: {written / 1e6:7.1f} / "
                        f"{content_length / 1e6:.1f} MB ({pct:.0f}%)",
                        end="", flush=True,
                    )
                else:
                    print(
                        f"\r  {dest.name}: {written / 1e6:7.1f} MB",
                        end="", flush=True,
                    )

    print()  # newline after progress
    tmp.replace(dest)  # rename completed files from `part` to `zip``
    return dest

def extract_zip(archive: Path, dest: Path, 
                remove_after: bool = False) -> Path:
    """
    Extract a zip from an archive into `dest`

    Params:
        archive (Path): The ``.zip`` file to extract
        dest (Path): Target directory where extracted files are sent (created if missing).
        remove_after (bool): Whether to delete the archive after successful extraction. Occurs if ``True``.

    Returns:
        dest (Path): The `dest` path directory. 
    """

    dest.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {archive.name} → {dest} ...")
    with zipfile.ZipFile(archive) as zf:
        # Security: check for path-traversal attacks in zip entries
        for info in zf.infolist():
            target = dest / info.filename
            resolved = target.resolve()
            if not str(resolved).startswith(str(dest.resolve())):
                raise ValueError(
                    f"Zip entry {info.filename!r} escapes target dir — "
                    f"possible path traversal attack."
                )
        zf.extractall(dest)
    if remove_after:
        archive.unlink()
        print(f"  Removed archive {archive.name}")
    return dest


def _count_images(directory: Path) -> int:
    """Count image files in a directory tree."""
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    return sum(
        1 for f in directory.rglob("*")
        if f.suffix.lower() in suffixes
    )

def _make_provenance(filename: str, url: str, path: Path) -> ProvenanceRecord:
    """Helper to build a ProvenanceRecord for a downloaded file."""
    return ProvenanceRecord(filename=filename, source_url=url, 
                            bytes=path.stat().st_size,
                            sha256=sha256_file(path),
                            downloaded_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))


# ═══════════════════════════════════════════════════════════════════════
# Source: GenImage subset (Harvard Dataverse split zip, fetched member-by-member)
# ═══════════════════════════════════════════════════════════════════════
def download_genimage_subset(selection: Path, dest: Path, workers: int = 8, limit: int | None = None) -> list[ProvenanceRecord]:
    """
    Download only the GenImage images listed in `selection` (a parquet/CSV with a ``path`` column such as the ``matched_balanced`` export) using HTTP Range requests against the Dataverse split zip, so the ~654 GB archive is never downloaded in full.

    Params:
        selection (Path): Parquet or CSV file whose ``path`` column holds archive member paths, i.e. ``GenImage/<generator>/<split>/<class>/<file>``.
        dest (Path): Directory the members are written under (their archive paths are preserved).
        workers (int): Number of concurrent range downloads.
        limit (int | None): Only fetch the first N selected images (useful for a dry run).
    """
    import pandas as pd
    from ai_detector.data.remote_zip import RemoteSplitZip, DOI

    df = pd.read_parquet(selection) if selection.suffix == ".parquet" else pd.read_csv(selection)
    wanted = list(dict.fromkeys(df["path"]))[:limit]
    print(f"  {len(wanted)} images requested from {selection.name}")

    zf = RemoteSplitZip(cache_dir=dest / "_cache", workers=workers)
    members = zf.read_directory(set(wanted))
    missing = [w for w in wanted if w not in members]
    if missing:
        print(f"  WARNING: {len(missing)} requested paths are not in the archive (e.g. {missing[:3]})")

    written, failed = zf.extract_members(members.values(), dest)
    print(f"  ✓ {len(written)} written, {len(members) - len(written) - len(failed)} already present, {len(failed)} failed")
    for name, err in failed[:10]:
        print(f"    FAILED {name}: {err}")

    url = f"https://dataverse.harvard.edu/dataset.xhtml?persistentId={DOI}"
    return [_make_provenance(name, url, dest / name) for name in members if (dest / name).exists()]


# ═══════════════════════════════════════════════════════════════════════
# Source: COCO val2017
# ═══════════════════════════════════════════════════════════════════════
COCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"
 
def download_coco(dest: Path) -> list[ProvenanceRecord]:
    """
    Download and extract COCO val2017 (~778 MB, 5 000 images).
 
    Result layout::
 
        dest/
        ├── val2017/
        │   ├── 000000000139.jpg
        │   ├── 000000000285.jpg
        │   └── ...  (5,000 images)
        ├── val2017.zip          (kept for re-verification)
        └── provenance.json
 
    This set serves as the ``test_ood_real`` split — a completely
    independent source of real photographs that the model never sees
    during training.  It lets us measure the false-positive rate on
    images that come from a different processing pipeline than GenImage's
    ImageNet-derived reals.
    """
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "val2017.zip"
    extracted = dest / "val2017"
 
    records: list[ProvenanceRecord] = []
 
    # ── Check if already extracted ──────────────────────────────────
    if extracted.exists():
        n = _count_images(extracted)
        if n >= 4_900:  # COCO val2017 has exactly 5,000
            print(f"  [skip] COCO val2017 already extracted ({n:,} images)")
            if archive.exists():
                records.append(_make_provenance(
                    "val2017.zip", COCO_VAL2017_URL, archive))
            return records
 
    # ── Download ────────────────────────────────────────────────────
    print("Downloading COCO val2017 (~778 MB) ...")
    stream_download(COCO_VAL2017_URL, archive)
 
    digest = sha256_file(archive)
    print(f"  sha256 = {digest}")
 
    records.append(ProvenanceRecord(
        filename="val2017.zip",
        source_url=COCO_VAL2017_URL,
        bytes=archive.stat().st_size,
        sha256=digest,
        downloaded_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    ))
 
    # ── Extract ─────────────────────────────────────────────────────
    extract_zip(archive, dest)
    n = _count_images(extracted)
    print(f"  ✓ COCO val2017: {n:,} images in {extracted}")
 
    return records
 
 
# ═══════════════════════════════════════════════════════════════════════
# Source: GenImage  (Google Drive via gdown)
# ═══════════════════════════════════════════════════════════════════════
GENIMAGE_GDRIVE_FOLDER = "1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS"
 
# Mapping from our internal snake_case names → Google Drive folder names.
# Folder names on Drive may vary from what's listed; the user can override.
GENIMAGE_GENERATORS = {
    "stable_diffusion_v_1_4": "Stable Diffusion V1.4",
    "stable_diffusion_v_1_5": "Stable Diffusion V1.5",
    "midjourney":             "Midjourney",
    "adm":                    "ADM",
    "glide":                  "GLIDE",
    "wukong":                 "Wukong",
    "vqdm":                   "VQDM",
    "biggan":                 "BigGAN",
}
 
 
def download_genimage_gdrive(
    dest: Path,
    file_id: str | None = None,
    folder_id: str | None = None,
) -> list[ProvenanceRecord]:
    """
    Download GenImage archives from Google Drive using ``gdown``.
 
    Two modes:
 
    1. **File mode** (``--gdrive-id <FILE_ID>``): download a single zip
       archive.  Use this when you've right-clicked a specific generator
       archive in the Drive web UI and copied its ID.
 
    2. **Folder mode** (``--gdrive-folder-id <FOLDER_ID>``): download an
       entire Drive folder.  Risky for large folders — Google's quota can
       cut you off.  Best used for small subsets or after you've verified
       the folder is what you expect.
 
    Result layout after extraction::
 
        dest/
        ├── Stable Diffusion V1.4/
        │   ├── train/
        │   │   ├── ai/
        │   │   │   ├── airplane/
        │   │   │   │   └── *.jpg
        │   │   │   └── ...
        │   │   └── nature/
        │   │       └── ...
        │   └── val/
        │       ├── ai/
        │       └── nature/
        └── provenance.json
 
    Parameters
    ----------
    dest : Path
        Root directory (e.g. ``data/raw/genimage``).
    file_id : str | None
        Google Drive file ID for a single archive.
    folder_id : str | None
        Google Drive folder ID (downloads everything inside).
 
    Raises
    ------
    SystemExit
        If ``gdown`` is not installed.
    ValueError
        If neither ``file_id`` nor ``folder_id`` is provided.
    """
    try:
        import gdown
    except ImportError:
        print(
            "ERROR: gdown is required for Google Drive downloads.\n"
            "Install it:  pip install gdown\n"
            "Or:          pip install gdown --break-system-packages",
            file=sys.stderr,
        )
        sys.exit(1)
 
    dest.mkdir(parents=True, exist_ok=True)
    records: list[ProvenanceRecord] = []
 
    if file_id:
        # ── Single file download ────────────────────────────────────
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading Google Drive file {file_id} ...")
        output = gdown.download(
            url, output=str(dest) + "/", quiet=False, resume=True,
        )
        if output is None:
            print("ERROR: gdown download failed. Possible quota limit.",
                  file=sys.stderr)
            print("Try again later or download manually from:\n"
                  f"  https://drive.google.com/file/d/{file_id}",
                  file=sys.stderr)
            return records

        assert isinstance(output, str)  # passed a str `output` path, not a file object
        out_path = Path(output)
        records.append(_make_provenance(
            out_path.name,
            f"https://drive.google.com/file/d/{file_id}",
            out_path,
        ))

        # Auto-extract if it's a zip
        if out_path.suffix.lower() == ".zip":
            extract_zip(out_path, dest)
            n = _count_images(dest)
            print(f"  ✓ Extracted. {n:,} images now in {dest}")

    elif folder_id:
        # ── Folder download ─────────────────────────────────────────
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        print(f"Downloading Google Drive folder {folder_id} ...")
        print("  (This may take a long time for large folders.)")
        print("  (Google Drive may throttle or block — see README.)")
        gdown.download_folder(
            url, output=str(dest), quiet=False)
        # gdown downloads files into dest with their Drive names.
        # We can't easily track individual files here, so record the folder.
        records.append(ProvenanceRecord(
            filename=f"folder_{folder_id}",
            source_url=url,
            bytes=0,  # folder — no single file size
            sha256="",  # folder — no single hash
            downloaded_utc=datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
        ))
        n = _count_images(dest)
        print(f"  ✓ Folder download complete. {n:,} images in {dest}")

    else:
        raise ValueError(
            "Provide either --gdrive-id (single file) or "
            "--gdrive-folder-id (folder)."
        )
 
    return records
 
 
def download_genimage_gdrive_sample(
    dest: Path,
    folder_id: str,
    n_per_subfolder: int = 500,
    seed: int = 42,
) -> list[ProvenanceRecord]:
    """
    Download only a random sample of images from one generator's Google
    Drive folder, instead of the whole thing.
 
    This only works if the folder contains **individual image files**
    (the typical unzipped layout: ``train/ai/``, ``train/nature/``,
    ``val/ai/``, ``val/nature/``). If the folder instead contains a
    single zip archive for the whole generator, sampling is impossible
    without downloading and extracting it first — this function detects
    that case and tells you plainly rather than silently doing the
    wrong thing.
 
    How it works
    ------------
    1. List the folder's contents via ``gdown.download_folder(...,
       skip_download=True)`` — this walks Drive's folder structure and
       returns file metadata (id, path) **without transferring any
       file bytes**. It is the Drive equivalent of ``ls -R``.
    2. Group the listed files by their immediate parent directory
       (e.g. everything under ``.../train/ai/`` is one group).
    3. Within each group, draw ``n_per_subfolder`` filenames using a
       seeded random sample — the seed makes the sample reproducible,
       which matters for your manifest and provenance records.
    4. Download only the sampled files with ``gdown.download()``,
       one at a time.
 
    Parameters
    ----------
    dest : Path
        Where sampled images are saved (e.g.
        ``data/raw/genimage/stable_diffusion_v_1_4``).
    folder_id : str
        The Google Drive folder ID for *one generator* (not the top-level
        GenImage folder — navigate into it first and copy the
        subfolder's ID from the address bar).
    n_per_subfolder : int
        How many files to sample from each train/val × ai/nature
        subfolder. 500 gives you up to 2,000 images for a generator
        with the usual four subfolders.
    seed : int
        Random seed for reproducible sampling.
 
    Returns
    -------
    list[ProvenanceRecord]
        One record per downloaded image.
    """
    try:
        import gdown
    except ImportError:
        print(
            "ERROR: gdown is required for Google Drive downloads.\n"
            "Install it:  pip install gdown",
            file=sys.stderr,
        )
        sys.exit(1)
 
    import random
 
    dest.mkdir(parents=True, exist_ok=True)
    records: list[ProvenanceRecord] = []
 
    # ── Step 1: list without downloading ─────────────────────────────
    print(f"Listing folder {folder_id} (no download yet) ...")
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        # gdown's stubs type this as list[str] | None, but with
        # skip_download=True it actually returns GdownFile objects
        # (with .path / .id attributes), not bare path strings.
        listing: list[Any] | None = gdown.download_folder(
            url, skip_download=True, quiet=True)
    except Exception as e:
        print(f"ERROR: could not list folder — {e}", file=sys.stderr)
        return records
 
    if not listing:
        print("ERROR: folder listing came back empty. Check the folder "
              "ID and that it's shared as 'Anyone with the link'.",
              file=sys.stderr)
        return records
 
    # ── Step 2: detect zip-packaged vs. individual-file layout ───────
    zip_entries = [f for f in listing if f.path.lower().endswith(".zip")]
    image_entries = [
        f for f in listing
        if Path(f.path).suffix.lower()
        in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    ]
 
    if zip_entries and not image_entries:
        print(
            "This generator's Drive folder contains "
            f"{len(zip_entries)} zip archive(s), not individual images:\n"
            + "\n".join(f"  - {f.path}" for f in zip_entries[:5])
            + ("\n  ..." if len(zip_entries) > 5 else "")
            + "\n\n"
            "Partial sampling isn't possible here — the images only "
            "become visible after extracting the zip. Options:\n"
            "  1. Download the full zip with download_genimage_gdrive() "
            "and subsample after extraction.\n"
            "  2. Use the 'genimage-subset' subcommand to fetch only the "
            "images you need from Harvard Dataverse."
        )
        return records
 
    if not image_entries:
        print(
            "ERROR: no recognizable image files or zip archives found "
            "in this folder. The layout may be nested differently than "
            "expected — inspect it manually at:\n  " + url,
            file=sys.stderr,
        )
        return records
 
    # ── Step 3: group by parent directory, then sample ───────────────
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for f in image_entries:
        parent = str(Path(f.path).parent)
        groups[parent].append(f)
 
    print(f"Found {len(image_entries)} images across {len(groups)} "
          f"subfolder(s):")
    rng = random.Random(seed)
    sampled: list = []
    for parent, files in sorted(groups.items()):
        k = min(n_per_subfolder, len(files))
        chosen = rng.sample(files, k)
        sampled.extend(chosen)
        print(f"  {parent}: sampling {k} of {len(files)}")
 
    # ── Step 4: download only the sampled files ───────────────────────
    # IMPORTANT: preserve the train/ai, train/nature, val/ai, val/nature
    # subfolder structure under `dest`. Flattening every file to
    # `dest / filename` would silently overwrite files whenever two
    # subfolders happen to share a filename (e.g. both "img_0238.jpg") —
    # a real risk here since GenImage's naming is numeric per subfolder.
    print(f"\nDownloading {len(sampled)} sampled images ...")
    for i, f in enumerate(sampled, start=1):
        # f.path looks like "Stable Diffusion V1.4/train/ai/img_0004.jpg".
        # Drop the top-level generator-folder segment (parts[0]) and
        # keep the rest (train/ai/img_0004.jpg) as the relative path
        # under dest, so subfolder identity is preserved on disk.
        rel_parts = Path(f.path).parts[1:]
        out_path = dest / Path(*rel_parts) if rel_parts else dest / Path(f.path).name
        out_path.parent.mkdir(parents=True, exist_ok=True)
 
        if out_path.exists():
            print(f"  [{i}/{len(sampled)}] [skip] {'/'.join(rel_parts)}")
            continue
        print(f"  [{i}/{len(sampled)}] {'/'.join(rel_parts)}")
        gdown.download(id=f.id, output=str(out_path), quiet=True)
 
        if out_path.exists():
            records.append(_make_provenance(
                out_path.name,
                f"https://drive.google.com/file/d/{f.id}",
                out_path,
            ))
 
    n = _count_images(dest)
    print(f"\n  ✓ Sampled download complete: {n:,} images in {dest}")
 
    return records
 
 
def extract_genimage_local(
    dest: Path,
    archive: Path,
) -> list[ProvenanceRecord]:
    """
    Extract a manually downloaded GenImage zip archive.
 
    Use this when you downloaded the zip from Google Drive via
    your browser (which avoids quota issues).
 
    Parameters
    ----------
    dest : Path
        Where to extract (e.g. ``data/raw/genimage``).
    archive : Path
        The ``.zip`` file to extract.
 
    Returns
    -------
    list[ProvenanceRecord]
        A provenance record for the archive.
    """
    if not archive.exists():
        print(f"ERROR: Archive not found: {archive}", file=sys.stderr)
        sys.exit(1)
 
    records: list[ProvenanceRecord] = []
    print(f"Hashing {archive.name} ...")
    records.append(_make_provenance(
        archive.name,
        f"local://{archive.resolve()}",
        archive,
    ))
 
    extract_zip(archive, dest)
    n = _count_images(dest)
    print(f"  ✓ Extracted. {n:,} images now in {dest}")
 
    return records


# ═══════════════════════════════════════════════════════════════════════
# Source: NTIRE  (HuggingFace Hub)
# ═══════════════════════════════════════════════════════════════════════
NTIRE_REPO_ID = "deepfakesMSU/NTIRE-RobustAIGenDetection-train"
 
NTIRE_NUM_SHARDS = 6  # shard_0.zip .. shard_5.zip

def download_ntire(
    dest: Path,
    shards: list[int] | None = None,
    token: str | None = None,
) -> list[ProvenanceRecord]:
    """
    Download the NTIRE Robust AI-Gen Detection training set from
    HuggingFace.

    The repo stores the dataset as flat per-shard archives at its root —
    ``shard_0.zip`` .. ``shard_5.zip`` (~20 GB each, ~115 GB total) —
    rather than as unpacked ``shard_N/`` directories. Each handler call
    downloads the requested shard archives with ``hf_hub_download`` and
    extracts them under ``dest``. For our project we typically need only
    ``shard_0`` as a ``test_wild`` set.

    Result layout::

        dest/
        ├── shard_0.zip
        ├── shard_0/
        │   ├── labels.csv
        │   └── images/
        │       ├── 0001.png
        │       └── ...
        ├── shard_1.zip
        ├── shard_1/
        │   └── ...
        └── provenance.json

    Parameters
    ----------
    dest : Path
        Where to save (e.g. ``data/raw/ntire``).
    shards : list[int] | None
        Which shard numbers to download (0-5).  ``None`` downloads all.
    token : str | None
        HuggingFace API token (needed if the repo is gated).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "ERROR: huggingface_hub is required for NTIRE downloads.\n"
            "Install it:  pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)
    records: list[ProvenanceRecord] = []

    wanted = shards if shards is not None else list(range(NTIRE_NUM_SHARDS))
    print(f"Downloading NTIRE shards: {wanted} ...")

    for s in wanted:
        filename = f"shard_{s}.zip"
        extracted = dest / f"shard_{s}"

        # Skip check: shard already extracted
        if extracted.exists() and _count_images(extracted) > 0:
            print(f"  [skip] {filename} already extracted "
                  f"({_count_images(extracted):,} images)")
            continue

        print(f"  Fetching {filename} ...")
        local_path = hf_hub_download(
            repo_id=NTIRE_REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=str(dest),
            token=token,
        )
        archive = Path(local_path)
        records.append(_make_provenance(
            filename,
            f"https://huggingface.co/datasets/{NTIRE_REPO_ID}/blob/main/{filename}",
            archive,
        ))

        extract_zip(archive, extracted)
        n = _count_images(extracted)
        print(f"  ✓ {filename}: {n:,} images in {extracted}")

    n_total = _count_images(dest)
    print(f"  ✓ NTIRE: {n_total:,} images total in {dest}")

    return records
 
# ═══════════════════════════════════════════════════════════════════════
# Source: RAISE  (Kaggle mirror or direct HTTP)
# ═══════════════════════════════════════════════════════════════════════
# Kaggle has a RAISE-TIFF x300 mirror that's ~7 GB and easy to access.
# The official site (loki.disi.unitn.it) requires a form submission.
# For automation, Kaggle is more practical.
RAISE_KAGGLE_SLUG = "mrutyunjaybiswal/raisetiff-uncompressed-images-dataset-x300"
 
def download_raise(dest: Path) -> list[ProvenanceRecord]:
    """
    Download the RAISE uncompressed TIFF dataset.
 
    Because the official RAISE website requires a web form submission
    and does not provide a stable direct-download URL, this handler
    guides you through two options:
 
    Option A — Kaggle CLI (recommended for automation)::
 
        pip install kaggle
        kaggle datasets download -d mrutyunjaybiswal/raisetiff-uncompressed-images-dataset-x300 -p data/raw/raise --unzip
 
    Option B — Manual download from the official RAISE website::
 
        1. Visit https://loki.disi.unitn.it/RAISE/download.html
        2. Select categories and camera model
        3. Fill out the form and download
        4. Place files in data/raw/raise/
 
    Then run::
 
        python scripts/download_images.py raise --extract data/raw/raise/archive.zip
 
    This handler **prints instructions** rather than automating the
    download, because both Kaggle API and RAISE require authentication
    that we should not hardcode.
    """
    dest.mkdir(parents=True, exist_ok=True)
    records: list[ProvenanceRecord] = []
 
    # Check if already present
    n = _count_images(dest)
    if n >= 200:
        print(f"  [skip] RAISE data already present ({n:,} images in {dest})")
        return records
 
    rule = "═" * 60
    print(
        f"{rule}\n"
        f"RAISE dataset download\n"
        f"{rule}\n"
        f"\n"
        f"RAISE requires manual download. Choose one option:\n"
        f"\n"
        f"OPTION A — Kaggle CLI (recommended):\n"
        f"  1. pip install kaggle\n"
        f"  2. Set up ~/.kaggle/kaggle.json  (API credentials)\n"
        f"  3. Run:\n"
        f"     kaggle datasets download -d {RAISE_KAGGLE_SLUG} "
        f"-p {dest} --unzip\n"
        f"\n"
        f"OPTION B — Official website:\n"
        f"  1. Visit https://loki.disi.unitn.it/RAISE/download.html\n"
        f"  2. Select RAISE-1k (smallest subset, ~35 GB)\n"
        f"  3. Download and place TIF files in:\n"
        f"     {dest}/\n"
        f"\n"
        f"After downloading, run this script again with --extract:\n"
        f"  python scripts/download_images.py raise "
        f"--dest {dest} --extract <archive_path>\n"
        f"{rule}"
    )
 
    return records
 
 
# ═══════════════════════════════════════════════════════════════════════
# Verification Utility
# ═══════════════════════════════════════════════════════════════════════
def verify_source(dest: Path, source: str) -> None:
    """
    Print a summary of what's on disk for a given source directory.
 
    Useful as a quick sanity check after downloading.
    """
    if not dest.exists():
        print(f"  {source}: directory does not exist ({dest})")
        return
 
    n = _count_images(dest)
    size_mb = sum(
        f.stat().st_size for f in dest.rglob("*") if f.is_file()
    ) / 1e6
 
    # Check for provenance
    prov = dest / "provenance.json"
    prov_status = "present" if prov.exists() else "MISSING"
 
    print(f"  {source}: {n:,} images, {size_mb:,.0f} MB, "
          f"provenance.json: {prov_status}")
 
    # List immediate subdirectories
    subdirs = sorted(
        d.name for d in dest.iterdir() if d.is_dir()
    )
    if subdirs:
        print(f"    subdirs: {', '.join(subdirs[:12])}"
              + (" ..." if len(subdirs) > 12 else ""))
 