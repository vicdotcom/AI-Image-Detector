"""
This script draws from the API configirations in the module `downloads.py` to download actual image data. 

Handles multiple data sources, each with different protocols
    ``genimage-subset``: Only the GenImage images listed in a selection file (e.g. `matched_balanced`), fetched from Harvard Dataverse with HTTP range requests. Recommended.
    ``genimage``: GenImage per-generator images (Google Drive via gdown)
    ``coco``: COCO val2017 (direct HTTP, ~778 MB)
    ``ntire``: NTIRE Robust AI-Gen Detection (HuggingFace Hub)
    ``raise``: RAISE uncompressed TIFFs (Kaggle or manual)
    ``verify``: Check what's on disk (no downloads)

    
Usage examples
--------------
Download only the bias-matched GenImage subset (exported by notebooks/01_metadata_EDA.ipynb) rather than the full ~650GB archive:
```
    python scripts/download_images.py genimage-subset --selection data/interim/genimage_matched_balanced.parquet
```
Use `--limit <int>` to download a smaller number of images or for a quick dry run.

Download one GenImage generator from Google Drive (need file ID):
```
    python scripts/download_images.py genimage --gdrive-id <FILE_ID>
```
Sample ~500 images per subfolder from a generator's Drive folder,
instead of the whole thing (only works if Drive exposes individual
files rather than a zip archive for that generator):
```
    python scripts/download_images.py genimage --gdrive-sample-folder-id <FOLDER_ID> --n-per-subfolder 500
```
Extract a manually downloaded GenImage zip:
```
    python scripts/download_images.py genimage --extract path/to/sdv14.zip
```
Download NTIRE shard 0 from HuggingFace:
```
    python scripts/download_images.py ntire --shards 0
```
Check what you've downloaded so far:
```
    python scripts/download_images.py verify
```

Provenance
----------
Every handler writes a ``provenance.json`` into its destination directory
recording what was downloaded, when, and its SHA-256 hash.  These files
should be committed to Git (they're small JSON) so that a collaborator
can verify they have the same data you do.


Dependencies
------------
- Core:       ``requests`` (already installed)
- GenImage:   ``pip install gdown``
- NTIRE:      ``pip install huggingface_hub``
- RAISE:      ``pip install kaggle``
"""


from __future__ import annotations
import argparse 
import sys 
from pathlib import Path

## Import modules
try:
    from ai_detector.data.download import(
        download_coco, 
        download_genimage_gdrive, download_genimage_gdrive_sample, download_genimage_subset,
        extract_genimage_local,
        download_ntire, 
        download_raise,
        extract_zip, verify_source, write_provenance
    )
except ImportError:
    _root = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_root))
    from ai_detector.data.download import(
    download_coco, 
    download_genimage_gdrive, download_genimage_gdrive_sample, download_genimage_subset,
    extract_genimage_local,
    download_ntire,
    download_raise, extract_zip, verify_source, write_provenance
)

## Default destinations (Relative to project root) ---------------
DEFAULT_DESTS = {
    "coco":     Path("data/raw/coco"),
    "genimage": Path("data/raw/genimage"),
    "ntire":    Path("data/raw/ntire"),
    "raise":    Path("data/raw/raise"),
}

## Main function --------------------------------------
def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
 
    if args.source is None:
        ap.print_help()
        return 0
 
    return args.func(args)

# ═══════════════════════════════════════════════════════════════════════
# Subcommand handlers
# ═══════════════════════════════════════════════════════════════════════
def cmd_genimage_subset(args: argparse.Namespace) -> int:
    """Handle the 'genimage-subset' subcommand — only the images listed in a selection file."""
    dest = args.dest or DEFAULT_DESTS["genimage"]
    if not args.selection.exists():
        print(f"ERROR: Selection file not found: {args.selection}", file=sys.stderr)
        return 1
    print(f"\n{'═' * 60}")
    print(f"  GenImage subset ({args.selection.name}) → {dest}")
    print(f"{'═' * 60}\n")

    records = download_genimage_subset(args.selection, dest, workers=args.workers, limit=args.limit)
    if records:
        prov = write_provenance(dest, records, source="genimage")
        print(f"\nProvenance → {prov}")
        print("COMMIT provenance.json to Git.")
    return 0


def cmd_coco(args: argparse.Namespace) -> int:
    """Handle the 'coco' subcommand."""
    dest = args.dest or DEFAULT_DESTS["coco"]
    print(f"\n{'═' * 60}")
    print(f"  COCO val2017 → {dest}")
    print(f"{'═' * 60}\n")
 
    records = download_coco(dest)
    if records:
        prov = write_provenance(dest, records, source="coco")
        print(f"\nProvenance → {prov}")
        print("COMMIT provenance.json to Git.")
    return 0
 
 
def cmd_genimage(args: argparse.Namespace) -> int:
    """Handle the 'genimage' subcommand."""
    dest = args.dest or DEFAULT_DESTS["genimage"]
    print(f"\n{'═' * 60}")
    print(f"  GenImage → {dest}")
    print(f"{'═' * 60}\n")
 
    records = []
 
    if args.extract:
        # ── Extract a local zip ─────────────────────────────────────
        archive = Path(args.extract)
        records = extract_genimage_local(dest, archive)
 
    elif args.gdrive_sample_folder_id:
        # ── Sample N images per subfolder from a generator's folder ──
        records = download_genimage_gdrive_sample(
            dest,
            folder_id=args.gdrive_sample_folder_id,
            n_per_subfolder=args.n_per_subfolder,
            seed=args.seed,
        )
 
    elif args.gdrive_id or args.gdrive_folder_id:
        # ── Download from Google Drive ──────────────────────────────
        records = download_genimage_gdrive(
            dest,
            file_id=args.gdrive_id,
            folder_id=args.gdrive_folder_id,
        )
    else:
        # ── No action specified — print guidance ────────────────────
        print(
            "GenImage download requires one of:\n"
            "\n"
            "  1. Download a single archive by Google Drive file ID:\n"
            "     python scripts/download_images.py genimage "
            "--gdrive-id <FILE_ID>\n"
            "\n"
            "  2. Download an entire Drive folder:\n"
            "     python scripts/download_images.py genimage "
            "--gdrive-folder-id <FOLDER_ID>\n"
            "\n"
            "  3. Sample N images per subfolder (train/ai, train/nature, "
            "etc.) instead of downloading everything — only works if "
            "the generator's Drive folder holds individual files rather "
            "than a zip archive:\n"
            "     python scripts/download_images.py genimage "
            "--gdrive-sample-folder-id <GENERATOR_FOLDER_ID> "
            "--n-per-subfolder 500\n"
            "\n"
            "  4. Extract a manually downloaded zip:\n"
            "     python scripts/download_images.py genimage "
            "--extract path/to/archive.zip\n"
            "\n"
            "To find folder/file IDs:\n"
            "  1. Open https://drive.google.com/drive/folders/"
            "1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS\n"
            "  2. Navigate into the generator you want (e.g. "
            "'Stable Diffusion V1.4')\n"
            "  3. Copy the ID from the address bar, or right-click a "
            "file → 'Get link' → copy the ID between /d/ and /view\n"
            "\n"
            "Tip: `python scripts/download_images.py genimage-subset "
            "--selection <file>` downloads only the images you select, "
            "with no folder IDs needed.\n"
        )
        return 0
 
    if records:
        extra = {}
        if args.gdrive_id:
            extra["gdrive_file_id"] = args.gdrive_id
        if args.gdrive_folder_id:
            extra["gdrive_folder_id"] = args.gdrive_folder_id
        prov = write_provenance(dest, records, source="genimage", extra=extra)
        print(f"\nProvenance → {prov}")
        print("COMMIT provenance.json to Git.")
    return 0
 
 
def cmd_ntire(args: argparse.Namespace) -> int:
    """Handle the 'ntire' subcommand."""
    dest = args.dest or DEFAULT_DESTS["ntire"]
    shards = args.shards  # None means all
 
    print(f"\n{'═' * 60}")
    shard_str = ", ".join(str(s) for s in shards) if shards else "all"
    print(f"  NTIRE (shards: {shard_str}) → {dest}")
    print(f"{'═' * 60}\n")
 
    records = download_ntire(dest, shards=shards, token=args.token)
    if records:
        prov = write_provenance(dest, records, source="ntire")
        print(f"\nProvenance → {prov}")
        print("COMMIT provenance.json to Git.")
    return 0
 
 
def cmd_raise(args: argparse.Namespace) -> int:
    """Handle the 'raise' subcommand."""
    dest = args.dest or DEFAULT_DESTS["raise"]
    print(f"\n{'═' * 60}")
    print(f"  RAISE → {dest}")
    print(f"{'═' * 60}\n")
 
    if args.extract:
        archive = Path(args.extract)
        if not archive.exists():
            print(f"ERROR: Archive not found: {archive}", file=sys.stderr)
            return 1
        extract_zip(archive, dest)
        print(f"  ✓ Extracted to {dest}")
        return 0
 
    records = download_raise(dest)
    if records:
        prov = write_provenance(dest, records, source="raise")
        print(f"\nProvenance → {prov}")
    return 0
 
 
def cmd_verify(args: argparse.Namespace) -> int:
    """Handle the 'verify' subcommand — show what's on disk."""
    print(f"\n{'═' * 60}")
    print("  Data verification summary")
    print(f"{'═' * 60}\n")
 
    for name, default_dest in DEFAULT_DESTS.items():
        dest = default_dest
        verify_source(dest, name)
    print()
    return 0
 
 
# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands for each source."""
    ap = argparse.ArgumentParser(
        prog="download_images",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="source", help="Data source to download")
 
    # ── genimage-subset ─────────────────────────────────────────────
    p_sub = sub.add_parser(
        "genimage-subset",
        help="Download only the GenImage images listed in a selection file "
             "(e.g. matched_balanced) from Harvard Dataverse via range requests",
    )
    p_sub.add_argument(
        "--selection", type=Path, required=True,
        help="Parquet/CSV with a 'path' column of GenImage archive paths",
    )
    p_sub.add_argument(
        "--dest", type=Path, default=None,
        help=f"Destination directory (default: {DEFAULT_DESTS['genimage']})",
    )
    p_sub.add_argument("--workers", type=int, default=32, help="Concurrent range downloads (default: 32; these are small, latency-bound requests)")
    p_sub.add_argument("--limit", type=int, default=None, help="Only fetch the first N images (dry run)")
    p_sub.set_defaults(func=cmd_genimage_subset)

    # ── coco ────────────────────────────────────────────────────────
    p_coco = sub.add_parser(
        "coco",
        help="Download COCO val2017 (~778 MB, 5000 images)",
    )
    p_coco.add_argument(
        "--dest", type=Path, default=None,
        help=f"Destination directory (default: {DEFAULT_DESTS['coco']})",
    )
    p_coco.set_defaults(func=cmd_coco)
 
    # ── genimage ────────────────────────────────────────────────────
    p_gen = sub.add_parser(
        "genimage",
        help="Download GenImage from Google Drive or extract local zip",
    )
    p_gen.add_argument(
        "--dest", type=Path, default=None,
        help=f"Destination directory (default: {DEFAULT_DESTS['genimage']})",
    )
    grp = p_gen.add_mutually_exclusive_group()
    grp.add_argument(
        "--gdrive-id", type=str, default=None,
        help="Google Drive file ID for a single archive",
    )
    grp.add_argument(
        "--gdrive-folder-id", type=str, default=None,
        help="Google Drive folder ID (downloads entire folder)",
    )
    grp.add_argument(
        "--gdrive-sample-folder-id", type=str, default=None,
        help="Google Drive folder ID for ONE generator; samples "
             "--n-per-subfolder images per train/val x ai/nature "
             "subfolder instead of downloading everything",
    )
    grp.add_argument(
        "--extract", type=str, default=None,
        help="Path to a locally downloaded zip to extract",
    )
    p_gen.add_argument(
        "--n-per-subfolder", type=int, default=500,
        help="Images to sample per subfolder when using "
             "--gdrive-sample-folder-id (default: 500)",
    )
    p_gen.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    p_gen.set_defaults(func=cmd_genimage)
 
    # ── ntire ───────────────────────────────────────────────────────
    p_ntire = sub.add_parser(
        "ntire",
        help="Download NTIRE from HuggingFace Hub",
    )
    p_ntire.add_argument(
        "--dest", type=Path, default=None,
        help=f"Destination directory (default: {DEFAULT_DESTS['ntire']})",
    )
    p_ntire.add_argument(
        "--shards", type=int, nargs="+", default=None,
        help="Shard numbers to download (e.g. --shards 0 1). "
             "Omit to download all.",
    )
    p_ntire.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace API token (if the repo is gated)",
    )
    p_ntire.set_defaults(func=cmd_ntire)
 
    # ── raise ───────────────────────────────────────────────────────
    p_raise = sub.add_parser(
        "raise",
        help="RAISE uncompressed TIFFs (prints instructions)",
    )
    p_raise.add_argument(
        "--dest", type=Path, default=None,
        help=f"Destination directory (default: {DEFAULT_DESTS['raise']})",
    )
    p_raise.add_argument(
        "--extract", type=str, default=None,
        help="Path to a downloaded RAISE archive to extract",
    )
    p_raise.set_defaults(func=cmd_raise)
 
    # ── verify ──────────────────────────────────────────────────────
    p_verify = sub.add_parser(
        "verify",
        help="Show what's currently on disk (no downloads)",
    )
    p_verify.set_defaults(func=cmd_verify)
 
    return ap


if __name__ == "__main__":
    sys.exit(main())

