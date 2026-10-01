"""
Uniform Image Preprocessing
---------------------------

This module actually resizes and re-encodes every surviving image to one fixed geometry and one fixed JPEG quality factor, regardless of class or source, thereby eliminating/minimzing any shortcut signal a machine learning model may learn from these features.

### Data flow

```
    manifest row (raw path, split, label)
            |
            v
    resize_and_reencode()   -- pure function, PIL.Image in -> PIL.Image out
            |
            v
    process_one()           -- adds disk I/O: open raw file, save processed file
            |
            v
    build_processed_dataset() -- fans process_one() out across a process pool
            |
            v
    data/processed/<split>/<ai|human>/<sha256>.jpg
            |
            v
    updated manifest (data/processed/manifest.parquet) ready for a
    torch.utils.data.Dataset in the training step.
```

Why re-encode at all, given the images already went through bias matching?
Matching only filters *which* files survive; it cannot make surviving files bit-identical in compression, because it never touches pixels. Re-encoding every image (both classes, all sources) at the same JPEG quality factor and the same canvas size removes `jpeg_qf`, `width`, and `height` as usable shortcut features (see notebooks/metadata_EDA.ipynb, section 8) without touching the deeper synthesis artifacts a CNN is meant to key on.
"""


from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False  # same safety setting as integrity.py
Image.MAX_IMAGE_PIXELS = 250_000_000


## ==================================================================================
## Configuration
## ==================================================================================
@dataclass(frozen= True)
class PreprocessConfig:
    """
    :param image_size: side length (pixels) of the final square output; the cropped square is resized to this value (a common input size for most vision models).
    :param jpeg_qf: JPEG quality factor (1-100) every output image is saved at, regardless of its original format or quality.
    :param crop_size: side length (pixels) of the center crop taken *before* resizing. This should be the lower bound of the size interval (`subset_v1.min_side`), so every image can be cropped without any resampling. If None, or larger than an image's shorter side, the crop falls back to the shorter side.
    :param resample: PIL resampling filter used for the resize step. It specifies how the new pixels should be calculated when the image size is changed. LANCZOS is the default and most optimal resampling method: it preserves high-frequency detail better than bilinear/box filters, which matters because forensic signal often concentrates in high frequencies.
    """

    image_size: int = 224
    jpeg_qf: int = 96
    crop_size: int | None = None
    resample: int = Image.Resampling.LANCZOS

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> PreprocessConfig:
        """
        Builds a `PreprocessConfig` from the full parsed `subset_v1.yaml`. `image_size` and `jpeg_qf` come from the `preprocessing:` block; `crop_size` comes from `matching.min_side`.
        """
        pre = raw.get("preprocessing", {})
        return cls(
            image_size=int(pre.get("image_size", 512)),
            jpeg_qf=int(pre.get("jpeg_qf", 98)),
            crop_size=int(raw["matching"]["min_side"]) if "min_side" in raw.get("matching", {}) else None,
        )


## ==================================================================================
## Pure pixel transform
## ==================================================================================
def resize_and_reencode(im: Image.Image, cfg: PreprocessConfig)-> Image.Image:
    """
    Center-crops an image to a square of `cfg.crop_size` (the lower bound of the size interval), then resizes that square to `cfg.image_size`. JPEG encoding happens last, in `process_one`.

    Data flow is as follows:
    ```
        PIL.Image (any mode, any size)
              |
              v  .convert("RGB")       -- strips alpha/palette/CMYK so every
              |                           output has exactly 3 channels
              v  center-crop to (crop_size, crop_size)   -- pixel-exact, no resampling
              |
              v  resize to (cfg.image_size, cfg.image_size)
              v
        PIL.Image, mode "RGB", size (cfg.image_size, cfg.image_size)
              |
              v  AI images only: JPEG save at cfg.jpeg_qf (in process_one)
                 Real images: saved as PNG (already at the target QF)
    ```

    Order matters: cropping first is lossless and discards pixels before any resampling cost is paid; resizing second means interpolation is applied once, to the decoded pixels; JPEG goes last so the final 8x8 block grid and quantization tables are identical for every image. Compressing before resizing would blur/misalign the block grid and leave a different artifact pattern depending on the source size.

    We first convert to RGB since we may have images that are in various Pillow image modes including but not limited to:
        - `RGB`- 3x8 bit pixels. True color. The most common format for standard images and web graphics (Red, Green, Blue).
        - `RGBA`- 4x8 bit pixels. True color with an alpha (transparency) mask. Essential for PNGs with see-through backgrounds.
        - `L`- 8-bit pixels, grayscale. Standard black-and-white images with 256 shades of gray.
        - `P`- 80-bit pixels mapped to any other mode using a color palette (indexed color). Often used in GIFs to reduce file size.
    Image modes such as `RGBA`, `L`, `P`, and so on could break the [3, H, W] tensor shape assumed in every downstream model.

    """
    im = im.convert("RGB")
    w, h = im.size
    side = min(w, h) if cfg.crop_size is None else min(cfg.crop_size, w, h)

    left = (w - side) // 2
    top = (h - side) // 2
    im = im.crop((left, top, left + side, top + side))

    if side != cfg.image_size:
        im = im.resize((cfg.image_size, cfg.image_size), resample=cfg.resample)
    return im

## ==================================================================================
## Single-image disk operation
## ==================================================================================
def process_one(job: tuple[Path, Path, PreprocessConfig, bool]) -> dict:
    """
    Opens one raw image, transforms it, saves the result, and reports what happened. Never raises an error. Failures are captured in the returned dict so one bad file can't crash a multi-hour batch job.

    Real images were already bias-matched to the target JPEG QF, so they are cropped and resized only and written losslessly (PNG) to avoid a second JPEG encode. AI images (originally lossless PNGs) are saved as JPEG at `cfg.jpeg_qf`.

    Params:
        job (tuple[Path, Path, PreprocessConfig, bool]): ``(src_abs_path, dst_abs_path, cfg, is_ai)`` tuple. Packed into a single tuple (rather than separate positional args) because `ProcessPoolExecutor.map` needs a single iterable of picklable arguments per call, and PreprocessConfig is a small frozen dataclass so it pickles cheaply.

    Returns:
        dict: Contains processed_path / processed_width / processed_height / processed_ok / processed_error. Designed to be assembled into a DataFrame and concatenated onto the input manifest column-wise.
    """
    src, dst, cfg, is_ai = job
    try:
        with Image.open(src) as im:
            out = resize_and_reencode(im, cfg)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if is_ai:
            out.save(dst, format="JPEG", quality=cfg.jpeg_qf)
        else:
            out.save(dst, format="PNG")
        return {
            "processed_path": str(dst),
            "processed_width": out.width,
            "processed_height": out.height,
            "processed_ok": True,
            "processed_error": None,
        }
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure
        # here must not crash the batch; it must be recorded and inspected.
        return {
            "processed_path": None,
            "processed_width": None,
            "processed_height": None,
            "processed_ok": False,
            "processed_error": f"{type(exc).__name__}: {exc}",
        }


## ==================================================================================
## Batch driver
## ==================================================================================
def build_processed_dataset(
    df: pd.DataFrame,
    raw_root: Path,
    processed_root: Path,
    cfg: PreprocessConfig,
    n_workers: int = 8,
    chunksize: int = 32,
) -> pd.DataFrame:
    """
    Fans `process_one` out across a process pool for every row of a manifest that already has `path`, `split`, `label`, and `sha256` columns.

    Destination layout:
        processed_root / <split> / ai / <sha256>.jpg      (label==1)
    processed_root / <split> / human / <sha256>.png   (label==0)

    Using sha256 as the filename (rather than the original filename) keeps
    names collision-free across sources without inventing a new ID scheme,
    and makes a partial rerun idempotent: re-running against a directory
    that already has some of these files just overwrites them with
    byte-identical output (same seed, same deterministic pipeline).

    Rows already marked `is_corrupt` should be filtered out by the caller
    before this is called -- this function assumes every row is expected to
    succeed and treats any failure as worth inspecting.
    """
    jobs = []
    for row in df.itertuples(index=False):
        src = raw_root / str(row.path)
        is_ai = row.label == 1
        label_dir = "ai" if is_ai else "human"
        ext = "jpg" if is_ai else "png"
        dst = processed_root / str(row.split) / label_dir / f"{row.sha256}.{ext}"
        jobs.append((src, dst, cfg, is_ai))

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for res in ex.map(process_one, jobs, chunksize=chunksize):
            results.append(res)

    out = df.reset_index(drop=True).copy()
    res_df = pd.DataFrame(results)
    return pd.concat([out, res_df], axis=1)