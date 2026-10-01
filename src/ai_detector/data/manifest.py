"""
Manifest Module
===============

This module creates a manifest that serves as the single source of truth 
for the project. It constructs an authoritative description of the dataset 
by transforming a collection of image files into a structured table where 
each row represents one image and each column represents a known property 
of that image. This is a standard practice in data engineering for 
structuring data.

With each image explicitly defined, the module enables better dataset 
splitting into train, evaluation, and testing sets.

Data Flow
---------

The transformation pipeline follows this structure:
```
    RAW DATA
        │
        ▼
    manifest.py
        │
        ▼
┌───────────────┐
│   MANIFEST    │
│               │
│ image 1       │
│ image 2       │
│ image 3       │
│ ...           │
└───────────────┘
    ▲    ▲    ▲
    │    │    │
train  test  eval
```

Module Dependencies
-------------------

This module draws from :mod:`integrity.py` to define the ideal train/
evaluation dataset split while minimizing data leakage. The overall 
architecture is as follows:
```
            RAW IMAGE FILE
                    │
                    ▼
                integrity.py
                    │
            "Measure the image"
                    │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
 SHA-256        pHash        Decoding
    │             │             │
    └─────────────┼─────────────┘
                    │
                    ▼
                manifest.py
                    │
            "Describe the image
            in the dataset"
                    │
                    ▼
               ImageRecord
                    │
                    ▼
                DataFrame
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    splitting   training    evaluation
```
"""

from __future__ import annotations # Type Hints
import pandas as pd
from PIL import Image

import json
import os
  # In this case it will be used to determine the number of CPU cores the program shoukd utilize to scan images

import subprocess # Used to run external programs, system commands, and shell scripts directly from your Python code

from concurrent.futures import ProcessPoolExecutor
  # This manifest may need to process thousands or hundreds of thousands of images
  # For each image process we: read file -> decode image -> compute hashing (sha256) -> compute pHash -> Estimate JPEG quality
  # Doing this sequentially is quite slow. Therefore, this module assigns each process to a particular CPU core to enable simultaneous processing

from dataclasses import dataclass, asdict, field, fields 
  # A convinent method of defining a structured data object

from datetime import datetime, timezone # For recording when the manifest was generated
from pathlib import Path
from typing import Iterable, Sequence # Importing custom type hints

## `integrity.py` imports
from .integrity import probe_decodability, sha256_file, phash, estimate_jpeg_quality
 # `.` before `integrity` is crucially important as this is not a top level module. Rather it is a relative import


## Supported Image Formats
IMAGE_SUFFIXES= {".jpg", ".jpeg", ".png", ".webp",".bmp", ".tif", ".tiff"}
  # Set of file extensions that this dataset scanner considers image files.


## Set Target Labels
LABEL_HUMAN= 0
LABEL_AI= 1

## ====================================================================================
## ImageRecord (THe Central structure of the Manifest being created)
## ====================================================================================
@dataclass(frozen= True) # Immutability: Once image info is recorded, it cannot be changed
class ImageRecord:
    """
    A storage structure of any particular image, fully described. Any info regarding the image cannot be changed once recorded.

    The `path` attribute is set relative to the data root to avoid it from breaking when moving to another system/interface (i.e.- docker, Colab). This means setting a path like `genimage/airplane/001.jpg` instead of `/home/john/myproject/data/genimage/airplane/001.jpg` to ensure it is not tied to a particular device
    """
    ## These are the descriptors of the image
    path: str  # Relative
    source: str  # Where the image was scraped from (or its generator)
    sha256: str
    phash: str

    ## ---- Labels -------------------------------------------------------------------
    label: int  # Human-Made (0), AI-Generated (1)
    generator: str
    split: str= "unassigned"  # Eventually this could contain: train, test or validation
    content_class: str| None = None  
      # i.e.: portrait, landscape, animal, architecture, etc...Prevents the model from exploiting content differences instead of generation differences
    group_id: int | None = None  # filled in after near-duplicate clustering


    ## ---- Image Properties -----------------------------------------------------------
    width: int = 0
    height: int = 0
    file_format: str = ""
      # Not necessarily the same as the file extensions defined above in IMAGE SUFFIXES
      # Pillow only reports only specific formats (JPEG, PNG, WEBP, TIFF) as file extensions can be misleading
    file_size_bytes: int = 0
    jpeg_qf: int | None = None
    mode: str = ""  
      # Pillow may report various modes (RGB, RGBA, L, CMYK, P). Image processing pipelines behave differently depending on channel structure
         # RGB = red, green, blue
         # RGBA = RGB + alpha
         # L = grayscale/luminance
         # CMYK = cyan, magenta, yellow, black
    

    ## ---- Image Health -------------------------------------------------------------
    is_corrupt: bool = False
    error: str | None = None

    @property
    def aspect_ratio(self)-> float:
        return self.width / self.height if self.height else 0.0
          # self.height check to avoid ZeroDivisionError
    
    @property
    def megapixels(self)-> float:
        return self.width * self.height / 1e6
      # A megapixel (MP) is 1 million pixels
      # This property computes the total number of pixels in an image by multplying height * width, then divide to 1 million to obtain the number of megapixels in an image
      # Megapixels measure spatial resultion. The higher its value, the higher the image's resolution.
    


## Columns of the image database
MANIFEST_COLUMNS: list[str] = [f.name for f in fields(ImageRecord)]


## ============================================================================
## Probing (For storing image information into one complete ImageRecord)
## ============================================================================
def probe_image(root: Path, abs_path: Path, 
                source: str, label: int, generator: str, content_class: str | None= None, split: str = "unassigned", compute_hashes: bool= True) -> ImageRecord:
    """
    Read an image file and return a fully-populated `ImageRecord`.

    This function performs the following operation:

        one image file
            │
            ▼
        probe_image()
            │
            ▼
        one structured record

    The function also checks for corrupted files via `probe_decodability()`. 
    Corrupted files are still stored in the image database, allowing users 
    to decide how to handle them later.

    :param root: The root path from the machine's memory or the working 
        directory path (e.g., ``C:/Desktop/project/data``)
    :type root: str

    :param abs_path: The full absolute path from the root to the image file 
        (e.g., ``C:/project/data/ai/midjourney/img001.jpg``). The relative 
        path is derived from this by removing machine-specific prefixes, 
        ensuring portability across systems (e.g., ``ai/midjourney/img001.jpg``)
    :type abs_path: str

    :param source: The source from which the image was scraped
    :type source: str

    :param label: Binary label indicating image origin:
        - ``0``: Human-made image
        - ``1``: AI-generated image
    :type label: int

    :param generator: The AI generative model that produced the image 
        (e.g., "Midjourney", "Wokong", etc.)
    :type generator: str

    :param content_class: The content category of the image 
        (e.g., portrait, landscape, animal, architecture, etc.)
    :type content_class: str

    :param split: The dataset split to which the image belongs 
        (training, validation, or testing)
    :type split: str

    :param compute_hashes: If ``True``, computes SHA-256 and perceptual 
        hashes for the image. Hashing can be computationally expensive, 
        so ``False`` is recommended when saving time/compute is important.
    :type compute_hashes: bool

    :returns: A fully-populated `ImageRecord` instance containing all 
        extracted metadata and hashes (if computed)
    :rtype: ImageRecord

    :raises FileNotFoundError: If the image file does not exist at 
        the specified path
    :raises ValueError: If invalid parameters are provided
    """

    ## Obtain relative path
    rel= str(abs_path.relative_to(root).as_posix())

    
    ## Decodability check
    decode= probe_decodability(abs_path)
      # Returns a DecodeResult (True/False) where if False the specific image error is mentioned
    if not decode.ok: # For corrupt images
        return ImageRecord(path= rel, source= source, 
                           sha256= sha256_file(abs_path) if compute_hashes else "", 
                             # We can still try to compute sha256 for corrupt images in cases the pixel data is intact but the header is corrupted
                           phash= "", 
                             # Nothing returned if the hashing doesn't work (For corrupt images) 
                           label= label, generator= generator, split= split, 
                           content_class= content_class, 
                           file_size_bytes= abs_path.stat().st_size, # File size in bytes
                           is_corrupt= True, 
                           error= decode.error)


    with Image.open(abs_path) as im: # Opening valid images
        width, height = im.size 
          # Image dimensions (1920, 1080)
          # If using tensors in PyTorch, this will need to be rearranged (C, 1920, 1080) 
        mode= im.mode # How the pixel data was encoded (RGB, RGBA, L, CMYK, P)
        fmt= im.format or abs_path.suffix.lstrip(".").upper()
          # Extract image format (JPEG, PNG, WEBP, TIFF) using the .format from Pillow or provide a fallback to the image's file extension name
        
    
    # Record image info for valid image files
    return ImageRecord(path= rel, source= source, 
                       sha256= sha256_file(abs_path) if compute_hashes else "", 
                       phash= phash(abs_path) if compute_hashes else "", 
                       label= label, generator= generator, split= split, 
                       content_class= content_class, 
                       width= width, height= height, mode= mode, file_format= fmt, 
                       file_size_bytes= abs_path.stat().st_size, 
                       jpeg_qf= estimate_jpeg_quality(abs_path), is_corrupt= False)


## Probe Image Helper Function
def _probe_star(args: tuple) -> ImageRecord:
    """
    This function serves as an adapter which unpacks the tuple of `probe_image()` arguments into indvidual arguments. 
    
    Rather than run the above `probe_image()` function directly, this function can be utilized to allow `ProcessToolExecutor` to pick each argument and serialize them (run them concurrently via available CPU cores)
    """
    return probe_image(*args)

## ====================================================================================
## Building the Manifest
## ====================================================================================
def build_manifest(jobs: Sequence[tuple], 
                   n_workers: int, 
                   chunksize: int = 64) -> pd.DataFrame:
    """
    This function scales the one-image operation to the entire dataset. Multiple images are probed in parallel across multiple CPU cores to return a `DataFrame` with exactly `MANIFEST_COLUMNS`.


    :param jobs: A sequence of argument tuples matching `probe_image()` function's signature:
        (abs_path, root, source, label, generator, content_class, split). i.e.- The full processing and info extraction pipeline applied on an image.
    :type jobs: list[tuple]

    :param n_workers: The maximum number of parallel worker processes to spawn via `ProcessPoolExecutor`. i.e.- How many cores will be utilized to process images in paralell.
    :type n_workers: int

    :param chunksize: The number of job items submitted together as a single batch to each worker process when using multiprocessing. i.e.- Rather than process a single image, each core processes them in batches of size `chunksize`
    :type chunksize: int

    :returns: Returns a pandas DataFrame consisting of the information regarding a particular image
    :rtype: pd.DataFrame

    The `ImageRecord` class is converted to `pd.DataFrame` via the below transformation:
    ```
    `ImageRecord`
        ↓
    `asdict()`
        ↓
    dictionary
        ↓
    pandas DataFrame
    ```
    """
    n_workers= n_workers or max(1, (os.cpu_count() or 2)- 1)
      # os.cpu_count returns the number of logical CPU cores in the system
      # max(1, ...) ensures at least one worker. 
      # If the number of workers is > 1, then we deduct 1 which means leaving one CPU available to avoid overloading the machine
    
    ## In the case of only 1 n_worker specified, images are simply processed sequenctially
    if n_workers == 1:
        records= [_probe_star(j) for j in jobs]
          # Utilizing only 1 worker can be useful when dealing with small datasets, debugging, or in environments where multiprocessing is inconvinient or unavailable

    ## Multiprocessing Case
    else:
        with ProcessPoolExecutor(max_workers= n_workers) as pool: # Creates worker processes
            records= list(pool.map(_probe_star, jobs, chunksize= chunksize))
              # jobs are then distributed among those processes
              # Chunksize specifies the number of images processed in a batch

    return pd.DataFrame([asdict(r) for r in records], columns= MANIFEST_COLUMNS)


def discover_images(root: Path, pattern: str = "**/*") -> list[Path]:
    """
    This function searches through the working directory (including subfolders) to find and sort image files.

    Images are sorted to ensure reproducibility. The assortment of image files will be different in each machine. If we were to randomly select a sample of images, the sample will always be different regardless of the random seed applied. Sorting therefore establishes a deterministic initial ordering prior to sampling.

    This is known as the reproducibility principle where randomness is only reproducible when the thing being randomized has a deterministic starting state.
    """
    return sorted(p for p in Path(root).glob(pattern) 
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
                    # glob("**/*") means recursively search through the directory tree 
                    # We check whether the image is actually a file (p.is_file()) and whether the extension (p.suffix_lower()) is supported (part of IMAGE_SUFFIXES)
                    # Images found are then sorted alphabetically


## ==========================================================================
## Manifest Versioning
## ==========================================================================
def _git_sha() -> str | None:
    """
    This function records the version of particular code where it returns the lates commit ID from Git.

    If Git is not active then `None`
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None

## Writing the Manifest
def save_manifest(df: pd.DataFrame, path: Path, note: str = "") -> dict:
    """
    Write a manifest to Parquet and emit a small, committable version record. Specifically, it exports the image metadata DataFrame into a binary Parquet file and generates a lighweight `.meta.json` snapshot file to track versions in Git.

    ```
    DataFrame
    │
    ├──► Parquet manifest
    │
    └──► metadata/version record
    ```

    Args:
        df (pd.DataFrame): The fully populated image manifest DataFrame generated downstream.
        path (Path): Target path where the `.parquet` file will be saved
        note (str= ""): An optional description or log note stored alongside the dataset versioning record.

    `.parquet` is preferred since because it preserves pandas data types better (than `.csv`) and is efficient for analytical datasets.

    The returned dict is what is committed to Git. The `parquet` file itself is gitignored as it can be quite heavy, though its changes can still be tracked via the sha256 hash of the manifest generated.
    """

    ## Save the .parquet manifest file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

    ## Dataset record
    record = {
        "manifest": path.name,
        "note": note,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), # Timestamp
        "git_sha": _git_sha(), # Git version
        "n_rows": int(len(df)),
        "sha256": sha256_file(path), 
          # Here we compute a sha265 hash got the entire manifest
          # It acts as a fingerprint for the manifest where if the manifest changes, the hash will change as well
        "counts": {
            "by_label": df["label"].value_counts().sort_index().to_dict(), 
              # Allows to check label distribution. In this particular case, class balance is needed
            "by_source": df["source"].value_counts().to_dict(), # Image source distribution
            "by_split": df["split"].value_counts().to_dict(), # Train/Test/Val distribution
            "by_generator": df["generator"].value_counts().to_dict(), # AI generator distribution
            "corrupt": int(df["is_corrupt"].sum()), # Number of corrupted images
        },
    }
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(record, indent=2, default=str))
    return record

## ==========================================================================
## Leakage Preventions (Preventing the model from shortcut learning)
## ==========================================================================
class LeakageError(AssertionError): # Custom excption definition
    """Raised when the same or similar content appears in more than one split."""
 
 
def assert_no_leakage(df: pd.DataFrame, keys: Iterable[str] = ("sha256", "group_id")) -> None:
    """
    This function checks whether particular hash identifiers occur across multiple splits.
    """
    problems = []
    for key in keys:
        if key not in df.columns:
            continue
        sub = df[df[key].notna() & (df[key] != "")]
        spans = sub.groupby(key)["split"].nunique()
        offenders = spans[spans > 1]
        if not offenders.empty:
            problems.append(f"{len(offenders)} distinct '{key}' values span multiple splits")
    if problems:
        raise LeakageError("; ".join(problems))