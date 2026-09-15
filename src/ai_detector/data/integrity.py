"""
Integrity, Fingerprinting and Duplicate-Detection Utilities

This is an image integrity and preprocessing module. Prior to training the AI image detector, we first need to understand the dataset feeding it. For instance, for a set of images within a dataset:
```
image_001.jpg
image_002.jpg
image_003.jpg
image_004.jpg
```

We may find:
```
image_001.jpg ───── exact copy ───── image_002.jpg

image_003.jpg ───── resized version ───── image_004.jpg
```
Which can cause data leakages in the event of splitting to training and evaluation sets.


This module provides utilities for assessing image integrity and preventing
data leakage in machine learning pipelines. It ensures that:
- Data leakage is prevented when splitting datasets into train/evaluation
  sets by detecting duplicate or near-duplicate images
- Images are uncorrupted and can be opened and processed



The module performs integrity checks at multiple levels of image representation:

```

                    IMAGE
                      │
           ┌──────────┼───────────┐
           ▼          ▼           ▼
         BYTES      VISUAL      ENCODING
           │          │           │
        SHA-256      pHash       JPEG QF
           │          │           │
      exact copy   similarity   compression


The full processing pipeline follows this structure:

```

                    IMAGE DATASET
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
      SHA-256 hashing          Decodability
             │                       │
             │                       ├── :func:`verify()`
             │                       └── :func:`load()`
             │
             ▼
      Exact duplicates
             │
             ▼
        pHash generation
             │
             ├── grayscale
             ├── resize 32x32
             ├── DCT
             ├── low frequencies
             ├── median threshold
             └── 64-bit hash
                     │
                     ▼
              Hamming distance
                     │
                     ▼
               LSH banding
                     │
                     ▼
          Near-duplicate pairs
                     │
                     ▼
                 Union-Find
                     │
                     ▼
              Duplicate groups
                     │
                     ▼
              TRAIN / TEST SPLIT
"""

from __future__ import annotations # Allows for dynamic type hints (i.e.- We could assign class type hints rather than only common types: int, str, dict....)

import numpy as np
from PIL import Image, ImageFile # Python's image processing library (Pillow)
import hashlib # Python's standard-library module for cryptographic hash functions
from dataclasses import dataclass # A convinient way of creating small classes primarily intended to store data
from pathlib import Path # Working with system file paths
from typing import Iterable, Sequence # Type Hints


## Pillow's safety settings
ImageFile.LOAD_TRUNCATED_IMAGES= False
 # Prevents loading truncated images
 # A truncated image is one where the file ends before the complete image data is present.

Image.MAX_IMAGE_PIXELS= 250_000_000
 # Prevents loading extremely large images. Max pixels is 250,000,000


## ==================================================================================
## Cryptographic Hashing (Are the files byte for byte identical?)
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    # Where `1 << 20` is a common bitwise left shift by 20 bits equivalent to 1*2^20= 1,048,576 bytes= 1MB. This is essentially saying, the 1 unit of chunk_size represents 1 MB 
    '''
    Streams a file through SHA-256 without loading it into memory

    We use a streaming framwork since some images are quite large (>= 20MB). Loading multiple large images at once can cause the program to crash.
    
    Returns a 64-character lowercase hex string. Two files with the same sha256 are identical

    '''
    h= hashlib.sha256() 
     # Hashing function that converts any input into a unique fixed 64-character string (256  bits). This is essentially a digital fingerprint

    ## Rather than load all the files at one go, we weould rather load them in small chunks
    with Path(path).open("rb") as f: 
         # "rb" means opening the file in read binary mode, thereby returning bytes
        for chunk in iter(lambda: f.read(chunk_size), b""): 
            # iter(no_argument_function, sentinel)
            # Creates an iterable where the function can be repeatedly called until the sentinel is observed telling it to stop
            h.update(chunk)
          # This means the program keeps reading the files in chunks until b"" where there are no more bytes
    return h.hexdigest()


## ==================================================================================
## Decodability
@dataclass(frozen= True) # frozen= True ensure immutability
class DecodeResult:
    ok: bool
    error: str | None = None
 # This function automatically creates a Python class for storing data. It saved us the hassle of defining specific __init__ methods, their variables and so on....
 # This simply gives the decoding function a structured result where for instance: `DecodeResult(ok=True)` means Image successfully decoded
 # Otherwise `DecodeResult(ok=False, error="OSError: image file is truncated")`would mean something went wrong

## The below function asks whether Pillow can decode the file image, rather than asking whether Pillow can simply open it.
def probe_decodability(path: Path)-> DecodeResult: 
    # Where the output of this function is True/False as per the defined class above
    """
    This function checks whether the image is fully decodable.

    A decodable image is simply an uncorrupted file. This means its pixels can be successfully read, decompressed, and processed without errors.

    An image file (i.e.- .jpg, .png) is normally structured as follows.
      - Header and Metadata- Dimensions (1920x1080), color profiles, EXIF (technical image details such as location, time and date, camera settings, device info)
      - Compressed pixel payload
    
    An image can therefore fail to decode via:
      - Header Corruption- The image file is garbage or the file may not even be an image at all
      - Truncation or Payload Corruption- The header is perfectly fine but the pixel stream is broken or missing (i.e.- The full image is not loaded). Can occur due to network connections dropping halfway through a download or web scrape.

    Therefore, a two-stage check is applied to ensure each aspect of the image is decodable:
      - Stage 1 (`Image.verify()`)- Checks header markers and basic formatting.
      - Stage 2 (`Image.load()`)- Uncompresses and decodes every single pixel.
    """
    try:
        with Image.open(path) as im:
            im.verify() # Header checks
            # im.verify() leaves the file in an unusable state so we have to open it again
        with Image.open(path) as im: 
            im.load() # Pixel checks
        return DecodeResult(ok= True)
    except Exception as exc:
        return DecodeResult(ok= False, error= f"{type(exc).__name__}: {exc}")
         # Returns the specidic decooding error whether caused by corrupted header or image


## ==================================================================================
## Perceptual Hashiing (For detecting near-duplicate images.)
## First we define the DCT matrix
def _dct_matrix(n: int) -> np.ndarray: 
    # `np.ndarray` is the underlying class object of the array itself
    r"""
    This functon serves as an input in the pHash function where the actual DCT transformation takes place. Via the DCT process, it helps pHash to know what kind of patterns (or frequencies) make up a particular image.

    The Discrete Cosine Transform (DCT) is a mathematical method that changes an image from the spatial domain (raw pixel values) to the frequency domain (cosine wave components). It essnetially splits the image into high and low frequency parts where:
      - Low frequencies (Top left)- Represent broad patterns, shaped, smooth gradients, overall illumination, large scale structure
      - High frequencies (Botton right)- Represents egdes, fine textures, sharp details, rapid color transitions
    
    It serves as a core image compression tool with most of the visual data packed into the top-left corner:
    ```
    Spatial Domain (Pixel Grid)            Frequency Domain (DCT Output)
    [ [120, 122, 125, ...],                [ [ DC ,  F1,  F2, ...],
    [118, 119, 121, ...],   === DCT ===>   [  F3,  F4,  F5, ...],
    [115, 117, 120, ...] ]                 [  F6,  F7,  F8, ...] ]
    (Brightness at (x, y))                 (Amplitude of Cosine Patterns)
    ```

    Where the [DC] term represents zero-frequency; simply the overall mean brightness of the entire image.

    To facilitate the DCT process, this function returns an nxn orthonormal matrix D. The actual transformation occurs as follows:
    
    $$
    DCT_freq= dct_matrix * image_pixel_columns * dct_matrix_transposed
    $$
    """
    k = np.arange(n).reshape(-1, 1)  # shape (n, 1) -> frequency index
    i = np.arange(n).reshape(1, -1)  # shape (1, n) -> spatial index
    d = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    d[0, :] *= np.sqrt(1 / n)
    d[1:, :] *= np.sqrt(2 / n)
    return d

_DCT32 = _dct_matrix(32)

## Perceptual Hashing
def phash(path_or_image: Path | Image.Image, hash_size: int = 8) -> str:
    """
    This function produces a percenptual hash. Unlike an ordinary hash function such as `sha_256_file()` above, this function checks visual similarity rather than strict identical copies.

    The rationale for this function is that say Image A is the photograph of a dog. Image B may be the same photograph though resized, cropped, made brighted/darker, and so on....

    A DCT transformation is applied to each image and an 8x8 grid containing the lowest frequencies are stored as shown below then hashed.
    ┌────────────────────┐
    │ █ █ █ ░ ░ ░ ░ ░    │
    │ █ █ █ ░ ░ ░ ░ ░    │
    │ █ █ █ ░ ░ ░ ░ ░    │
    │ ░ ░ ░              │
    │ ░ ░ ░              │
    │                    │
    └────────────────────┘Image

    From the above image, the top left 8x8 grid is selected after applying the DCT transformation as it contains the most visual information.

    The 8x8 frequency matrix is hashed to return a hexadecimal hash string.

    Known failure mode: near-uniform images (flat black, flat sky) have almost
    no low-frequency structure, so their hashes collapse together and produce
    false "duplicates". While this function is still useful to ensure diverse training input, threshold flags should be examined manually.
    """

    if isinstance(path_or_image, Image.Image):
        im= path_or_image # If the image is already open/loaded or modified in memory
    else:
        im= Image.open(path_or_image) # If a file path is given or the image is not yet opened.
    im= im.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
     # `.convert("L")` converts images to grayscale ("L" means luminance/grayscale in Pillow)
         # pHash tries to determine structural similarity (e.g.- A red car and blue car images may be different but when all in grayscale they are structurally similar)
     # .resize((32, 32) rescales all images making it easier to assess similarities between images of various sizes and resolutions
     # Lanczos is a high quality image resampling algorithm
    
    pixels = np.asarray(im, dtype=np.float64) # Conversion of image to (32, 32) array of pixels for DCT transformation
    freq = _DCT32 @ pixels @ _DCT32.T # DCT transformation (32, 32)
    low = freq[:hash_size, :hash_size] # Out of the 32x32 frequencies, retain the 8x8 lowest frequencies
 
    flat = low.flatten()
    median = np.median(flat[1:]) # exclude DC from median
    bits = flat > median 
      # Converting frequencies into (0,1) bits for hashing
      # Median is used as it divides the values into roughly equal groups
    bits[0] = False  # Exclude the DC term (which is the first term in the matrix) by forcing it to False as it only measures overall brightness

    # Hash generation
    packed = np.packbits(bits) # The 8x8 boolean bits are packed into a binary valued array that convertes then into 8x8 bytes to facilitate hashing             
    return packed.tobytes().hex()
     # The hashing function returns a 16-chatacter hexadecimal string which is used to compare if an image is similar to another via Hamming Distance

     
## Hamming Distance
def hamming(hex_a: str, hex_b: str) -> int:
    """
    This is the core function that tells us whether any two images are similar via computing the Hamming Distance between their pHashes (`hex_a` and `hex_b`). **Note:** *Hamming distance can onle be computed between two image pairs*

    Hamming Distance is a metric that measures how different two equal-length sequences (strings (i.e.- hashes), bit arrays, vectors, etc) are from each other. From the pHashes obtained:
        - 0 to 5 differing bits: Almost certainly the same image (or minor edits/compression).
        - 6 to 10 differing bits: Similar content with moderate edits.
        - greater than 12 differing bits: Completely different images

    The hashes are however not compared directly:
        1. The pHash hexadecimal strings for each image are first converted into integers 
        2. `^` (Exclusive OR (XOR) operation) then compares the integer values bitwise. 
        3. The comparison result is comverted into a binary representation via `bin()` (since `int` objects do not have a `count()` method. `str` objects do)
        4. If bits are the same (0), if different (1). The number of differences is the Hamming Distance.

    **Example**: Suppose we compare two 8-bit hex hashes: ``"a5"`` and ``"f5"``

    1. Convert to integer:

       - ``"a5"`` -> ``10100101``
       - ``"f5"`` -> ``11110101``

    2. Bitwise XOR::

         10100101  (a5)
        ^11110101  (f5)
        -----------
         01010000  (XOR int Result)

    3. Convert to binary ``str``:

       ``bin(...)`` -> ``'0b1010000'``

    4. Count the differing bits:

       ``count("1")`` -> ``2``

    **Result**: Hamming Distance of ``2``
    """
    return bin(int(hex_a, 16) ^ int(hex_b, 16)).count("1")

## ==================================================================================
## JPEG Quality Factor Estimation (How were the images encoded?)
_STD_LUMINANCE_Q = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ], dtype=np.float64,
)

def estimate_jpeg_quality(path: Path) -> int | None:
    """
    This function computes a metric known as the JPEG Quality Factor (QF) that was applied when the image was encoded.

    JPEG Quality Factor (QF) is a fingerprint of which pipeline produced the file. In practice, image data could come from various sources. If we scraped the AI-generated image set from one website and the "human" set from another website, then those two websites likely have two different re-encoding pipelines. The model may then try to "cheat" by finding the easiest seperable signal in the training distribution (i.e.- the encoder signal). This function allows us to mitigate this risk.

    Note that this only works for JPEG files. Non-JEPGs return `None`.

    JPEG QF is found by comparing the JPEG's quantization matrix against the standard `_STD_LUMINANCE_Q` quantization matrix defined above. JPEG QF is then estimated from this comparison.
    """

    try:
        with Image.open(path) as im:
            if im.format != "JPEG":
                return None  # For Non-JPEGs
            qtables= getattr(im, "quantization", None) # The JPEG's quantization table
            if not qtables:
                return None # IF no quantization table is found (by Pillow)
            table= np.asarray(qtables[0], dtype=np.float64).reshape(8, 8)
              # The JPEG's actual quantization matrix
    except Exception:
        return None

    # Coefficients saturated at 1 or 255 carry no information about S.
    mask = (table > 1) & (table < 255)
    if not mask.any():
        return 100 if table.mean() <= 1.5 else 1
 
    scale = ((100.0 * table[mask] - 50.0) / _STD_LUMINANCE_Q[mask]).mean()
    scale = float(np.clip(scale, 1e-6, None))
 
    quality = 100.0 - scale / 2.0 if scale < 100.0 else 5000.0 / scale
    return int(round(float(np.clip(quality, 1, 100))))
            
## ==========================================================================================
## Duplicate Grouping (An efficient method of grouping possibly similar images together)
class _UnionFind:
    """
    Draws from the Hamming Distance between any two images (See `hamming()` function within the same module). This class contructs a transitive closure where if Image A is similar to Image B and Image B is similar to Image C then A, B, C should belong in the same group (if A~B and B~C then A, B, C$).

    Extremely useful particularly when splitting images into training and evaluation sets as it prevents any data leakage caused by duplicate or near-duplicate images.
    """

    def __init__(self, n: int) -> None:
        self.parent= list(range(n)) 
          # Which group representative an image should belong to
          # e.g: if n=5 then self.parent= [0,1,2,3,4] where each values are groups

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]] 
              # Path compression to speed up lookup operations
            x= self.parent[x]
        return x 
          # Finds the root of item x
    
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
              # Connects the set containing a with the set containing b by setting one set's root as the parent of the other.

## Grouping function for similar pairs
def near_duplicate_pairs(hashes: Sequence[str], 
                         max_distance: int= 5, # 0-5 Hamming distance means images are similar
                         n_bands: int= 8) -> list[tuple[int, int, int]]:
    """
    This function finds pairs of images whose 64-bit perceptual hashes (``phash()``) differ by at most ``max_distance`` bits.

    Rather than simply performing pairwise comparisons between images that can result to millions of combinations, we implement Banded Locality-Sensitive Hashing (LSH) which is an algorithmic technique used to find approximate nearest neighbours or similar items in massive datasets.

    Banded LSH works by:
        1. Splitting each 64-bit hash integer into ``n_bands``
        2. By the Piegonhole Principle, if two hashes differ by at most ``max_distance`` bits, those differing bits can occupy at most ``max_distance`` of the ``n_bands`` bands, so at least one band must match be 100% identical between the two hashes (as long as ``n_bands > max_distance``).

    Returns a list of (``index_a``, ``index_b``, ``distance``) with ``index_a`` < ``index_b``
    """
    band_bits = 64 // n_bands
    buckets: list[dict[int, list[int]]] = [{} for _ in range(n_bands)]
 
    ints = [int(h, 16) for h in hashes]
    for idx, value in enumerate(ints):
        for b in range(n_bands):
            key = (value >> (b * band_bits)) & ((1 << band_bits) - 1)
            buckets[b].setdefault(key, []).append(idx)
 
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int, int]] = []
    for bucket in buckets:
        for members in bucket.values():
            if len(members) < 2 or len(members) > 5000:
                # Huge buckets are almost always degenerate images (flat black
                # frames). Skipping them keeps this from blowing up; they get
                # caught by the exact-hash check instead.
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b_ = members[i], members[j]
                    key = (a, b_) if a < b_ else (b_, a)
                    if key in seen:
                        continue
                    seen.add(key)
                    dist = bin(ints[a] ^ ints[b_]).count("1")
                    if dist <= max_distance:
                        pairs.append((key[0], key[1], dist))
    return pairs
 
 
def group_ids_from_pairs(n_items: int, pairs: Iterable[tuple[int, int, int]]) -> np.ndarray:
    """
    This function turns pairwise links into a clean array of group IDs, where each image index receives an integer label. 

    Output shape: (n_items,) of int. Items with no matches get their own
    singleton group. These labels become ``group_id`` in the manifest and are what you pass to a GroupShuffleSplit so that an entire duplicate cluster lands on one side of the train/test line.
    """
    uf = _UnionFind(n_items)
    for a, b, _ in pairs:
        uf.union(a, b)
    roots = np.array([uf.find(i) for i in range(n_items)])
    # Relabel to a dense 0..k-1 range so the ids are readable.
    _, dense = np.unique(roots, return_inverse=True)
    return dense
 
"""
Summary workflow for 5. Duplicate Grouping
[64-bit Perceptual Hashes]
           │
           ▼
[Split into 4x 16-bit Bands] ──> Bucket items by band value
           │
           ▼
[Compare within Buckets]    ──> Calculate Hamming distance via XOR
           │
           ▼
[Near-Duplicate Pairs]      ──> e.g., (0, 3), (3, 7), (12, 19)
           │
           ▼
[Union-Find Transitive Grouping] ─> Merges (0, 3) + (3, 7) into {0, 3, 7}
           │
           ▼
[Dense Re-indexing]         ──> Returns array of group IDs: e.g., [0, 1, 1, 0, 2...]
"""
