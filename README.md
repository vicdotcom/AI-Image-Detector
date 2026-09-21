# AI Image Detection
As artificial intelligence advances, the boundary between authentic and synthetic imagery is becoming increasingly difficult to distinguish. This is an end-to-end image classifier that detects whether images are **AI-generated** or **human-made**. Currently a work-in-progress.

**Current stage:** Image data download and validation (i.e.- Checking for corrupted images, recording image metadata: dimensions, JPEG quality, image source, specific AI generator, real/AI labels, etc....) pipeline is complete. Metadata-level EDA and bias-matching are complete, and image-level EDA (near-duplicate clustering across all sources) is done. Currently working on image preprocessing and train/validation/test splitting prior to employing a deep learning model for training and evaluation.


## Table of Contents
- [AI Image Detection](#ai-image-detection)
  - [Table of Contents](#table-of-contents)
  - [1. Problem Definition](#1-problem-definition)
  - [2. Dataset Strategy](#2-dataset-strategy)
    - [Image Sources](#image-sources)
    - [Bias-Matching](#bias-matching)
    - [Preprocessing Strategy](#preprocessing-strategy)
    - [Image integrity checks (`Integrity.py`)](#image-integrity-checks-integritypy)
    - [Manifest Construction (`manifest.py`, `build_manifest.py`)](#manifest-construction-manifestpy-build_manifestpy)
    - [Splitting Philosophy](#splitting-philosophy)
  - [3. Project Structure (So far)](#3-project-structure-so-far)
  - [4. Setup](#4-setup)
  - [5. Usage](#5-usage)
  - [6. References](#6-references)


## 1. Problem Definition

We may want to build a model that distinguishes between the following image types:

- **Fully synthetic:** text-to-image output - Stable Diffusion,
  Midjourney, DALL·E, etc.
- **Human-authored:** camera photos, scanned art, hand-made
  digital illustration.


<!-- - **AI-edited / hybrid** (in-painting, generative fill, upscaling, style
  transfer): This is currently out of the scope of this project, though robustness to these image types will be evaluated at a later stage. -->

<!-- Formally, this is binary classification problem where: given a pixel tensor $x \in \mathbb{R}^{H \times W \times 3}$ (i.e.- the numerical representation of an image), we predict whether the image is human-made or AI-generated ($y \in \{\text{human, AI}\}$)
using a probabilistic estimate within range $[0,1]$ where values closer to 1 indicate a higher likelihood of being AI-generated. -->

This is however not a straight-forward task. The model, rather than distinguish genuine image artifacts, may instead "cheat" and use secondary image characteristics perform the classification: 

  - A model trained on one generator family (e.g.- Midjourney) tends to learn that family's fingerprint rather than "AI-ness" in general, so accuracy can collapse on an unseen generator (e.g.- DALL-E), a form of [distribution shift](https://parasdahal.com/notes/distribution-shift/).
  - If the images from each class differ systematically in resolution/format/compression, etc..., a model can learn *that* instead (shortcut learning)
  - Image duplicates or near duplicates can present a form of data leakage if they are spread between train/validation/test splits

We aim to produce the best probabilistic estimate from a model fit to a specific distribution. That is: *image is likely AI-generated (model score 0.91)*. Our objective and scope for the project is therefore as follows: 
> Build a binary image classifier that, given a single still image, outputs a calibrated probability that the image was fully synthesized by a generative model.

> In building the classifier, we also construct an image download and preprocessing pipeline that actively minimizes the abovementioned problems. Minimizing shortcut signals leads to improved image classification accuracy [Grommelt et al. (2024)](https://arxiv.org/abs/2403.17608).

## 2. Dataset Strategy

### Image Sources
Candidate sources include:
- **[GenImage](https://github.com/gendetection/UnbiasedGenImage)** (*Primary image source*) - ~1M
real (ImageNet) / fake pairs across 8 generators, with deliberate bias controls (matched sizes, controlled JPEG compression).
- **[NTIRE Robust AIGen Detection](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train)** - 42 generators, unlabeled augmentations. Used later as a true wild/out-of-distribution test.
-  **[COCO](https://cocodataset.org/#overview)** - solely real images, used to assess the false-positive rate on an unseen real-image source.
- **[RAISE](https://loki.disi.unitn.it/RAISE/)** - uncompressed RAW-derived images; the hardest real-image shift.

**Metadata-level EDA (`01_metadata_EDA.ipynb`):** before downloading actual image, the [GenImage metadata CSV](https://dataverse.harvard.edu/file.xhtml?fileId=9659368&version=2.0) (dimensions, generator, JPEG quality, class label) is analyzed on its own. This is what makes it possible to plan a dataset subset and catch shortcut learning risks without touching the images themselves.


### Bias-Matching

Bias-matching is a data filtering technique designed to eliminate shortcut learning. A generative AI model can output images with distinct metadata: exact canvas dimensions (e.g.- 1024x1024) and consistent JPEG Quality Factors (QF), while *in contrast*, real photos come in thousands of random resolutions and compression levels. If a raw dataset is fed to a deep learning model, the network may quickly exploit these shortcut signals to classify images.

We filter for real and fake images that share the same metadata profile, thereby eliminating predictive signal from image metadata (`01_metadata_EDA.ipynb`). The bias-matching applied is asymmetric: AI-generated images are left untouched while real images that do not fit the metadata profile are filtered out. This is because AI-generated images occupy a narrower band of metadata space than real images, as they are constrained to their specific generators.

**Findings from the GenImage metadata (2,681,150 images, 9 generators/classes):**
- All AI images are PNGs (QF = 100) with generator-specific sizes: 128x128 (BigGAN), 256x256 (ADM, GLIDE, VQDM), 512x512 (Stable Diffusion v1.4/v1.5, Wukong) and 1024x1024 (Midjourney). Real (ImageNet, `nature`) images vary widely in size (modal 500x375) and quality.
- The modal real-image QF is **96** (903,382 of 1,331,167 real images, 67.86%), so matching is done at QF = 96 with zero tolerance, with real-image sides restricted to 450-550 px.
- Asymmetric matching retains only **41,753 of 1,331,167 real images (3.14%)**, so a large starting pool is paramount.
- Only the generators producing ~512x512 images are kept (Stable Diffusion v1.4, v1.5, Wukong, Midjourney), leaving 677,994 AI images (50.22%). These are then undersampled to balance the classes and equalise images per generator: **83,505 images (41,753 human / 41,752 AI, 10,438 per AI generator)**.
- After matching, image `width`/`height` no longer carry label information, but `jpeg_qf` still fully determines the label (feature importance 1.0) since every AI image remains a QF = 100 PNG while every real image is QF = 96. This signal is removed at the preprocessing stage below.

![alt text](image.png)

The matched, balanced selection is exported to `data/interim/genimage_matched_balanced.parquet` and used as the index for downloading only those images.

### Preprocessing Strategy

Following [Grommelt et al. (2024)](https://arxiv.org/abs/2403.17608), the residual dimension/compression differences are normalized directly:
- **Re-encode the AI images at JPEG QF = 96** to match the real images' compression.
- **Content balancing:** *"We then sampled the same number of generated images for each 512x512 generator. To avoid disparities in content distribution between natural and generated images, we ensured an equal number of natural and generated images per ImageNet class."*
- Crop/resize both real and AI images to a uniform 512x512 so a transform is never applied to only one class.

### Image integrity checks (`Integrity.py`)

Every downloaded image is validated before being added to a manifest (see below [Manifest Construction](#manifest-construction) section) with validation methods including: 
  - **Corruption/truncation probing**- Checks whether the image files van be decoded (i.e.- Images can be opened and their metadata can be read)
  - **Duplicate checking**- Done be generating a unique SHA-256 hash per image. Exact image duplicates have the same hash.
  - **Perceptual hashing**- A unique perceptual hash is generated per image for detecting similar looking images. 
  - **JPEG quality estimation**- Computing the JPEG QF values of images. Crucial for image preprocessing and bias reduction.  

See `integrity.py` for full method descriptions.

### Manifest Construction (`manifest.py`, `build_manifest.py`)

A manifest is simply a table dataset of per-image metadata and labels (one row per image) that downstream steps (bias-matching, preprocessing, splitting) read and write rather than touching raw files. It's structured as follows:

| image_path | source | generator | label | width | height | jpeg_qf | sha256 |
|---|---|---|---|---|---|---|---|
| data/raw/genimage/sd_v1_4/000123.jpg | genimage | stable_diffusion_v_1_4 | fake | 512 | 512 | 92 | a1b2c3... |
| data/raw/coco/000456.jpg | coco | real | real | 640 | 480 | 88 | d4e5f6... |

The per-source outputs of the download, integrity and manifest recording steps (GenImage, Tiny GenImage, NTIRE, COCO, RAISE) are then combined into one unified manifest, which is the single input where everything downstream (i.e.- preprocessing, splitting, train/val/test) is built from.

### Splitting Philosophy

Naive random splitting leaks information via image duplicates/near-duplicates, similar content groups (e.g.- Images of a dog, cars, tables, planes, e.t.c.), and generator or source-specific signatures. Splits here are instead group-aware (duplicate clusters never cross a split), stratified by class/generator/content category, and include held-out-generator test sets so cross-generator generalization and not just in-distribution accuracy, gets measured. The manifest is split as follows:

| Split | Sources / Generators |
|---|---|
| `train` | stable_diffusion_v_1_4, stable_diffusion_v_1_5, wukong |
| `test_ood_genimage` | midjourney |
| `test_wild` | NTIRE |
| `test_ood_real` | COCO |
| `test_ood_real_uncompressed` | RAISE |

Within the `train` pool itself, a further **group-aware train/val/test_in_dist split** is applied: images are first clustered by near-duplicate/perceptual-hash groups (and, where applicable, paired real/fake origin), and whole groups - never individual images - are assigned to train, val, or test_in_dist. This prevents near-duplicate or paired images from appearing on both sides of a split, which would let the model "memorize" rather than generalize and would inflate validation/test metrics in a way that doesn't hold up on genuinely unseen data.

Starting scale is intentionally small to get the pipeline and evaluation trustworthy before scaling up to the full ~500GB GenImage dataset (or including additional image sources for training and testing).

## 3. Project Structure (So far)

```yaml
ai-image-detector/
├── README.md                          # this file
├── pyproject.toml                     # dependencies, dev tools (ruff, pytest), editable install
├── .gitignore                         # excludes data/, secrets, caches, notebook outputs
│
├── configs/
│   └── data/
│       └── subset_v1.yaml             # dataset selection specifications: in/out-of distribution generators, data splits, bias-matching thresholds
│
├── data/                              # gitignored - nothing here is committed
│   ├── raw/                           # immutable downloads (coco/, genimage/, ...)
│   ├── interim/                       # work-in-progress manifests, EDA parquet files
│   └── processed/                     # final train/val/test manifests (not yet produced)
│
├── notebooks/
│   ├── 01_metadata_EDA.ipynb # metadata-only EDA; decides dataset composition before downloading images
│   └── 02_image_EDA.ipynb       # image-level EDA: combines per-source manifests (138,805 images), drops corrupt/all-black images (29 removed), clusters near-duplicates and assigns splits
│
├── reports/
│   ├── figures/                       # plots obtained from notebook EDA
│   ├── manifests/                     # final train/val/test manifests
│
├── scripts/                           # for one-off tasks such as downloading metadata/images, building manifests, etc.
│   ├── setup_git.sh                   # one-time repo/branch initialization
│   ├── download_genimage_metadata.py  # fetches the small GenImage metadata CSV (not the images)
│   ├── download_images.py             # downloads actual image data: coco | genimage | genimage-subset | ntire | raise | plus download verification
│   └── build_manifest.py              # builds the dataset manifest from images on disk (next step)
│
├── src/ai_detector/                   # installable package - the reusable logic notebooks/scripts import
│   ├── data/
│   │   ├── integrity.py               # corruption checks, SHA-256, perceptual hashing, JPEG quality estimation
│   │   ├── manifest.py                # manifest schema (ImageRecord) and read/write logic
│   │   ├── selection.py               # applies configs/data/*.yaml: bias-matching, stratified sampling
│   │   ├── viz.py                     # reusable EDA plotting functions
│   │   └── download.py                # per-source download handlers used by scripts/download_images.py
│   └── utils/                         # logging, seeding, and other shared helpers
│
└── tests/
    └── test_data_integrity.py         # unit tests for image checking and manifest reading/writing modules in src
```


## 4. Setup

```bash
git clone <this-repo>
cd Ai_Image_Detector

python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -e ".[dev]"          # editable install + pytest/ruff/jupyterlab/HuggingFace

bash scripts/setup_git.sh        # one-time: branch setup (read it first)
pytest                           # confirm everything is wired correctly
```

`-e` (editable) means edits under `src/ai_detector/` take effect immediately
without reinstalling - verify with:

```bash
python -c "from ai_detector.data.integrity import phash; print('ok')"
```

## 5. Usage

To reproduce the project's current state from scratch, run these in order:

**Step 1: Download GenImage metadata (not the images yet)**

```bash
python scripts/download_genimage_metadata.py --dest data/raw/genimage_meta
```

Fetches the small metadata CSV describing GenImage's ~1M images (dimensions,
generator, JPEG quality) and writes a `provenance.json` recording the source
URL, SHA-256, and download time. This is what lets you plan a dataset subset
without touching the 500GB of actual images.

**Step 2: Metadata-level EDA**

```bash
jupyter lab notebooks/01_metadata_EDA.ipynb
```

Explores class/generator/size/compression distributions from the metadata
CSV, quantifies dataset shortcuts (e.g. how well a model could classify
real-vs-fake using *only* image dimensions and JPEG quality), and produces
`configs/data/subset_v1.yaml` - the dataset selection spec used downstream.

**Step 3 - Download the selected GenImage subset**

The full GenImage archive on [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FAKDIHF)
is ~654 GB split across 500 zip parts. Rather than downloading it, the notebook
exports the `matched_balanced` selection to
`data/interim/genimage_matched_balanced.parquet` and the downloader fetches only
those images using HTTP range requests:

```bash
# Quick dry run
python scripts/download_images.py genimage-subset --selection data/interim/genimage_matched_balanced.parquet --limit 50

# Full selection (re-running skips images already on disk)
python scripts/download_images.py genimage-subset --selection data/interim/genimage_matched_balanced.parquet
```

The first run downloads the archive's ~380 MB central directory (cached in
`data/raw/genimage/_cache/`). Other sources (`coco`, `ntire`, `raise`) follow the
same CLI pattern; run `python scripts/download_images.py --help` for details, and
`python scripts/download_images.py verify` to check what's on disk.

Every download handler writes a `provenance.json` alongside the data,
recording what was downloaded, when, and its checksum - commit these files
(they're small); actual images (`data/`) are gitignored.

## 6. References

- Grommelt, P., Weiss, L., Pfreundt, F.-J., & Keuper, J. (2024). *Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets*. arXiv:2403.17608. https://arxiv.org/abs/2403.17608

  ```bibtex
  @misc{grommelt2024fakejpegrevealingcommon,
        title={Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets}, 
        author={Patrick Grommelt and Louis Weiss and Franz-Josef Pfreundt and Janis Keuper},
        year={2024},
        eprint={2403.17608},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2403.17608}, 
  }
  ```
