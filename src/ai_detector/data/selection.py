"""
This module turns raw metadata table into a leakage-safe set of splits. 

The key idea behind this module is to control for any experimental bias where the AI image detector may "cheat" its way to accuracy by simply because the AI-generated images happen to have different JPEG quality, dimensions, generators, or duplicated content than human-made images.

It relies on the image checks (i.e.- hashing, decodability, JPEG quality) computed using `integrity.py` and the sbsequent metadata manifest created by `manifest.py`.

```
Raw image collection
       │
       ▼
   integrity.py
       │
       ▼
Check files / hashes / grouping
       │
       ▼
   manifest.py
       │
       ▼
Create metadata manifest
       │
       ▼
  selection.py
       │
       ├── Normalize column names
       │
       ├── Match real-image metadata to fake-image metadata
       │
       ├── Measure metadata shortcuts
       │
       ├── Prevent train/test leakage
       │
       ├── Create train / validation / test / OOD
       │
       └── Optionally create a small pilot dataset
       │
       ▼
Final experimental dataset
```
"""

from __future__ import annotations
from typing import Any, cast, Sequence # Allows type hinting into Any datatype

from dataclasses import dataclass, field, fields
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import yaml
  # The experiment parameters are written directly into a yaml file

from .integrity import near_duplicate_pairs, group_ids_from_pairs
  # For checking for near-duplicates and assigning them into groups for images downloaded across multiple sources


## ==================================================================================
## Configuration
## ==================================================================================
@dataclass
class SubsetConfig:
    """
    Custom data structure for the dataset's and model's configuration parameters. Includes:
      - In distribution and out of distribution generators
      - Image matching parameters to ensure consistency across image dimensions and statistics
      - validation and in-distribution test set splits
      - Optional pilot dataset construction parameters
    """
    name: str # Name of the current config (e.g.- baseline, pilot, strict matching)
    seed: int = 42
    real_generator_token: str = "nature" 
      # We are sourcing the human-made images from GenImage. This is how they label them
    train_generators: list[str] = field(default_factory= list)
      # The AI generators implemented in the training data
      # field(default_factory= list) creates a brand new empty list for each class SubsetConfig object 
    ood_generators: list[str] = field(default_factory= list)
      # AI generators not part of the training set that are used for testing/evaluation
   
    # Image matching parameters (matching)
    min_side: int | None = None
    max_side: int | None = None
    jpeg_qf: int | None = None
    jpeg_qf_tolerance: int = 0 # Allows a small range around the JPEG QF value

    # Split fractions (Fractions of image groups rather than for individual images) (splits)
    val_fraction: float = 0.1 # 10% of groups becomes part of the validation set
    test_in_dist_fraction: float = 0.1 # 10% of groups becomes part of the in-distribution test set

    # (pilot)
    pilot_n_per_stratum: int | None = None
      # Allows us to construct a small version of the dataset
      # e.g.- `pilot_n_per_stratum = 50` means 50 images per generator/label/split combination

    
    # Constructing an object from a yaml file 
    @classmethod
    def from_yaml(cls, path: Path) -> SubsetConfig:
        """
        This method constructs the object from reading a yaml file in the `path` specified.
        """
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) 
          # where .safe_load() converts YAML into Python objects
          # This operation should return a dictionary of the SubsetConfig attributes and their values/elements for a particular SubsetConfig object

        # Extracting configuration sections
        matching = raw.pop("matching", {}) or {}
        splits = raw.pop("splits", {}) or {}
        pilot = raw.pop("pilot", {}) or {}
          # If the yaml contains these sections, their data will be extracted from the `raw` dictionary via `.pop()`. Otherwise an empty dictionary will be returned if the config section or data within the section is unavailable 

        merged= {** raw, # Populate this dictionary with the data obtained from `raw`
                 "min_side": matching.get("min_side"), 
                   # .get() returns the value for a dictionary key
                 "max_side": matching.get("max_side"),
                 "jpeg_qf": matching.get("jpeg_qf"),
                 "jpeg_qf_tolerance": matching.get("jpeg_qf_tolerance", 0),
                 "val_fraction": splits.get("val_fraction", 0.1),
                 "test_in_dist_fraction": splits.get("test_in_dist_fraction", 0.1),
                 "pilot_n_per_stratum": pilot.get("n_per_stratum"),
                }

        # The names of the fields declared in SubsetConfig class. This ensures only recognized fields are returned
        known= {f.name for f in cls.__dataclass_fields__.values()} 
        return cls(**{k: v for k, v in merged.items() if k in known})


## ==================================================================================
## Column Normalization
## ==================================================================================
CANONICAL_COLUMNS = {
    # canonical name -> candidate names seen in the wild as we may be importing images from various sources
    "generator": ["generator", "model", "source_model"],
    "width": ["width", "w", "img_width"],
    "height": ["height", "h", "img_height"],
    "jpeg_qf": ["compression_rate", "jpeg_qf", "quality", "qf", "quality_factor"],
    "content_class": ["content_class", "class_id", "class", "label_id", "wnid"],
    "path": ["path", "filepath", "file", "filename", "image_path"],
    "split": ["split", "subset", "partition"],
}

def normalize_columns(df: pd.DataFrame, extra: dict[str, str] | None =  None) -> pd.DataFrame:
    """
    Renaming source-specific columns into consistent canonical names. 

    It receives a `DataFrame` and optionally, a manually specified mapping (`extra`) of `dict` type. 

    To manually define mappings add to the `extra` parameter a dictionary with the column name to be changed as key and the intended column name as value for instance:
    ```python
    extra = {"model_generator":"generator"}
    ```
    
    If no `extra` mappings are specified, the function will perform a default mapping for any of the below cases found:
    ```python
    CANONICAL_COLUMNS = {
    # canonical name -> candidate names seen in the wild as we may be importing images from various sources
    "generator": ["generator", "model", "source_model"],
    "width": ["width", "w", "img_width"],
    "height": ["height", "h", "img_height"],
    "jpeg_qf": ["compression_rate", "jpeg_qf", "quality", "qf", "quality_factor"],
    "content_class": ["content_class", "class_id", "class", "label_id", "wnid"],
    "path": ["path", "filepath", "file", "filename", "image_path"],
    "split": ["split", "subset", "partition"],
    }
    ```

    """

    mapping: dict[str, str] = {}
    lowered= {c.lower(): c for c in df.columns}
      # this becomes a dictionary with the lowercase name as key and the original name as a value (e.g.- "width": "WIDTH")
    for canonical, candidates in CANONICAL_COLUMNS.items(): # For every canonical name.....
        for cand in candidates: # .....It checks the candidate names....
            if cand in lowered and lowered[cand] != canonical: 
                # ....and if the candidate name is found but is not a canonical name.....
                mapping[lowered[cand]] = canonical 
                  # ....the candidate name is renamed to fit the canonical name
                  # Returns something like: "img_height":"height"
                break
    if extra:
        mapping.update(extra) 
          # This occurs after the above canpnical mapping so whatever is specified in extra takes precedence
    return df.rename(columns= mapping) 
      # Canonical names are applied to the image metadata dataset



## ==================================================================================
## Bias Matching
## ==================================================================================
def apply_matching(df: pd.DataFrame, cfg: SubsetConfig)-> pd.DataFrame:
    r"""
    This function restricts real images so their metadata occupies approximately the same region as the generated images.

    Bias matching is a data filtering technique designed to eliminate shortcut learning. Generative AI models output images with rigid, predictable metadata: exact canvas dimensions (e.g.- 1024x1024) and consistent JPEG Quality Factors (QF). In contrast, real photos come in thousands of random resolutions and compression levels. If a raw dataset is fed to a deep learning model, the model neural network may quickly learn via a shortcut rule.

    Bias matching therefore forces real and fake images to share the exact same metadata profile, thereby eliminating any predictive signal from image metadata.

    Generative models cannot easiy change their output compression during dataset collection. Also, they are very consistent in their dimensions and encoding. Therefore we apply an asymetric bias matching approach:
      1. AI-generated images are left untouched as they already occupy a narrow, constrained band of space
      2. Any human-made images that fall outside the (`size` and `jpeg_qf` range) occupied by the fakes are removed
      3. A simple classifier is trained on metadata alone. If we get an accuracy close to 50% (akin to a random guess) shortcut learning is successfully eliminated. The higher the accuracy score, the more bias is inherent in the metadata (see `shortcut_probe()`)

    ****Consequence****: It is possible to remain with far less human-made images.

    Params:
      df (pd.DataFrame): The metadata/image dataset. Ensure columns `generator`, `height`, `width`, and `jpeg_qf` are present.
      cfg (class SubsetConfig): The bias matching configuration parameters. See `class SubsetConfig` for the structure. Can take in a `yaml` file in this structure. Note that when restricting real images per dimension, if `min_side` and `max_side` in the `.yaml` file is `None`, then `apply_matching()` defaults to restricting the real images per modal height and width for each generator.

    Returns:
     matched_df (pd.DataFrame): A dataframe with real and AI-generated images that are consistent across dimensions and JPEG quality factor (JPEG QF).
    """

    # Identifying real images
    is_real= df["generator"] == cfg.real_generator_token # Returns Boolean
    keep= pd.Series(True, index= df.index)

    ## This part of the code proposes restricting the reals to the modal dimensions of the AI generated images. We found it to produce far too few real images however.

    if cfg.min_side is None and cfg.max_side is None:
        # Per-generator modal (width, height) matching
        accepted_sizes: set[tuple[int, int]] = set()
        for _, g in df.loc[~is_real].groupby("generator"):
            modal_size = cast(
                tuple[int, int],
                g[["width", "height"]].value_counts().idxmax(),
            )
            accepted_sizes.add(modal_size)
              # value_counts().idxmax() finds the most frequent (width, height) pair
              # rather than the mode of width and height independently, which could
              # recombine into a size the generator never actually produced

        real_pairs = pd.Series(list(zip(df["width"], df["height"])), index=df.index)
        matches_mode = real_pairs.isin(accepted_sizes)
        keep &= ~is_real | matches_mode

    # Minimum size (to be retained)
    if cfg.min_side is not None:
        keep &= ~is_real | ((df["width"] >= cfg.min_side) & (df["height"] >= cfg.min_side))

    # Maximum size
    if cfg.max_side is not None:
        keep &= ~is_real | ((df["width"] <= cfg.max_side) & (df["height"] <= cfg.max_side))

    # JPEG QF
    if cfg.jpeg_qf is not None and "jpeg_qf" in df.columns:
        lo = cfg.jpeg_qf - cfg.jpeg_qf_tolerance
        hi = cfg.jpeg_qf + cfg.jpeg_qf_tolerance
        keep &= ~is_real | df["jpeg_qf"].between(lo, hi)
          # Keeps images with JPEG QFs that are within a range
 
    return df[keep].copy() # Returns the filtered dataset with real images restricted

def shortcut_probe(df: pd.DataFrame, 
                   feature_cols: tuple[str, ...] = ("width", "height", "jpeg_qf"), 
                   n_folds: int = 5,
                   scoring: str = 'accuacy', 
                    # Default for balanced classes.
                   seed: int = 0) -> float:
    """
    This function checks whether real and AI-generated images can be distinguished without looking at the image itself. 

    A simple decision tree is fit on metadata only and returns 5-fold CV accuracy. 
      - ~0.50  -> metadata carries no label information. This is the score we want to achieve.
      - ~0.95  -> a model can 'solve' your benchmark without vision at all. Shortcut learning

    Params:
      df (pd.DataFrame): The image metadata dataset
      feature_cols (tuple[str, ...]): The input features to the simple classifier. Default features are `("width", "height", "jpeg_qf")` therefore ensure these are present in your `df` DataFrame otherwise specify the features explicitly.
      n_folds (int): Number C-V folds created. **Default**: 5
      scoring (str): The classification scoring metric applied. Default is `accuracy` which is good for balanced classes and also because we care about performance on every class. If data is imbalanced, `balanced_accuracy` is recommended. 
      seed (int): For reproducibility

    Returns:
      score (float): Average classification accuracy score across `n_folds`
    """

    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    cols= [c for c in feature_cols if c in df.columns]
    X= df[cols].fillna(-1).to_numpy() # Missing metadata is replaced with -1
    y= (df['generator'] != "nature").astype(int).to_numpy()
      # Real images: 0
      # AI generated (anything else): 1
    
    if len(np.unique(y)) < 2:
        return float("nan")
      # In the event the DataFrame contains only one type of image (either only real or AI-generated), "nan" is returned
    clf= DecisionTreeClassifier(max_depth= 3, random_state= seed)
    return float(cross_val_score(clf, X, y, cv= n_folds, n_jobs= -1, scoring= scoring).mean())
      # Returns average accuracy score across 5 corss-validation folds
      # We use a simple classifier as the goal is to identify where an obvious metadata shortcut is present


## ==================================================================================
## Image Data Splitting
## ==================================================================================
def assign_genimage_splits(df: pd.DataFrame, 
                           cfg: SubsetConfig, 
                           group_col: str = "group_id") -> pd.DataFrame:
    """
    Assigns the following train/validation/test splits:
      - `train`: Model training
      - `val`: For evaluation to find the best performing model
      - `test_in_dist`: Consists of images produced by the same generators that produced the images in the `train` set
      - `test_ood_genimage`: Images produced by completely new generators that were not in the `train` set

    Also handles how human-made images are split. The metadata manifest database should contain clusters of similar human-made and AI-generated images groups e.g.:
    ```
    group 123
      real original
      generated version A from generator X
      generated version B from generator Y
      generated version C from generator Z
    ```

    Basically, this function implements a group-aware splitting methodology where rather than splitting row-wise, it splits according to the number groups in the dataset. This therefore prevents the following:
      - Similar images from appearing in both training and val/test sets (data leakeage)
      - Real image duplication when specifiying the generators in the training and testing sets 

    Split order occurs as follows:
    ```
    OOD
    ↓
    test
    ↓
    validation
    ↓
    remaining → train
    ```
    """

    rng= np.random.default_rng(cfg.seed) # Deterministic randomness

    # Start with everything unassigned. The function then progressively assigns rows
    df= df.copy()
    df['split']= "unassigned"

    # Identify generator categories
    is_real= df['generator'] == cfg.real_generator_token # Real images
    is_ood_gen= df['generator'].isin(cfg.ood_generators) # OOD Generators
    is_train_gen= df['generator'].isin(cfg.train_generators) 
      # All generators eligible for train/val/test in distribution set
    
    # Collect real groups
    ood_test_split_real= 0.25 # Percentage of image groups to reserve for out of distribution testing
    real_groups = df.loc[is_real, group_col].dropna().unique() # Take a list of real groups
    rng.shuffle(real_groups) # Randomly shuffle them
    n_ood_real = int(len(real_groups) * ood_test_split_real)   
      # reserve a specified percentage and number of groups for OOD testing (real images)
    ood_real_groups = set(real_groups[:n_ood_real].tolist()) 
      # Set containing group Ids for real images for OOD testing
    df.loc[is_ood_gen, "split"] = "test_ood_genimage"
    df.loc[is_real & df[group_col].isin(ood_real_groups), "split"] = "test_ood_genimage"

    pool_mask = (is_train_gen | (is_real & ~df[group_col].isin(ood_real_groups)))
    pool_groups = df.loc[pool_mask, group_col].dropna().unique()
    rng.shuffle(pool_groups)
 
    n = len(pool_groups)
    n_test = int(n * cfg.test_in_dist_fraction)
    n_val = int(n * cfg.val_fraction)
    test_g = set(pool_groups[:n_test].tolist())
    val_g = set(pool_groups[n_test:n_test + n_val].tolist())
 
    df.loc[pool_mask & df[group_col].isin(test_g), "split"] = "test_in_dist"
    df.loc[pool_mask & df[group_col].isin(val_g), "split"] = "val"
    df.loc[pool_mask & (df["split"] == "unassigned"), "split"] = "train"
    return df
 

def stratified_pilot(
    df: pd.DataFrame,
    n_per_stratum: int,
    strata: tuple[str, ...] = ("split", "label", "generator"),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Deterministically down-sample the manifest for pipeline development.

    For development speed, it creates a smaller dataset while preserving important categories.

    A stratum is a subgroup defined by some combination of variables i.e.: `("split", "label", "generator")` which means groups such as:
    ```
    (train, real, nature)
    (train, fake, stable_diffusion)
    (train, fake, dalle)

    (val, real, nature)
    (val, fake, stable_diffusion)

    (test_in_dist, real, nature)
    ...
    ```
    The pilot takes up to `n_per_stratum` examples from each.
 
    Stratifying on (split, label, generator) guarantees the pilot is a
    miniature of the full set rather than an accidental pile of one generator.
    """
    cols = [c for c in strata if c in df.columns]
    return (
        df.groupby(cols, group_keys=False, observed=True)
          .apply(lambda g: g.sample(min(len(g), n_per_stratum), random_state=seed))
          .reset_index(drop=True))


## ==================================================================================
## Cross-Source Manifest Combinaton
## ==================================================================================
def combine_manifests(paths: Sequence[Path]) -> pd.DataFrame:
    """
    Loads every per-source manifest parquet and concatenates them into one DataFrame.

    Params:
      paths (Sequence[Path]): parquet files, one per `build_manifest.py` run.

    Returns:
      pd.DataFrame: concatenated manifest, columns normalized via `normalize_columns` index reset (row order across sources is otherwise meaningless and stale per-fil indices would collide).
    """

    frames= []
    for p in paths:
        df= pd.read_parquet(p)
        df= normalize_columns(df)
        frames.append(df)
        
    combined= pd.concat(frames, ignore_index= True)
    return combined

## ==================================================================================
## Cross-Source Near-Duplicate Grouping
## ==================================================================================
def assign_group_ids(df: pd.DataFrame, phash_col: str = "phash", 
    max_distance: int =5, n_bands: int = 4) -> pd.DataFrame:
    """
    Runs banded-LSH near-duplicate clustering (see `integrity.near_duplicate_pairs`) over the FULL combined manifest and writes the resulting `group_id` column.

    This must run on the combined manifest, not per-source, because a duplicate cluster crossing between image soruces (i.e.- the same underlying photo appearing in two datasets) is the kind of leakage that should be avoided if we are to perform an out-of-distribution test.

    Rows with missing/empty phash (typically `is_corrupt == True`) are given their own singleton group so they never accidentally cluster with a valid image just because both hashes are blank.

    Params:
      df (pd.DataFrame): combined manifest with a `phash_col` column of 16-hex-char strings. `max_distance` and `n_bands` are forwarded to `near_duplicate_pairs`; keep these equal to `dedup.phash_max_distance` / `dedup.n_bands` in the `subset_v1.yaml` config so the image EDA notebook and any script agree.
    
    Returns:
      pd.DataFrame: copy of df with an integer `group_id` column added.
    """

    df = df.copy().reset_index(drop=True)
    has_hash = df[phash_col].notna() & (df[phash_col] != "")

    hashes = df.loc[has_hash, phash_col].tolist()
    pairs = near_duplicate_pairs(hashes, max_distance=max_distance, n_bands=n_bands)
    local_groups = group_ids_from_pairs(len(hashes), pairs)

    df["group_id"] = -1  # placeholder; -1 rows get singleton ids below
    df.loc[has_hash, "group_id"] = local_groups

    # Corrupt / hash-less rows: give each its own unique group id so they
    # can never silently cluster with anything.
    next_id = int(df["group_id"].max()) + 1
    n_missing = int((~has_hash).sum())
    df.loc[~has_hash, "group_id"] = np.arange(next_id, next_id + n_missing)
    return df


## ==================================================================================
## Full-Dataset Split Assignment
## ==================================================================================
def assign_full_splits(df: pd.DataFrame, cfg: SubsetConfig,
    group_col: str = "group_id",
    genimage_sources: tuple[str, ...] = ("genimage", "unbiased_genimage"),
    generator_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Extends `assign_genimage_splits` to the whole combined, multi-source
    manifest.

    GenImage / Unbiased GenImage rows go through the group-aware, generator-aware logic in `assign_genimage_splits` exactly as before. Every other source is a dedicated held-out evaluation set by *design*, so its split is a fixed lookup, never a computed fraction:

        coco  -> test_ood_real               (false-positive rate on unseen reals)
        ntire -> test_wild                   (blind: unknown generators + transforms)
        raise -> test_ood_real_uncompressed  (hardest compression shift)

    Params:
      df: combined manifest with `source`, `generator`, `group_id` columns.
      cfg: SubsetConfig (same one driving `assign_genimage_splits`).
      genimage_sources: which `source` values are routed through the generator-based logic; everything else falls through to the fixed lookup table.
      generator_aliases: optional lookup from a dataset's unique generator names to the names used in `cfg.train_generators` / `cfg.ood_generators`.

    Returns:
      pd.DataFrame: copy of df with `split` fully assigned. Raises nothing itself — always follow this with `assert_no_leakage`.
    
    Raises:
      ValueError: Raised rather than silently leaving rows `unassigned` if a source appears that neither branch recognizes. This ensures all sources are accounted for.
    """
    df = df.copy()
    df["split"] = "unassigned"

    is_genimage = df["source"].isin(genimage_sources)
    genimage_df = df[is_genimage].copy()
    if generator_aliases:
        genimage_df["generator"] = genimage_df["generator"].replace(generator_aliases)
    genimage_part = assign_genimage_splits(genimage_df, cfg, group_col=group_col)
    df.loc[genimage_part.index, "split"] = genimage_part["split"]

    fixed_lookup = {
        "coco": "test_ood_real",
        "ntire": "test_wild",
        "raise": "test_ood_real_uncompressed",
    }
    for source_name, split_name in fixed_lookup.items():
        df.loc[df["source"] == source_name, "split"] = split_name

    unresolved = (df["split"] == "unassigned").sum()
    if unresolved:
        bad_sources = df.loc[df["split"] == "unassigned", "source"].unique().tolist()
        raise ValueError(
            f"{unresolved} rows have no split logic (unknown sources: {bad_sources}). "
            "Add them to genimage_sources or fixed_lookup."
        )
    return df


## ==================================================================================
## Real Image Pooling and Redistribution
## ==================================================================================
def redistribute_real_images(df: pd.DataFrame,
                             cfg: SubsetConfig,
                             group_col: str = "group_id",
                             label_col: str = "label",
                             source_col: str = "source",
                             genimage_sources: tuple[str, ...] = ("genimage", "unbiased_genimage", "genimage_unbiased"),
                             ) -> pd.DataFrame:
    """
    
    This function redistributes real images across remaining AI generators following the removal of images below a certain size/resolution threshold (see ``metadata_EDA.ipynb``).

    Pools every real GenImage image and randomly redistributes them across the configured generators so that each generator has (as close as possible to) as many real images as fake ones, thereby ensuring a balanced daatset.

    In GenImage, a real image's `generator` is merely the folder it shipped in. Real images are exchangeable across folders, so reals from generators that were discarded (not in `cfg.train_generators` or `cfg.ood_generators`) are valid extra reals for the generators we keep.

    The redistribution is done over GROUPS (see `assign_group_ids`), never over rows, so near-duplicates always travel together. Because the reals take on a configured generator name, the later `assign_genimage_splits` call routes them through the exact same in-distribution (train/val/test_in_dist) or OOD path as that generator's fakes. No real group can therefore end up on both sides of a split boundary.

    Only GenImage rows (`source_col` in `genimage_sources`) are touched. Rows from any other source are returned unchanged. Fakes are NEVER dropped or relabelled beyond canonical spelling; the only rows ever removed are surplus reals beyond what is needed to balance.

    Steps:
      1. Fakes from configured generators get the spelling used in the config (matching is case-insensitive i.e.- `Midjourney` == `midjourney`). Fakes from generators outside the config (expected to have been dropped already) are left untouched and flagged in the `unconfigured_fake` column, with a warning if any exist
      2. Real groups that also contain a kept fake (cross-label near-duplicates) are pinned to that fake's generator, as both must land in the same split
      3. The remaining real groups are shuffled (`cfg.seed`) and each is given to the generator with the largest outstanding real deficit
      4. Surplus real groups, once every generator is balanced, are dropped (the only rows removed)

    Params:
      df (pd.DataFrame): GenImage-only metadata with `generator`, `group_col` and `label_col` (0= real, 1= AI-generated). If `label_col` is absent, reals are identified using `cfg.real_generator_token`. Call this after `assign_group_ids` and (typically) after `apply_matching`.
      cfg (SubsetConfig): Supplies the generators, `seed` and the real token.
      group_col (str): Near-duplicate group column
      label_col (str): Binary label column
      source_col (str): Column identifying the dataset source. If absent, every row is treated as GenImage.
      genimage_sources (tuple[str, ...]): `source_col` values considered GenImage.

    Returns:
      pd.DataFrame: A copy where GenImage reals carry a configured generator name (surplus reals removed) and all fakes and non-GenImage rows are retained. The original folder is kept in a new `source_folder` column and `unconfigured_fake` flags fakes from generators outside the config.

    Raises:
      ValueError: If a configured generator has no fake images.

    Warns:
      If there are too few real images to match a generator's fakes. The shortfall is reported rather than filled by oversampling.
    """
    rng = np.random.default_rng(cfg.seed)
    canon = {g.lower(): g for g in [*cfg.train_generators, *cfg.ood_generators]}

    full = df.copy()
    is_gi = full[source_col].isin(genimage_sources) if source_col in full.columns else pd.Series(True, index=full.index)
    others = full[~is_gi].copy()
    df = full[is_gi].copy()

    gen_key = df["generator"].astype(str).str.lower()
    if label_col in df.columns:
        is_real = df[label_col] == 0
    else:
        is_real = gen_key == cfg.real_generator_token.lower()
    is_fake_kept = ~is_real & gen_key.isin(canon)

    df["unconfigured_fake"] = ~is_real & ~is_fake_kept
    if df["unconfigured_fake"].any():
        extra = sorted(df.loc[df["unconfigured_fake"], "generator"].astype(str).unique())
        warnings.warn(f"Fakes from generators outside the config are present (not balanced against): {extra}")

    df["source_folder"] = df["generator"]
    df.loc[is_fake_kept, "generator"] = gen_key[is_fake_kept].map(canon)

    # Number of real images each generator needs in order to match its fakes
    target = df.loc[is_fake_kept, "generator"].value_counts().reindex(list(canon.values()), fill_value=0)
    empty = target[target == 0].index.tolist()
    if empty:
        raise ValueError(f"No fake images found for configured generators: {empty}")
    assigned = {g: 0 for g in target.index}

    real_sizes = df.loc[is_real].groupby(group_col).size()
    group_to_gen: dict[Any, str] = {}

    # Real groups sharing a cluster with a kept fake are pinned to that fake's generator
    fake_group_gen = (
        df.loc[is_fake_kept].groupby([group_col, "generator"]).size().rename("n").reset_index()
          .sort_values([group_col, "n"], ascending=[True, False])
          .drop_duplicates(group_col).set_index(group_col)["generator"]
    )
    pinned = real_sizes.index.intersection(fake_group_gen.index)
    for gid in pinned:
        g = fake_group_gen[gid]
        group_to_gen[gid] = g
        assigned[g] += int(real_sizes[gid])

    # Everything else is shuffled and handed to whichever generator is furthest from its target
    free = real_sizes.index.difference(pinned).to_numpy().copy()
    rng.shuffle(free)
    gens = list(target.index)
    for gid in free:
        deficits = np.array([target[g] - assigned[g] for g in gens])
        if deficits.max() <= 0:
            break
        g = gens[int(deficits.argmax())]
        group_to_gen[gid] = g
        assigned[g] += int(real_sizes[gid])

    short = {g: int(target[g] - assigned[g]) for g in gens if assigned[g] < target[g]}
    if short:
        warnings.warn(f"Not enough real images to balance generators (missing reals): {short}")

    real_new_gen = df[group_col].map(group_to_gen)
    keep_real = is_real & real_new_gen.notna()
    df.loc[keep_real, "generator"] = real_new_gen[keep_real]
    df = df[~is_real | keep_real]
    if not others.empty:
        others["source_folder"] = others["generator"]
        others["unconfigured_fake"] = False
        df = pd.concat([df, others]).sort_index()
    return df