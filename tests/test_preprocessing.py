"""
Tests for src/ai_detector/preprocessing/image_ops.py

Run with: ``pytest -q tests/test_preprocessing.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_detector.preprocessing.image_ops import (
    PreprocessConfig, resize_and_reencode, process_one, build_processed_dataset,
)


@pytest.fixture
def cfg() -> PreprocessConfig:
    return PreprocessConfig(image_size=64, jpeg_qf=90)


def _make_image(w: int, h: int, mode: str = "RGB") -> Image.Image:
    arr = np.random.default_rng(0).integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    im = Image.fromarray(arr)
    return im.convert(mode)


## Shape / geometry -----------------------------------------------------
def test_output_is_exact_square(cfg):
    im = _make_image(300, 150)  # wide rectangle
    out = resize_and_reencode(im, cfg)
    assert out.size == (cfg.image_size, cfg.image_size)


def test_handles_portrait_and_landscape(cfg):
    for w, h in [(300, 150), (150, 300), (64, 64), (63, 65)]:
        out = resize_and_reencode(_make_image(w, h), cfg)
        assert out.size == (cfg.image_size, cfg.image_size)


def test_non_rgb_modes_are_converted(cfg):
    gray = _make_image(200, 200, mode="L")
    out = resize_and_reencode(gray, cfg)
    assert out.mode == "RGB"  # every output must have exactly 3 channels


## Disk round-trip --------------------------------------------------------
def test_process_one_writes_expected_qf(tmp_path, cfg):
    src = tmp_path / "src.png"
    _make_image(200, 200).save(src)
    dst = tmp_path / "out" / "img.jpg"

    result = process_one((src, dst, cfg, True))
    assert result["processed_ok"] is True
    assert dst.exists()

    with Image.open(dst) as im:
        assert im.format == "JPEG"
        assert im.size == (cfg.image_size, cfg.image_size)


def test_process_one_reports_failure_without_raising(tmp_path, cfg):
    missing_src = tmp_path / "does_not_exist.jpg"
    dst = tmp_path / "out.jpg"
    result = process_one((missing_src, dst, cfg, True))
    assert result["processed_ok"] is False
    assert result["processed_error"] is not None
    assert not dst.exists()


## Batch driver -------------------------------------------------------------
def test_build_processed_dataset_layout(tmp_path, cfg):
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    raw_root.mkdir()

    rows = []
    for i, (label, split) in enumerate([(1, "train"), (0, "train"), (1, "val")]):
        fname = f"img_{i}.png"
        _make_image(120, 120).save(raw_root / fname)
        rows.append({"path": fname, "label": label, "split": split, "sha256": f"hash{i}"})
    df = pd.DataFrame(rows)

    out = build_processed_dataset(df, raw_root, processed_root, cfg, n_workers=2)

    assert out["processed_ok"].all()
    assert (processed_root / "train" / "ai" / "hash0.jpg").exists()
    assert (processed_root / "train" / "human" / "hash1.png").exists()
    assert (processed_root / "val" / "ai" / "hash2.jpg").exists()

def test_real_images_saved_as_lossless_png(tmp_path, cfg):
    src = tmp_path / "src.jpg"
    _make_image(200, 200).save(src, quality=96)
    dst = tmp_path / "out" / "img.png"
    assert process_one((src, dst, cfg, False))["processed_ok"] is True
    with Image.open(dst) as im:
        assert im.format == "PNG"
        assert im.size == (cfg.image_size, cfg.image_size)
