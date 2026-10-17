"""Unit tests and acceptance tests for Part 1: Data Collection & Curation."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from vidgen.data.manifest import (
    MANIFEST_COLUMNS,
    create_empty_manifest,
    read_manifest,
    write_manifest,
)
from vidgen.data.moving_mnist import MovingMNISTGenerator
from vidgen.data.curation import (
    CurationConfig,
    compute_laplacian_sharpness,
    compute_motion_score,
    detect_black_borders,
    compute_dhash,
    hamming_distance,
    estimate_aesthetic_score,
    run_curation_pipeline_end_to_end,
)
from vidgen.data.dataset import CuratedVideoDataset
from vidgen.data.buckets import AspectRatioBucketSampler


def test_manifest_schema_and_io(tmp_path):
    """Verify manifest creation and atomic Parquet read/write."""
    empty_df = create_empty_manifest()
    assert list(empty_df.columns) == MANIFEST_COLUMNS

    test_file = tmp_path / "test_manifest.parquet"
    row = {col: "test" if "score" not in col and "duration" not in col and "fps" not in col and "frames" not in col and "height" not in col and "width" not in col and "is_safe" not in col else 0 for col in MANIFEST_COLUMNS}
    row["is_safe"] = True
    df = pd_from_rows([row])

    write_manifest(df, test_file)
    assert test_file.exists()

    loaded = read_manifest(test_file)
    assert len(loaded) == 1
    assert loaded.iloc[0]["id"] == "test"


def pd_from_rows(rows):
    import pandas as pd
    return pd.DataFrame(rows)


def test_moving_mnist_generation():
    """Verify Moving MNIST clip and image generation."""
    gen = MovingMNISTGenerator(height=64, width=64, fps=8.0)

    # 1. Video clip
    video, meta = gen.generate_clip("clip_001", digit=3, frames=17, is_image=False)
    assert video.shape == (17, 64, 64, 3)
    assert meta["frames"] == 17
    assert meta["media_type"] == "video"
    assert "Digit 3" in meta["caption_short"]
    cap_dict = json.loads(meta["caption_json"])
    assert cap_dict["subject"] == "digit 3"

    # 2. Image pair
    img, meta_img = gen.generate_clip("img_001", digit=7, is_image=True)
    assert img.shape == (1, 64, 64, 3)
    assert meta_img["frames"] == 1
    assert meta_img["media_type"] == "image"


def test_quality_and_signal_filters():
    """Verify signal processing filters: sharpness, motion, borders, dHash."""
    gen = MovingMNISTGenerator(height=64, width=64, fps=8.0)

    # Sharp vs Blurry
    sharp_clip, _ = gen.generate_clip("sharp", digit=5, frames=5, inject_blur=False)
    blurry_clip, _ = gen.generate_clip("blurry", digit=5, frames=5, inject_blur=True)

    sharp_score = compute_laplacian_sharpness(sharp_clip[0])
    blurry_score = compute_laplacian_sharpness(blurry_clip[0])
    assert sharp_score > blurry_score, f"Expected sharp ({sharp_score}) > blurry ({blurry_score})"

    # Moving vs Static
    moving_clip, _ = gen.generate_clip("moving", digit=2, frames=17, speed=2.0)
    static_clip, _ = gen.generate_clip("static", digit=2, frames=17, inject_static=True)

    moving_score = compute_motion_score(moving_clip)
    static_score = compute_motion_score(static_clip)
    assert moving_score > static_score
    assert static_score < 0.1

    # Letterbox border detection
    letterbox_clip, _ = gen.generate_clip("lb", digit=1, frames=5, inject_letterbox=True)
    crop = detect_black_borders(letterbox_clip[0])
    assert crop is not None
    top, left, bottom, right = crop
    assert top >= 8, f"Expected top crop >= 8, got {top}"
    assert bottom <= 56, f"Expected bottom crop <= 56, got {bottom}"

    # Perceptual hash
    h1 = compute_dhash(sharp_clip[0])
    h2 = compute_dhash(sharp_clip[0])
    h3 = compute_dhash(blurry_clip[0])
    assert hamming_distance(h1, h2) == 0
    assert len(h1) in (16, 64)


def test_bucket_sampler():
    """Verify AspectRatioBucketSampler groups identical shapes into batches."""
    shapes = [
        (17, 64, 64),
        (17, 64, 64),
        (33, 64, 64),
        (17, 64, 64),
        (33, 64, 64),
    ]
    sampler = AspectRatioBucketSampler(shapes, batch_size=2, drop_last=False, shuffle=False)
    batches = list(sampler)
    assert len(batches) >= 2
    # Ensure items inside each batch have matching shape
    for batch in batches:
        b_shapes = [shapes[idx] for idx in batch]
        assert all(s == b_shapes[0] for s in b_shapes)


def test_curated_video_dataset(tmp_path):
    """Verify CuratedVideoDataset shapes [3, T, H, W] in [-1, 1]."""
    curation_dir = tmp_path / "data"
    df, stats = run_curation_pipeline_end_to_end(num_clips=50, output_dir=curation_dir)

    manifest_path = curation_dir / "stage6_captioned.parquet"
    dataset = CuratedVideoDataset(
        manifest_path=manifest_path,
        frames=17,
        height=64,
        width=64,
    )
    assert len(dataset) > 0

    item = dataset[0]
    pixels = item["pixels"]
    assert pixels.shape == (3, 17, 64, 64), f"Expected (3, 17, 64, 64), got {pixels.shape}"
    assert pixels.min() >= -1.0
    assert pixels.max() <= 1.0
    assert isinstance(item["prompt"], str)


def test_end_to_end_1000_clips_acceptance(tmp_path):
    """Part 1 Acceptance Test:

    Run 1,000 clips end-to-end through all 7 curation stages.
    Generates 200 kept caption reviews and 50 rejected clip reviews for threshold tuning.
    """
    curation_dir = tmp_path / "curation_1000"
    df, stats = run_curation_pipeline_end_to_end(
        num_clips=1000,
        output_dir=curation_dir,
        seed=42,
    )

    # 1. Check all 1,000 clips were processed
    assert len(df) == 1000
    assert stats["total_clips"] == 1000
    assert stats["kept_clips"] > 700, f"Expected >700 kept clips, got {stats['kept_clips']}"
    assert stats["rejected_clips"] > 50, f"Expected rejection of edge cases, got {stats['rejected_clips']}"

    # 2. Check all 7 parquet stages were written to disk
    for stage_i in range(1, 7):
        stage_file = list(curation_dir.glob(f"stage{stage_i}_*.parquet"))
        assert len(stage_file) == 1, f"Missing stage {stage_i} parquet file"

    # 3. Check statistics report was generated
    stats_file = curation_dir / "statistics_report.json"
    assert stats_file.exists()
    with open(stats_file, "r") as f:
        loaded_stats = json.load(f)
    assert loaded_stats["total_clips"] == 1000

    # 4. Check 200 random kept captions for human review
    kept_review_file = curation_dir / "review_200_captions.json"
    assert kept_review_file.exists()
    with open(kept_review_file, "r") as f:
        kept_200 = json.load(f)
    assert len(kept_200) == 200
    assert "caption_short" in kept_200[0]
    assert "caption_long" in kept_200[0]
    assert "caption_json" in kept_200[0]

    # 5. Check 50 rejected clips for threshold adjustment
    rej_review_file = curation_dir / "review_50_rejected.json"
    assert rej_review_file.exists()
    with open(rej_review_file, "r") as f:
        rej_50 = json.load(f)
    assert len(rej_50) == 50
    assert "filter_status" in rej_50[0]
    # Verify rejection reasons exist (blurry, static_video, chaotic_motion, near_duplicate, etc.)
    rejection_reasons = {r["filter_status"] for r in rej_50}
    assert any("rejected:" in reason for reason in rejection_reasons)
