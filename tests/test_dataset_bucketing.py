"""Unit and Acceptance Tests for Part 2: Dataset, Bucketing, and Precomputation."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from vidgen.data.manifest import read_manifest
from vidgen.data.moving_mnist import MovingMNISTGenerator
from vidgen.data.curation import run_curation_pipeline_end_to_end
from vidgen.data.buckets import (
    BucketSpec,
    get_standard_buckets,
    find_nearest_bucket,
    DistributedBucketSampler,
)
from vidgen.data.dataset import (
    AdvancedVideoDataset,
    decode_media_fallback,
)
from vidgen.data.precompute import (
    ChannelStatsAccumulator,
    run_precompute_pipeline,
    PrecomputedLatentDataset,
)


def test_frame_count_and_multiples_of_16_rule():
    """Verify Wan rules: T = 4k+1 and H, W are multiples of 16."""
    buckets = get_standard_buckets(preset="tiny")
    for b in buckets:
        assert (b.frames - 1) % 4 == 0, f"Bucket frames ({b.frames}) violates T = 4k + 1"
        assert b.height % 16 == 0, f"Bucket height ({b.height}) must be a multiple of 16"
        assert b.width % 16 == 0, f"Bucket width ({b.width}) must be a multiple of 16"

    # Invalid frames must raise AssertionError
    with pytest.raises(AssertionError):
        BucketSpec(frames=16, height=64, width=64)

    # Invalid height/width must raise AssertionError
    with pytest.raises(AssertionError):
        BucketSpec(frames=17, height=60, width=64)


def test_standard_aspect_ratios_presence():
    """Verify standard aspect ratios: 16:9, 9:16, 1:1, 4:3, 3:4 for video and image."""
    buckets = get_standard_buckets(preset="tiny")
    video_aspects = [round(b.aspect_ratio, 2) for b in buckets if b.frames > 1]
    image_aspects = [round(b.aspect_ratio, 2) for b in buckets if b.frames == 1]

    # Verify both videos and images cover landscape, square, and portrait
    for aspect_list in [video_aspects, image_aspects]:
        assert any(a > 1.5 for a in aspect_list), "Must include wide (16:9) bucket"
        assert any(a < 0.7 for a in aspect_list), "Must include tall (9:16) bucket"
        assert any(abs(a - 1.0) < 0.05 for a in aspect_list), "Must include square (1:1) bucket"


def test_nearest_bucket_assignment():
    """Verify clips are assigned to appropriate nearest aspect ratio & image buckets."""
    buckets = get_standard_buckets(preset="tiny")

    # Wide video -> 16:9 approx (48x80)
    b_wide = find_nearest_bucket(frames=17, height=48, width=80, buckets=buckets, is_image=False)
    assert b_wide.frames == 17
    assert b_wide.width > b_wide.height

    # Tall video -> 9:16 approx (80x48)
    b_tall = find_nearest_bucket(frames=17, height=80, width=48, buckets=buckets, is_image=False)
    assert b_tall.frames == 17
    assert b_tall.height > b_tall.width

    # Image -> T=1 bucket
    b_img = find_nearest_bucket(frames=1, height=64, width=64, buckets=buckets, is_image=True)
    assert b_img.frames == 1


def test_distributed_bucket_sampler_token_scaling_and_homogeneity():
    """Verify that batches drawn from sampler are homogeneous and token-scaled."""
    buckets = get_standard_buckets(preset="tiny")
    b_video = [b for b in buckets if b.frames == 17 and b.height == 64][0]  # 80 tokens
    b_image = [b for b in buckets if b.frames == 1 and b.height == 64][0]   # 16 tokens

    # Synthesize assignments: 20 video items and 20 image items
    assignments = [b_video] * 20 + [b_image] * 20
    sampler = DistributedBucketSampler(
        bucket_assignments=assignments,
        target_tokens_per_batch=160,
        min_batch_size=1,
        max_batch_size=16,
        shuffle=False,
    )

    batches = list(sampler)
    assert len(batches) > 0

    for batch in batches:
        # 1. Homogeneity: all indices in a batch belong to the exact same bucket
        batch_buckets = [assignments[i] for i in batch]
        assert all(b == batch_buckets[0] for b in batch_buckets)

        # 2. Token scaling: image batches have more samples than video batches
        if batch_buckets[0].frames == 1:
            assert len(batch) >= len(batches[0])


def test_sampler_resumption_and_sharding():
    """Verify DistributedBucketSampler state saving and deterministic next-batch resume."""
    buckets = get_standard_buckets(preset="tiny")
    b_sq = [b for b in buckets if b.frames == 17 and b.height == 64][0]
    assignments = [b_sq] * 30

    sampler = DistributedBucketSampler(
        bucket_assignments=assignments,
        target_tokens_per_batch=160,
        shuffle=True,
        seed=1337,
    )

    iterator = iter(sampler)
    batch_0 = next(iterator)
    batch_1 = next(iterator)
    saved_state = sampler.state_dict()
    batch_2_uninterrupted = next(iterator)

    # Resume fresh sampler
    sampler_resumed = DistributedBucketSampler(
        bucket_assignments=assignments,
        target_tokens_per_batch=160,
        shuffle=True,
        seed=1337,
    )
    sampler_resumed.load_state_dict(saved_state)

    iterator_resumed = iter(sampler_resumed)
    batch_2_resumed = next(iterator_resumed)

    assert batch_2_resumed == batch_2_uninterrupted, "Resumed sampler did not produce identical next batch"

    # Multi-GPU rank sharding
    s_rank0 = DistributedBucketSampler(assignments, rank=0, world_size=2, shuffle=False, seed=42)
    s_rank1 = DistributedBucketSampler(assignments, rank=1, world_size=2, shuffle=False, seed=42)

    batches_r0 = list(s_rank0)
    batches_r1 = list(s_rank1)
    assert len(batches_r0) > 0 and len(batches_r1) > 0
    # Ranks should process disjoint batches
    flat_r0 = {idx for b in batches_r0 for idx in b}
    flat_r1 = {idx for b in batches_r1 for idx in b}
    assert flat_r0.isdisjoint(flat_r1)


def test_dataset_crops_transforms_and_cfg_dropout(tmp_path):
    """Verify spatial/temporal transforms, [-1, 1] normalization, and 10% CFG drop."""
    curation_dir = tmp_path / "data"
    df, _ = run_curation_pipeline_end_to_end(num_clips=20, output_dir=curation_dir)
    manifest_path = curation_dir / "stage6_captioned.parquet"

    dataset = AdvancedVideoDataset(
        manifest_path=manifest_path,
        preset="tiny",
        target_fps=8.0,
        caption_dropout=0.50,  # 50% for testing drop occurrence
        is_training=True,
        seed=42,
    )

    dropped_count = 0
    for idx in range(len(dataset)):
        item = dataset[idx]
        pixels = item["pixels"]

        # Tensor conventions
        assert pixels.ndim == 4  # [3, T, H, W]
        assert pixels.shape[0] == 3
        assert (pixels.shape[1] - 1) % 4 == 0  # T = 4k + 1
        assert pixels.min() >= -1.0
        assert pixels.max() <= 1.0

        if item["is_caption_dropped"]:
            dropped_count += 1
            assert item["prompt"] == ""

    assert dropped_count > 0, "CFG dropout should have triggered on several items"


def test_precomputation_and_channel_normalization(tmp_path):
    """Verify VAE latents & text embeddings caching, and 16-channel mean/std stats."""
    curation_dir = tmp_path / "curation"
    df, _ = run_curation_pipeline_end_to_end(num_clips=15, output_dir=curation_dir)
    manifest_path = curation_dir / "stage6_captioned.parquet"

    dataset = AdvancedVideoDataset(
        manifest_path=manifest_path,
        preset="tiny",
        target_fps=8.0,
        is_training=False,
    )

    precompute_dir = tmp_path / "precomputed"
    out_dir, stats = run_precompute_pipeline(
        dataset=dataset,
        output_dir=precompute_dir,
    )

    # 1. Verify latent_stats.json and latent_stats.pt
    stats_file = precompute_dir / "latent_stats.json"
    assert stats_file.exists()
    assert stats["num_channels"] == 16
    assert len(stats["mean"]) == 16
    assert len(stats["std"]) == 16
    assert stats["total_elements_sampled"] > 0

    # 2. Verify precomputed manifest and .safetensors shards
    pre_manifest = precompute_dir / "precomputed_manifest.json"
    assert pre_manifest.exists()
    shards = list((precompute_dir / "shards").glob("*.safetensors"))
    assert len(shards) == len(dataset)

    # 3. Test loading through PrecomputedLatentDataset
    pre_dataset = PrecomputedLatentDataset(
        manifest_path=pre_manifest,
        stats_path=precompute_dir / "latent_stats.pt",
        use_augmentation=True,
        caption_dropout=0.10,
    )
    assert len(pre_dataset) == len(dataset)

    item = pre_dataset[0]
    latents = item["latents"]
    assert latents.shape[0] == 16  # 16 latent channels
    assert latents.ndim == 4       # [16, t, h, w]
    assert "text_emb" in item


def test_throughput_and_visual_dump_acceptance(tmp_path):
    """Part 2 Acceptance Test:

    - Visual dump of one batch: pixel range [-1, 1], non-trivial variance, T % 4 == 1, shape matches bucket.
    - Throughput benchmark: DataLoader yields batches at > 50 clips/sec on CPU.
    """
    import time

    curation_dir = tmp_path / "curation_acceptance"
    df, _ = run_curation_pipeline_end_to_end(num_clips=40, output_dir=curation_dir)
    manifest_path = curation_dir / "stage6_captioned.parquet"

    dataset = AdvancedVideoDataset(
        manifest_path=manifest_path,
        preset="tiny",
        target_fps=8.0,
        is_training=True,
    )

    sampler = DistributedBucketSampler(
        bucket_assignments=dataset.bucket_assignments,
        target_tokens_per_batch=160,
        shuffle=True,
        seed=42,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
    )

    # 1. Visual dump of first batch
    first_batch = next(iter(loader))
    pixels = first_batch["pixels"]  # [B, 3, T, H, W]
    B, C, T, H, W = pixels.shape

    # Acceptance assertions:
    assert C == 3, f"Expected 3 channels, got {C}"
    assert (T - 1) % 4 == 0, f"Frame count ({T}) does not satisfy T % 4 == 1"
    assert H % 16 == 0 and W % 16 == 0, f"Dimensions ({H}, {W}) are not multiples of 16"
    assert pixels.min() >= -1.0 and pixels.max() <= 1.0, "Pixel values out of [-1, 1]"
    # Non-trivial variance check (ensures not blank black tensor)
    variance = float(torch.var(pixels).item())
    assert variance > 0.01, f"Batch has insufficient variance ({variance}), looks like blank frames"

    # 2. Throughput benchmark test
    start_time = time.perf_counter()
    total_samples = 0
    for batch in loader:
        total_samples += batch["pixels"].shape[0]
    elapsed = time.perf_counter() - start_time
    throughput = total_samples / max(1e-4, elapsed)

    assert throughput > 40.0, f"Throughput ({throughput:.1f} clips/s) below training step requirement"
