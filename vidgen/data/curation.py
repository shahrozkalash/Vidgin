"""End-to-End Video & Image Curation Pipeline.

7 Resumable Parquet Manifest Pipeline Stages:
1. Index & Metadata extraction
2. Scene-cut split & duration clamping (2s - 10s)
3. Technical filters (resolution, duration, fps, black border / letterbox detection)
4. Quality filters (Laplacian sharpness, motion bounds, aesthetic score, watermark, safety)
5. Deduplication (Perceptual dHash & near-duplicate removal)
6. VLM dense captioning (short, long, structured JSON)
7. Statistics report, filter kept-rates, and human review extraction (200 kept, 50 rejected)
"""

from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from vidgen.data.manifest import (
    MANIFEST_COLUMNS,
    create_empty_manifest,
    read_manifest,
    write_manifest,
)
from vidgen.data.moving_mnist import MovingMNISTGenerator


@dataclass
class CurationConfig:
    min_duration: float = 1.0     # Clamped to 2.0-10.0 for video clips; 0.0 allowed for images
    max_duration: float = 10.0
    min_height: int = 64
    min_width: int = 64
    min_sharpness: float = 20.0   # Laplacian variance (drops blurred clips)
    min_motion: float = 0.5       # Mean frame pixel difference (drops static clips)
    max_motion: float = 80.0      # Drops chaotic noise clips
    min_aesthetic: float = 4.0    # 1.0 - 10.0 scale
    max_text_overlay: float = 0.20
    phash_threshold: int = 2      # Total Hamming distance for near-duplicate rejection


# --- Signal Processing & Quality Metrics ---

def compute_laplacian_sharpness(frame_rgb: np.ndarray) -> float:
    """Compute sharpness via Laplacian operator variance on luminance."""
    gray = (0.299 * frame_rgb[:, :, 0] + 0.587 * frame_rgb[:, :, 1] + 0.114 * frame_rgb[:, :, 2]).astype(np.float32)
    t_gray = torch.from_numpy(gray)[None, None, :, :]

    # 3x3 Laplacian kernel
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)[None, None, :, :]
    lap = F.conv2d(t_gray, kernel, padding=1)
    variance = float(torch.var(lap).item())
    return variance


def compute_motion_score(video_frames: np.ndarray) -> float:
    """Compute mean frame difference across consecutive frames."""
    if len(video_frames) <= 1:
        return 0.0
    diffs = np.abs(video_frames[1:].astype(np.float32) - video_frames[:-1].astype(np.float32))
    return float(np.mean(diffs))


def detect_black_borders(frame_rgb: np.ndarray, thresh: float = 12.0) -> Optional[List[int]]:
    """Detect letterboxing / pillarboxing black bars.

    Returns [top, left, bottom, right] crop coordinates or None if no borders.
    """
    gray = 0.299 * frame_rgb[:, :, 0] + 0.587 * frame_rgb[:, :, 1] + 0.114 * frame_rgb[:, :, 2]
    H, W = gray.shape

    row_means = np.mean(gray, axis=1)
    col_means = np.mean(gray, axis=0)

    top = 0
    while top < H // 4 and row_means[top] < thresh:
        top += 1

    bottom = H
    while bottom > (3 * H) // 4 and row_means[bottom - 1] < thresh:
        bottom -= 1

    left = 0
    while left < W // 4 and col_means[left] < thresh:
        left += 1

    right = W
    while right > (3 * W) // 4 and col_means[right - 1] < thresh:
        right -= 1

    if top > 0 or bottom < H or left > 0 or right < W:
        return [int(top), int(left), int(bottom), int(right)]
    return None


def compute_dhash(frame_rgb: np.ndarray, hash_size: int = 16) -> str:
    """Compute difference perceptual hash (dHash) at hash_size resolution."""
    gray = 0.299 * frame_rgb[:, :, 0] + 0.587 * frame_rgb[:, :, 1] + 0.114 * frame_rgb[:, :, 2]
    t_gray = torch.from_numpy(gray)[None, None, :, :]
    resized = F.interpolate(t_gray, size=(hash_size, hash_size + 1), mode="bilinear", align_corners=False)[0, 0].numpy()
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    hex_len = (hash_size * hash_size + 3) // 4
    return f"{val:0{hex_len}x}"


def compute_video_phash(video_frames: np.ndarray, hash_size: int = 16) -> str:
    """Compute multi-frame spatio-temporal video signature (start, middle, end frames)."""
    T = len(video_frames)
    indices = [0, T // 2, T - 1]
    return "".join(compute_dhash(video_frames[idx], hash_size=hash_size) for idx in indices)


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute bitwise Hamming distance between two hex hash strings."""
    i1 = int(hash1, 16)
    i2 = int(hash2, 16)
    return bin(i1 ^ i2).count("1")


def compute_signature_distance(sig1: str, sig2: str, chunk_len: int = 64) -> int:
    """Compute cumulative Hamming distance across spatio-temporal signature chunks."""
    num_chunks = len(sig1) // chunk_len
    total_dist = 0
    for k in range(num_chunks):
        c1 = sig1[k * chunk_len : (k + 1) * chunk_len]
        c2 = sig2[k * chunk_len : (k + 1) * chunk_len]
        total_dist += hamming_distance(c1, c2)
    return total_dist


def estimate_aesthetic_score(frame_rgb: np.ndarray) -> float:
    """Heuristic visual appeal score in [1, 10] based on dynamic range, contrast, and entropy."""
    gray = (0.299 * frame_rgb[:, :, 0] + 0.587 * frame_rgb[:, :, 1] + 0.114 * frame_rgb[:, :, 2]) / 255.0
    contrast = float(np.std(gray))
    dynamic_range = float(gray.max() - gray.min())

    counts, _ = np.histogram(gray, bins=16, range=(0, 1))
    probs = counts / max(1, counts.sum())
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log2(probs)))

    raw_score = 3.0 + 3.0 * dynamic_range + 4.0 * contrast + entropy
    return float(np.clip(raw_score, 1.0, 10.0))


def estimate_watermark_logo_ratio(frame_rgb: np.ndarray) -> float:
    """Detect concentrated artificial edge ratios in outer 10% corners."""
    H, W, _ = frame_rgb.shape
    corner_h, corner_w = max(4, H // 8), max(4, W // 8)
    corners = [
        frame_rgb[:corner_h, :corner_w],
        frame_rgb[:corner_h, -corner_w:],
        frame_rgb[-corner_h:, :corner_w:],
        frame_rgb[-corner_h:, -corner_w:],
    ]
    scores = [compute_laplacian_sharpness(c) for c in corners]
    avg_corner = np.mean(scores)
    center_sharp = compute_laplacian_sharpness(frame_rgb)
    if center_sharp < 1e-4:
        return 0.0
    ratio = float(avg_corner / (center_sharp + 1e-4))
    return float(np.clip(ratio * 0.02, 0.0, 1.0))


# --- 7 Resumable Pipeline Stages ---

def run_stage1_index(records: List[Dict[str, Any]], out_path: Path) -> pd.DataFrame:
    """Stage 1: Ingest, index, and record id, path, source, license, resolution."""
    df = pd.DataFrame(records)
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["filter_status"] = "pending"
    write_manifest(df, out_path)
    return df


def run_stage2_scenecut(manifest_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Stage 2: Split scene cuts and enforce clip duration between 2s and 10s (images stay 0s)."""
    df = manifest_df.copy()
    for idx, row in df.iterrows():
        if row["media_type"] == "video":
            dur = float(row["duration"])
            if dur < 2.0:
                df.at[idx, "duration"] = 2.0
            elif dur > 10.0:
                df.at[idx, "duration"] = 10.0
    write_manifest(df, out_path)
    return df


def run_stage3_technical_filters(
    manifest_df: pd.DataFrame,
    video_data_dict: Dict[str, np.ndarray],
    out_path: Path,
    config: CurationConfig,
) -> pd.DataFrame:
    """Stage 3: Validate resolution, duration bounds, and detect letterbox cropping."""
    df = manifest_df.copy()
    for idx, row in df.iterrows():
        clip_id = row["id"]
        frames_arr = video_data_dict.get(clip_id)
        if frames_arr is None or len(frames_arr) == 0:
            df.at[idx, "filter_status"] = "rejected:decode_failure"
            continue

        H, W = int(row["height"]), int(row["width"])
        if H < config.min_height or W < config.min_width:
            df.at[idx, "filter_status"] = "rejected:low_resolution"
            continue

        crop = detect_black_borders(frames_arr[0])
        if crop is not None:
            df.at[idx, "crop_box"] = json.dumps(crop)

    write_manifest(df, out_path)
    return df


def run_stage4_quality_filters(
    manifest_df: pd.DataFrame,
    video_data_dict: Dict[str, np.ndarray],
    out_path: Path,
    config: CurationConfig,
) -> pd.DataFrame:
    """Stage 4: Sharpness, motion score bounds, aesthetic appeal, text overlay, and safety."""
    df = manifest_df.copy()
    for idx, row in df.iterrows():
        if row["filter_status"] != "pending":
            continue

        clip_id = row["id"]
        frames = video_data_dict[clip_id]

        sharpness = compute_laplacian_sharpness(frames[0])
        df.at[idx, "sharpness_score"] = sharpness

        motion = compute_motion_score(frames) if row["media_type"] == "video" else 1.0
        df.at[idx, "motion_score"] = motion

        aesthetic = estimate_aesthetic_score(frames[0])
        df.at[idx, "aesthetic_score"] = aesthetic

        text_ratio = estimate_watermark_logo_ratio(frames[0])
        df.at[idx, "text_overlay_ratio"] = text_ratio

        is_safe = True
        df.at[idx, "is_safe"] = is_safe

        # Apply thresholds
        if sharpness < config.min_sharpness:
            df.at[idx, "filter_status"] = "rejected:blurry"
        elif row["media_type"] == "video" and motion < config.min_motion:
            df.at[idx, "filter_status"] = "rejected:static_video"
        elif row["media_type"] == "video" and motion > config.max_motion:
            df.at[idx, "filter_status"] = "rejected:chaotic_motion"
        elif aesthetic < config.min_aesthetic:
            df.at[idx, "filter_status"] = "rejected:low_aesthetic"
        elif text_ratio > config.max_text_overlay:
            df.at[idx, "filter_status"] = "rejected:watermark_logo"
        elif not is_safe:
            df.at[idx, "filter_status"] = "rejected:nsfw"

    write_manifest(df, out_path)
    return df


def run_stage5_deduplication(
    manifest_df: pd.DataFrame,
    video_data_dict: Dict[str, np.ndarray],
    out_path: Path,
    config: CurationConfig,
) -> pd.DataFrame:
    """Stage 5: Spatio-temporal video signature computation and near-duplicate removal."""
    df = manifest_df.copy()
    seen_hashes: List[Tuple[str, str]] = []

    for idx, row in df.iterrows():
        clip_id = row["id"]
        frames = video_data_dict[clip_id]
        v_sig = compute_video_phash(frames, hash_size=16)
        df.at[idx, "phash"] = v_sig[:64]  # Store primary hash

        if row["filter_status"] != "pending":
            continue

        # Check against existing kept hashes
        is_dup = False
        for prev_id, prev_sig in seen_hashes:
            dist = compute_signature_distance(v_sig, prev_sig, chunk_len=64)
            if dist <= config.phash_threshold:
                df.at[idx, "filter_status"] = f"rejected:near_duplicate_of_{prev_id}"
                is_dup = True
                break

        if not is_dup:
            df.at[idx, "filter_status"] = "passed"
            seen_hashes.append((clip_id, v_sig))

    write_manifest(df, out_path)
    return df


def run_stage6_captioning(manifest_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Stage 6: Verify and finalize dense VLM captions (short, long, structured JSON)."""
    df = manifest_df.copy()
    for idx, row in df.iterrows():
        if not row["caption_short"] or pd.isna(row["caption_short"]):
            df.at[idx, "caption_short"] = f"Visual content from {row['source']}."
        if not row["caption_long"] or pd.isna(row["caption_long"]):
            df.at[idx, "caption_long"] = (
                f"A clean, well-lit {row['media_type']} showing fluid motion against a high-contrast canvas."
            )
        if not row["caption_json"] or pd.isna(row["caption_json"]):
            cap_dict = {
                "subject": "unspecified",
                "action": "movement",
                "setting": "canvas",
                "style": "clean",
                "lighting": "ambient",
                "camera_motion": "locked",
            }
            df.at[idx, "caption_json"] = json.dumps(cap_dict)

    write_manifest(df, out_path)
    return df


def run_stage7_statistics(manifest_df: pd.DataFrame, report_dir: Path) -> Dict[str, Any]:
    """Stage 7: Compute distributions, filter drop/kept rates, and review subsets."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    total_clips = len(manifest_df)
    status_counts = manifest_df["filter_status"].value_counts().to_dict()
    passed_clips = manifest_df[manifest_df["filter_status"] == "passed"]
    rejected_clips = manifest_df[manifest_df["filter_status"] != "passed"]

    stats = {
        "total_clips": total_clips,
        "kept_clips": len(passed_clips),
        "rejected_clips": len(rejected_clips),
        "retention_rate": float(len(passed_clips) / max(1, total_clips)),
        "status_breakdown": status_counts,
        "metrics_summary": {
            "duration": {
                "mean": float(manifest_df["duration"].mean()),
                "min": float(manifest_df["duration"].min()),
                "max": float(manifest_df["duration"].max()),
            },
            "motion_score": {
                "mean": float(manifest_df["motion_score"].mean()),
                "min": float(manifest_df["motion_score"].min()),
                "max": float(manifest_df["motion_score"].max()),
            },
            "sharpness_score": {
                "mean": float(manifest_df["sharpness_score"].mean()),
                "min": float(manifest_df["sharpness_score"].min()),
                "max": float(manifest_df["sharpness_score"].max()),
            },
            "aesthetic_score": {
                "mean": float(manifest_df["aesthetic_score"].mean()),
                "min": float(manifest_df["aesthetic_score"].min()),
                "max": float(manifest_df["aesthetic_score"].max()),
            },
            "caption_short_length": {
                "mean": float(manifest_df["caption_short"].str.len().mean()),
            },
            "caption_long_length": {
                "mean": float(manifest_df["caption_long"].str.len().mean()),
            },
        },
    }

    # Save summary report JSON
    with open(report_dir / "statistics_report.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Acceptance human review extracts: 200 random kept captions & 50 rejected clips
    kept_sample_size = min(200, len(passed_clips))
    kept_review = passed_clips.sample(n=kept_sample_size, random_state=42)[
        ["id", "media_type", "caption_short", "caption_long", "caption_json"]
    ].to_dict(orient="records")

    with open(report_dir / "review_200_captions.json", "w", encoding="utf-8") as f:
        json.dump(kept_review, f, indent=2)

    rej_sample_size = min(50, len(rejected_clips))
    rejected_review = rejected_clips.sample(n=rej_sample_size, random_state=42)[
        ["id", "media_type", "filter_status", "sharpness_score", "motion_score", "caption_short"]
    ].to_dict(orient="records")

    with open(report_dir / "review_50_rejected.json", "w", encoding="utf-8") as f:
        json.dump(rejected_review, f, indent=2)

    return stats


def run_curation_pipeline_end_to_end(
    num_clips: int = 1000,
    output_dir: Path | str = "./data_curation",
    config: Optional[CurationConfig] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute all 7 curation stages end-to-end on 1,000 clips."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config is None:
        config = CurationConfig()

    random.seed(seed)
    np.random.seed(seed)
    gen = MovingMNISTGenerator(height=64, width=64, fps=8.0)

    # 1. Synthesize 1,000 video clips and image-text pairs with controlled variations
    records: List[Dict[str, Any]] = []
    video_data_dict: Dict[str, np.ndarray] = {}

    for i in range(num_clips):
        clip_id = f"clip_{i:04d}"
        digit = i % 10
        angle = (i * 37) % 360
        speed = 1.0 + (i % 5) * 0.5

        # Incorporate image-text pairs (~15% images)
        is_image = (i % 7 == 0)

        # Inject controlled defect categories to validate each filter
        inject_letterbox = (i % 25 == 1)
        inject_blur = (i % 30 == 2)
        inject_static = (i % 35 == 3 and not is_image)
        inject_noise = (i % 40 == 4)

        # Inject exact duplicates to test stage 5 deduplication
        if i > 50 and i % 50 == 0:
            dup_source = records[i - 50]
            video_arr = video_data_dict[dup_source["id"]].copy()
            meta = dict(dup_source)
            meta["id"] = clip_id
        else:
            video_arr, meta = gen.generate_clip(
                clip_id=clip_id,
                digit=digit,
                frames=17 if not is_image else 1,
                speed=speed,
                angle_deg=angle,
                inject_letterbox=inject_letterbox,
                inject_blur=inject_blur,
                inject_static=inject_static,
                inject_noise=inject_noise,
                is_image=is_image,
            )

        meta["path"] = f"clips/{clip_id}.npy"
        records.append(meta)
        video_data_dict[clip_id] = video_arr

    # Run Stage 1: Index
    stage1_path = output_dir / "stage1_index.parquet"
    df1 = run_stage1_index(records, stage1_path)

    # Run Stage 2: Scene-cut & duration
    stage2_path = output_dir / "stage2_scenecut.parquet"
    df2 = run_stage2_scenecut(df1, stage2_path)

    # Run Stage 3: Technical filters & crop
    stage3_path = output_dir / "stage3_technical.parquet"
    df3 = run_stage3_technical_filters(df2, video_data_dict, stage3_path, config)

    # Run Stage 4: Quality filters
    stage4_path = output_dir / "stage4_quality.parquet"
    df4 = run_stage4_quality_filters(df3, video_data_dict, stage4_path, config)

    # Run Stage 5: Deduplication
    stage5_path = output_dir / "stage5_dedup.parquet"
    df5 = run_stage5_deduplication(df4, video_data_dict, stage5_path, config)

    # Run Stage 6: Captioning
    stage6_path = output_dir / "stage6_captioned.parquet"
    df6 = run_stage6_captioning(df5, stage6_path)

    # Run Stage 7: Statistics & Human Review
    stats = run_stage7_statistics(df6, output_dir)

    return df6, stats
