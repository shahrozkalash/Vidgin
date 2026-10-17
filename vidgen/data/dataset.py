"""Advanced Video and Image Dataset Loader with Bucketing and Augmentation.

Features:
- PyAV / torchvision video decoding with graceful fallback
- Temporal resampling to target FPS (Wan uses 16)
- Random temporal crop ensuring T = 4k + 1
- Short-side resize and random/center spatial crop matching bucket (H, W)
- Image and video joint mixing (Images as T=1 clips)
- Random caption variant selection & 10% CFG null-prompt drop
- Pixel tensor convention: [3, T, H, W] in [-1.0, 1.0]
"""

from __future__ import annotations
import json
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from vidgen.data.manifest import read_manifest
from vidgen.data.buckets import BucketSpec, get_standard_buckets, find_nearest_bucket


def decode_video_pyav(
    video_path: str | Path,
    target_fps: float = 16.0,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """Decode video using PyAV with temporal resampling.

    Returns:
        frames: [T, H, W, 3] in uint8
        original_fps: float
    """
    import av

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    original_fps = float(stream.average_rate or 30.0)

    # Decode all RGB frames
    raw_frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        raw_frames.append(img)
        if max_frames is not None and len(raw_frames) >= max_frames:
            break
    container.close()

    if not raw_frames:
        raise ValueError(f"Could not decode any frames from {video_path}")

    # Temporal resampling if original_fps differs from target_fps
    num_raw = len(raw_frames)
    duration = num_raw / max(1.0, original_fps)
    target_num_frames = max(1, round(duration * target_fps))

    if target_num_frames == num_raw:
        resampled_indices = list(range(num_raw))
    else:
        resampled_indices = np.linspace(0, num_raw - 1, target_num_frames, dtype=int).tolist()

    resampled_frames = np.stack([raw_frames[i] for i in resampled_indices], axis=0)
    return resampled_frames, original_fps


def decode_media_fallback(
    path: str | Path,
    target_fps: float = 16.0,
    video_data_dict: Optional[Dict[str, np.ndarray]] = None,
    clip_id: Optional[str] = None,
) -> np.ndarray:
    """Decode video or image from disk, in-memory cache, or synthetic fallback."""
    # 1. Check in-memory dict first
    if clip_id is not None and video_data_dict and clip_id in video_data_dict:
        return video_data_dict[clip_id]

    path = Path(path)
    if not path.exists():
        # Fallback synthetic frame with non-trivial pattern
        seed_val = abs(hash(str(path))) % 10
        from vidgen.data.moving_mnist import render_digit
        digit_img = render_digit(seed_val, size=14)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[25:39, 25:39] = np.repeat((digit_img * 255.0).astype(np.uint8)[:, :, None], 3, axis=-1)
        return frame[None, :, :, :]

    # 2. Image formats
    if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            arr = np.array(rgb)  # [H, W, 3]
            return arr[None, :, :, :]  # [1, H, W, 3]

    # 3. Video formats (.mp4, .avi, etc.)
    try:
        frames, _ = decode_video_pyav(path, target_fps=target_fps)
        return frames
    except Exception:
        pass

    # 4. Numpy arrays (.npy)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim == 3:  # [H, W, 3]
            return arr[None, :, :, :]
        return arr

    return np.zeros((1, 64, 64, 3), dtype=np.uint8)


class AdvancedVideoDataset(Dataset):
    """Production-grade Video and Image Dataset with spacetime bucketing and augmentations.

    Guarantees:
    - Frame counts satisfy T = 4k + 1
    - Spatial dimensions match assigned bucket (H, W)
    - Pixel range is strictly [-1.0, 1.0]
    """

    def __init__(
        self,
        manifest_path: str | Path,
        preset: str = "tiny",
        target_fps: float = 16.0,
        image_ratio: float = 0.20,
        caption_dropout: float = 0.10,
        is_training: bool = True,
        seed: int = 42,
        video_data_dict: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.preset = preset
        self.target_fps = target_fps
        self.image_ratio = image_ratio
        self.caption_dropout = caption_dropout
        self.is_training = is_training
        self.seed = seed
        self.video_data_dict = video_data_dict or {}

        # Load and filter manifest
        df = read_manifest(self.manifest_path)
        self.df = df[df["filter_status"] == "passed"].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No passed clips found in manifest {manifest_path}")

        # Standard aspect ratio buckets
        self.buckets = get_standard_buckets(preset=preset)

        # Pre-assign every clip to its nearest bucket
        self.bucket_assignments: List[BucketSpec] = []
        for _, row in self.df.iterrows():
            is_img = (row["media_type"] == "image")
            frames = int(row["frames"]) if pd.notna(row["frames"]) else (1 if is_img else 17)
            h = int(row["height"]) if pd.notna(row["height"]) else 64
            w = int(row["width"]) if pd.notna(row["width"]) else 64
            bucket = find_nearest_bucket(frames, h, w, self.buckets, is_image=is_img)
            self.bucket_assignments.append(bucket)

    def __len__(self) -> int:
        return len(self.df)

    def get_bucket(self, idx: int) -> BucketSpec:
        """Return the pre-assigned BucketSpec for item idx."""
        return self.bucket_assignments[idx]

    def _temporal_crop_or_pad(self, frames: torch.Tensor, target_t: int) -> torch.Tensor:
        """Crop or pad frames tensor [T, C, H, W] to exactly target_t frames (T = 4k+1)."""
        cur_t = frames.shape[0]

        if cur_t == target_t:
            return frames

        if cur_t > target_t:
            # Crop
            if self.is_training:
                start = random.randint(0, cur_t - target_t)
            else:
                start = (cur_t - target_t) // 2
            return frames[start : start + target_t]

        # Pad by repeating last frame or cycling
        pad_len = target_t - cur_t
        if cur_t == 1:
            # Image: repeat across all T frames
            return frames.repeat(target_t, 1, 1, 1)

        last_frame = frames[-1:].repeat(pad_len, 1, 1, 1)
        return torch.cat([frames, last_frame], dim=0)

    def _spatial_resize_and_crop(
        self,
        frames: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """Resize short side to match aspect ratio, then center/random crop to (target_h, target_w).

        Input: [T, C, H, W]
        """
        T, C, H, W = frames.shape
        if (H, W) == (target_h, target_w):
            return frames

        # Compute scaling scale so that both dimensions are at least target dimensions
        scale = max(target_h / H, target_w / W)
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))

        # Bilinear interpolation
        resized = F.interpolate(
            frames,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

        # Spatial crop
        diff_h = new_h - target_h
        diff_w = new_w - target_w

        if self.is_training:
            top = random.randint(0, max(0, diff_h))
            left = random.randint(0, max(0, diff_w))
        else:
            top = max(0, diff_h // 2)
            left = max(0, diff_w // 2)

        cropped = resized[:, :, top : top + target_h, left : left + target_w]
        return cropped

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        clip_id = row["id"]
        bucket = self.bucket_assignments[idx]

        # 1. Decode raw media frames [T_raw, H_raw, W_raw, 3] in uint8
        raw_arr = decode_media_fallback(
            row["path"],
            target_fps=self.target_fps,
            video_data_dict=self.video_data_dict,
            clip_id=clip_id,
        )

        # Convert to float tensor [T, 3, H, W] in [0.0, 1.0]
        t_frames = torch.from_numpy(raw_arr).permute(0, 3, 1, 2).float() / 255.0

        # 2. Temporal crop or pad to bucket.frames (T = 4k + 1)
        t_frames = self._temporal_crop_or_pad(t_frames, target_t=bucket.frames)

        # 3. Spatial resize short side and crop to bucket (height, width)
        t_frames = self._spatial_resize_and_crop(t_frames, target_h=bucket.height, target_w=bucket.width)

        # 4. Pixel normalization to [-1.0, 1.0] and rearrange to [3, T, H, W]
        video_tensor = t_frames.permute(1, 0, 2, 3) * 2.0 - 1.0

        # Verify Wan conventions
        assert (video_tensor.shape[1] - 1) % 4 == 0, f"Frames {video_tensor.shape[1]} != 4k+1"
        assert video_tensor.shape[2:] == (bucket.height, bucket.width)

        # 5. Captions: randomly choose variant & apply 10% CFG drop
        caption_choices = []
        if pd.notna(row["caption_long"]) and row["caption_long"]:
            caption_choices.append(str(row["caption_long"]))
        if pd.notna(row["caption_short"]) and row["caption_short"]:
            caption_choices.append(str(row["caption_short"]))
        if not caption_choices:
            caption_choices.append("A video clip.")

        selected_prompt = random.choice(caption_choices)

        # 10% CFG dropout
        is_dropped = False
        if self.is_training and (random.random() < self.caption_dropout):
            selected_prompt = ""  # Empty prompt for CFG unconditional generation
            is_dropped = True

        return {
            "id": clip_id,
            "media_type": row["media_type"],
            "pixels": video_tensor,  # [3, T, H, W] in [-1, 1]
            "prompt": selected_prompt,
            "is_caption_dropped": is_dropped,
            "bucket": bucket.to_tuple(),  # (frames, height, width)
        }


# Alias for backward compatibility
CuratedVideoDataset = AdvancedVideoDataset
