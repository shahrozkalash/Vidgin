"""Video and Image PyTorch Dataset loading from curated Parquet manifests.

Tensor conventions:
- Video Pixels: [3, T, H, W] in [-1.0, 1.0] with T = 4k + 1
- Image Pixels: [3, 1, H, W] or repeated across T
- Captions: short, long, and structured metadata
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from vidgen.data.manifest import read_manifest


class CuratedVideoDataset(Dataset):
    """PyTorch Dataset loading curated video-text and image-text pairs."""

    def __init__(
        self,
        manifest_path: str | Path,
        frames: int = 17,
        height: int = 64,
        width: int = 64,
        caption_key: str = "caption_long",
        video_data_dict: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.frames = frames
        self.height = height
        self.width = width
        self.caption_key = caption_key
        self.video_data_dict = video_data_dict or {}

        # Validate T = 4k + 1
        assert (frames - 1) % 4 == 0, f"Frames ({frames}) must satisfy T = 4k + 1"

        # Load and filter manifest to passed clips
        df = read_manifest(self.manifest_path)
        self.df = df[df["filter_status"] == "passed"].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        clip_id = row["id"]

        # Fetch frames
        if clip_id in self.video_data_dict:
            raw_frames = self.video_data_dict[clip_id]  # [T_raw, H, W, 3] in [0, 255]
        else:
            # Fallback mock for testing
            is_image = (row["media_type"] == "image")
            t_count = 1 if is_image else self.frames
            raw_frames = np.zeros((t_count, self.height, self.width, 3), dtype=np.uint8)

        # Handle letterbox crop if present
        if pd.notna(row["crop_box"]) and row["crop_box"]:
            try:
                top, left, bottom, right = json.loads(row["crop_box"])
                raw_frames = raw_frames[:, top:bottom, left:right, :]
            except Exception:
                pass

        # Convert to float tensor [T, C, H, W] in [0, 1]
        t_frames = torch.from_numpy(raw_frames).permute(0, 3, 1, 2).float() / 255.0

        # Resize spatial dimensions if needed
        if t_frames.shape[-2:] != (self.height, self.width):
            t_frames = torch.nn.functional.interpolate(
                t_frames,
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )

        # Handle temporal dimension: pad/repeat to self.frames (T = 4k + 1)
        cur_t = t_frames.shape[0]
        if cur_t == 1:
            # Image: repeat across all T frames
            t_frames = t_frames.repeat(self.frames, 1, 1, 1)
        elif cur_t < self.frames:
            # Pad by repeating last frame
            pad_count = self.frames - cur_t
            last_frame = t_frames[-1:].repeat(pad_count, 1, 1, 1)
            t_frames = torch.cat([t_frames, last_frame], dim=0)
        elif cur_t > self.frames:
            t_frames = t_frames[: self.frames]

        # Reshape to [3, T, H, W] and normalize to [-1.0, 1.0]
        video_tensor = t_frames.permute(1, 0, 2, 3) * 2.0 - 1.0

        prompt = str(row[self.caption_key]) if pd.notna(row[self.caption_key]) else "A video."
        caption_dict = json.loads(row["caption_json"]) if pd.notna(row["caption_json"]) else {}

        return {
            "id": clip_id,
            "media_type": row["media_type"],
            "pixels": video_tensor,  # [3, T, H, W] in [-1, 1]
            "prompt": prompt,
            "caption_short": row["caption_short"],
            "caption_long": row["caption_long"],
            "caption_json": caption_dict,
        }
