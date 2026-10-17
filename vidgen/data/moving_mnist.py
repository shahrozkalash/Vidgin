"""Moving MNIST and Synthetic Video Generator for Track A Data Curation.

Generates reproducible video clips and image-text pairs with ground truth:
- Digit shapes, motion vectors, bounces, speed
- Spatial variety in initial positions
- Letterboxed borders (for crop testing)
- Blurry / static / chaotic noise injections (for quality filter validation)
- Duplicates (for perceptual hash deduplication testing)
- Short, long, and dense structured JSON captions
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F


# 5x7 bitmap representation for digits 0-9
DIGIT_BITMAPS = {
    0: ["11111", "10001", "10001", "10001", "10001", "10001", "11111"],
    1: ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    2: ["11111", "00001", "00001", "11111", "10000", "10000", "11111"],
    3: ["11111", "00001", "00001", "11111", "00001", "00001", "11111"],
    4: ["10001", "10001", "10001", "11111", "00001", "00001", "00001"],
    5: ["11111", "10000", "10000", "11111", "00001", "00001", "11111"],
    6: ["11111", "10000", "10000", "11111", "10001", "10001", "11111"],
    7: ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    8: ["11111", "10001", "10001", "11111", "10001", "10001", "11111"],
    9: ["11111", "10001", "10001", "11111", "00001", "00001", "11111"],
}


def render_digit(digit: int, size: int = 14) -> np.ndarray:
    """Render a digit into an anti-aliased (size x size) bitmap."""
    pattern = DIGIT_BITMAPS[digit % 10]
    grid = np.zeros((7, 5), dtype=np.float32)
    for r in range(7):
        for c in range(5):
            if pattern[r][c] == "1":
                grid[r, c] = 1.0

    t_grid = torch.from_numpy(grid)[None, None, :, :]
    upscaled = F.interpolate(t_grid, size=(size, size), mode="bilinear", align_corners=False)
    return upscaled[0, 0].numpy()


class MovingMNISTGenerator:
    """Generates synthetic video clips and images for Wan data pipeline."""

    def __init__(self, height: int = 64, width: int = 64, fps: float = 8.0):
        self.height = height
        self.width = width
        self.fps = fps

    def generate_clip(
        self,
        clip_id: str,
        digit: int = 3,
        frames: int = 17,
        speed: float = 1.5,
        angle_deg: float = 45.0,
        start_pos: Optional[Tuple[float, float]] = None,
        inject_letterbox: bool = False,
        inject_blur: bool = False,
        inject_static: bool = False,
        inject_noise: bool = False,
        is_image: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate a single video tensor [T, H, W, 3] in [0, 255] uint8 and metadata."""
        if is_image:
            frames = 1

        duration = frames / self.fps if not is_image else 0.0
        digit_size = 14
        digit_img = render_digit(digit, size=digit_size)

        # Trajectory
        rad = math.radians(angle_deg)
        vx = speed * math.cos(rad)
        vy = speed * math.sin(rad)
        if inject_static:
            vx, vy = 0.0, 0.0

        if start_pos is not None:
            x, y = float(start_pos[0]), float(start_pos[1])
        else:
            # Deterministic, unique position per clip
            h_val = abs(hash(clip_id))
            max_x = max(1, self.width - digit_size - 4)
            max_y = max(1, self.height - digit_size - 4)
            x = float(2 + (h_val % max_x))
            y = float(2 + ((h_val // 101) % max_y))

        video_frames = []
        for t in range(frames):
            frame = np.zeros((self.height, self.width), dtype=np.float32)

            if inject_noise:
                frame = np.random.uniform(0.0, 1.0, (self.height, self.width)).astype(np.float32)
            else:
                ix = int(round(x))
                iy = int(round(y))

                ix = max(0, min(self.width - digit_size, ix))
                iy = max(0, min(self.height - digit_size, iy))

                frame[iy : iy + digit_size, ix : ix + digit_size] = np.maximum(
                    frame[iy : iy + digit_size, ix : ix + digit_size], digit_img
                )

                x += vx
                y += vy
                if x <= 0 or x >= (self.width - digit_size):
                    vx = -vx
                if y <= 0 or y >= (self.height - digit_size):
                    vy = -vy

            if inject_blur:
                t_f = torch.from_numpy(frame)[None, None, :, :]
                kernel = torch.ones((1, 1, 5, 5), dtype=torch.float32) / 25.0
                t_f = F.conv2d(t_f, kernel, padding=2)
                frame = t_f[0, 0].numpy()

            if inject_letterbox:
                frame[:8, :] = 0.0
                frame[-8:, :] = 0.0

            rgb = np.repeat(np.clip(frame * 255.0, 0, 255).astype(np.uint8)[:, :, None], 3, axis=-1)
            video_frames.append(rgb)

        video_arr = np.stack(video_frames, axis=0)  # [T, H, W, 3]

        if inject_static or is_image:
            direction_str = "positioned statically on the canvas"
        elif abs(vx) > abs(vy):
            direction_str = "moving horizontally right" if vx > 0 else "moving horizontally left"
        else:
            direction_str = "moving vertically down" if vy > 0 else "moving vertically up"

        media_name = "image" if is_image else "video clip"
        short_caption = f"Digit {digit} {direction_str} on dark background."
        long_caption = (
            f"A high-contrast synthetic {media_name} showing a bright white numeral {digit} "
            f"situated on a deep black background. The digit is {direction_str} across the frame "
            f"with constant smooth velocity and crisp anti-aliased contours under fixed ambient lighting."
        )
        caption_dict = {
            "subject": f"digit {digit}",
            "action": f"{direction_str}",
            "setting": "deep black minimalist synthetic canvas",
            "style": "clean vector-raster graphics",
            "lighting": "emissive monochrome white foreground",
            "camera_motion": "static camera, fixed frame of reference",
            "is_synthetic": True,
        }

        meta = {
            "id": clip_id,
            "media_type": "image" if is_image else "video",
            "source": "moving_mnist",
            "license": "MIT",
            "duration": float(duration),
            "fps": float(self.fps),
            "frames": int(frames),
            "height": int(self.height),
            "width": int(self.width),
            "caption_short": short_caption,
            "caption_long": long_caption,
            "caption_json": json.dumps(caption_dict),
            "digit": digit,
            "injected_defect": (
                "letterbox"
                if inject_letterbox
                else "blur"
                if inject_blur
                else "static"
                if inject_static
                else "noise"
                if inject_noise
                else "none"
            ),
        }

        return video_arr, meta
