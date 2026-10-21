"""Track A synthetic data: Moving MNIST generator + torch Dataset.

Produces real pixel clips [3, T, H, W] in [-1, 1] with controllable digit
trajectories, speeds, letterboxed black borders, static clips, and
near-duplicates — mirroring src/utils/movingMnist.ts in the studio app.
Digits are rendered from deterministic 5x7 bitmaps (no downloads needed).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

# 5x7 bitmap font for digits 0-9 ("#" = ink).
_DIGIT_BITMAPS = {
    0: ["#####", "#...#", "#...#", "#...#", "#...#", "#...#", "#####"],
    1: ["..#..", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    2: ["#####", "....#", "....#", "#####", "#....", "#....", "#####"],
    3: ["#####", "....#", "....#", "#####", "....#", "....#", "#####"],
    4: ["#...#", "#...#", "#...#", "#####", "....#", "....#", "....#"],
    5: ["#####", "#....", "#....", "#####", "....#", "....#", "#####"],
    6: ["#####", "#....", "#....", "#####", "#...#", "#...#", "#####"],
    7: ["#####", "....#", "....#", "...#.", "..#..", "..#..", "..#.."],
    8: ["#####", "#...#", "#...#", "#####", "#...#", "#...#", "#####"],
    9: ["#####", "#...#", "#...#", "#####", "....#", "....#", "#####"],
}

DIRECTIONS = {
    "left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1),
    "up-left": (-1, -1), "up-right": (1, -1), "down-left": (-1, 1), "down-right": (1, 1),
}


def render_digit(digit: int, size: int) -> np.ndarray:
    """Render one digit as a (size, size) float32 array in [0, 1]."""
    bmp = _DIGIT_BITMAPS[int(digit) % 10]
    small = np.array([[1.0 if c == "#" else 0.0 for c in row] for row in bmp],
                     dtype=np.float32)  # 7 x 5
    # nearest-neighbor upscale to (size, size)
    ys = (np.linspace(0, 7, size, endpoint=False)).astype(int)
    xs = (np.linspace(0, 5, size, endpoint=False)).astype(int)
    return small[ys[:, None], xs[None, :]]


def generate_moving_mnist_clip(
    seed: int,
    num_digits: int = 2,
    frame_count: int = 17,
    height: int = 64,
    width: int = 64,
    speeds: list[float] | None = None,
    directions: list[str] | None = None,
    digits: list[int] | None = None,
    digit_size: int = 24,
    letterbox: bool = True,
    static: bool = False,
) -> tuple[np.ndarray, str]:
    """Generate one clip.

    Returns (clip, caption) where clip is float32 [3, T, H, W] in [-1, 1].
    """
    rng = np.random.RandomState(seed)
    if digits is None:
        digits = [int(rng.randint(0, 10)) for _ in range(num_digits)]
    if directions is None:
        names = list(DIRECTIONS)
        directions = [names[int(rng.randint(0, len(names)))] for _ in range(num_digits)]
    if speeds is None:
        speeds = [float(rng.uniform(1.0, 3.0)) for _ in range(num_digits)]
    if static:
        speeds = [0.0] * num_digits

    border = 8 if letterbox else 0
    roam_h, roam_w = height - 2 * border, width - 2 * border

    stamps = [render_digit(d, digit_size) for d in digits]
    # start positions inside the roam area
    pos = [
        np.array([
            rng.uniform(0, max(roam_w - digit_size, 1)),
            rng.uniform(0, max(roam_h - digit_size, 1)),
        ])
        for _ in range(num_digits)
    ]
    vel = [
        np.array(DIRECTIONS[d], dtype=np.float32) * s
        for d, s in zip(directions, speeds)
    ]

    frames = np.zeros((frame_count, height, width), dtype=np.float32)
    for t in range(frame_count):
        canvas = np.zeros((height, width), dtype=np.float32)
        for i in range(num_digits):
            x, y = pos[i]
            xi, yi = int(round(x)) + border, int(round(y)) + border
            x0, x1 = max(xi, 0), min(xi + digit_size, width)
            y0, y1 = max(yi, 0), min(yi + digit_size, height)
            sx0, sy0 = x0 - xi, y0 - yi
            canvas[y0:y1, x0:x1] = np.maximum(
                canvas[y0:y1, x0:x1], stamps[i][sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
            )
            # bounce
            pos[i] = pos[i] + vel[i]
            for ax, lim in ((0, roam_w - digit_size), (1, roam_h - digit_size)):
                if lim <= 0:
                    continue
                if pos[i][ax] < 0:
                    pos[i][ax] = -pos[i][ax]
                    vel[i][ax] = -vel[i][ax]
                elif pos[i][ax] > lim:
                    pos[i][ax] = 2 * lim - pos[i][ax]
                    vel[i][ax] = -vel[i][ax]
        frames[t] = canvas

    clip = np.stack([frames, frames, frames], axis=0)  # [3, T, H, W]
    clip = clip * 2.0 - 1.0

    if static:
        caption = " and ".join(f"static digit {d}" for d in digits)
    else:
        caption = " and ".join(f"digit {d} moving {dr}" for d, dr in zip(digits, directions))
    return clip.astype(np.float32), caption


class MovingMNISTDataset(Dataset):
    """Deterministic per-index dataset: item i is seeded by base_seed + i,
    so it is reproducible across runs, workers, and checkpoint resumes."""

    def __init__(
        self,
        num_clips: int,
        height: int = 64,
        width: int = 64,
        frames: int = 17,
        num_digits: int = 2,
        base_seed: int = 0,
        letterbox: bool = True,
        shard: tuple[int, int] | None = None,
    ):
        self.num_clips = num_clips
        self.height = height
        self.width = width
        self.frames = frames
        self.num_digits = num_digits
        self.base_seed = base_seed
        self.letterbox = letterbox
        # Optional weight-averaging pool shard (vidgen/pool.py): local
        # index j maps to global clip index positions[j], so item content
        # (seeded by base_seed + global index) is identical to the
        # unsharded dataset's item at that global index. No shard = the
        # original identity mapping.
        self.shard = shard
        self._positions: list[int] | None = None
        if shard is not None:
            from vidgen.pool import shard_positions

            self._positions = shard_positions(num_clips, shard)
            if not self._positions:
                raise ValueError(
                    f"data shard {shard} selects no clips "
                    f"(num_clips={num_clips} too small)"
                )

    def __len__(self) -> int:
        if self._positions is not None:
            return len(self._positions)
        return self.num_clips

    def __getitem__(self, i: int):
        if self._positions is not None:
            i = self._positions[int(i)]
        clip, caption = generate_moving_mnist_clip(
            seed=self.base_seed + int(i),
            num_digits=self.num_digits,
            frame_count=self.frames,
            height=self.height,
            width=self.width,
            letterbox=self.letterbox,
        )
        return torch.from_numpy(clip), caption
