"""Part 1: lightweight CPU quality/signal filters (numpy).

Thresholds mirror README.md / AUDIT.md:
  sharpness  Var(Laplacian) >= 20.0   (on 0-255 scale)
  motion     0.5 <= mean|frame diff| <= 80.0  (drops static clips and chaos)
  aesthetic  entropy-based score >= 4.0
"""

from __future__ import annotations

import numpy as np

SHARPNESS_MIN = 20.0
MOTION_MIN, MOTION_MAX = 0.5, 80.0
AESTHETIC_MIN = 4.0


def _to_gray_uint8(clip: np.ndarray) -> np.ndarray:
    """clip [3, T, H, W] in [-1, 1] -> gray [T, H, W] uint8."""
    gray = clip.mean(axis=0)  # [T, H, W]
    return np.clip((gray * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)


def laplacian_variance(frame: np.ndarray) -> float:
    """Variance of the discrete Laplacian (sharpness proxy)."""
    f = frame.astype(np.float32)
    lap = (
        -4 * f[1:-1, 1:-1]
        + f[:-2, 1:-1] + f[2:, 1:-1] + f[1:-1, :-2] + f[1:-1, 2:]
    )
    return float(lap.var())


def sharpness(clip: np.ndarray) -> float:
    gray = _to_gray_uint8(clip)
    return float(np.mean([laplacian_variance(gray[t]) for t in range(gray.shape[0])]))


def motion(clip: np.ndarray) -> float:
    gray = _to_gray_uint8(clip).astype(np.float32)
    return float(np.abs(gray[1:] - gray[:-1]).mean())


def aesthetic(clip: np.ndarray) -> float:
    """Mean per-frame Shannon entropy of the grayscale histogram (0..8 bits)."""
    gray = _to_gray_uint8(clip)
    scores = []
    for t in range(gray.shape[0]):
        hist, _ = np.histogram(gray[t], bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        scores.append(float(-(hist * np.log2(hist)).sum()))
    return float(np.mean(scores))


def passes_quality(clip: np.ndarray) -> dict:
    """Return per-filter scores and an overall pass flag."""
    s, m, a = sharpness(clip), motion(clip), aesthetic(clip)
    ok = (s >= SHARPNESS_MIN) and (MOTION_MIN <= m <= MOTION_MAX) and (a >= AESTHETIC_MIN)
    return {
        "sharpness": s, "sharpness_ok": s >= SHARPNESS_MIN,
        "motion": m, "motion_ok": MOTION_MIN <= m <= MOTION_MAX,
        "aesthetic": a, "aesthetic_ok": a >= AESTHETIC_MIN,
        "pass": ok,
    }
