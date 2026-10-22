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


def brightness(clip: np.ndarray) -> float:
    """Mean grayscale level on the 0..255 scale (darkness/overexposure)."""
    return float(_to_gray_uint8(clip).mean())


def passes_quality(clip: np.ndarray, thresholds: dict | None = None) -> dict:
    """Return per-filter scores and an overall pass flag.

    ``thresholds`` optionally overrides any of the module defaults:
    sharpness_min, motion_min, motion_max, aesthetic_min, plus the new
    brightness_min / brightness_max bounds (defaults 0..255 = no bound,
    so historical behaviour is unchanged unless a caller opts in).
    These scores are heuristics, not universal quality truths: a valid
    night scene is dark, a valid portrait can be near-static. Callers
    preparing real datasets should set thresholds for their footage
    instead of treating the defaults as laws.
    """
    t = {
        "sharpness_min": SHARPNESS_MIN, "motion_min": MOTION_MIN,
        "motion_max": MOTION_MAX, "aesthetic_min": AESTHETIC_MIN,
        "brightness_min": 0.0, "brightness_max": 255.0,
    }
    if thresholds:
        unknown = set(thresholds) - set(t)
        if unknown:
            raise ValueError(f"unknown quality thresholds: {sorted(unknown)}")
        t.update(thresholds)
    s, m, a, b = sharpness(clip), motion(clip), aesthetic(clip), brightness(clip)
    ok = (s >= t["sharpness_min"] and t["motion_min"] <= m <= t["motion_max"]
          and a >= t["aesthetic_min"]
          and t["brightness_min"] <= b <= t["brightness_max"])
    return {
        "sharpness": s, "sharpness_ok": s >= t["sharpness_min"],
        "motion": m, "motion_ok": t["motion_min"] <= m <= t["motion_max"],
        "aesthetic": a, "aesthetic_ok": a >= t["aesthetic_min"],
        "brightness": b,
        "brightness_ok": t["brightness_min"] <= b <= t["brightness_max"],
        "pass": ok,
    }
