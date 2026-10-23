"""Non-learned video-quality metrics for generated clips.

Phase 6 of the targeted improvement programme (2026-10-22). Every
metric here is a plain statistic over pixels — no pretrained evaluator
and no external service, per Vidgin's ownership rule. These are
PROXIES with real limitations, documented per metric; they catch
regressions and degenerate outputs (frozen frames, flicker storms,
mode collapse, verbatim memorization). They cannot judge whether a
clip matches its prompt's meaning — prompt adherence needs either a
learned text-video evaluator (excluded: it would be someone else's
brain) or human review of the saved sample videos, which is what the
evaluation report is for.

Clips are uint8 arrays [T, H, W, 3].
"""

from __future__ import annotations

import numpy as np


def _brightness(clip: np.ndarray) -> np.ndarray:
    """Per-frame mean brightness, shape [T]."""
    return _as_float255(clip).mean(axis=(1, 2, 3))


def _as_float255(clip: np.ndarray) -> np.ndarray:
    """Normalize a clip to float pixels on the 0..255 scale.

    This module's metrics are defined on 0..255 pixels, and every
    in-repo caller passes uint8 — but a float clip in [-1, 1] (the
    model's native range) used to slip through silently: casts to
    int16 truncated nearly every value to -1/0/1 and the metrics
    reported ~0 motion and ~0 distance for real, moving videos
    (hit live in the 2026-10-23 paired-prompt check). Float clips
    are now rescaled explicitly: [-1, 1] or [0, 1] -> [0, 255].
    Integer clips pass through untouched.
    """
    a = clip.astype(np.float32)
    if clip.dtype.kind == "f" and a.size and float(a.max()) <= 1.5:
        if float(a.min()) < 0.0:
            a = (a + 1.0) * 127.5
        else:
            a = a * 255.0
    return a


def motion_mean_absdiff(clip: np.ndarray) -> float:
    """Mean |frame[t] - frame[t-1]| over the clip (0 = frozen video)."""
    if clip.shape[0] < 2:
        return 0.0
    a = _as_float255(clip)
    diff = np.abs(a[1:] - a[:-1])
    return float(diff.mean())


def flicker_score(clip: np.ndarray) -> float:
    """Brightness second-difference: high-frequency luminance jitter.

    Mean |b[t] - 2*b[t-1] + b[t-2]| over time. Smooth pans change
    brightness ~linearly (second difference ~0); flicker oscillates,
    scoring high. A static clip scores 0 — read together with motion.
    """
    if clip.shape[0] < 3:
        return 0.0
    b = _brightness(clip)
    second = b[2:] - 2 * b[1:-1] + b[:-2]
    return float(np.abs(second).mean())


def sharpness(clip: np.ndarray) -> float:
    """Mean gradient magnitude (higher = crisper edges, less blur)."""
    gray = clip.astype(np.float32).mean(axis=-1)
    gx = np.abs(np.diff(gray, axis=2)).mean()
    gy = np.abs(np.diff(gray, axis=1)).mean()
    return float(gx + gy)


def clip_signature(clip: np.ndarray, size: int = 16) -> np.ndarray:
    """Downscaled float frames [T, size, size, 3] for cheap distances."""
    t, h, w, _ = clip.shape
    # Stride-sample to at most `size` per side, then block-mean to exact.
    ys = np.linspace(0, h - 1, size).astype(int)
    xs = np.linspace(0, w - 1, size).astype(int)
    return _as_float255(clip[:, ys][:, :, xs])


def clip_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between two clips' signatures."""
    return float(np.abs(clip_signature(a) - clip_signature(b)).mean())


def diversity(clips: list[np.ndarray]) -> float:
    """Mean pairwise clip distance (near 0 = mode collapse)."""
    if len(clips) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(len(clips)):
        for j in range(i + 1, len(clips)):
            total += clip_distance(clips[i], clips[j])
            pairs += 1
    return total / pairs


def memorization_distance(clips: list[np.ndarray],
                          train_clips: list[np.ndarray]) -> dict:
    """Nearest-training-clip distance per generated clip.

    A generated clip whose distance to its nearest training clip is
    near zero is a copy, not a creation. Returns mean/min over the
    generated set (only meaningful with a real training set to compare
    against; the report says so when none was supplied).
    """
    if not clips or not train_clips:
        return {"mean_nearest": None, "min_nearest": None}
    nearest = [min(clip_distance(c, t) for t in train_clips) for c in clips]
    return {"mean_nearest": float(np.mean(nearest)),
            "min_nearest": float(np.min(nearest))}


def frozen_pair_fraction(clip: np.ndarray, eps: float = 0.5) -> float:
    """Fraction of consecutive frame pairs that are (near-)identical.

    A generated clip whose frames barely change is frozen, whatever
    its per-frame sharpness says. ``eps`` is a mean-absolute-difference
    threshold on the clips' pixel scale (0..255 for the uint8 clips
    this module evaluates; 0.5 ≈ sub-gray-level average change).
    1.0 = a still image, 0.0 = every frame moves.
    """
    if len(clip) < 2:
        return 1.0
    diffs = np.abs(np.diff(_as_float255(clip), axis=0)).mean(axis=(1, 2, 3))
    return float(np.mean(diffs < eps))


def evaluate_clip_set(clips: list[np.ndarray],
                      train_clips: list[np.ndarray] | None = None) -> dict:
    """Aggregate metrics for a set of generated clips."""
    if not clips:
        raise ValueError("no clips to evaluate")
    return {
        "clips": len(clips),
        "motion_mean_absdiff": float(np.mean(
            [motion_mean_absdiff(c) for c in clips])),
        "flicker_score": float(np.mean([flicker_score(c) for c in clips])),
        "sharpness": float(np.mean([sharpness(c) for c in clips])),
        "diversity": diversity(clips),
        "frozen_pair_fraction": float(np.mean(
            [frozen_pair_fraction(c) for c in clips])),
        "memorization": memorization_distance(clips, train_clips or []),
        "prompt_adherence": (
            "not measured: requires a learned text-video evaluator, "
            "which Vidgin's ownership rule excludes; review the saved "
            "sample videos instead"),
    }
