"""Tests for Phases 6/7: non-learned evaluation metrics + MP4 export."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vidgen.evaluate import (
    diversity,
    evaluate_clip_set,
    flicker_score,
    memorization_distance,
    motion_mean_absdiff,
)
from vidgen.video_io import probe_mp4, write_mp4

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _clip(fill_fn, t=17, h=32, w=32):
    frames = np.zeros((t, h, w, 3), dtype=np.uint8)
    for i in range(t):
        frames[i] = fill_fn(i)
    return frames


def test_mp4_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    frames = rng.uniform(-1, 1, size=(17, 32, 32, 3)).astype(np.float32)
    path = str(tmp_path / "clip.mp4")
    write_mp4(frames, path, fps=16)
    count, width, height = probe_mp4(path)
    assert count == 17 and width == 32 and height == 32
    assert os.path.getsize(path) > 0


def test_metrics_discriminate():
    frozen = _clip(lambda i: np.full((32, 32, 3), 100, np.uint8))
    assert motion_mean_absdiff(frozen) == 0.0
    assert flicker_score(frozen) == 0.0

    flickery = _clip(lambda i: np.full((32, 32, 3),
                                       200 if i % 2 else 20, np.uint8))
    assert flicker_score(flickery) > 100

    moving = _clip(lambda i: np.full((32, 32, 3), 20 + i * 5, np.uint8))
    assert motion_mean_absdiff(moving) == pytest.approx(5.0)
    assert flicker_score(moving) < flicker_score(flickery)


def test_diversity_and_memorization():
    a = _clip(lambda i: np.full((32, 32, 3), 30, np.uint8))
    b = _clip(lambda i: np.full((32, 32, 3), 220, np.uint8))
    assert diversity([a, a]) == 0.0
    assert diversity([a, b]) > 100
    memo = memorization_distance([a], [a, b])
    assert memo["mean_nearest"] == 0.0  # exact copy detected
    far = memorization_distance([b], [a])
    assert far["mean_nearest"] > 100
    report = evaluate_clip_set([a, b], [a])
    assert report["clips"] == 2
    assert "not measured" in report["prompt_adherence"]


def test_evaluate_checkpoint_smoke(tmp_path):
    """End-to-end: tiny trained-for-2-steps DiT + untrained VAE -> report."""
    from vidgen.train import DiTTrainer
    from vidgen.train_vae import VAETrainer

    vae_tr = VAETrainer(preset="tiny", seed=0, num_clips=8, batch_size=2,
                        ckpt_dir=None)
    vae_path = str(tmp_path / "vae.pt")
    vae_tr.save(vae_path)
    dit_tr = DiTTrainer(preset="tiny", seed=0, num_clips=8, batch_size=2,
                        ckpt_dir=None, vae_ckpt=vae_path)
    dit_tr.train_step()
    dit_tr.train_step()
    dit_path = str(tmp_path / "dit.pt")
    dit_tr.save(dit_path)

    out_dir = str(tmp_path / "eval")
    proc = subprocess.run(
        [sys.executable, "scripts/evaluate_checkpoint.py",
         "--dit-ckpt", dit_path, "--vae-ckpt", vae_path,
         "--prompts", "people walking", "--seeds", "0",
         "--steps", "4", "--out-dir", out_dir],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.load(open(os.path.join(out_dir, "report.json")))
    assert report["candidate"]["metrics"]["clips"] == 1
    assert os.path.isfile(os.path.join(out_dir, "candidate", "clip_000.mp4"))
    count, _w, _h = probe_mp4(os.path.join(out_dir, "candidate", "clip_000.mp4"))
    assert count == 17
    assert os.path.isfile(os.path.join(out_dir, "report.md"))
