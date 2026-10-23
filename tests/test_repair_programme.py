"""Repair programme (2026-10-23) regression tests.

- The LR scheduler must track OPTIMIZER steps: a step skipped for
  non-finite loss/gradients must not advance the schedule (both
  trainers). Before the fix, a burst of bad steps burned the LR
  horizon while the weights stood still.
- MP4 export is atomic: a failed export leaves neither the
  destination nor a temp file behind; a successful one leaves no
  temp file either.
- The evaluation metrics are defined on 0..255 pixels but must not
  silently report ~0 for float clips in the model's [-1, 1] range
  (that trap produced a false "frozen video" reading live).
"""

import os

import numpy as np
import pytest
import torch

from vidgen.evaluate import (clip_distance, frozen_pair_fraction,
                             motion_mean_absdiff)
from vidgen.train import DiTTrainer
from vidgen.train_vae import VAETrainer
from vidgen import video_io


def _poison_first_param(module):
    p = next(module.parameters())
    saved = p.detach().clone()
    with torch.no_grad():
        p.fill_(float("nan"))
    return p, saved


def _restore(p, saved):
    with torch.no_grad():
        p.copy_(saved)


def test_dit_scheduler_waits_out_skipped_steps():
    torch.manual_seed(0)
    tr = DiTTrainer(preset="tiny", batch_size=1, device="cpu",
                    lr_schedule="cosine", total_steps=100,
                    warmup_steps=0)
    assert tr.sched is not None
    epoch0 = tr.sched.last_epoch
    p, saved = _poison_first_param(tr.vae)
    rec = tr.train_step()
    assert rec.get("skipped_nonfinite") is True
    assert tr.sched.last_epoch == epoch0, (
        "scheduler advanced on a skipped step — LR would decay "
        "without any training happening")
    _restore(p, saved)
    rec = tr.train_step()
    assert "skipped_nonfinite" not in rec
    assert tr.sched.last_epoch == epoch0 + 1


def test_vae_scheduler_waits_out_skipped_steps():
    torch.manual_seed(0)
    tr = VAETrainer(preset="tiny", batch_size=1, num_clips=4,
                    device="cpu", lr_schedule="cosine",
                    total_steps=100, warmup_steps=0)
    assert tr.sched is not None
    epoch0 = tr.sched.last_epoch
    p, saved = _poison_first_param(tr.vae)
    rec = tr.train_step()
    assert rec.get("skipped_nonfinite") is True
    assert tr.sched.last_epoch == epoch0
    _restore(p, saved)
    rec = tr.train_step()
    assert "skipped_nonfinite" not in rec
    assert tr.sched.last_epoch == epoch0 + 1


# ------------------------------------------------------------ MP4 export

def test_failed_mp4_export_leaves_nothing(monkeypatch, tmp_path):
    def boom(uint8, path, fps):
        # simulate a writer that dies after creating a partial file
        with open(path, "wb") as f:
            f.write(b"partial")
        raise RuntimeError("encoder died")

    monkeypatch.setattr(video_io, "_write_mp4_cv2", boom)
    monkeypatch.setattr(video_io, "_write_mp4_ffmpeg", boom)
    dest = str(tmp_path / "clip.mp4")
    frames = np.zeros((4, 16, 16, 3), dtype=np.float32)
    with pytest.raises(RuntimeError):
        video_io.write_mp4(frames, dest)
    assert not os.path.exists(dest)
    assert os.listdir(tmp_path) == []  # no temp leftovers either


def test_successful_mp4_export_leaves_no_temp(tmp_path):
    rng = np.random.default_rng(0)
    frames = rng.uniform(-1, 1, (5, 16, 16, 3)).astype(np.float32)
    dest = str(tmp_path / "ok.mp4")
    video_io.write_mp4(frames, dest)
    assert os.path.isfile(dest) and os.path.getsize(dest) > 0
    assert os.listdir(tmp_path) == ["ok.mp4"]


# ------------------------------------------------------- metric ranges

def test_metrics_agree_between_uint8_and_model_range():
    rng = np.random.default_rng(3)
    clip8 = rng.integers(0, 255, (17, 32, 32, 3), dtype=np.uint8)
    clipf = clip8.astype(np.float32) / 127.5 - 1.0  # model range
    m8, mf = motion_mean_absdiff(clip8), motion_mean_absdiff(clipf)
    assert m8 > 10.0  # the clip genuinely moves
    assert abs(m8 - mf) < 1.0, (m8, mf)
    other8 = rng.integers(0, 255, (17, 32, 32, 3), dtype=np.uint8)
    otherf = other8.astype(np.float32) / 127.5 - 1.0
    d8 = clip_distance(clip8, other8)
    df = clip_distance(clipf, otherf)
    assert d8 > 5.0 and abs(d8 - df) < 1.0, (d8, df)
    assert clip_distance(clip8, clip8) == 0.0
    assert frozen_pair_fraction(clipf) == frozen_pair_fraction(clip8)
