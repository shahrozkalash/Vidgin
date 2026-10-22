"""Tests for Phase 4/5: LR schedules, AMP, grad accumulation, throughput."""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vidgen.train import DiTTrainer
from vidgen.train_utils import build_lr_scheduler, validate_train_options
from vidgen.train_vae import VAETrainer


def _opt(lr=1.0):
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([p], lr=lr)


def test_warmup_then_cosine_values():
    opt = _opt()
    sched = build_lr_scheduler(opt, warmup_steps=2, total_steps=6,
                               schedule="cosine")
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(6):
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[0] == pytest.approx(0.5)   # warmup epoch 0: (0+1)/2
    assert lrs[1] == pytest.approx(1.0)  # warmup done
    assert lrs[2] == pytest.approx(1.0)  # cosine progress 0
    assert lrs[-1] == pytest.approx(0.05, abs=0.02)  # cosine floor
    assert all(b <= a + 1e-9 for a, b in zip(lrs[2:], lrs[3:]))  # decays


def test_schedule_validation():
    with pytest.raises(ValueError, match="total_steps"):
        build_lr_scheduler(_opt(), schedule="cosine")  # no horizon
    with pytest.raises(ValueError, match="unknown lr schedule"):
        build_lr_scheduler(_opt(), schedule="banana")
    assert build_lr_scheduler(_opt()) is None  # historical default
    with pytest.raises(ValueError):
        validate_train_options(0, "none")
    with pytest.raises(ValueError):
        validate_train_options(1, "fp32")


def test_dit_trainer_schedule_accum_amp(tmp_path):
    tr = DiTTrainer(preset="tiny", seed=0, num_clips=8, batch_size=2,
                    ckpt_dir=None, warmup_steps=2, lr_schedule="cosine",
                    total_steps=6, grad_accum=2, amp="bf16")
    lrs = []
    for _ in range(3):
        rec = tr.train_step()
        assert math.isfinite(rec["loss"])
        assert rec["secs"] > 0 and rec["clips_per_sec"] > 0
        lrs.append(rec["lr"])
    assert tr.step == 3  # optimizer steps, not micro-batches
    path = str(tmp_path / "dit_sched.pt")
    tr.save(path)

    from vidgen.checkpoint import load_checkpoint
    payload = load_checkpoint(path)
    assert payload["train_config"]["grad_accum"] == 2
    assert payload["train_config"]["amp"] == "bf16"
    assert payload["train_config"]["lr_schedule"] == "cosine"
    assert "sched" in payload

    tr2 = DiTTrainer.resume(path, seed=0, num_clips=8, batch_size=2,
                            ckpt_dir=None)
    assert tr2.train_config == tr.train_config
    assert tr2.sched is not None and tr2.grad_accum == 2
    rec = tr2.train_step()
    assert math.isfinite(rec["loss"])


def test_vae_trainer_schedule_accum_amp(tmp_path):
    tr = VAETrainer(preset="tiny", seed=0, num_clips=8, batch_size=2,
                    ckpt_dir=None, warmup_steps=1, lr_schedule="linear",
                    total_steps=4, grad_accum=2, amp="bf16")
    for _ in range(2):
        rec = tr.train_step()
        assert math.isfinite(rec["loss"])
        assert rec["clips_per_sec"] > 0
    path = str(tmp_path / "vae_sched.pt")
    tr.save(path)
    from vidgen.checkpoint import load_checkpoint
    payload = load_checkpoint(path)
    assert payload["train_config"]["lr_schedule"] == "linear"
    tr2 = VAETrainer.resume(path, seed=0, num_clips=8, batch_size=2,
                            ckpt_dir=None)
    assert tr2.train_config == tr.train_config
    assert math.isfinite(tr2.train_step()["loss"])


def test_defaults_reproduce_historical_behaviour(tmp_path):
    tr = DiTTrainer(preset="tiny", seed=0, num_clips=8, batch_size=2,
                    ckpt_dir=None)
    assert tr.sched is None and tr.scaler is None
    assert tr.train_config == {"warmup_steps": 0, "lr_schedule": "none",
                               "total_steps": None, "grad_accum": 1,
                               "amp": "none"}
    path = str(tmp_path / "plain.pt")
    tr.save(path)
    from vidgen.checkpoint import load_checkpoint
    payload = load_checkpoint(path)
    assert "sched" not in payload and "scaler" not in payload
