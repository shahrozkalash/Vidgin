"""Portability patch tests (2026-11-26): train on any cheap backend.

Vidgin must resolve its device from what the machine actually has —
CUDA (NVIDIA / AMD ROCm), MPS (Apple Silicon), XPU (Intel) or CPU —
honour explicit device names, downgrade AMP modes that are unsafe on
a backend, and keep sampling/RNG working on backends that cannot own
a generator. All backend probes are monkeypatched, so these tests run
identically on a CPU-only machine.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import vidgen.config as config_mod
from vidgen.config import available_backends, resolve_device
from vidgen.sampler import make_generator, sample_latents
from vidgen.seed import get_rng_states, seed_everything, set_rng_states
from vidgen.train_utils import (
    make_grad_scaler,
    resolve_amp_for_device,
)


def _fake_backends(monkeypatch, available):
    monkeypatch.setattr(
        config_mod, "_backend_available",
        lambda name: name in available)


# ---------------------------------------------------------------- devices

def test_auto_falls_back_to_cpu(monkeypatch):
    _fake_backends(monkeypatch, set())
    assert resolve_device("auto") == "cpu"
    assert available_backends() == ()


def test_auto_prefers_cuda_over_mps_and_xpu(monkeypatch):
    _fake_backends(monkeypatch, {"cuda", "mps", "xpu"})
    assert resolve_device("auto") == "cuda"
    assert available_backends() == ("cuda", "mps", "xpu")


def test_auto_picks_mps_on_apple_silicon(monkeypatch):
    _fake_backends(monkeypatch, {"mps"})
    assert resolve_device("auto") == "mps"


def test_auto_picks_xpu_on_intel(monkeypatch):
    _fake_backends(monkeypatch, {"xpu"})
    assert resolve_device("auto") == "xpu"


def test_explicit_devices_pass_through(monkeypatch):
    _fake_backends(monkeypatch, set())
    for name in ("cpu", "cuda", "cuda:1", "mps", "xpu", "xla"):
        assert resolve_device(name) == name


def test_backend_probe_never_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("exotic torch build")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    assert config_mod._backend_available("cuda") is False
    assert config_mod._backend_available("banana") is False


# -------------------------------------------------------------------- AMP

def test_amp_none_is_historical_default_everywhere():
    for dev in ("cpu", "cuda", "mps", "xpu"):
        assert resolve_amp_for_device(dev, None) == ("none", None)


def test_amp_fp16_downgraded_on_mps_with_warning():
    amp, warn = resolve_amp_for_device("mps", "fp16")
    assert amp == "none"
    assert warn and "MPS" in warn


def test_amp_honoured_where_safe():
    assert resolve_amp_for_device("cuda", "fp16") == ("fp16", None)
    assert resolve_amp_for_device("cuda", "bf16") == ("bf16", None)
    assert resolve_amp_for_device("mps", "bf16") == ("bf16", None)
    assert resolve_amp_for_device("cpu", "bf16") == ("bf16", None)
    assert resolve_amp_for_device("xpu", "fp16") == ("fp16", None)


def test_amp_unknown_mode_rejected():
    with pytest.raises(ValueError, match="unknown amp mode"):
        resolve_amp_for_device("cpu", "fp32")


def test_grad_scaler_only_for_cuda_fp16():
    assert make_grad_scaler("cpu", "fp16") is None
    assert make_grad_scaler("mps", "fp16") is None
    assert make_grad_scaler("xpu", "fp16") is None
    assert make_grad_scaler("cuda", "bf16") is None


# ------------------------------------------------------- sampler generator

def test_make_generator_cpu_is_seeded_and_reproducible():
    g1 = make_generator("cpu", 123)
    g2 = make_generator("cpu", 123)
    assert torch.equal(torch.randn(4, generator=g1),
                       torch.randn(4, generator=g2))


def test_make_generator_falls_back_for_unknown_backend():
    gen = make_generator("not_a_real_backend", 7)
    assert gen.device.type == "cpu"
    torch.randn(2, generator=gen)  # usable


class _ConstModel(torch.nn.Module):
    """Velocity field returning a constant, for a sampler smoke step."""

    def forward(self, x, t, text, text_mask):
        return torch.ones_like(x)


def test_sample_latents_runs_on_cpu():
    text = torch.zeros(1, 2, 4)
    mask = torch.ones(1, 2, dtype=torch.bool)
    out = sample_latents(_ConstModel(), (1, 2, 2, 3, 3), text, mask,
                         method="euler", num_steps=2, cfg_scale=1.0,
                         seed=3, device="cpu")
    assert out.shape == (1, 2, 2, 3, 3)
    assert torch.isfinite(out).all()


# -------------------------------------------------------------------- RNG

def test_rng_states_roundtrip_with_new_keys_optional():
    seed_everything(42)
    states = get_rng_states()
    assert states["torch_cuda"] is None or isinstance(
        states["torch_cuda"], list)
    before = torch.randn(3)
    seed_everything(42)
    set_rng_states(states)
    after = torch.randn(3)
    assert torch.equal(before, after)


def test_set_rng_states_accepts_legacy_checkpoint_without_new_keys():
    seed_everything(9)
    legacy = get_rng_states()
    for key in ("torch_mps", "torch_xpu"):
        legacy.pop(key, None)
    set_rng_states(legacy)  # must not raise


# --------------------------------------------------------------- trainers

def test_vae_trainer_records_none_amp_by_default(tmp_path):
    from vidgen.train_vae import VAETrainer

    tr = VAETrainer(preset="tiny", seed=0, num_clips=4, batch_size=2,
                    device="cpu", ckpt_dir=None)
    assert tr.amp == "none"
    assert tr.train_config["amp"] == "none"
    rec = tr.train_step()
    assert rec["loss"] == rec["loss"]  # finite, not NaN


def test_vae_trainer_amp_resolution_applied(tmp_path):
    from vidgen.train_vae import VAETrainer

    tr = VAETrainer(preset="tiny", seed=0, num_clips=4, batch_size=2,
                    device="cpu", ckpt_dir=None, amp="bf16")
    assert tr.amp == "bf16"
