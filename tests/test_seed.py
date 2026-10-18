"""Part 7: seed determinism / divergence."""

import random

import numpy as np
import pytest
import torch

from vidgen.seed import get_rng_states, seed_everything, set_rng_states


def _draw():
    return (random.random(), float(np.random.rand()), float(torch.randn(1).item()))


def test_seed_everything_determinism():
    seed_everything(7)
    a = _draw()
    seed_everything(7)
    b = _draw()
    assert a == b


def test_seed_everything_divergence():
    seed_everything(7)
    a = _draw()
    seed_everything(8)
    b = _draw()
    assert a != b


def test_rng_states_roundtrip_reproduces_draws():
    # Explicit save/restore contract: snapshot the RNGs mid-stream, draw,
    # restore the snapshot, and the next draws must match bit-for-bit.
    seed_everything(123)
    _draw()  # advance all three RNGs
    snapshot = get_rng_states()
    a = _draw()
    set_rng_states(snapshot)
    b = _draw()
    assert a == b


def test_set_rng_states_accepts_cpu_tensors():
    # The restore path must force CPU: torch.set_rng_state() rejects CUDA
    # tensors, which is what a checkpoint loaded with map_location='cuda'
    # would carry. On a CPU-only machine this asserts the no-op case.
    seed_everything(5)
    snapshot = get_rng_states()
    set_rng_states(snapshot)  # must not raise
    assert snapshot["torch"].device.type == "cpu"
    assert snapshot["torch_cuda"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_seed_determinism():
    seed_everything(7)
    a = float(torch.randn(1, device="cuda").item())
    seed_everything(7)
    b = float(torch.randn(1, device="cuda").item())
    assert a == b
