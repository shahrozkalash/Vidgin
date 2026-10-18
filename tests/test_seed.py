"""Part 7: seed determinism / divergence."""

import random

import numpy as np
import pytest
import torch

from vidgen.seed import seed_everything


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_seed_determinism():
    seed_everything(7)
    a = float(torch.randn(1, device="cuda").item())
    seed_everything(7)
    b = float(torch.randn(1, device="cuda").item())
    assert a == b
