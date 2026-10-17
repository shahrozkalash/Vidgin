"""Unit tests for deterministic seeding across Python, NumPy, and PyTorch."""

import random
import numpy as np
import torch
import pytest

from vidgen.utils.seed import seed_everything, seed_worker


def test_seed_everything_determinism():
    """Verify seed_everything produces bit-for-bit identical numbers."""
    seed_everything(1337)
    py_1 = random.random()
    np_1 = np.random.randn(5)
    th_1 = torch.randn(5)

    seed_everything(1337)
    py_2 = random.random()
    np_2 = np.random.randn(5)
    th_2 = torch.randn(5)

    assert py_1 == py_2
    np.testing.assert_array_equal(np_1, np_2)
    torch.testing.assert_close(th_1, th_2)


def test_seed_everything_divergence():
    """Verify different seeds produce divergent numbers."""
    seed_everything(1337)
    th_1 = torch.randn(10)

    seed_everything(1338)
    th_2 = torch.randn(10)

    assert not torch.allclose(th_1, th_2)


@pytest.mark.gpu
def test_gpu_seed_determinism():
    """GPU-only test: verify CUDA RNG determinism (skipped on CPU)."""
    assert torch.cuda.is_available()
    seed_everything(42)
    t1 = torch.randn(10, device="cuda")

    seed_everything(42)
    t2 = torch.randn(10, device="cuda")

    torch.testing.assert_close(t1, t2)
