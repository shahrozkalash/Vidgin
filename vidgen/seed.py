"""Deterministic seeding for reproducible training (Part 7)."""

from __future__ import annotations

import random


def seed_everything(seed: int, num_threads: int = 2) -> int:
    """Seed python, numpy, and torch RNGs; fix CPU thread count for bitwise
    reproducibility across checkpoint resume boundaries."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(num_threads)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


def get_rng_states() -> dict:
    """Snapshot all RNG states for checkpointing.

    CUDA RNG states are stored as CPU tensors so saved checkpoints are
    device-portable (a checkpoint saved on a GPU machine must load cleanly
    on a CPU-only machine).
    """
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().cpu(),
        "torch_cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None,
    }


def set_rng_states(state: dict) -> None:
    """Restore RNG states saved by get_rng_states()."""
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        # get_rng_state_all() returns one state per device id, in order;
        # states were stored on CPU for portability, so move each back to
        # its own device before restoring.
        torch.cuda.set_rng_state_all(
            [s.to(f"cuda:{i}") for i, s in enumerate(cuda_states)]
        )
