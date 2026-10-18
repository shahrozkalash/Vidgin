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
    """Snapshot all RNG states for checkpointing."""
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
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
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
