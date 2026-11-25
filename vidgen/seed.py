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
        # Apple / Intel accelerators: torch.manual_seed already seeds
        # them on modern builds, but seed explicitly where the API
        # exists so older builds behave the same. Never fatal.
        try:
            if torch.backends.mps.is_available():
                torch.mps.manual_seed_all(seed)
        except Exception:
            pass
        try:
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and xpu.is_available():
                xpu.manual_seed_all(seed)
        except Exception:
            pass
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

    # The numpy state's key array is stored as a torch tensor (not an
    # ndarray): a checkpoint then holds only tensors and plain python
    # values, so torch.load(weights_only=True) opens it on ANY torch
    # version without a numpy allowlist (see vidgen/checkpoint.py).
    name, keys, pos, has_gauss, cached = np.random.get_state()
    states = {
        "python": random.getstate(),
        "numpy": (name, torch.from_numpy(keys.astype(np.int64)), int(pos),
                  int(has_gauss), float(cached)),
        "torch": torch.get_rng_state().cpu(),
        "torch_cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None,
    }
    # Extra backends are additive keys: old checkpoints (without them)
    # and old code (ignoring them) both keep working.
    try:
        if torch.backends.mps.is_available():
            states["torch_mps"] = torch.mps.get_rng_state().cpu()
    except Exception:
        pass
    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            states["torch_xpu"] = [s.cpu() for s in xpu.get_rng_state_all()]
    except Exception:
        pass
    return states


def set_rng_states(state: dict) -> None:
    """Restore RNG states saved by get_rng_states().

    States are stored on CPU, but a checkpoint loaded with
    map_location='cuda' (or loaded on a GPU machine) may carry them on CUDA:
    torch.set_rng_state() only accepts a CPU tensor, so force CPU at restore.
    torch.cuda.set_rng_state_all() accepts CPU ByteTensors and routes them to
    the right device internally.
    """
    import numpy as np
    import torch

    random.setstate(state["python"])
    name, keys, *rest = state["numpy"]
    if isinstance(keys, torch.Tensor):  # current format (see get_rng_states)
        keys = keys.cpu().numpy().astype(np.uint32)
    np.random.set_state((name, keys, *rest))
    torch.set_rng_state(state["torch"].cpu())
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        if not isinstance(cuda_states, (list, tuple)):
            cuda_states = [cuda_states]
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])
    mps_state = state.get("torch_mps")
    if mps_state is not None:
        try:
            if torch.backends.mps.is_available():
                torch.mps.set_rng_state(mps_state.cpu())
        except Exception:
            pass
    xpu_states = state.get("torch_xpu")
    if xpu_states is not None:
        try:
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and xpu.is_available():
                if not isinstance(xpu_states, (list, tuple)):
                    xpu_states = [xpu_states]
                xpu.set_rng_state_all([s.cpu() for s in xpu_states])
        except Exception:
            pass
