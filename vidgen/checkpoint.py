"""Part 7: checkpoint save/resume with full RNG state.

Restoring a checkpoint reproduces the next training step bit-for-bit
(Part 7 audit), because python/numpy/torch RNG states travel with the
model and optimizer states.
"""

from __future__ import annotations

import torch

from vidgen.seed import get_rng_states, set_rng_states


def save_checkpoint(path: str, payload: dict) -> None:
    payload = dict(payload)
    payload["rng"] = get_rng_states()
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if "rng" in payload:
        set_rng_states(payload["rng"])
    return payload
