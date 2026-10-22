"""Part 7: checkpoint save/resume with full RNG state.

Restoring a checkpoint reproduces the next training step bit-for-bit
(Part 7 audit), because python/numpy/torch RNG states travel with the
model and optimizer states.
"""

from __future__ import annotations

import os

import torch

from vidgen.seed import get_rng_states, set_rng_states


def save_checkpoint(path: str, payload: dict) -> None:
    """Save a checkpoint ATOMICALLY (write temp file, then rename).

    A crash, OOM kill or cloud-session wipe mid-save previously left a
    truncated file at the checkpoint path, destroying the only resume
    point. The temp file lives in the same directory (same filesystem,
    so the rename is atomic) and a partial save never touches the real
    path; a leftover ``.tmp`` file is the visible sign of an interrupted
    save and can be deleted safely.
    """
    payload = dict(payload)
    payload["rng"] = get_rng_states()
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if "rng" in payload:
        set_rng_states(payload["rng"])
    return payload
