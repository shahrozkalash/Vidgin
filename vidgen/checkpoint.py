"""Part 7: checkpoint save/resume with full RNG state.

Restoring a checkpoint reproduces the next training step bit-for-bit
(Part 7 audit), because python/numpy/torch RNG states travel with the
model and optimizer states.

Loading is SAFE by default (2026-10-28): checkpoints are unpickled with
``weights_only=True`` plus a small allowlist (the numpy types inside the
saved RNG state, and any config classes a caller names). Plain
``torch.load`` executes arbitrary code embedded in a pickle, and pool
workers exchange checkpoints between machines. A trusted legacy file
holding other objects can still be opened with VIDGIN_TRUST_CHECKPOINTS=1.
"""

from __future__ import annotations

import os
import pickle

import torch

from vidgen.seed import get_rng_states, set_rng_states

TRUST_ENV = "VIDGIN_TRUST_CHECKPOINTS"


def save_checkpoint(path: str, payload: dict) -> None:
    """Save a checkpoint ATOMICALLY (write temp file, then rename).

    A crash, OOM kill or cloud-session wipe mid-save previously left a
    truncated file at the checkpoint path, destroying the only resume
    point. The temp file lives in the same directory (same filesystem,
    so the rename is atomic) and a partial save never touches the real
    path; a leftover ``.tmp`` file is the visible sign of an interrupted
    save and can be deleted safely.

    The temp file is fsynced before the rename: without it a power loss
    or VM preemption can leave the renamed checkpoint pointing at data
    that never reached the disk (a zero-length or truncated file).
    """
    payload = dict(payload)
    payload["rng"] = get_rng_states()
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _numpy_safe_globals() -> list:
    """The numpy objects a pickled ``np.random.get_state()`` refers to,
    under both their numpy 2 and numpy 1 qualified names."""
    import numpy as np

    try:
        from numpy._core.multiarray import _reconstruct
    except ImportError:  # numpy 1.x
        from numpy.core.multiarray import _reconstruct
    return [
        np.ndarray, np.dtype, type(np.dtype(np.uint32)),
        (_reconstruct, "numpy._core.multiarray._reconstruct"),
        (_reconstruct, "numpy.core.multiarray._reconstruct"),
    ]


def load_checkpoint(path: str, map_location: str = "cpu", *,
                    restore_rng: bool = False,
                    safe_classes: tuple = ()) -> dict:
    """Load a checkpoint payload without executing pickled code.

    ``restore_rng``: also restore the saved python/numpy/torch RNG
    states. Off by default — only a training RESUME wants that; sampling,
    evaluation, pooling and weights-only init previously had their random
    streams silently rewound by every load.

    ``safe_classes``: extra classes (e.g. config dataclasses) the payload
    may contain, added to the unpickling allowlist.
    """
    try:
        with torch.serialization.safe_globals(
                _numpy_safe_globals() + list(safe_classes)):
            payload = torch.load(path, map_location=map_location,
                                 weights_only=True)
    except pickle.UnpicklingError as e:
        if os.environ.get(TRUST_ENV) != "1":
            raise ValueError(
                f"{path}: refusing to unpickle objects outside the "
                f"checkpoint allowlist ({e}). Unpickling can run arbitrary "
                f"code; if you trust this file, set {TRUST_ENV}=1 to load "
                f"it anyway.") from e
        payload = torch.load(path, map_location=map_location,
                             weights_only=False)
    if restore_rng and "rng" in payload:
        set_rng_states(payload["rng"])
    return payload


def inference_weights(payload: dict, use_ema: bool = True):
    """(dit_state, text_enc_state, used_ema) for sampling/evaluation.

    A run trained with ``--ema-decay`` carries exponential-moving-average
    weights, which usually sample noticeably better than the raw
    weights; they are preferred unless ``use_ema`` is False.
    """
    ema = payload.get("ema")
    if use_ema and ema is not None:
        return ema["dit"], ema["text_enc"], True
    return payload["dit"], payload["text_enc"], False
