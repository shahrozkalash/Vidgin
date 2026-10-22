"""Shared training utilities: LR schedules, AMP, gradient scaling.

Phase 4/5 of the targeted improvement programme (2026-10-22). Both
trainers (DiT and VAE) use these helpers so a run's learning-rate
schedule, mixed precision, and gradient-accumulation behaviour are
identical in kind and recorded in every checkpoint's ``train_config``.

Design rules:
* Defaults reproduce the pre-Phase-4 behaviour exactly: constant LR
  (no scheduler), no autocast, no GradScaler, accumulation of 1.
* A cosine/linear schedule REQUIRES a total step count — decaying
  toward an unknown horizon would be a silent quality change.
* GradScaler exists only for fp16 on CUDA (bf16 needs no scaling, and
  CPU training is fp32 by default).
"""

from __future__ import annotations

import contextlib
import math

import torch

LR_SCHEDULES = ("none", "cosine", "linear")
AMP_MODES = ("none", "bf16", "fp16")
_COSINE_FLOOR = 0.05  # cosine decays to 5% of peak LR, not to zero


def build_lr_scheduler(optimizer, warmup_steps: int = 0,
                       total_steps: int | None = None,
                       schedule: str = "none"):
    """LambdaLR implementing warmup -> decay, or None for constant LR.

    ``schedule="none"`` with ``warmup_steps=0`` returns None (the
    historical behaviour). Warmup alone (schedule "none", warmup > 0)
    still returns a scheduler: linear ramp, then constant.
    """
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"unknown lr schedule {schedule!r}; choose {LR_SCHEDULES}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if schedule != "none" and total_steps is None:
        raise ValueError(
            f"lr schedule {schedule!r} needs total_steps so the decay "
            "has a horizon; pass the run's planned step count")
    if total_steps is not None and total_steps <= warmup_steps:
        raise ValueError(
            f"total_steps ({total_steps}) must exceed warmup_steps ({warmup_steps})")
    if schedule == "none" and warmup_steps == 0:
        return None

    def lr_lambda(epoch: int) -> float:
        if warmup_steps and epoch < warmup_steps:
            return (epoch + 1) / warmup_steps
        if schedule == "none" or total_steps is None:
            return 1.0
        progress = (epoch - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if schedule == "cosine":
            return _COSINE_FLOOR + (1 - _COSINE_FLOOR) * 0.5 * (
                1.0 + math.cos(math.pi * progress))
        return max(0.0, 1.0 - progress)  # linear

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def amp_dtype(amp: str):
    if amp == "bf16":
        return torch.bfloat16
    if amp == "fp16":
        return torch.float16
    return None


def amp_context(device, amp: str):
    """Autocast context for the forward/loss, or a no-op context."""
    dtype = amp_dtype(amp)
    if dtype is None:
        return contextlib.nullcontext()
    dev_type = torch.device(str(device)).type
    return torch.autocast(device_type=dev_type, dtype=dtype)


def make_grad_scaler(device, amp: str):
    """GradScaler only where it is needed: fp16 on CUDA."""
    if amp == "fp16" and torch.device(str(device)).type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


def validate_train_options(grad_accum: int, amp: str) -> None:
    if grad_accum < 1:
        raise ValueError(f"grad_accum must be >= 1, got {grad_accum}")
    if amp not in AMP_MODES:
        raise ValueError(f"unknown amp mode {amp!r}; choose {AMP_MODES}")
