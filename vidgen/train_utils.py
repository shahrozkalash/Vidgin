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


class NonFiniteGuard:
    """Detects NaN/Inf training steps and protects the weights from them.

    A single non-finite loss (bad batch, loss spike under mixed
    precision, corrupted clip) previously flowed straight into the
    optimizer and poisoned the model permanently — every later step
    stayed NaN and the run was garbage without any message saying when
    it happened. The trainers now check the loss and gradient norm
    BEFORE stepping: a bad step is skipped (gradients discarded, no
    weight update) and counted; if bad steps repeat beyond
    ``max_bad`` the run stops loudly, because at that point something
    is systematically wrong (usual causes: learning rate too high,
    corrupted data, or fp16 without a scaler).
    """

    def __init__(self, max_bad: int = 10):
        self.max_bad = max_bad
        self.bad_steps = 0
        self.consecutive_bad = 0

    def check(self, *values: float) -> bool:
        """True when every value is finite (the step may proceed)."""
        import math

        if all(math.isfinite(float(v)) for v in values):
            self.consecutive_bad = 0
            return True
        self.bad_steps += 1
        self.consecutive_bad += 1
        if self.consecutive_bad > self.max_bad:
            raise RuntimeError(
                f"training stopped: {self.consecutive_bad} consecutive "
                "non-finite (NaN/Inf) steps — the loss or gradients "
                "blew up. Lower the learning rate, check the data for "
                "corrupt clips, and if using --amp fp16 make sure the "
                "run is on CUDA (the grad scaler only exists there).")
        return False


def guarded_optimizer_step(optimizer, scaler, guard: NonFiniteGuard,
                           loss_value: float, grad_norm: float) -> str:
    """Step the optimizer unless the step is non-finite; returns the outcome.

    Returns ``"stepped"``, ``"skipped_nonfinite"`` (the guard rejected the
    loss/gradients) or ``"skipped_overflow"`` (fp16 GradScaler found inf
    gradients and lowered its scale — routine early in fp16 training).

    With a GradScaler, gradient overflow is the SCALER's job, not the
    guard's: the guard only judges the loss there. Previously the guard
    also rejected the inf grad norm of an ordinary overflow step and
    skipped without ``scaler.update()``, so the scale never dropped and
    the very next ``unscale_()`` raised "unscale_() has already been
    called on this optimizer since the last update()" — every --amp fp16
    run on CUDA died at its first overflow (found 2026-10-28). Every
    path now ends with ``update()`` after the caller's ``unscale_()``.
    """
    if scaler is None:
        if guard.check(loss_value, grad_norm):
            optimizer.step()
            return "stepped"
        optimizer.zero_grad()
        return "skipped_nonfinite"
    if not guard.check(loss_value):
        optimizer.zero_grad()
        scaler.update()  # consume this step's inf checks; lowers the scale
        return "skipped_nonfinite"
    scale_before = scaler.get_scale()
    scaler.step(optimizer)  # skips internally when grads overflowed
    scaler.update()
    if scaler.get_scale() < scale_before:
        return "skipped_overflow"
    return "stepped"


class ModelEMA:
    """Exponential moving average of one or more modules' weights.

    Diffusion models sample noticeably better from EMA weights than from
    the raw, noisy end-of-training weights. The shadow copy is fp32 and
    lives on the modules' device; it never feeds back into training, so a
    run with EMA follows exactly the same trajectory as one without.

    The effective decay ramps up as ``min(decay, (1 + n) / (10 + n))``
    over the first updates, so short runs and fresh warm starts are not
    dominated by their random/donor initialization.
    """

    def __init__(self, modules: dict, decay: float):
        if not 0.0 < float(decay) < 1.0:
            raise ValueError(f"ema decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {}
        self.reset(modules)

    @torch.no_grad()
    def reset(self, modules: dict) -> None:
        """Restart the average from the modules' current weights."""
        self.updates = 0
        self.shadow = {
            name: {k: (v.detach().float().clone() if v.is_floating_point()
                       else v.detach().clone())
                   for k, v in m.state_dict().items()}
            for name, m in modules.items()}

    @torch.no_grad()
    def update(self, modules: dict) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for name, m in modules.items():
            shadow = self.shadow[name]
            for k, v in m.state_dict().items():
                if v.is_floating_point():
                    shadow[k].lerp_(v.detach().float(), 1.0 - d)
                else:
                    shadow[k].copy_(v)

    def state_dict(self) -> dict:
        out = {"decay": self.decay, "updates": self.updates}
        out.update(self.shadow)
        return out

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        self.updates = int(state["updates"])
        for name, shadow in self.shadow.items():
            for k, v in state[name].items():
                shadow[k].copy_(v)


def oom_guidance(context: str = "training") -> str:
    """Practical recovery steps printed when CUDA runs out of memory."""
    return (
        f"CUDA out of memory during {context}. In order of preference:\n"
        "  1. halve --batch-size (activation memory scales with it);\n"
        "  2. add --grad-checkpoint (recomputes activations; ~30% slower, "
        "much less memory) and recover the effective batch with "
        "--grad-accum;\n"
        "  3. add --amp bf16 (halves activation memory on Ampere+ GPUs);\n"
        "  4. train a smaller preset, or fewer frames per clip.\n"
        "Your checkpoints up to the last saved step are intact — "
        "resume with --resume <latest checkpoint> instead of restarting.")
