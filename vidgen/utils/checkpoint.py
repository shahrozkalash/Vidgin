"""Full-fidelity checkpoint save and resume system.

Preserves:
- Model state dict
- EMA weights state dict
- Optimizer state dict
- LR Scheduler state dict (optional)
- Global step and epoch
- RNG state: Python, NumPy, PyTorch CPU, and PyTorch CUDA
- Data sampler state
- Model configuration
"""

from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


def get_rng_state() -> Dict[str, Any]:
    """Capture full RNG state across Python, NumPy, and PyTorch."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda"] = None
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore exact RNG state across Python, NumPy, and PyTorch."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    save_path: str | Path,
    step: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Optional[Any] = None,
    lr_scheduler: Optional[Any] = None,
    sampler_state: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
) -> Path:
    """Save an atomic, reproducible checkpoint.

    Args:
        save_path: Destination path for checkpoint (.pt).
        step: Current global training step.
        epoch: Current dataset epoch.
        model: PyTorch model.
        optimizer: PyTorch optimizer.
        ema: Optional EMA model wrapper.
        lr_scheduler: Optional learning rate scheduler.
        sampler_state: Optional dictionary tracking data iterator/sampler state.
        config: Optional configuration dictionary.

    Returns:
        Path to the saved checkpoint.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP / compiled models if present
    raw_model = model.module if hasattr(model, "module") else model

    checkpoint = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": get_rng_state(),
        "sampler_state": sampler_state if sampler_state is not None else {},
    }

    if ema is not None:
        checkpoint["ema_state_dict"] = ema.state_dict()

    if lr_scheduler is not None:
        checkpoint["lr_scheduler_state_dict"] = lr_scheduler.state_dict()

    if config is not None:
        checkpoint["config"] = config if isinstance(config, dict) else str(config)

    # Save to temporary file first for atomic write safety
    temp_path = save_path.with_suffix(".tmp")
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, save_path)
    return save_path


def resume_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional[Any] = None,
    lr_scheduler: Optional[Any] = None,
    restore_rng: bool = True,
) -> Tuple[int, int, Dict[str, Any]]:
    """Resume training state from a checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint.
        model: PyTorch model to restore weights into.
        optimizer: Optional optimizer to restore state into.
        ema: Optional EMA wrapper to restore weights into.
        lr_scheduler: Optional scheduler to restore.
        restore_rng: If True, restores exact RNG states for bit-for-bit reproducibility.

    Returns:
        Tuple of (step, epoch, sampler_state).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if ema is not None and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])

    if lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    if restore_rng and "rng_state" in checkpoint:
        set_rng_state(checkpoint["rng_state"])

    step = checkpoint.get("step", 0)
    epoch = checkpoint.get("epoch", 0)
    sampler_state = checkpoint.get("sampler_state", {})

    return step, epoch, sampler_state
