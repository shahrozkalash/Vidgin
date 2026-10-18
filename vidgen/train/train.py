"""Full Flow Matching Trainer for Wan Video Latent Diffusion.

Features:
- Rectified Flow Loss with shifted logit-normal timestep distribution (default s=5.0).
- Timestep input scaled by 1000.0.
- 10% Classifier-Free Guidance (CFG) text dropout to empty string representation.
- AdamW optimizer with betas=(0.9, 0.95), weight_decay=0.01, linear warmup + cosine decay.
- Gradient clipping (max_norm=1.0).
- Automatic Mixed Precision (BF16 autocast with FP32 master weights).
- Exponential Moving Average (EMA) of weights (decay 0.999 for short, 0.9999 for long runs).
- Dynamic timestep-binned loss logging ([0.0-0.1], ..., [0.9-1.0]).
- Profiling: tokens per second and Model FLOPs Utilization (MFU).
- Invariant global batch size scaling: B_global = B_local * world_size * grad_accum.
- Resumable from atomic checkpoints across any GPU count.
"""

from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW

from vidgen.train.losses import RectifiedFlowLoss
from vidgen.train.ema import EMAModel
from vidgen.train.profiler import TrainingProfiler
from vidgen.train.parallel import (
    ParallelismConfig,
    calculate_memory_footprint_gb,
    compute_invariant_batch_schedule,
    ReshardingCoordinator,
)
from vidgen.utils.checkpoint import save_checkpoint, resume_checkpoint, get_rng_state, set_rng_state


class WanTrainer:
    """Production-grade trainer for Wan Video Diffusion Transformers."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 50000,
        grad_clip: float = 1.0,
        ema_decay: float = 0.999,
        shift: float = 5.0,
        text_drop_prob: float = 0.1,
        mixed_precision: str = "bf16",
        device: torch.device | str = "cpu",
        checkpoint_dir: str | Path = "./checkpoints",
        world_size: int = 1,
        rank: int = 0,
        target_global_batch_size: int = 16,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.world_size = max(1, world_size)
        self.rank = rank
        self.target_global_batch_size = target_global_batch_size

        # Invariant batch calculation
        self.local_batch, self.grad_accum, self.effective_global_batch = (
            compute_invariant_batch_schedule(target_global_batch_size, self.world_size)
        )

        # Loss function
        self.loss_fn = RectifiedFlowLoss(
            shift=shift,
            text_drop_prob=text_drop_prob,
        ).to(self.device)

        # Optimizer: fused AdamW if supported on CUDA, else standard AdamW
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )

        # EMA Model
        self.ema = EMAModel(self.model, decay=ema_decay)

        # Profiler
        self.profiler = TrainingProfiler(
            model=self.model,
            num_gpus=self.world_size,
        )

        # State tracking
        self.global_step = 0
        self.epoch = 0
        self.step_in_epoch = 0

    def get_lr(self, step: int) -> float:
        """Compute learning rate with linear warmup and cosine decay."""
        if step < self.warmup_steps:
            return self.learning_rate * float(step + 1) / float(max(1, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
        return self.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    def update_lr(self, step: int) -> float:
        """Set learning rate for optimizer parameter groups."""
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def train_step(
        self,
        x_0: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        null_embedding: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, Any]:
        """Perform a single training step with gradient accumulation and clipping.

        Args:
            x_0: Clean video latent batch [B_local, C, T, H, W] or tokens [B_local, L, D].
            text_embeddings: Text representation [B_local, L_txt, D_txt].
            text_mask: Attention mask for text.
            null_embedding: Null representation for CFG drop.
            token_mask: Padding mask for video tokens.
            generator: RNG generator for deterministic testing.

        Returns:
            Dictionary of step metrics.
        """
        self.model.train()
        self.profiler.start_step()

        # Update learning rate schedule
        current_lr = self.update_lr(self.global_step)

        # Forward pass under autocast
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        autocast_dtype = (
            torch.bfloat16
            if self.mixed_precision == "bf16"
            else (torch.float16 if self.mixed_precision == "fp16" else torch.float32)
        )

        with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=(self.mixed_precision != "fp32")):
            loss, metrics = self.loss_fn(
                model=self.model,
                x_0=x_0.to(self.device),
                text_embeddings=text_embeddings.to(self.device),
                text_mask=text_mask.to(self.device) if text_mask is not None else None,
                null_embedding=null_embedding.to(self.device) if null_embedding is not None else None,
                token_mask=token_mask.to(self.device) if token_mask is not None else None,
                generator=generator,
            )
            # Scale loss by gradient accumulation steps
            scaled_loss = loss / self.grad_accum

        scaled_loss.backward()

        self.step_in_epoch += 1
        grad_norm = 0.0

        # Optimizer step upon completing gradient accumulation cycle
        if self.step_in_epoch % self.grad_accum == 0:
            # Gradient clipping
            grad_norm = float(
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            )

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # Update Exponential Moving Average
            self.ema.update(self.model)

            self.global_step += 1

        # Profile execution
        seq_len = x_0.shape[1] if x_0.ndim == 3 else (x_0.shape[2] * x_0.shape[3] * x_0.shape[4])
        perf_metrics = self.profiler.end_step(
            global_batch_size=self.effective_global_batch,
            seq_len=seq_len,
        )

        step_summary = {
            "step": self.global_step,
            "loss": float(loss.item()),
            "lr": current_lr,
            "grad_norm": grad_norm,
            "tokens_per_sec": perf_metrics.get("tokens_per_sec", 0.0),
            "mfu_percent": perf_metrics.get("mfu_percent", 0.0),
            "effective_global_batch": self.effective_global_batch,
            "binned_losses": self.loss_fn.get_binned_losses(),
        }

        return step_summary

    def save(self, tag: Optional[str] = None) -> Path:
        """Save training checkpoint."""
        filename = f"checkpoint_step_{self.global_step:06d}.pt" if tag is None else f"checkpoint_{tag}.pt"
        path = self.checkpoint_dir / filename
        return save_checkpoint(
            save_path=path,
            step=self.global_step,
            epoch=self.epoch,
            model=self.model,
            optimizer=self.optimizer,
            ema=self.ema,
            config={
                "world_size": self.world_size,
                "learning_rate": self.learning_rate,
                "grad_accum": self.grad_accum,
                "global_batch": self.effective_global_batch,
            },
        )

    def resume(
        self,
        checkpoint_path: str | Path,
        new_world_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Resume training from checkpoint, optionally adjusting for a new GPU count."""
        step, epoch, sampler_state = resume_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            ema=self.ema,
        )
        self.global_step = step
        self.epoch = epoch

        reshard_info = {}
        if new_world_size is not None and new_world_size != self.world_size:
            reshard_info = ReshardingCoordinator.adjust_sampler_for_new_world_size(
                saved_step=step,
                saved_epoch=epoch,
                old_world_size=self.world_size,
                old_grad_accum=self.grad_accum,
                old_local_batch=self.local_batch,
                new_world_size=new_world_size,
            )
            self.world_size = new_world_size
            self.local_batch = reshard_info["new_local_batch"]
            self.grad_accum = reshard_info["new_grad_accum"]
            self.effective_global_batch = reshard_info["new_global_batch"]

        return {
            "resumed_step": step,
            "resumed_epoch": epoch,
            "reshard_info": reshard_info,
        }
