"""Few-Step Distillation & Rectified-Flow Reflow for Video DiT (Part 11).

Goal:
Distill a 30-50 step rectified flow model into an ultra-fast 4 to 8 step generator:
1. Rectified Flow Reflow (2-Rectified Flow):
   - Problem: Standard flow matching generates curved probability ODE trajectories between noise x_1 and data x_0.
   - Solution: Generate clean data samples x_hat_0 using 50 Euler steps from teacher model with seed noise x_1.
   - Pair (x_hat_0, x_1) represents endpoints of an empirical flow trajectory.
   - Train student on straight linear interpolations x_t = (1 - t) * x_hat_0 + t * x_1 with constant velocity v = x_1 - x_hat_0.
   - Mathematical theorem: 2-Rectified Flow provably straightens trajectories, allowing accurate 4-step Euler integration.
2. Few-Step Solvers:
   - 4-step, 6-step, and 8-step time grids tailored for flow matching.
3. Consistency & Distribution-Matching Distillation:
   - Step-skipping loss matching student predictions directly across multiple teacher steps.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np


class RectifiedFlowReflow:
    """Manages 2-Rectified Flow dataset generation and student training."""

    def __init__(self, teacher_steps: int = 50):
        self.teacher_steps = teacher_steps

    @torch.no_grad()
    def generate_reflow_pairs(
        self,
        teacher_model: nn.Module,
        batch_size: int,
        latent_shape: Tuple[int, int, int, int],  # [C, T, H, W]
        context: torch.Tensor,
        device: torch.device,
        guidance_scale: float = 5.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates (x_1, x_hat_0) endpoint pairs using teacher ODE integration.

        Returns:
            x_1: Initial Gaussian noise [B, C, T, H, W]
            x_hat_0: Model-generated video latents [B, C, T, H, W]
        """
        C, T, H, W = latent_shape
        x_1 = torch.randn(batch_size, C, T, H, W, device=device)
        x_t = x_1.clone()

        # Teacher integration from sigma=1.0 down to 0.0
        sigmas = torch.linspace(1.0, 0.0, self.teacher_steps + 1, device=device)

        for i in range(self.teacher_steps):
            s_curr = sigmas[i]
            s_next = sigmas[i + 1]
            dt = s_next - s_curr  # negative

            t_tensor = s_curr.unsqueeze(0).expand(batch_size)
            v_cond = teacher_model(x_t, t_tensor, context)

            if guidance_scale > 1.0:
                null_context = torch.zeros_like(context)
                v_uncond = teacher_model(x_t, t_tensor, null_context)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond

            # Euler step
            x_t = x_t + dt * v

        x_hat_0 = x_t
        return x_1, x_hat_0

    def compute_student_reflow_loss(
        self,
        student_model: nn.Module,
        x_1: torch.Tensor,
        x_hat_0: torch.Tensor,
        context: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes rectified flow loss on straight synthetic reflow trajectory.
        
        x_t = (1 - t) * x_hat_0 + t * x_1
        Target velocity: v_target = x_1 - x_hat_0 (constant along straight path!)
        """
        B = x_1.shape[0]
        device = x_1.device

        # Sample uniform t in [0, 1]
        t = torch.rand(B, device=device)
        t_expand = t.view(B, 1, 1, 1, 1)

        # Straight-line trajectory interpolation
        x_t = (1.0 - t_expand) * x_hat_0 + t_expand * x_1

        # Target velocity
        target_v = x_1 - x_hat_0

        # Predict student velocity
        pred_v = student_model(x_t, t, context, text_mask=text_mask)

        # MSE Loss
        loss = F.mse_loss(pred_v, target_v)
        return loss, pred_v


class FewStepFlowSolver:
    """Specialized few-step (4, 6, 8 steps) flow matching sampler."""

    @staticmethod
    def get_time_grid(steps: int = 4, shift: float = 3.0) -> List[float]:
        """Calculates shifted time grid optimized for 4 to 8 step generation."""
        assert steps in (4, 6, 8, 10), f"Supported few-step counts are 4, 6, 8, 10, got {steps}"
        u = np.linspace(1.0, 0.0, steps + 1)
        # Shift formula: sigma = s*u / (1 + (s-1)*u)
        sigmas = (shift * u) / (1.0 + (shift - 1.0) * u)
        return sigmas.tolist()

    @staticmethod
    @torch.no_grad()
    def sample_few_steps(
        model: nn.Module,
        x_init: torch.Tensor,
        context: torch.Tensor,
        steps: int = 4,
        shift: float = 3.0,
        guidance_scale: float = 1.0,  # Typically low or 1.0 for distilled models
    ) -> torch.Tensor:
        """Runs few-step Euler integration."""
        sigmas = FewStepFlowSolver.get_time_grid(steps=steps, shift=shift)
        x = x_init.clone()
        B = x.shape[0]

        for i in range(len(sigmas) - 1):
            s_curr = sigmas[i]
            s_next = sigmas[i + 1]
            dt = s_next - s_curr

            t_tensor = torch.tensor([s_curr], dtype=x.dtype, device=x.device).expand(B)
            v = model(x, t_tensor, context)

            if guidance_scale > 1.0:
                null_context = torch.zeros_like(context)
                v_uncond = model(x, t_tensor, null_context)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)

            x = x + dt * v

        return x
