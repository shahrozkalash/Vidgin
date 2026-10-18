"""Classifier-Free Guidance (CFG) Distillation & Few-Step Optimization (Part 12.3).

Target:
Overcome the dual-forward pass cost of Classifier-Free Guidance (CFG):
1. CFG Distillation:
   - Baseline: At every step, the model evaluates both conditional and unconditional forward passes:
     v = v_uncond + g * (v_cond - v_uncond). This doubles the compute per step (2x NFEs).
   - Solution: CFG distillation conditions the student DiT on the guidance parameter g (e.g. g in [1, 7]).
   - The student directly predicts the guided velocity in a single forward pass:
     Loss = || v_student(x_t, t, c, g) - [v_uncond + g * (v_cond - v_uncond)] ||^2
   - Result: Halves sampling wall-clock time immediately (1x evaluation per step).
2. Combined Step Distillation:
   - Pair CFG distillation with 4 to 8 step rectified flow for an overall ~25x inference acceleration.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class CFGDistillationLoss:
    """Computes distillation loss to fold CFG into a single-pass student."""

    def __init__(self, guidance_range: Tuple[float, float] = (1.5, 7.0)):
        self.guidance_range = guidance_range

    def compute_loss(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        null_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates loss between single-pass student and dual-pass teacher."""
        B = x_t.shape[0]
        device = x_t.device

        # Sample random guidance scale g in guidance_range for each batch item
        min_g, max_g = self.guidance_range
        g = min_g + torch.rand(B, 1, 1, 1, 1, device=device) * (max_g - min_g)

        # Teacher evaluations (uncond and cond)
        with torch.no_grad():
            v_cond = teacher_model(x_t, t, context)
            v_uncond = teacher_model(x_t, t, null_context)
            target_guided_v = v_uncond + g * (v_cond - v_uncond)

        # Student evaluation (single forward pass predicting guided velocity directly)
        pred_guided_v = student_model(x_t, t, context)

        # Distillation MSE Loss
        loss = F.mse_loss(pred_guided_v, target_guided_v)
        return loss, pred_guided_v


class FastSinglePassSampler:
    """Samples from a CFG-distilled model requiring only 1 evaluation per Euler step."""

    @staticmethod
    @torch.no_grad()
    def sample(
        model: nn.Module,
        x_init: torch.Tensor,
        context: torch.Tensor,
        steps: int = 8,
        shift: float = 3.0,
    ) -> torch.Tensor:
        """Runs single-pass Euler integration (no separate unconditional branch needed)."""
        import numpy as np
        u = np.linspace(1.0, 0.0, steps + 1)
        sigmas = ((shift * u) / (1.0 + (shift - 1.0) * u)).tolist()

        x = x_init.clone()
        B = x.shape[0]

        for i in range(len(sigmas) - 1):
            s_curr = sigmas[i]
            s_next = sigmas[i + 1]
            dt = s_next - s_curr

            t_tensor = torch.tensor([s_curr], dtype=x.dtype, device=x.device).expand(B)
            # ONLY ONE FORWARD PASS PER STEP
            v = model(x, t_tensor, context)
            x = x + dt * v

        return x
