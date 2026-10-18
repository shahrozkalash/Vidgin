"""Flow Matching Sampler for Wan Video Latent Diffusion.

Features:
- Time schedule: u = linspace(1, 0, steps + 1)
- Shifted time grid: sigma(u) = s * u / (1 + (s - 1) * u) with parameter s (default s=5.0 for Wan)
- Integration schemes:
  * Euler: 1st order flow step x_{next} = x_i + (sigma_{next} - sigma_i) * v
  * Heun: 2nd order predictor-corrector
  * UniPC: Unified predictor-corrector for flow ODEs
  * DPM-Solver++ (Flow): 2nd order multistep flow solver
- Classifier-Free Guidance (CFG):
  * v = v_uncond + g * (v_cond - v_uncond)
  * Supports custom negative prompts in the unconditional branch
- Strict validation:
  * Oracle test: A model returning (eps - x0) recovers x0 exactly for any step count
"""

from __future__ import annotations
import math
from typing import Callable, Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


def get_time_schedule(
    steps: int,
    shift: float = 5.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute shifted time grid sigma in [1.0, 0.0].

    u = linspace(1.0, 0.0, steps + 1)
    sigma = s * u / (1 + (s - 1) * u)

    When u = 1.0: sigma = s / s = 1.0 (pure noise)
    When u = 0.0: sigma = 0 / 1 = 0.0 (clean data)
    Larger s pushes intermediate steps toward higher noise levels, allocating
    more capacity to global structural layout for high-res/long clips.
    """
    assert steps >= 1, f"Steps must be >= 1, got {steps}"
    u = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    if abs(shift - 1.0) < 1e-6:
        return u
    sigma = (shift * u) / (1.0 + (shift - 1.0) * u)
    return sigma


class FlowSampler:
    """Numerical ODE Solver for Rectified Flow Latent Video Diffusion.

    Flow formulation:
    x_t = (1 - sigma) * x0 + sigma * eps
    Target velocity: v = d(x_t)/d(sigma) = eps - x0
    Reverse flow ODE:
    dx / d(sigma) = v(x, sigma)
    Step from sigma_i to sigma_{next} (< sigma_i):
    Delta_sigma = sigma_{next} - sigma_i < 0
    x_{next} = x_i + Delta_sigma * v
    """

    SUPPORTED_SOLVERS = ["euler", "heun", "unipc", "dpm_solver"]

    def __init__(
        self,
        steps: int = 30,
        shift: float = 5.0,
        solver: str = "euler",
        guidance_scale: float = 5.0,
    ):
        self.steps = steps
        self.shift = shift
        self.solver = solver.lower()
        self.guidance_scale = guidance_scale

        if self.solver not in self.SUPPORTED_SOLVERS:
            raise ValueError(
                f"Unknown solver '{solver}'. Supported: {self.SUPPORTED_SOLVERS}"
            )

    def apply_cfg(
        self,
        v_cond: torch.Tensor,
        v_uncond: torch.Tensor,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply Classifier-Free Guidance extrapolation.

        v_guided = v_uncond + g * (v_cond - v_uncond)
        When g = 1.0: v_guided = v_cond
        When g > 1.0: amplifies conditioned direction away from null/negative prompt
        """
        g = self.guidance_scale if guidance_scale is None else guidance_scale
        if abs(g - 1.0) < 1e-6:
            return v_cond
        return v_uncond + g * (v_cond - v_uncond)

    @torch.no_grad()
    def sample(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor, bool], torch.Tensor],
        x_init: torch.Tensor,
        steps: Optional[int] = None,
        shift: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        solver: Optional[str] = None,
        callback: Optional[Callable[[int, torch.Tensor, float], None]] = None,
    ) -> torch.Tensor:
        """Execute reverse ODE sampling from pure noise to clean latents.

        Args:
            model_fn: Callable(x, sigma_timestep, is_uncond) -> velocity tensor v.
                      sigma_timestep is typically sigma * 1000.
            x_init: Initial noise tensor eps ~ N(0, I) of shape [B, C, t, h, w].
            steps: Optional override for sampling steps.
            shift: Optional override for time schedule shift factor.
            guidance_scale: Optional override for CFG scale.
            solver: Optional override for numerical ODE solver.
            callback: Optional callback(step_idx, current_latents, current_sigma).

        Returns:
            x_0: Clean generated latents [B, C, t, h, w].
        """
        num_steps = steps if steps is not None else self.steps
        s = shift if shift is not None else self.shift
        g = guidance_scale if guidance_scale is not None else self.guidance_scale
        method = (solver if solver is not None else self.solver).lower()

        if method not in self.SUPPORTED_SOLVERS:
            raise ValueError(f"Unknown solver '{method}'")

        device = x_init.device
        dtype = x_init.dtype

        # 1. Compute shifted time schedule
        timesteps = get_time_schedule(num_steps, shift=s, device=device, dtype=dtype)

        x = x_init.clone()

        def get_guided_velocity(curr_x: torch.Tensor, curr_sigma: torch.Tensor) -> torch.Tensor:
            """Evaluate model with CFG guidance."""
            # Model timestep scaling: sigma in [0, 1] scaled to [0, 1000]
            t_input = curr_sigma * 1000.0

            if abs(g - 1.0) < 1e-6:
                # Guidance scale 1: evaluate conditional model only
                return model_fn(curr_x, t_input, False)

            # Evaluate conditioned and unconditioned passes
            v_cond = model_fn(curr_x, t_input, False)
            v_uncond = model_fn(curr_x, t_input, True)
            return self.apply_cfg(v_cond, v_uncond, guidance_scale=g)

        # 2. Iterative ODE integration
        if method == "euler":
            # 1st-order Euler method
            for i in range(num_steps):
                sigma_curr = timesteps[i]
                sigma_next = timesteps[i + 1]
                dt = sigma_next - sigma_curr  # dt < 0 since sigma is falling

                v = get_guided_velocity(x, sigma_curr)
                x = x + dt * v

                if callback is not None:
                    callback(i, x, float(sigma_next.item()))

        elif method == "heun":
            # 2nd-order Heun's method (Improved Euler)
            for i in range(num_steps):
                sigma_curr = timesteps[i]
                sigma_next = timesteps[i + 1]
                dt = sigma_next - sigma_curr

                # Stage 1: Predictor
                v1 = get_guided_velocity(x, sigma_curr)
                x_pred = x + dt * v1

                if sigma_next.item() > 0.0:
                    # Stage 2: Corrector
                    v2 = get_guided_velocity(x_pred, sigma_next)
                    v_avg = 0.5 * (v1 + v2)
                    x = x + dt * v_avg
                else:
                    x = x_pred

                if callback is not None:
                    callback(i, x, float(sigma_next.item()))

        elif method == "unipc":
            # Unified Predictor-Corrector (Flow Midpoint / UniPC-2)
            for i in range(num_steps):
                sigma_curr = timesteps[i]
                sigma_next = timesteps[i + 1]
                dt = sigma_next - sigma_curr

                # Predictor at current point
                v_curr = get_guided_velocity(x, sigma_curr)

                # Midpoint intermediate evaluation
                sigma_mid = 0.5 * (sigma_curr + sigma_next)
                x_mid = x + 0.5 * dt * v_curr
                v_mid = get_guided_velocity(x_mid, sigma_mid)

                # Step using midpoint velocity
                x = x + dt * v_mid

                if callback is not None:
                    callback(i, x, float(sigma_next.item()))

        elif method == "dpm_solver":
            # DPM-Solver++ (2nd order multistep flow solver)
            # Uses velocity from previous step to construct linear interpolation
            prev_v = None
            prev_sigma = None

            for i in range(num_steps):
                sigma_curr = timesteps[i]
                sigma_next = timesteps[i + 1]
                dt = sigma_next - sigma_curr

                v_curr = get_guided_velocity(x, sigma_curr)

                if prev_v is None or i == 0:
                    # First step falls back to Euler
                    x = x + dt * v_curr
                else:
                    # 2nd-order multistep Adams-Bashforth style extrapolation for flow
                    # v(t) ~ v_curr + (t - sigma_curr) * (v_curr - prev_v) / (sigma_curr - prev_sigma)
                    d_sigma = sigma_curr - prev_sigma
                    if abs(d_sigma.item()) > 1e-7:
                        deriv = (v_curr - prev_v) / d_sigma
                        # Integral of v(t) dt from sigma_curr to sigma_next
                        x = x + dt * v_curr + 0.5 * (dt ** 2) * deriv
                    else:
                        x = x + dt * v_curr

                prev_v = v_curr
                prev_sigma = sigma_curr

                if callback is not None:
                    callback(i, x, float(sigma_next.item()))

        return x


# ==============================================================================
# Oracle Model for Exact Verification
# ==============================================================================

class OracleFlowModel:
    """Exact Oracle model for validating flow ODE solvers.

    For a known ground-truth latent x0 and initial noise eps, the exact
    velocity field is:
        v(x_t, sigma) = eps - x0

    Under this velocity field:
        x_t = (1 - sigma) * x0 + sigma * eps
        dx / d(sigma) = eps - x0 = constant

    Any Euler step:
        x_{next} = x_i + (sigma_{next} - sigma_i) * (eps - x0)
    At sigma = 0:
        x_0 = eps + (0 - 1) * (eps - x0) = x0 (EXACTLY, bit-exact to tolerance).
    """

    def __init__(self, x0: torch.Tensor, eps: torch.Tensor):
        self.x0 = x0
        self.eps = eps
        # Constant true velocity
        self.true_v = eps - x0

    def __call__(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        is_uncond: bool = False,
    ) -> torch.Tensor:
        # Return exact target velocity
        return self.true_v.clone()
