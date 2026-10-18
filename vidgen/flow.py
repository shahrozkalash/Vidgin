"""Part 7: rectified flow training objective.

README convention: x_0 = Gaussian noise, x_1 = data latents.
  x_t = (1 - t) * x_0 + t * x_1,   velocity target v = x_1 - x_0.
Timesteps follow the shifted logit-normal schedule (s = 5.0, Part 7 audit):
  sigma(u) = s*u / (1 + (s-1)*u),  u ~ Normal(0, 1),  t = sigma(sigmoid(u)).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_shifted_logit_normal(shape: tuple, s: float = 5.0,
                                generator: torch.Generator | None = None,
                                device: str = "cpu") -> torch.Tensor:
    """Sample timesteps t in (0, 1) from the shifted logit-normal schedule."""
    u = torch.randn(shape, generator=generator, device=device)
    sig = torch.sigmoid(u)
    return (s * sig) / (1.0 + (s - 1.0) * sig)


def rectified_flow_targets(x1: torch.Tensor, t: torch.Tensor,
                           generator: torch.Generator | None = None):
    """Given data latents x1 [B,...] and timesteps t [B] (broadcastable),
    return (x_t, velocity_target). Noise drawn from the global torch RNG so
    checkpoint RNG snapshots keep runs bit-reproducible."""
    if t.dim() == 1:
        t = t.view(-1, *([1] * (x1.dim() - 1)))
    x0 = torch.randn_like(x1) if generator is None else torch.randn(
        x1.shape, generator=generator, device=x1.device, dtype=x1.dtype)
    xt = (1.0 - t) * x0 + t * x1
    return xt, x1 - x0


def flow_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def timestep_bins(t: torch.Tensor, num_bins: int = 10) -> torch.Tensor:
    """Bin indices in [0, num_bins) for diagnostic per-timestep loss tracking."""
    return torch.clamp((t * num_bins).long(), 0, num_bins - 1)
