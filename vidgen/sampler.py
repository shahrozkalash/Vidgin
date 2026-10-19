"""Part 8: ODE samplers (Euler, Midpoint, RK4) with classifier-free guidance.

Integrates the learned velocity field from noise (t=0) to data (t=1).
With an oracle model returning the exact constant velocity (x1 - x0), every
method recovers x1 with error < 1e-5 (Part 8 audit: oracle exact recovery).

Optional prefix conditioning (Stage 1 upgrade, "extend" mode): pass
``prefix_latent`` ([B, C, prefix_t, h, w]) to clamp the first ``prefix_t``
latent frames to the straight-line rectified-flow path from their initial
noise to the context. This is the flow-matching analogue of RePaint
inpainting: at each integration step the context region is replaced by
(1 - t) * noise + t * context, so the generated remainder must continue
from the given past. Used by vidgen.extend for segment chaining.
"""

from __future__ import annotations

import torch


def _cfg_velocity(model, x, t_batch, text, text_mask, null_text, null_mask, cfg_scale):
    v_cond = model(x, t_batch, text, text_mask)
    if null_text is None or cfg_scale == 1.0:
        return v_cond
    v_uncond = model(x, t_batch, null_text, null_mask)
    return v_uncond + cfg_scale * (v_cond - v_uncond)


@torch.no_grad()
def sample_latents(model, latent_shape: tuple,
                   text: torch.Tensor, text_mask: torch.Tensor,
                   null_text: torch.Tensor | None = None,
                   null_mask: torch.Tensor | None = None,
                   method: str = "euler", num_steps: int = 20,
                   cfg_scale: float = 7.5, seed: int = 0,
                   device: str = "cpu",
                   prefix_latent: torch.Tensor | None = None,
                   prefix_t: int = 0) -> torch.Tensor:
    """Generate latents [B,C,t,h,w] from noise. method in {euler, midpoint, rk4}.

    With ``prefix_latent`` given, the first ``prefix_t`` latent frames are
    clamped to the rectified-flow interpolation between their initial noise
    and the context, so generation continues from a fixed past.
    """
    if method not in ("euler", "midpoint", "rk4"):
        raise ValueError(f"Unknown sampler {method!r}")
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    b = latent_shape[0]
    x = torch.randn(latent_shape, generator=gen, device=device)
    if prefix_t:
        if prefix_latent is None:
            raise ValueError("prefix_t>0 requires prefix_latent")
        pl = prefix_latent.to(device=device, dtype=x.dtype)
        expect = (b, latent_shape[1], prefix_t, latent_shape[3], latent_shape[4])
        if tuple(pl.shape) != expect:
            raise ValueError(
                f"prefix_latent shape {tuple(pl.shape)} incompatible with "
                f"prefix_t={prefix_t}, expected {expect}")
        if not 0 < prefix_t < latent_shape[2]:
            raise ValueError(
                f"prefix_t={prefix_t} must be within (0, latent_t={latent_shape[2]})")
        x0_prefix = x[..., :prefix_t, :, :].clone()
    ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    dt = 1.0 / num_steps

    def v(x_, t_):
        t_batch = torch.full((b,), t_, device=device)
        return _cfg_velocity(model, x_, t_batch, text, text_mask,
                             null_text, null_mask, cfg_scale)

    for i in range(num_steps):
        t = float(ts[i])
        if method == "euler":
            x = x + dt * v(x, t)
        elif method == "midpoint":
            k1 = v(x, t)
            x = x + dt * v(x + 0.5 * dt * k1, t + 0.5 * dt)
        else:  # rk4
            k1 = v(x, t)
            k2 = v(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = v(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = v(x + dt * k3, t + dt)
            x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if prefix_t:
            # flow-matching inpaint: context follows its straight-line path
            t_next = float(ts[i + 1])
            x[..., :prefix_t, :, :] = (1.0 - t_next) * x0_prefix + t_next * pl
    return x
