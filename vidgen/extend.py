"""Stage 1 upgrade: long-video generation by segment chaining.

The DiT only ever generates one native window (Tiny: 17 frames). To exceed it,
we generate segment 1 normally, then for each following segment:

  1. VAE-encode the last ``ctx_px`` pixel frames of the accumulated video
     (the causal VAE maps ctx_px = 4*ctx_latent - 3 pixel frames to exactly
     ``ctx_latent`` latent frames) to get context latents ``mu * latent_scale``
     (mu, not a random reparameterization draw, for determinism);
  2. sample a full new window with :func:`vidgen.sampler.sample_latents` while
     clamping the first ``ctx_latent`` latent frames to the rectified-flow
     straight-line path from noise to that context (flow-matching inpainting),
     so the generated remainder continues from the given past;
  3. decode the window and stitch: keep only the genuinely new frames, with a
     short linear crossfade over ``blend`` frames at the join to hide the
     VAE-reconstruction seam.

Why this design: the DiT has no cross-segment attention and was trained on
fixed windows, so true end-to-end conditioning is out of reach without
retraining. Latent-space prefix clamping is the closest principled option the
architecture allows — the model's first latent frames see the actual encoded
past (causal receptive field), and the crossfade removes the small
reconstruction discontinuity. Pixel-only chaining (independent segments +
crossfade) would jump visibly at every join because motion would restart.

Determinism: segment k uses ``seed + k``, so a fixed base seed reproduces the
whole video. ``ctx_latent`` frames of context cost no new frames; each chained
segment adds ``frames - (4*ctx_latent - 3)`` new frames.

Playback: the project plays clips at ~290 ms/frame, so ~103 frames ~= 30 s.
"""

from __future__ import annotations

import numpy as np
import torch

from vidgen.config import VidgenConfig
from vidgen.sampler import sample_latents

#: project playback rate (ms per frame); ~103 frames ~= 30 s.
FRAME_MS = 290


def context_pixel_frames(ctx_latent: int) -> int:
    """Pixel frames that encode to ``ctx_latent`` latent frames.

    From the causal VAE geometry: decode maps latent_t -> 4*latent_t - 3
    pixels, and encode inverts it for lengths satisfying T = 4k+1.
    """
    if ctx_latent < 1:
        raise ValueError(f"ctx_latent must be >= 1, got {ctx_latent}")
    return 4 * ctx_latent - 3


def new_frames_per_segment(cfg: VidgenConfig, ctx_latent: int) -> int:
    return cfg.frames - context_pixel_frames(ctx_latent)


def segments_for_seconds(cfg: VidgenConfig, seconds: float, ctx_latent: int) -> int:
    """Segment count whose output reaches at least ``seconds`` at FRAME_MS."""
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}")
    target = int(np.ceil(seconds * 1000.0 / FRAME_MS))
    per = new_frames_per_segment(cfg, ctx_latent)
    if target <= cfg.frames:
        return 1
    import math
    return 1 + math.ceil((target - cfg.frames) / per)


def _to_pixel_tensor(frames: np.ndarray) -> torch.Tensor:
    """[T,H,W,3] float32 [-1,1] -> [1,3,T,H,W]."""
    return torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def _to_frame_array(clip: torch.Tensor) -> np.ndarray:
    """[1,3,T,H,W] [-1,1] -> [T,H,W,3] float32."""
    return clip[0].permute(1, 2, 3, 0).detach().cpu().numpy()


def _crossfade_stitch(acc: np.ndarray, window: np.ndarray,
                      ctx_px: int, blend: int) -> np.ndarray:
    """Append the genuinely new frames of ``window`` to ``acc``.

    ``window[:ctx_px]`` reconstructs ``acc[-ctx_px:]``; the join is hidden by
    morphing the last ``blend`` frames of acc from the original tail toward
    the reconstruction (a convex blend, so values stay in [-1, 1]), then
    appending ``window[ctx_px:]``. Each chained segment adds exactly
    ``frames - ctx_px`` new frames.
    """
    if not 0 <= blend <= ctx_px:
        raise ValueError(f"blend={blend} must be within [0, ctx_px={ctx_px}]")
    out = acc.copy()
    if blend:
        tail_orig = acc[-blend:].astype(np.float64)
        tail_recon = window[ctx_px - blend:ctx_px].astype(np.float64)
        alpha = (np.arange(1, blend + 1) / (blend + 1)).reshape(-1, 1, 1, 1)
        out[-blend:] = ((1 - alpha) * tail_orig + alpha * tail_recon
                        ).astype(np.float32)
    return np.concatenate([out, window[ctx_px:]], axis=0)


@torch.no_grad()
def generate_long(dit, vae, text_enc, cfg: VidgenConfig, prompt: str,
                  segments: int = 1, steps: int = 20, method: str = "euler",
                  cfg_scale: float = 7.5, seed: int = 0, device: str = "cpu",
                  ctx_latent: int = 2, blend: int = 2) -> np.ndarray:
    """Generate a chained clip. Returns [T_total, H, W, 3] float32 in [-1, 1]."""
    if segments < 1:
        raise ValueError(f"segments must be >= 1, got {segments}")
    if not 0 < ctx_latent < cfg.latent_t:
        raise ValueError(
            f"ctx_latent={ctx_latent} must be within (0, latent_t={cfg.latent_t})")
    ctx_px = context_pixel_frames(ctx_latent)
    if ctx_px >= cfg.frames:
        raise ValueError(f"context ({ctx_px}px) must be shorter than the "
                         f"window ({cfg.frames}px)")

    text, text_mask = text_enc([prompt])
    null_text, null_mask = text_enc.null_embedding()
    text, text_mask = text.to(device), text_mask.to(device)
    null_text, null_mask = null_text.to(device), null_mask.to(device)
    lat_shape = (1, cfg.latent_channels, cfg.latent_t, cfg.latent_h, cfg.latent_w)

    def decode_window(z: torch.Tensor) -> np.ndarray:
        return _to_frame_array(vae.decode(z / cfg.latent_scale))

    # segment 1: unconditional-on-past, exactly the sample.py path
    z = sample_latents(dit, lat_shape, text, text_mask, null_text, null_mask,
                       method=method, num_steps=steps, cfg_scale=cfg_scale,
                       seed=seed, device=device)
    frames = decode_window(z)

    for k in range(1, segments):
        tail = _to_pixel_tensor(frames[-ctx_px:]).to(device)
        mu, _ = vae.encode(tail)                    # [1,16,ctx_latent,h,w]
        prefix = (mu * cfg.latent_scale).to(device)
        z = sample_latents(dit, lat_shape, text, text_mask, null_text, null_mask,
                           method=method, num_steps=steps, cfg_scale=cfg_scale,
                           seed=seed + k, device=device,
                           prefix_latent=prefix, prefix_t=ctx_latent)
        window = decode_window(z)
        frames = _crossfade_stitch(frames, window, ctx_px, blend)

    assert np.isfinite(frames).all(), "chained output contains NaN/Inf"
    return frames.astype(np.float32)
