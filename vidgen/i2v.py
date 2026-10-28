"""Image-to-video: animate a still image into a clip that starts from it.

Why this works without retraining:

* The VAE is causal and encodes the first frame on its own (T = 4k+1 ->
  k+1 latent frames, frame 0 -> latent frame 0). A single image therefore
  encodes to latent frame 0 of any video that begins with it, identical
  up to float rounding (~1e-7; see tests/test_i2v_voice.py).
* :func:`vidgen.sampler.sample_latents` can clamp leading latent frames to
  the straight rectified-flow path ``(1 - t) * noise + t * context`` (the
  flow-matching inpainting already used for segment chaining in
  vidgen/extend.py). That path is precisely the ``x_t`` the DiT sees for
  every frame during training, so the clamped first frame is in
  distribution and the model generates the remaining frames to continue
  from it, steered by the text prompt.

The decoded first frame is the VAE's reconstruction of the image (the
encoder is strictly causal; the decoder's normalization spans the whole
window, which moves frame 0 by ~1e-3, well under one grey level), and
every later frame is generated.
"""

from __future__ import annotations

import numpy as np
import torch

from vidgen.config import VidgenConfig
from vidgen.sampler import sample_latents


def load_image(path: str, height: int, width: int) -> np.ndarray:
    """Read an image as uint8 [H, W, 3] at the model's resolution.

    The image is center-cropped to the target aspect ratio first (never
    stretched), then resized with a high-quality filter.
    """
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        src_w, src_h = im.size
        target = width / height
        if src_w / src_h > target:             # too wide: crop the sides
            new_w = max(1, round(src_h * target))
            left = (src_w - new_w) // 2
            im = im.crop((left, 0, left + new_w, src_h))
        elif src_w / src_h < target:           # too tall: crop top/bottom
            new_h = max(1, round(src_w / target))
            top = (src_h - new_h) // 2
            im = im.crop((0, top, src_w, top + new_h))
        im = im.resize((width, height), Image.LANCZOS)
        return np.asarray(im, dtype=np.uint8).copy()


@torch.no_grad()
def image_latent(vae, image: np.ndarray, cfg: VidgenConfig,
                 device: str = "cpu") -> torch.Tensor:
    """Encode a uint8 [H, W, 3] image to the DiT's first latent frame:
    ``[1, C, 1, latent_h, latent_w]``, already multiplied by latent_scale.
    Uses the VAE mean (not a random draw), so it is deterministic."""
    want = (cfg.height, cfg.width, 3)
    if image.dtype != np.uint8 or image.shape != want:
        raise ValueError(f"image must be uint8 {want} (H,W,3), got "
                         f"{image.dtype} {image.shape}; use load_image()")
    x = torch.from_numpy(image).to(torch.float32) / 127.5 - 1.0
    x = x.permute(2, 0, 1).reshape(1, 3, 1, cfg.height, cfg.width)
    mu, _ = vae.encode(x.to(device))
    return mu * cfg.latent_scale


@torch.no_grad()
def generate_from_image(dit, vae, text_enc, cfg: VidgenConfig,
                        image: np.ndarray, prompt: str, steps: int = 20,
                        method: str = "euler", cfg_scale: float = 7.5,
                        seed: int = 0, device: str = "cpu",
                        depth: torch.Tensor | None = None) -> np.ndarray:
    """One native window that starts from ``image``.

    Returns frames ``[T, H, W, 3]`` float32 in [-1, 1].
    """
    prefix = image_latent(vae, image, cfg, device)
    text, text_mask = text_enc([prompt])
    null_text, null_mask = text_enc.null_embedding()
    lat_shape = (1, cfg.latent_channels, cfg.latent_t, cfg.latent_h,
                 cfg.latent_w)
    z = sample_latents(dit, lat_shape, text.to(device), text_mask.to(device),
                       null_text.to(device), null_mask.to(device),
                       method=method, num_steps=steps, cfg_scale=cfg_scale,
                       seed=seed, device=device, prefix_latent=prefix,
                       prefix_t=1, depth=depth)
    clip = vae.decode(z / cfg.latent_scale)              # [1,3,T,H,W]
    frames = clip[0].permute(1, 2, 3, 0).cpu().numpy()
    assert np.isfinite(frames).all(), "image-to-video output contains NaN/Inf"
    return frames.astype(np.float32)
