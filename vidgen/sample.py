"""Part 8: generate a clip from a trained DiT checkpoint.

Pipeline: sample latents with an ODE sampler (+CFG) -> VAE decode -> save
frames (.npy always; PNG/GIF too when Pillow is installed).

Usage:
  python -m vidgen.sample --ckpt checkpoints/tiny-dit/latest.pt \\
      --prompt "digit 3 moving left" --steps 20 --outdir outputs/clip1
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from vidgen.checkpoint import load_checkpoint
from vidgen.config import get_preset, resolve_device
from vidgen.dit.model import DiT
from vidgen.sampler import sample_latents
from vidgen.seed import seed_everything
from vidgen.text.encoder import TinyTextEncoder
from vidgen.vae.model import CausalVAE


def save_frames(frames: np.ndarray, outdir: str) -> dict:
    """frames: [T, H, W, 3] float32 in [-1, 1]. Returns paths written."""
    os.makedirs(outdir, exist_ok=True)
    out = {}
    npy_path = os.path.join(outdir, "frames.npy")
    np.save(npy_path, frames)
    out["npy"] = npy_path
    try:
        from PIL import Image

        uint8 = ((frames * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        png_paths = []
        for i, f in enumerate(uint8):
            p = os.path.join(outdir, f"frame_{i:03d}.png")
            Image.fromarray(f).save(p)
            png_paths.append(p)
        gif_path = os.path.join(outdir, "clip.gif")
        imgs = [Image.fromarray(f) for f in uint8]
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=62)
        out["png_frames"] = png_paths
        out["gif"] = gif_path
    except ImportError:
        print("Pillow not installed: wrote .npy only (pip install pillow for PNG/GIF)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="digit 3 moving left")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--method", default="euler", choices=["euler", "midpoint", "rk4"])
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default="outputs/clip")
    ap.add_argument("--vae-ckpt", default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    payload = load_checkpoint(args.ckpt, map_location=device)
    assert payload["kind"] == "dit", f"not a DiT checkpoint: {args.ckpt}"
    cfg = get_preset(payload["preset"])

    vae = CausalVAE().to(device).eval()
    if args.vae_ckpt or payload.get("vae_ckpt"):
        vp = load_checkpoint(args.vae_ckpt or payload["vae_ckpt"], map_location=device)
        vae.load_state_dict(vp["vae"])
    text_enc = TinyTextEncoder(d_t=cfg.text_dim, max_len=cfg.text_max_len).to(device)
    dit = DiT(cfg).to(device)
    dit.load_state_dict(payload["dit"])
    text_enc.load_state_dict(payload["text_enc"])

    text, text_mask = text_enc([args.prompt])
    null_text, null_mask = text_enc.null_embedding()
    lat_shape = (1, cfg.latent_channels, cfg.latent_t, cfg.latent_h, cfg.latent_w)
    z = sample_latents(dit, lat_shape, text.to(device), text_mask.to(device),
                       null_text.to(device), null_mask.to(device),
                       method=args.method, num_steps=args.steps,
                       cfg_scale=args.cfg, seed=args.seed, device=device)
    with torch.no_grad():
        clip = vae.decode(z / cfg.latent_scale)  # [1,3,T,H,W] in [-1,1]
    frames = clip[0].permute(1, 2, 3, 0).cpu().numpy()  # [T,H,W,3]
    assert frames.shape == (cfg.frames, cfg.height, cfg.width, 3)
    assert frames.min() >= -1.0 - 1e-5 and frames.max() <= 1.0 + 1e-5
    out = save_frames(frames, args.outdir)
    print(f"sampled clip: shape={frames.shape} range=[{frames.min():.3f},{frames.max():.3f}]")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
