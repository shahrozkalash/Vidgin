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
from vidgen.config import VidgenConfig, get_preset, resolve_device
from vidgen.depth import depth_to_latent_grid
from vidgen.dit.model import DiT
from vidgen.extend import generate_long, new_frames_per_segment, segments_for_seconds
from vidgen.sampler import sample_latents
from vidgen.seed import seed_everything
from vidgen.text.encoder import TinyTextEncoder
from vidgen.vae.model import CausalVAE


def save_frames(frames: np.ndarray, outdir: str, duration_ms: int = 62) -> dict:
    """frames: [T, H, W, 3] float32 in [-1, 1]. Returns paths written.

    ``duration_ms`` sets the GIF frame duration; 62 keeps the original fast
    preview, 290 matches the project's standard playback rate (~5 s for a
    17-frame clip; the long-clip path uses 290).
    """
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
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                     duration=duration_ms)
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
    # Stage 1 upgrade: long video via segment chaining (vidgen.extend).
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="target duration; segments are chained to reach it "
                         "(~290 ms/frame). 0 = single native window (old behavior)")
    ap.add_argument("--segments", type=int, default=0,
                    help="explicit segment count; overrides --seconds")
    ap.add_argument("--ctx-latent", type=int, default=2,
                    help="latent frames of past used as chaining context")
    ap.add_argument("--blend", type=int, default=2,
                    help="pixel frames crossfaded at each segment join")
    ap.add_argument("--frame-ms", type=int, default=0,
                    help="GIF frame duration ms; 0 = 62 for single clips, "
                         "290 for chained clips")
    ap.add_argument("--depth-npy", default=None,
                    help="optional uint8 [T,H,W] depth map .npy (larger = "
                         "nearer) to steer a depth-trained checkpoint with "
                         "a fixed 3D layout (vidgen/depth.py); single "
                         "window only, not with --seconds/--segments")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    payload = load_checkpoint(args.ckpt, map_location=device)
    assert payload["kind"] == "dit", f"not a DiT checkpoint: {args.ckpt}"
    # The checkpoint's stored config is authoritative (it records e.g.
    # use_depth, which decides whether the DiT has a depth stem at all);
    # older checkpoints without one fall back to the named preset.
    if payload.get("config"):
        cfg = VidgenConfig.from_dict(payload["config"])
    else:
        cfg = get_preset(payload["preset"])

    vae = CausalVAE().to(device).eval()
    if args.vae_ckpt or payload.get("vae_ckpt"):
        vp = load_checkpoint(args.vae_ckpt or payload["vae_ckpt"], map_location=device)
        vae.load_state_dict(vp["vae"])
    text_enc = TinyTextEncoder(d_t=cfg.text_dim, max_len=cfg.text_max_len).to(device)
    dit = DiT(cfg).to(device)
    dit.load_state_dict(payload["dit"])
    text_enc.load_state_dict(payload["text_enc"])

    segments = args.segments
    if segments <= 0 and args.seconds > 0:
        segments = segments_for_seconds(cfg, args.seconds, args.ctx_latent)
    chained = segments > 1
    if chained and args.depth_npy:
        raise SystemExit("--depth-npy steers a single native window; "
                         "it cannot be combined with --seconds/--segments.")
    if chained:
        frames = generate_long(dit, vae, text_enc, cfg, args.prompt,
                               segments=segments, steps=args.steps,
                               method=args.method, cfg_scale=args.cfg,
                               seed=args.seed, device=device,
                               ctx_latent=args.ctx_latent, blend=args.blend)
        expect_t = cfg.frames + (segments - 1) * new_frames_per_segment(
            cfg, args.ctx_latent)
        assert frames.shape == (expect_t, cfg.height, cfg.width, 3), frames.shape
    else:
        text, text_mask = text_enc([args.prompt])
        null_text, null_mask = text_enc.null_embedding()
        depth = None
        if args.depth_npy:
            if not cfg.use_depth:
                raise SystemExit(
                    "--depth-npy given, but this checkpoint was trained "
                    "without depth conditioning (use_depth=False); the map "
                    "would have no input to enter through.")
            arr = np.load(args.depth_npy)
            want = (cfg.frames, cfg.height, cfg.width)
            if arr.dtype != np.uint8 or arr.shape != want:
                raise SystemExit(
                    f"--depth-npy must be uint8 {want} (T,H,W), got "
                    f"{arr.dtype} {arr.shape}")
            d = torch.from_numpy(arr).to(torch.float32) / 255.0
            depth = depth_to_latent_grid(
                d.view(1, 1, *want).to(device),
                cfg.latent_t, cfg.latent_h, cfg.latent_w)
        lat_shape = (1, cfg.latent_channels, cfg.latent_t, cfg.latent_h, cfg.latent_w)
        z = sample_latents(dit, lat_shape, text.to(device), text_mask.to(device),
                           null_text.to(device), null_mask.to(device),
                           method=args.method, num_steps=args.steps,
                           cfg_scale=args.cfg, seed=args.seed, device=device,
                           depth=depth)
        with torch.no_grad():
            clip = vae.decode(z / cfg.latent_scale)  # [1,3,T,H,W] in [-1,1]
        frames = clip[0].permute(1, 2, 3, 0).cpu().numpy()  # [T,H,W,3]
        assert frames.shape == (cfg.frames, cfg.height, cfg.width, 3)
    assert frames.min() >= -1.0 - 1e-5 and frames.max() <= 1.0 + 1e-5
    frame_ms = args.frame_ms or (290 if chained else 62)
    out = save_frames(frames, args.outdir, duration_ms=frame_ms)
    dur = frames.shape[0] * frame_ms / 1000.0
    print(f"sampled clip: shape={frames.shape} (~{dur:.1f}s at {frame_ms}ms/frame) "
          f"range=[{frames.min():.3f},{frames.max():.3f}]")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
