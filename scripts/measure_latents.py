#!/usr/bin/env python3
"""Measure a trained VAE's latents + reconstruction quality on a dataset.

Phase 3 tool (2026-10-22). Run this AFTER VAE training and BEFORE DiT
training: it reports the latent mean/std the DiT will actually see,
the data-driven ``--latent-scale`` recommendation (1 / global std),
and reconstruction MSE/PSNR so a weak VAE is caught early.

Usage:
    python scripts/measure_latents.py \
        --vae-ckpt /path/to/vae.pt --preset small \
        --data-dir /path/to/dataset --max-clips 200 \
        --out latent_report.json --preview-dir previews \
        [--min-psnr 15]

Exit code is 1 when --min-psnr is given and the measured PSNR falls
below it (a gate for training pipelines), otherwise 0.
Without --data-dir it measures on synthetic Moving MNIST — useful as a
tool smoke test only, never as evidence about real data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from vidgen.checkpoint import load_checkpoint
from vidgen.config import get_preset, resolve_device
from vidgen.data.clips import FolderClipDataset
from vidgen.data.moving_mnist import MovingMNISTDataset
from vidgen.vae.model import CausalVAE
from vidgen.vae.stats import (
    measure_latent_stats, save_recon_mp4, save_recon_preview,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vae-ckpt", required=True, help="trained VAE checkpoint")
    ap.add_argument("--preset", default="tiny",
                    help="preset giving the clip geometry (frames/size)")
    ap.add_argument("--data-dir", default=None,
                    help="prepared real-clip dataset; default: synthetic "
                         "Moving MNIST (smoke test only)")
    ap.add_argument("--max-clips", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    ap.add_argument("--preview-dir", default=None,
                    help="also save input-vs-reconstruction PNG previews "
                         "for the first 3 clips into this directory")
    ap.add_argument("--preview-mp4", default=None,
                    help="also save a side-by-side original|reconstruction "
                         "MP4 of the first clip to this path (shows "
                         "temporal distortion numbers can hide)")
    ap.add_argument("--min-psnr", type=float, default=None,
                    help="fail (exit 1) if reconstruction PSNR is below this")
    args = ap.parse_args()

    device = resolve_device(args.device)
    cfg = get_preset(args.preset)
    payload = load_checkpoint(args.vae_ckpt, map_location=str(device))
    if payload.get("kind") != "vae":
        print(f"error: {args.vae_ckpt} is not a VAE checkpoint", file=sys.stderr)
        return 2
    vae = CausalVAE().to(device)
    vae.load_state_dict(payload["vae"])

    if args.data_dir:
        dataset = FolderClipDataset(
            args.data_dir, frames=cfg.frames, height=cfg.height,
            width=cfg.width)
        source = args.data_dir
    else:
        dataset = MovingMNISTDataset(
            args.max_clips, height=cfg.height, width=cfg.width,
            frames=cfg.frames, base_seed=999)
        source = "synthetic Moving MNIST (smoke test only)"

    report = measure_latent_stats(
        vae, dataset, max_clips=args.max_clips,
        batch_size=args.batch_size, device=str(device))
    report["vae_ckpt"] = args.vae_ckpt
    report["dataset"] = source
    report["current_latent_scale"] = cfg.latent_scale

    print(json.dumps(report, indent=2))
    print(f"\nRecommended: python -m vidgen.train ... "
          f"--latent-scale {report['recommended_latent_scale']}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"Report written: {args.out}")
    if args.preview_dir:
        os.makedirs(args.preview_dir, exist_ok=True)
        for i in range(min(3, len(dataset))):
            clip, _caption = dataset[i]
            out = os.path.join(args.preview_dir, f"recon_{i}.png")
            save_recon_preview(vae, clip, out, device=str(device))
            print(f"Preview: {out}")
    if args.preview_mp4:
        clip, _caption = dataset[0]
        save_recon_mp4(vae, clip, args.preview_mp4, device=str(device))
        print(f"Side-by-side MP4: {args.preview_mp4}")
    if args.min_psnr is not None and report["recon_psnr_db"] is not None \
            and report["recon_psnr_db"] < args.min_psnr:
        print(f"GATE FAILED: PSNR {report['recon_psnr_db']} dB < "
              f"{args.min_psnr} dB — do not start DiT training on this VAE",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
