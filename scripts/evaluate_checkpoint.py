#!/usr/bin/env python3
"""Repeatable checkpoint evaluation: fixed prompts + seeds, saved MP4s,
non-learned quality metrics, and a baseline-vs-candidate report.

Phase 6/7 tool (2026-10-22). Generates the SAME prompts with the SAME
seeds and settings for the candidate checkpoint (and --baseline when
given), saves every clip as a verified MP4, computes the pixel-level
metrics from vidgen/evaluate.py, and writes report.json + report.md.

Prompt adherence is deliberately NOT scored: measuring it honestly
needs a learned text-video evaluator, which Vidgin's ownership rule
excludes. The saved MP4s are the prompt-adherence check — a human
watches them.

Usage:
    python scripts/evaluate_checkpoint.py \
        --dit-ckpt /path/to/dit.pt --vae-ckpt /path/to/vae.pt \
        [--baseline /path/to/old_dit.pt] \
        [--prompts "prompt one,prompt two" | --prompts-file prompts.txt] \
        [--seeds 0,1] [--steps 12] [--train-dir /path/to/dataset] \
        --out-dir /path/to/eval_out
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from vidgen.checkpoint import load_checkpoint
from vidgen.config import VidgenConfig, get_preset, resolve_device
from vidgen.data.clips import FolderClipDataset
from vidgen.dit.model import DiT
from vidgen.evaluate import evaluate_clip_set
from vidgen.sampler import sample_latents
from vidgen.seed import seed_everything
from vidgen.text.encoder import build_text_encoder
from vidgen.text.tokenizer import CaptionTokenizer, tokenizer_from_dict
from vidgen.vae.model import CausalVAE
from vidgen.video_io import write_mp4

DEFAULT_PROMPTS = [
    "people walking across a city street",
    "cars driving through a busy intersection",
    "a crowd moving through the city",
    "the camera pans across a busy city skyline",
    "a street scene changing from day to night",
]


def generate_clips(dit_ckpt: str, vae_ckpt: str | None, prompts, seeds,
                   steps: int, cfg_scale: float, method: str,
                   device, out_dir: str) -> list[np.ndarray]:
    """Generate one clip per (prompt, seed); save MP4 + npy per clip."""
    payload = load_checkpoint(dit_ckpt, map_location=str(device))
    if payload.get("kind") != "dit":
        raise ValueError(f"{dit_ckpt} is not a DiT checkpoint")
    cfg = (VidgenConfig.from_dict(payload["config"])
           if payload.get("config") else get_preset(payload["preset"]))
    vae_path = vae_ckpt or payload.get("vae_ckpt")
    if not vae_path:
        raise ValueError("no VAE checkpoint given (--vae-ckpt) and the "
                         "DiT checkpoint does not name one")
    vp = load_checkpoint(vae_path, map_location=str(device))
    vae = CausalVAE().to(device).eval()
    vae.load_state_dict(vp["vae"])

    tokenizer = (tokenizer_from_dict(payload["tokenizer"])
                 if payload.get("tokenizer") is not None else None)
    if getattr(cfg, "text_tokenizer", "hash") == "caption" and tokenizer is None:
        raise ValueError("caption-mode checkpoint carries no tokenizer "
                         "payload; it is incomplete")
    text_enc = build_text_encoder(cfg, tokenizer=tokenizer).to(device)
    text_enc.load_state_dict(payload["text_enc"])
    text_enc.eval()
    dit = DiT(cfg).to(device)
    dit.load_state_dict(payload["dit"])
    dit.eval()

    os.makedirs(out_dir, exist_ok=True)
    clips: list[np.ndarray] = []
    index = []
    with torch.no_grad():
        for prompt in prompts:
            text, text_mask = text_enc([prompt])
            null_text, null_mask = text_enc.null_embedding()
            for seed in seeds:
                lat_shape = (1, cfg.latent_channels, cfg.latent_t,
                             cfg.latent_h, cfg.latent_w)
                z = sample_latents(
                    dit, lat_shape, text.to(device), text_mask.to(device),
                    null_text.to(device), null_mask.to(device),
                    method=method, num_steps=steps, cfg_scale=cfg_scale,
                    seed=seed, device=str(device))
                clip = vae.decode(z / cfg.latent_scale)
                frames = clip[0].permute(1, 2, 3, 0).cpu().numpy()
                name = f"clip_{len(clips):03d}"
                write_mp4(frames, os.path.join(out_dir, name + ".mp4"),
                          fps=16.0)
                np.save(os.path.join(out_dir, name + ".npy"), frames)
                uint8 = ((frames * 0.5 + 0.5).clip(0, 1) * 255).round().astype(np.uint8)
                clips.append(uint8)
                index.append({"file": name + ".mp4", "prompt": prompt,
                              "seed": seed})
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return clips


def load_train_clips(train_dir: str, cfg, limit: int = 32) -> list[np.ndarray]:
    ds = FolderClipDataset(train_dir, frames=cfg.frames, height=cfg.height,
                           width=cfg.width)
    clips = []
    for i in range(min(limit, len(ds))):
        clip, _ = ds[i]
        arr = clip.permute(1, 2, 3, 0).numpy()
        clips.append(((arr * 0.5 + 0.5).clip(0, 1) * 255).round().astype(np.uint8))
    return clips


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dit-ckpt", required=True)
    ap.add_argument("--vae-ckpt", default=None)
    ap.add_argument("--baseline", default=None,
                    help="older DiT checkpoint to compare against, same "
                         "prompts/seeds/settings")
    ap.add_argument("--prompts", default=None,
                    help="comma-separated prompts (default: city theme set)")
    ap.add_argument("--prompts-file", default=None,
                    help="one prompt per line; overrides --prompts")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--method", default="euler")
    ap.add_argument("--train-dir", default=None,
                    help="training dataset for the memorization check")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            prompts = [l.strip() for l in f if l.strip()]
    elif args.prompts:
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    else:
        prompts = DEFAULT_PROMPTS
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = resolve_device(args.device)
    seed_everything(0)
    os.makedirs(args.out_dir, exist_ok=True)

    payload = load_checkpoint(args.dit_ckpt, map_location="cpu")
    cfg = (VidgenConfig.from_dict(payload["config"])
           if payload.get("config") else get_preset(payload["preset"]))
    train_clips = (load_train_clips(args.train_dir, cfg)
                   if args.train_dir else None)

    report: dict = {"prompts": prompts, "seeds": seeds,
                    "steps": args.steps, "cfg": args.cfg,
                    "train_clips_compared": len(train_clips) if train_clips else 0}
    candidate = generate_clips(
        args.dit_ckpt, args.vae_ckpt, prompts, seeds, args.steps,
        args.cfg, args.method, device,
        os.path.join(args.out_dir, "candidate"))
    report["candidate"] = {
        "checkpoint": args.dit_ckpt,
        "metrics": evaluate_clip_set(candidate, train_clips),
    }
    if args.baseline:
        baseline = generate_clips(
            args.baseline, args.vae_ckpt, prompts, seeds, args.steps,
            args.cfg, args.method, device,
            os.path.join(args.out_dir, "baseline"))
        report["baseline"] = {
            "checkpoint": args.baseline,
            "metrics": evaluate_clip_set(baseline, train_clips),
        }

    with open(os.path.join(args.out_dir, "report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    lines = ["# Vidgin checkpoint evaluation", "",
             f"Prompts: {len(prompts)} | seeds: {seeds} | steps: {args.steps} "
             f"| cfg: {args.cfg}", ""]
    for label in ("baseline", "candidate"):
        if label not in report:
            continue
        m = report[label]["metrics"]
        lines += [f"## {label}: {report[label]['checkpoint']}", "",
                  f"- clips: {m['clips']}",
                  f"- motion (mean abs frame diff): {m['motion_mean_absdiff']:.3f}",
                  f"- flicker score: {m['flicker_score']:.3f}",
                  f"- sharpness: {m['sharpness']:.3f}",
                  f"- diversity: {m['diversity']:.3f}",
                  f"- memorization mean nearest distance: {m['memorization']['mean_nearest']}",
                  f"- prompt adherence: {m['prompt_adherence']}", ""]
    with open(os.path.join(args.out_dir, "report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"Reports + MP4s in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
