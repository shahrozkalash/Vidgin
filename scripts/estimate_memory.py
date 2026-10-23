#!/usr/bin/env python3
"""Estimate Vidgin training memory BEFORE starting a run (Phase 10).

Counts parameters exactly (models built on the meta device — no real
memory is allocated, so even the 14B preset is safe to measure on a
laptop) and turns them into a training-memory budget:

  weights (fp32 master)   params x 4 bytes
  AdamW state (m + v)     params x 8 bytes
  gradients (fp32)        params x 4 bytes
  -----------------------------------------
  fixed total             params x 16 bytes   (x 14 with --amp bf16:
                                               bf16 grads + bf16 copy)

Activations are the wild card: the estimate uses the standard
transformer approximation (~34 bytes per token per dim per layer for
stored activations without checkpointing; ~2 bytes per token per dim
per layer PLUS one full layer with --grad-checkpoint) and is labelled
an estimate — measure the real peak with the trainers' gpu_mem_mb
step records on the target hardware.

Usage:
    python scripts/estimate_memory.py --preset small --batch-size 2
    python scripts/estimate_memory.py --preset vidgen480p \
        --batch-size 1 --grad-checkpoint --amp bf16
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from vidgen.config import get_preset  # noqa: E402
from vidgen.dit.model import DiT  # noqa: E402
from vidgen.text.encoder import build_text_encoder  # noqa: E402
from vidgen.vae.model import CausalVAE  # noqa: E402

GPUS = [("Free T4 (Colab/Kaggle)", 16), ("RTX 4090 / L4", 24),
        ("A100 / H100 (40 GB)", 40), ("A100 / H100 (80 GB)", 80)]


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def estimate(preset: str, batch_size: int, grad_checkpoint: bool,
             amp: str) -> dict:
    cfg = get_preset(preset)
    with torch.device("meta"):
        dit = DiT(cfg)
        text = build_text_encoder(cfg)
        vae = CausalVAE()
    dit_p, text_p, vae_p = count_params(dit), count_params(text), count_params(vae)
    total = dit_p + text_p + vae_p

    bytes_per_param = 14 if amp == "bf16" else 16
    fixed_gb = total * bytes_per_param / 1e9

    tokens = cfg.token_t * cfg.token_h * cfg.token_w
    dim = cfg.dit.dim
    layers = cfg.dit.layers
    if grad_checkpoint:
        act_bytes = tokens * dim * (2 * layers + 34) * batch_size
    else:
        act_bytes = tokens * dim * 34 * layers * batch_size
    act_gb = act_bytes / 1e9
    vae_act_gb = (batch_size * cfg.frames * cfg.height * cfg.width
                  * 3 * 4 * 8) / 1e9  # rough VAE activation allowance

    total_gb = fixed_gb + act_gb + vae_act_gb
    return {
        "preset": preset,
        "dit_params": dit_p, "text_params": text_p, "vae_params": vae_p,
        "total_params": total,
        "tokens_per_clip": tokens,
        "fixed_gb": round(fixed_gb, 2),
        "activation_gb_est": round(act_gb, 2),
        "vae_activation_gb_est": round(vae_act_gb, 2),
        "total_gb_est": round(total_gb, 2),
        "fits": {name: total_gb <= gb * 0.92 for name, gb in GPUS},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
    args = ap.parse_args()

    r = estimate(args.preset, args.batch_size, args.grad_checkpoint, args.amp)
    print(f"Preset:          {r['preset']}")
    print(f"DiT params:      {r['dit_params']:,}")
    print(f"Text params:     {r['text_params']:,}")
    print(f"VAE params:      {r['vae_params']:,}")
    print(f"Total params:    {r['total_params']:,}")
    print(f"Tokens/clip:     {r['tokens_per_clip']:,}")
    print(f"Fixed (weights+AdamW+grads): {r['fixed_gb']} GB")
    print(f"DiT activations (estimate):  {r['activation_gb_est']} GB "
          f"(batch {args.batch_size}, grad_checkpoint={args.grad_checkpoint})")
    print(f"VAE activations (allowance): {r['vae_activation_gb_est']} GB")
    print(f"TOTAL estimate:  {r['total_gb_est']} GB")
    for name, _gb in GPUS:
        print(f"  fits {name}: {'YES' if r['fits'][name] else 'NO'}")
    print("Note: fixed memory is exact (counted parameters); activations "
          "are an approximation — confirm with the gpu_mem_mb records "
          "in the training log on the real hardware.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
