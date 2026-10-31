#!/usr/bin/env python3
"""Grow a trained VAE checkpoint into a wider VAE geometry.

Takes a VAE checkpoint trained at the historical shape (latent 16 /
base 32) and writes a WEIGHTS-ONLY checkpoint at the target preset's
VAE shape (the Small preset: latent 32 / base 64), with every learned
weight transplanted into the widened tensors and the new capacity
zero-initialized so the grown network reproduces the donor's function
at init as closely as the architecture allows (see
vidgen/vae/grow.py for the method and its one honest exception, the
mid ResBlocks' GroupNorm statistics).

The output carries the donor's step number but NO optimizer state:
train it as a fresh run via the normal trainer flag:

  python -m vidgen.train_vae --preset small \\
      --init-weights grown_vae.pt --data-dir <clips> ...

Usage:
  python scripts/grow_vae.py step_020500.pt \\
      --to small --out grown_step_020500.pt
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vidgen.config import PRESETS  # noqa: E402
from vidgen.vae.grow import grow_vae_checkpoint_file  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Transplant a trained VAE checkpoint's weights "
                    "into a wider VAE geometry (weights-only output).")
    ap.add_argument("input", metavar="VAE_CKPT",
                    help="trained VAE checkpoint at the smaller shape")
    ap.add_argument("--to", default="small", metavar="PRESET",
                    choices=sorted(PRESETS),
                    help="target preset whose VAE shape to grow into "
                         "(default small: latent 32 / base 64)")
    ap.add_argument("--out", required=True, metavar="PATH",
                    help="where to write the grown checkpoint")
    args = ap.parse_args(argv)

    try:
        meta = grow_vae_checkpoint_file(args.input, args.out, args.to)
    except (ValueError, RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"grown VAE {meta['source_preset']} (step "
          f"{meta['source_step']}) {meta['source_shape']} -> "
          f"{meta['target_preset']} {meta['target_shape']} -> {args.out}")
    print(f"  transplanted {meta['vae_copied_params']:,} of "
          f"{meta['vae_total_params']:,} target params "
          f"({100.0 * meta['copied_fraction']:.1f}% carried from the "
          f"source; the rest is new capacity that starts silent)")
    print(f"  source weights used: {meta['source_weights']}; "
          f"optimizer dropped — fresh optimizer on the grown run")
    print(f"  train it with: python -m vidgen.train_vae "
          f"--preset {meta['target_preset']} "
          f"--init-weights {args.out} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
