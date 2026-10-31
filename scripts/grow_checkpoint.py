#!/usr/bin/env python3
"""Grow a trained DiT checkpoint into a larger preset (model growth).

Takes a DiT checkpoint trained at a smaller preset and writes a
WEIGHTS-ONLY checkpoint in the target preset's geometry, with every
learned weight transplanted to its semantically correct position and
the new capacity fresh-initialized (see vidgen/grow.py for the method
and its honest limits). Train the grown model as a fresh run via the
normal trainer flag:

  python -m vidgen.train --preset vidgen480p \\
      --init-weights grown_480p.pt --vae-ckpt <vae ckpt> ...

The VAE is not grown here — within one shape it transfers exactly
via vidgen.train_vae --init-weights; a VAE shape change uses
scripts/grow_vae.py (vidgen/vae/grow.py).

Usage:
  python scripts/grow_checkpoint.py small_latest.pt \\
      --to vidgen480p --out grown_480p.pt
  python scripts/grow_checkpoint.py small_latest.pt \\
      --to small --out /dev/null   # (same geometry: refused, use the
                                   #  checkpoint directly)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vidgen.config import PRESETS  # noqa: E402
from vidgen.grow import grow_checkpoint_file  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Transplant a trained DiT checkpoint's weights "
                    "into a larger preset (weights-only output).")
    ap.add_argument("input", metavar="DIT_CKPT",
                    help="trained DiT checkpoint at the smaller preset")
    ap.add_argument("--to", required=True, metavar="PRESET",
                    choices=sorted(PRESETS),
                    help="target (larger) preset")
    ap.add_argument("--out", required=True, metavar="PATH",
                    help="where to write the grown checkpoint")
    ap.add_argument("--seed", type=int, default=1234,
                    help="seed for fresh-initializing the new capacity "
                         "(default 1234)")
    ap.add_argument("--depth", action="store_true",
                    help="give the grown model a depth-conditioning "
                         "stem even if the source was trained without "
                         "one (zero-init: the stem starts silent, "
                         "like a fresh depth model). By default the "
                         "source's depth mode is inherited.")
    args = ap.parse_args(argv)

    try:
        meta = grow_checkpoint_file(args.input, args.out, args.to,
                                    seed=args.seed,
                                    force_depth=args.depth)
    except (ValueError, RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"grown {meta['source_preset']} (step {meta['source_step']}) "
          f"-> {meta['target_preset']} -> {args.out}")
    print(f"  transplanted {meta['copied_params']:,} of "
          f"{meta['total_params']:,} target params "
          f"({100.0 * meta['copied_fraction']:.1f}% carried from the "
          f"source; the rest is fresh capacity to train)")
    print(f"  block map (target layer -> source layer): "
          f"{meta['block_map']}")
    print(f"  text transfer: {meta['text_transfer']}"
          + (f" ({meta.get('text_words_placed', 0)} words placed, "
             f"{meta.get('text_bucket_collisions', 0)} bucket collisions)"
             if meta["text_transfer"] == "embeddings_only" else ""))
    print(f"  train it with: python -m vidgen.train "
          f"--preset {meta['target_preset']} --init-weights {args.out} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
