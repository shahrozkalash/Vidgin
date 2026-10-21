#!/usr/bin/env python3
"""Coordinator tool for the weight-averaging pool: merge one round.

Takes the checkpoints of 2+ pool workers that all started a round from
the SAME checkpoint and trained the SAME number of steps on their own
--data-shard slices, and averages their model weights into the next
pooled checkpoint (DiLoCo-style; see vidgen/pool.py and
PYTHON_IMPLEMENTATION.md, "Weight-averaging pool").

Only floating-point weight tensors are averaged. With --base (the
checkpoint the round started from) the merge is one outer step:
    out = base + outer_lr * (mean - base)
Without --base the output is the plain mean and --outer-lr must be 1.0.

The output is a WEIGHTS-ONLY checkpoint (no optimizer/sampler/RNG
state): the next round starts from it via either trainer's
--init-weights, which continues step numbering from its stored step.

Usage:
  python scripts/average_checkpoints.py \
      worker1/latest.pt worker2/latest.pt worker3/latest.pt \
      --base pooled_round3.pt --outer-lr 1.0 \
      --out pooled_round4.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vidgen.pool import load_payload_file, merge_payloads  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Average one pool round's worker checkpoints into "
                    "the next pooled checkpoint.")
    ap.add_argument("inputs", nargs="+", metavar="WORKER_CKPT",
                    help="2 or more worker checkpoints from the same "
                         "round (same kind, preset, config and step)")
    ap.add_argument("--base", default=None, metavar="PATH",
                    help="the checkpoint the round started from; enables "
                         "the DiLoCo outer step base + outer_lr*(mean-base)")
    ap.add_argument("--outer-lr", type=float, default=1.0,
                    help="outer step size (default 1.0). Values other "
                         "than 1.0 require --base.")
    ap.add_argument("--out", required=True, metavar="PATH",
                    help="where to write the pooled checkpoint")
    args = ap.parse_args(argv)

    if len(args.inputs) < 2:
        ap.error("pooling a round needs at least 2 worker checkpoints")
    try:
        payloads = [load_payload_file(p) for p in args.inputs]
        base = load_payload_file(args.base) if args.base else None
        merged = merge_payloads(payloads, base=base,
                                outer_lr=args.outer_lr,
                                input_names=list(args.inputs),
                                base_name=args.base)
    except (ValueError, RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    # Plain torch.save (NOT vidgen.checkpoint.save_checkpoint): a pooled
    # checkpoint is a weights-only artifact — it must not acquire this
    # process's RNG state or pretend to carry optimizer state.
    torch.save(merged, args.out)
    print(f"pooled {len(args.inputs)} workers -> {args.out} "
          f"(kind={merged['kind']} preset={merged['preset']} "
          f"step={merged['step']} outer_lr={args.outer_lr} "
          f"base={args.base})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
