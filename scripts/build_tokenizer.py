#!/usr/bin/env python3
"""Build Vidgin's caption tokenizer from a dataset's own captions.

Phase 2 of the targeted improvement programme (2026-10-22): trains a
CaptionTokenizer (vidgen/text/tokenizer.py) on every caption found in
one or more prepared dataset directories (their manifest.jsonl plus any
manifest_<split>.jsonl, deduplicated), saves it as JSON, and reports
vocabulary size, in-corpus coverage, and how many word distinctions the
OLD crc32 word-hash would have collapsed for this exact vocabulary.

Usage:
    python scripts/build_tokenizer.py --data-dir /path/to/dataset \
        --out /path/to/tokenizer.json [--max-vocab 8192] [--min-freq 1]

Then train with it:
    python -m vidgen.train --data-dir /path/to/dataset \
        --tokenizer-vocab /path/to/tokenizer.json ...
The tokenizer is stored inside every checkpoint, so resume and
sampling rebuild it automatically — the JSON is only needed to START
a caption-mode run.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidgen.text.tokenizer import CaptionTokenizer, audit_hash_collisions


def collect_captions(data_dirs: list[str]) -> list[str]:
    captions: list[str] = []
    seen: set[tuple[str, str]] = set()
    for data_dir in data_dirs:
        manifests = [os.path.join(data_dir, "manifest.jsonl")]
        manifests += sorted(glob.glob(os.path.join(data_dir, "manifest_*.jsonl")))
        for manifest in manifests:
            if not os.path.isfile(manifest):
                continue
            with open(manifest, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    key = (rec.get("file", ""), rec.get("caption", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    if rec.get("caption"):
                        captions.append(rec["caption"])
    return captions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", action="append", required=True,
                    help="prepared dataset directory (repeatable)")
    ap.add_argument("--out", required=True, help="output tokenizer JSON path")
    ap.add_argument("--max-vocab", type=int, default=8192,
                    help="maximum vocabulary size including special tokens")
    ap.add_argument("--min-freq", type=int, default=1,
                    help="drop words appearing fewer times than this")
    args = ap.parse_args()

    captions = collect_captions(args.data_dir)
    if not captions:
        print("error: no captions found in the given dataset directories",
              file=sys.stderr)
        return 2
    tokenizer = CaptionTokenizer.train(
        captions, max_vocab=args.max_vocab, min_freq=args.min_freq)
    tokenizer.save(args.out)

    words = tokenizer.words[4:]
    collisions = audit_hash_collisions(words, buckets=8192)
    collided = sum(len(ws) for ws in collisions.values())
    print(f"Captions:        {len(captions)}")
    print(f"Vocabulary:      {len(tokenizer)} tokens (incl. 4 specials)")
    print(f"OOV rate:        {tokenizer.oov_rate(captions):.4f} on the training captions")
    print(f"Old hash scheme: {collided} of these words would have shared a "
          f"bucket with another word ({len(collisions)} colliding buckets)")
    print(f"Saved:           {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
