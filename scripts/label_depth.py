#!/usr/bin/env python3
"""Add (or replace) depth maps on an already prepared dataset.

``prepare_clips.py --depth`` labels clips as they are fetched; this labels
a dataset that already exists — e.g. to make the MiDaS teacher maps
DepthBrain trains from, or (once a DepthBrain has passed its agreement
check) to give a round DepthBrain maps:

  # MiDaS (default; needs internet once)
  python scripts/label_depth.py --data /data/round1
  # a DepthBrain that passed its agreement check (no internet)
  python scripts/label_depth.py --data /data/round5 --depth-source depthbrain \\
      --depthbrain-ckpt checkpoints/depthbrain/depthbrain.pt

Clips that already have a map are skipped unless ``--overwrite``. Each
labelled entry records ``"depth"`` and ``"depth_source"``; the manifest is
rewritten atomically (temp file + rename) every ``--flush-every`` clips,
so an interrupted run keeps its progress and simply continues when rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from vidgen.config import resolve_device
from vidgen.data.clips import MANIFEST_NAME, _load_manifest
from vidgen.depth import (DEFAULT_DEPTH_SOURCE, DEPTH_SOURCES,
                          depth_source_id, estimate_depth)


def _write_manifest(path: str, entries: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in entries:
            f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def label_dataset(data: str, source: str = DEFAULT_DEPTH_SOURCE,
                  ckpt: str | None = None, device: str = "cpu",
                  overwrite: bool = False, limit: int | None = None,
                  flush_every: int = 50, manifest_name: str = MANIFEST_NAME,
                  allow_unverified: bool = False) -> dict:
    """Label ``data`` in place; returns ``{"labelled", "skipped", "source"}``."""
    entries = _load_manifest(data, manifest_name)
    # Fails fast (before any clip) on a missing or unverified checkpoint.
    sid = depth_source_id(source, ckpt, device, allow_unverified)
    path = os.path.join(data, manifest_name)
    done = skipped = 0
    for rec in entries:
        if limit is not None and done >= limit:
            break
        have = rec.get("depth") and os.path.isfile(
            os.path.join(data, str(rec["depth"])))
        if have and not overwrite:
            skipped += 1
            continue
        clip = np.load(os.path.join(data, rec["file"]))
        dname = rec["file"][:-4] + "_depth.npy"
        np.save(os.path.join(data, dname),
                estimate_depth(clip, device=device, source=source, ckpt=ckpt,
                               allow_unverified=allow_unverified))
        rec["depth"], rec["depth_source"] = dname, sid
        done += 1
        if done % flush_every == 0:
            _write_manifest(path, entries)
    if done:
        _write_manifest(path, entries)
    return {"labelled": done, "skipped": skipped, "source": sid}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="prepared dataset folder")
    ap.add_argument("--depth-source", choices=list(DEPTH_SOURCES),
                    default=DEFAULT_DEPTH_SOURCE)
    ap.add_argument("--depthbrain-ckpt", default=None,
                    help="DepthBrain checkpoint (default: $VIDGIN_DEPTHBRAIN "
                         "or checkpoints/depthbrain/depthbrain.pt)")
    ap.add_argument("--allow-unverified-depthbrain", action="store_true",
                    help="use a DepthBrain that has NOT passed its "
                         "agreement check (experiments only)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace maps that already exist")
    ap.add_argument("--limit", type=int, default=None,
                    help="label at most this many clips")
    ap.add_argument("--flush-every", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--manifest", default=MANIFEST_NAME)
    args = ap.parse_args(argv)
    try:
        out = label_dataset(args.data, args.depth_source, args.depthbrain_ckpt,
                            resolve_device(args.device), args.overwrite,
                            args.limit, args.flush_every, args.manifest,
                            args.allow_unverified_depthbrain)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")
    print(f"labelled {out['labelled']} clip(s), skipped {out['skipped']} "
          f"that already had a map — {out['source']}")


if __name__ == "__main__":
    main()
