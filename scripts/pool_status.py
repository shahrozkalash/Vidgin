#!/usr/bin/env python3
"""Coordinator helper: which pool checkpoints are merge-ready?

Given checkpoint files and/or directories of them, prints each file's
kind / preset / step (and whether it is itself a pooled checkpoint),
then groups the readable ones by (kind, preset, step): any group of
2+ files is one finished round waiting to be merged with
scripts/average_checkpoints.py, and the exact merge command is printed.

Usage:
  python scripts/pool_status.py ~/pool_inbox/
  python scripts/pool_status.py worker1/latest.pt worker2/latest.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vidgen.pool import checkpoint_summary, load_payload_file  # noqa: E402


def collect_paths(paths: list[str]) -> list[str]:
    """Expand directories to their *.pt files (sorted); files pass through."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(
                os.path.join(p, f) for f in os.listdir(p)
                if f.endswith(".pt")))
        else:
            out.append(p)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Report pool checkpoint steps and merge-ready groups.")
    ap.add_argument("paths", nargs="+", metavar="PATH",
                    help="checkpoint files and/or directories of them")
    args = ap.parse_args(argv)

    files = collect_paths(args.paths)
    if not files:
        print("no checkpoint files found")
        return 0
    groups: dict[tuple, list[str]] = defaultdict(list)
    for path in files:
        try:
            s = checkpoint_summary(load_payload_file(path))
        except Exception as e:  # unreadable file: report, don't die
            print(f"{path}: ERROR {e}")
            continue
        pool_note = (f" pooled(workers={s['pool_workers']})"
                     if s["is_pool"] else "")
        print(f"{path}: kind={s['kind']} preset={s['preset']} "
              f"step={s['step']}{pool_note}")
        if s["kind"] is not None and s["step"] is not None:
            groups[(s["kind"], s["preset"], s["step"])].append(path)
    ready = {k: v for k, v in groups.items() if len(v) >= 2}
    if ready:
        print("\nmerge-ready groups (same kind, preset, step):")
        for (kind, preset, step), members in sorted(ready.items()):
            print(f"  {kind} preset={preset} step={step}: "
                  f"{len(members)} workers")
            cmd = " ".join(members)
            print(f"    python scripts/average_checkpoints.py {cmd} "
                  f"--out pooled_{kind}_step{step}.pt")
    else:
        print("\nno merge-ready groups yet (a round needs 2+ checkpoints "
              "with the same kind, preset and step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
