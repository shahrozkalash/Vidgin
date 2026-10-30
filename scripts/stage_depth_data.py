#!/usr/bin/env python3
"""Copy DepthBrain training data from a slow mount (Google Drive) to local disk.

Trainers stall when they read thousands of small .npy files through the
Drive FUSE mount, so the cloud notebook (notebooks/train_depthbrain.ipynb)
stages first: for each dataset folder this copies exactly what
``scripts/train_depthbrain.py`` reads — ``manifest.jsonl``, every clip
``.npy`` it lists and that clip's ``*_depth.npy`` MiDaS map — into
``<dst>/<folder name>/``. Nothing else (videos, previews, backups) is copied.

Folder names are kept (the trainer's held-out split falls back to
folder name + file when a manifest has no ``source_url``). Two sources with
the same folder name get their parent's name prepended (``round1__packed``).

Re-running is safe and cheap: a file already staged with the same size is
skipped, and the manifest is written last, so a half-finished copy never
looks complete. ``<dst>/staged.json`` lists the staged folders for the
trainer's ``--data``; check them with ``train_depthbrain.py --dry-run``.

Usage:
  python scripts/stage_depth_data.py /content/drive/MyDrive/vidgin-data/round1 \\
      /content/drive/MyDrive/vidgin-data/round2 --dst /content/depth_data
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidgen.data.clips import MANIFEST_NAME, _load_manifest


def staged_names(sources: list[str]) -> list[str]:
    """Destination folder name per source: its own name, or parent__name
    when two sources share a name."""
    norm = [os.path.normpath(s) for s in sources]
    base = [os.path.basename(s) for s in norm]
    names = [f"{os.path.basename(os.path.dirname(s))}__{b}"
             if base.count(b) > 1 else b for s, b in zip(norm, base)]
    if len(set(names)) != len(names):
        raise ValueError(f"cannot give the sources distinct folder names: "
                         f"{sources}")
    return names


def _copy(src: str, dst: str, always: bool = False) -> bool:
    """Copy unless already staged with the same size; True if copied."""
    if not always and os.path.isfile(dst) \
            and os.path.getsize(dst) == os.path.getsize(src):
        return False
    tmp = dst + ".part"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return True


def stage(sources: list[str], dst: str, quiet: bool = False) -> dict:
    """Stage every source folder under ``dst``; returns the counts and
    writes ``<dst>/staged.json``. Rows whose clip or depth map is missing
    at the source are reported (``missing``), not silently dropped — the
    manifest is copied as-is, so ``--dry-run`` shows the mismatch."""
    os.makedirs(dst, exist_ok=True)
    rows = []
    for src, name in zip(sources, staged_names(sources)):
        out = os.path.join(dst, name)
        os.makedirs(out, exist_ok=True)
        recs = _load_manifest(src)
        copied = skipped = missing = 0
        for i, rec in enumerate(recs):
            for key in ("file", "depth"):
                fname = rec.get(key)
                if not isinstance(fname, str) or not fname:
                    missing += 1
                    continue
                s = os.path.join(src, fname)
                if not os.path.isfile(s):
                    missing += 1
                    continue
                d = os.path.join(out, fname)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                if _copy(s, d):
                    copied += 1
                else:
                    skipped += 1
            if not quiet and (i + 1) % 200 == 0:
                print(f"  {name}: {i + 1}/{len(recs)} clips", flush=True)
        # The manifest goes last: a staged folder with a manifest is complete.
        _copy(os.path.join(src, MANIFEST_NAME), os.path.join(out, MANIFEST_NAME),
              always=True)
        row = {"source": src, "staged": out, "manifest_lines": len(recs),
               "files_copied": copied, "files_already_staged": skipped,
               "files_missing_at_source": missing}
        rows.append(row)
        if not quiet:
            print(f"{src} -> {out}: {len(recs)} manifest lines, "
                  f"{copied} files copied, {skipped} already staged, "
                  f"{missing} missing at source")
    result = {"dirs": [r["staged"] for r in rows], "datasets": rows}
    with open(os.path.join(dst, "staged.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sources", nargs="+",
                    help="dataset folders (manifest.jsonl + clips + depth maps)")
    ap.add_argument("--dst", required=True, help="local folder to stage into")
    args = ap.parse_args(argv)
    result = stage(args.sources, args.dst)
    missing = sum(r["files_missing_at_source"] for r in result["datasets"])
    print(f"staged {len(result['dirs'])} folder(s) into {args.dst} "
          f"(list: {os.path.join(args.dst, 'staged.json')})")
    if missing:
        print(f"WARNING: {missing} clip/depth files listed in the manifests "
              "are missing at the source — --dry-run will fail until they "
              "are fixed", file=sys.stderr)


if __name__ == "__main__":
    main()
