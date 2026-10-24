#!/usr/bin/env python3
"""Vidgin helper bundle: captions, dataset checking, and dataset packing.

One small command-line tool for the jobs around training:

  # Write proper captions for a prepared clip folder
  python scripts/vidgin_helper.py captions --data-dir /path/to/clips

  # Check that a folder is complete and trainer-ready
  python scripts/vidgin_helper.py check /path/to/clips \
      --frames 33 --height 256 --width 256 --require-depth

  # Merge finished groups/chunks into one training dataset
  python scripts/vidgin_helper.py pack /path/to/group1 /path/to/group2 \
      --labels group1,group2 --out /path/to/dataset

The caption command delegates to the existing free-tier Gemini tool in
``scripts/caption_clips.py``.  Check and pack are fully local: they use
no network and no external model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Make both the repository root (for ``vidgen``) and this script's folder
# (for ``caption_clips``) importable no matter where the tool is run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vidgen.data.dataset_tools import (  # noqa: E402
    DEFAULT_DARK_THRESHOLD,
    _read_manifest_strict,
    audit_captions,
    check_dataset,
    find_near_duplicates,
    format_caption_audit,
    format_dataset_report,
    format_pack_report,
    format_split_report,
    make_splits,
    pack_datasets,
    verify_splits,
)


def _caption_dataset(*args, **kwargs):
    """Lazy wrapper so check/pack work even if caption deps are absent."""

    from caption_clips import caption_dataset

    return caption_dataset(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vidgin_helper.py",
        description="Vidgin helper bundle: captions, clip checking, and dataset packing.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    captions = sub.add_parser(
        "captions", help="write proper vision captions for a prepared clip folder"
    )
    captions.add_argument("--data-dir", required=True, help="prepared clips directory")
    captions.add_argument(
        "--backend",
        choices=["gemini", "local", "auto"],
        default="gemini",
        help="captioning backend: 'gemini' (Free Tier API), 'local' (BLIP vision model), or 'auto'",
    )
    captions.add_argument("--limit", type=int, default=None,
                          help="maximum clips to inspect in this run (default: all)")
    captions.add_argument("--max-calls", type=int, default=240,
                          help="free-tier API call cap for this run (default: 240)")
    captions.add_argument("--model", default="gemini-3.8-flash",
                          help="Gemini vision model (default: gemini-3.8-flash)")
    captions.add_argument("--overwrite", action="store_true",
                          help="re-caption clips already captioned by Gemini or local")
    captions.add_argument("--pace", type=float, default=6.0,
                          help="seconds between free-tier requests (default: 6.0)")

    check = sub.add_parser(
        "check", help="report missing, broken, dark, or mismatched clips"
    )
    check.add_argument("data_dir", help="prepared clips directory")
    check.add_argument("--frames", type=int, default=None, help="expected frames per clip")
    check.add_argument("--height", type=int, default=None, help="expected clip height")
    check.add_argument("--width", type=int, default=None, help="expected clip width")
    check.add_argument("--require-depth", action="store_true",
                       help="treat a missing depth reference/file as an error")
    check.add_argument("--dark-threshold", type=float, default=DEFAULT_DARK_THRESHOLD,
                       help=f"sampled brightness below this is 'too dark' (default: {DEFAULT_DARK_THRESHOLD})")
    check.add_argument("--detect-duplicates", action="store_true",
                       help="also hash clip files and report exact duplicates (slower)")
    check.add_argument("--strict", action="store_true",
                       help="exit non-zero when warnings are found, not only errors")
    check.add_argument("--json", dest="json_path", default=None,
                       help="also write the full report as JSON to this path")

    pack = sub.add_parser(
        "pack", help="merge prepared clip folders into one training dataset"
    )
    pack.add_argument("sources", nargs="+", help="prepared clips directories to merge")
    pack.add_argument("--out", required=True, help="output dataset directory")
    pack.add_argument("--labels", default=None,
                      help="comma-separated short label for each source, in order; "
                           "labels become file-name prefixes")
    pack.add_argument("--overwrite", action="store_true",
                      help="replace the output directory if it already contains files")
    pack.add_argument("--json", dest="json_path", default=None,
                      help="also write the pack report as JSON to this path")

    split = sub.add_parser(
        "split",
        help="write deterministic train/val/test split manifests into a dataset",
    )
    split.add_argument("data_dir", help="prepared clips directory to split")
    split.add_argument("--ratios", default="0.8,0.1,0.1",
                       help="train,val,test shares (comma-separated, sum to 1)")
    split.add_argument("--seed", type=int, default=1234,
                       help="split seed; the same seed reproduces the same split")
    split.add_argument("--overwrite", action="store_true",
                       help="regenerate split manifests that already exist")

    caudit = sub.add_parser(
        "captions-audit",
        help="audit caption quality: empty/duplicated/generic/search-query captions",
    )
    caudit.add_argument("data_dir", help="prepared clips directory")
    caudit.add_argument("--json", dest="json_path", default=None,
                        help="also write the audit as JSON to this path")

    vsplit = sub.add_parser(
        "verify-splits",
        help="verify split manifests are group-disjoint (no source leakage)",
    )
    vsplit.add_argument("data_dir", help="prepared clips directory with splits")

    ndup = sub.add_parser(
        "near-duplicates",
        help="find perceptually near-duplicate clips (re-encodes/crops of one source)",
    )
    ndup.add_argument("data_dir", help="prepared clips directory")
    ndup.add_argument("--threshold", type=float, default=0.01,
                      help="max mean signature distance to call two clips duplicates")
    ndup.add_argument("--max-clips", type=int, default=512,
                      help="cap on clips compared (the check is pairwise)")

    return parser


def _write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "captions":
        print(
            f"Captioning clips in {args.data_dir} using {args.backend} backend (model: {args.model})...",
            flush=True,
        )
        processed, skipped, failed = _caption_dataset(
            data_dir=args.data_dir,
            backend=args.backend,
            limit=args.limit,
            max_calls=args.max_calls,
            model=args.model,
            overwrite=args.overwrite,
            pace_seconds=args.pace,
        )
        print("=" * 50)
        print(f"Captioned: {processed}")
        print(f"Skipped:   {skipped} (already captioned)")
        print(f"Failed:    {failed}")
        return 0 if failed == 0 else 1

    if args.command == "check":
        report = check_dataset(
            args.data_dir,
            expected_frames=args.frames,
            expected_height=args.height,
            expected_width=args.width,
            require_depth=args.require_depth,
            dark_threshold=args.dark_threshold,
            detect_duplicates=args.detect_duplicates,
        )
        print(format_dataset_report(report))
        if args.json_path:
            _write_json(args.json_path, report.to_dict())
            print(f"JSON report: {args.json_path}")
        if report.errors or (args.strict and report.warnings):
            return 1
        return 0

    if args.command == "pack":
        labels = None
        if args.labels:
            labels = [part.strip() for part in args.labels.split(",")]
        try:
            report = pack_datasets(
                args.sources,
                args.out,
                labels=labels,
                overwrite=args.overwrite,
            )
        except (ValueError, FileNotFoundError, FileExistsError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(format_pack_report(report))
        if args.json_path:
            _write_json(args.json_path, report.to_dict())
            print(f"JSON report: {args.json_path}")
        return 0

    if args.command == "split":
        try:
            parts = [float(p) for p in args.ratios.split(",")]
            if len(parts) != 3:
                raise ValueError("--ratios needs exactly three numbers: train,val,test")
            report = make_splits(
                args.data_dir,
                ratios=(parts[0], parts[1], parts[2]),
                seed=args.seed,
                overwrite=args.overwrite,
            )
        except (ValueError, FileNotFoundError, FileExistsError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(format_split_report(report))
        return 0

    if args.command == "captions-audit":
        try:
            entries = _read_manifest_strict(args.data_dir)
        except (ValueError, FileNotFoundError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        audit = audit_captions(entries)
        print(format_caption_audit(audit))
        if args.json_path:
            _write_json(args.json_path, audit.to_dict())
            print(f"JSON report: {args.json_path}")
        return 0 if audit.ok else 1

    if args.command == "verify-splits":
        try:
            result = verify_splits(args.data_dir)
        except (ValueError, FileNotFoundError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2))
        if result["small_dataset_warning"]:
            print("WARNING: fewer than 50 clips — validation metrics on "
                  "this dataset are not meaningful yet.")
        return 0 if result["ok"] else 1

    if args.command == "near-duplicates":
        try:
            pairs = find_near_duplicates(
                args.data_dir, threshold=args.threshold,
                max_clips=args.max_clips)
        except (ValueError, FileNotFoundError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if not pairs:
            print("No near-duplicate pairs found.")
        for a, b, d in pairs:
            print(f"  {a} ~ {b}  (distance {d:.4f})")
        print(f"{len(pairs)} near-duplicate pair(s).")
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
