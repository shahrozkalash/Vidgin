#!/usr/bin/env python3
"""Assemble the caption engine vs Gemini head-to-head (the "arena").

This builds inputs only. The caption engine (scripts/caption_engine.py) is
used as-is: its ``--eval-suite`` measures, its ``make_ab_pack`` draws the
blind pairs and its ``--ab-score`` scores them. Nothing here captions a
clip or calls a model.

Three steps (each is a subcommand):

1. ``build SOURCE OUT`` — SOURCE is a captioned chunk folder (Stage 2
   chunk3: 120 clips, all Gemini-captioned). Takes ``--n`` (120) Gemini rows
   and writes, into a NEW folder OUT (never over an existing one)::

     OUT/reference/<chunk>/manifest.jsonl  the Gemini rows, byte-for-byte
     OUT/reference/.gemini_baseline.json   immutable snapshot of those rows
     OUT/engine/<chunk>/manifest.jsonl     the same clips, captions removed
     OUT/engine/<chunk>/clip_*.npy (+ _depth.npy)   the 120 clips
     OUT/pack.json, OUT/README.md          what was taken, how to run it

   The source is only read. If it has a ``.gemini_baseline.json`` (in the
   folder or its parent, or given with ``--source-baseline``), every taken
   row must still match it, or the build stops: a changed baseline is
   never copied.
2. Caption ``OUT/engine`` with the engine (owner's run), then ``ab OUT``:
   the engine's ``make_ab_pack`` draws ``--pairs`` (30) blind pairs with
   ``--seed``; the judge's copy goes to ``OUT/judge/`` (the pack plus one
   frame strip per pair) and the key to ``OUT/sealed/``. Give the judge
   ONLY ``OUT/judge/`` — the rest of OUT holds the Gemini captions.
3. After judging, ``score OUT RATED_PACK`` opens the sealed key with the
   engine's ``score_ab`` and writes ``OUT/ab_result.json`` for
   ``caption_engine.py --eval-suite OUT/engine OUT/reference --ab-result``.

The same SOURCE, ``--n``, ``--pairs`` and ``--seed`` give the same pack,
byte for byte.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import caption_engine as engine

N_CLIPS = 120
N_PAIRS = 30
SEED = 42
# Caption fields removed from the engine's copy of a row: what the engine
# writes, and the Gemini-specific fields it must not see.
_STRIP_PREFIXES = ("caption", "gemini", "vidgin_")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _row_bytes(line: bytes) -> bytes:
    return line.rstrip(b"\r\n")


def find_source_baseline(source: Path) -> Path | None:
    """The engine's ``.gemini_baseline.json`` covering SOURCE: in the chunk
    folder itself or in its parent (a baseline snapshotted on the whole
    Stage 2 tree)."""
    for d in (source, source.parent):
        p = engine.baseline_path(d)
        if p.is_file():
            return p
    return None


def engine_row(rec: dict) -> dict:
    """The row the engine captions: everything but the captions."""
    return {k: v for k, v in rec.items()
            if not k.startswith(_STRIP_PREFIXES) and k not in engine.FIELDS}


def build(source: str, out: str, n: int = N_CLIPS, seed: int = SEED,
          source_baseline: str | None = None) -> dict:
    src, dst = Path(source), Path(out)
    manifest = src / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"{manifest} not found")
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"{dst} exists and is not empty — an eval "
                              "pack is never rebuilt in place; use a new folder")
    raw = manifest.read_bytes().splitlines(keepends=True)
    gemini = [i for i, line in enumerate(raw)
              if line.strip() and json.loads(line).get("caption_source") == "gemini"]
    if len(gemini) < n:
        raise ValueError(f"{manifest}: only {len(gemini)} Gemini-captioned "
                         f"rows, need {n}")
    chosen = gemini if len(gemini) == n else \
        sorted(random.Random(seed).sample(gemini, n))

    # The source's immutable baseline, if any, must still hold for every row.
    baseline_file = (Path(source_baseline) if source_baseline
                     else find_source_baseline(src))
    if baseline_file is not None and not baseline_file.is_file():
        raise FileNotFoundError(f"{baseline_file} not found")
    baseline_rows = 0
    if baseline_file is not None:
        base = json.loads(baseline_file.read_text(encoding="utf-8"))
        rel = manifest.relative_to(baseline_file.parent).as_posix()
        for i in chosen:
            want = base.get(f"{rel}::{i}")
            if want is None:
                raise ValueError(f"{manifest} row {i + 1} is not in "
                                 f"{baseline_file}")
            if _sha(raw[i]) != want:
                raise ValueError(f"{manifest} row {i + 1} no longer matches "
                                 f"its Gemini baseline ({baseline_file}); "
                                 "refusing to build from a changed baseline")
            baseline_rows += 1

    chunk = src.name
    ref_dir, eng_dir = dst / "reference" / chunk, dst / "engine" / chunk
    ref_dir.mkdir(parents=True)
    eng_dir.mkdir(parents=True)
    ref_lines, eng_lines, clips = [], [], []
    for i in chosen:
        rec = json.loads(raw[i])
        clip = engine.find_clip(manifest, rec)
        if clip is None:
            raise FileNotFoundError(f"{manifest} row {i + 1}: clip not found")
        rel_clip = clip.resolve().relative_to(src.resolve())
        files = [rel_clip]
        depth = clip.with_name(clip.stem + "_depth.npy")
        if depth.is_file():
            files.append(depth.resolve().relative_to(src.resolve()))
        for f in files:
            (eng_dir / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / f, eng_dir / f)
        ref_lines.append(raw[i] if raw[i].endswith(b"\n") else raw[i] + b"\n")
        eng_lines.append((json.dumps(engine_row(rec), ensure_ascii=False,
                                     separators=(",", ":")) + "\n").encode())
        clips.append({"row": i, "clip": rel_clip.as_posix(),
                      "depth": len(files) > 1, "sha256": _sha(raw[i])})
    (ref_dir / "manifest.jsonl").write_bytes(b"".join(ref_lines))
    (eng_dir / "manifest.jsonl").write_bytes(b"".join(eng_lines))
    with contextlib.redirect_stdout(io.StringIO()):
        engine.snapshot_baseline(dst / "reference")

    info = {
        "tool": "build_caption_eval_pack.py", "source": str(src),
        "chunk": chunk, "seed": seed, "clips": n,
        "source_gemini_rows": len(gemini),
        "source_baseline": (str(baseline_file) if baseline_file else None),
        "source_baseline_rows_checked": baseline_rows,
        "rows": clips,
    }
    (dst / "pack.json").write_text(json.dumps(info, indent=2) + "\n",
                                   encoding="utf-8")
    (dst / "README.md").write_text(_readme(chunk, n), encoding="utf-8")
    return info


def _readme(chunk: str, n: int) -> str:
    return f"""# Caption engine vs Gemini — eval pack ({chunk}, {n} clips)

- `reference/` — the Gemini captions, byte-identical to the source rows
  (`python scripts/caption_engine.py --verify <pack>/reference` checks them
  against `reference/.gemini_baseline.json`).
- `engine/` — the same {n} clips, captions removed, for the engine.

1. Caption the clips (owner's run, local model):
   `python scripts/caption_engine.py <pack>/engine --backend ollama --model <model>`
2. Draw the blind pairs:
   `python scripts/build_caption_eval_pack.py ab <pack>`
   Give the judge ONLY `<pack>/judge/`. `<pack>/sealed/` holds the key.
3. After judging (ratings filled in `judge/vidgin_ab_pack.md`):
   `python scripts/build_caption_eval_pack.py score <pack> <rated pack>`
4. The full verdict:
   `python scripts/caption_engine.py --eval-suite <pack>/engine <pack>/reference --ab-result <pack>/ab_result.json`
"""


def pair_keys(out: str) -> tuple[set, set]:
    """Join keys (relative chunk folder, clip basename) of the engine and
    reference trees — the keys the engine's eval suite pairs by."""
    def keys(base: Path) -> set:
        found = set()
        for m in engine.manifests(base):
            for line in m.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                name = next((str(e[k]) for k in ("clip_path", "video_path",
                                                  "path", "file", "filename")
                             if isinstance(e.get(k), str)), None)
                if name:
                    found.add((m.parent.relative_to(base).as_posix(),
                               Path(name).name))
        return found
    return keys(Path(out) / "engine"), keys(Path(out) / "reference")


def _frame_strip(clip_path: Path, frames: int = 4) -> bytes:
    a = np.load(clip_path, mmap_mode="r")
    if a.ndim == 3:
        a = np.repeat(a[..., None], 3, axis=-1)
    idx = np.linspace(0, a.shape[0] - 1, frames).round().astype(int)
    strip = np.concatenate([np.asarray(a[i])[..., :3] for i in idx], axis=1)
    if strip.dtype != np.uint8:
        strip = np.clip(strip * (255.0 if strip.max() <= 1.0 else 1.0),
                        0, 255).astype(np.uint8)
    return engine.png_bytes(np.ascontiguousarray(strip))


def make_ab(out: str, pairs: int = N_PAIRS, seed: int | None = None) -> dict:
    """Blind pairs via the engine's own ``make_ab_pack``; split into the
    judge's folder and the sealed key."""
    dst = Path(out)
    info = json.loads((dst / "pack.json").read_text(encoding="utf-8"))
    seed = info["seed"] if seed is None else seed
    judge, sealed = dst / "judge", dst / "sealed"
    if judge.exists() or sealed.exists():
        raise FileExistsError(f"{judge} or {sealed} exists — the A/B pack is "
                              "drawn once; judging may already have started")
    eng, ref = dst / "engine", dst / "reference"
    available = len(engine.ab_pairs(eng, ref))
    if available < pairs:
        raise ValueError(f"only {available} clips have an engine caption "
                         f"(caption_source=vlm); need {pairs} — run the "
                         "caption engine on OUT/engine first")
    pack, key = engine.make_ab_pack(eng, ref, pairs, seed=seed)
    judge.mkdir()
    sealed.mkdir()
    shutil.move(str(key), str(sealed / key.name))
    shutil.move(str(pack), str(judge / pack.name))
    mapping = json.loads((sealed / key.name).read_text(encoding="utf-8"))
    (judge / "frames").mkdir()
    for i, entry in sorted(mapping["pairs"].items(), key=lambda kv: int(kv[0])):
        clip = eng / entry["clip"]
        (judge / "frames" / f"pair_{int(i):02d}.png").write_bytes(
            _frame_strip(clip))
    (judge / "README.md").write_text(
        "# Blind caption ratings\n\nFor each pair in `vidgin_ab_pack.md`, "
        "look at `frames/pair_NN.png` (4 frames from the clip, left to "
        "right) and rate Caption A and Caption B 1-5. The order of A and B "
        "is random per pair. Fill in the ratings in the pack file and "
        "return it.\n", encoding="utf-8")
    return {"judge": str(judge), "sealed": str(sealed / key.name),
            "pairs": len(mapping["pairs"]), "seed": seed}


def score(out: str, rated: str) -> dict:
    """Open the sealed key and score the filled pack (engine's score_ab)."""
    dst = Path(out)
    result = engine.score_ab(dst / "sealed" / "vidgin_ab_key.json", rated)
    (dst / "ab_result.json").write_text(json.dumps(result) + "\n",
                                        encoding="utf-8")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="assemble the eval pack from a chunk")
    b.add_argument("source", help="captioned chunk folder (Stage 2 chunk3)")
    b.add_argument("out", help="new folder for the pack")
    b.add_argument("--n", type=int, default=N_CLIPS)
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--source-baseline", default=None,
                   help="the source's .gemini_baseline.json when it is not in "
                        "the chunk folder or its parent")
    a = sub.add_parser("ab", help="draw the blind A/B pack (after captioning)")
    a.add_argument("out")
    a.add_argument("--pairs", type=int, default=N_PAIRS)
    a.add_argument("--seed", type=int, default=None,
                   help="default: the build's seed")
    s = sub.add_parser("score", help="score a rated pack with the sealed key")
    s.add_argument("out")
    s.add_argument("rated")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        info = build(args.source, args.out, args.n, args.seed,
                     args.source_baseline)
        e, r = pair_keys(args.out)
        print(f"EVAL PACK {args.out}: {info['clips']} clips, "
              f"{len(e & r)} engine/reference pairs, source baseline "
              f"{info['source_baseline'] or 'absent (not checked)'} "
              f"({info['source_baseline_rows_checked']} rows checked)")
    elif args.cmd == "ab":
        res = make_ab(args.out, args.pairs, args.seed)
        print(f"A/B PACK: {res['pairs']} pairs -> {res['judge']} (give the "
              f"judge only this folder); key sealed in {res['sealed']}")
    else:
        score(args.out, args.rated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
