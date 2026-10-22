#!/usr/bin/env python3
"""One reproducible evaluation run for Vidgin (Phase 10, 2026-10-22).

Runs the whole evidence chain in one command and writes ONE
machine-readable report:

  1. dataset health        (vidgen.data.dataset_tools.check_dataset)
  2. caption audit           (empty/duplicated/generic/search-query)
  3. tokenizer coverage      (OOV / UNK rates, when --tokenizer-vocab)
  4. split leakage check     (verify_splits, when splits exist)
  5. VAE reconstruction      (scripts/measure_latents.py, when --vae-ckpt)
  6. DiT sample evaluation   (scripts/evaluate_checkpoint.py, when
                              --dit-ckpt: fixed prompts/seeds, temporal
                              metrics, MP4s saved under --out-dir)

The report records provenance — git commit, seed, Python/torch
versions, dataset path and clip count — so two runs can be compared
honestly. ``--compare <old report>`` adds per-metric deltas. Statuses
are pass/warn/fail/skipped and are NEVER manufactured: a step that did
not run says skipped, a failed check says fail.

Usage:
    python scripts/run_evaluation.py --data-dir /path/to/dataset \
        --out report.json [--tokenizer-vocab tok.json] \
        [--vae-ckpt vae.pt] [--dit-ckpt dit.pt --preset small] \
        [--compare previous_report.json]

Exit code: 0 when no step failed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidgen.data.dataset_tools import (  # noqa: E402
    _read_manifest_strict,
    audit_captions,
    check_dataset,
    verify_splits,
)
from vidgen.text.tokenizer import load_tokenizer  # noqa: E402

STATUS_RANK = {"pass": 0, "skipped": 0, "warn": 1, "fail": 2}


def _worst(statuses: list[str]) -> str:
    return max(statuses, key=lambda s: STATUS_RANK[s]) if statuses else "skipped"


def _git_commit(repo: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


def _run_json_step(cmd: list[str]) -> tuple[int, dict | None]:
    """Run a repo script that prints/writes JSON; return (rc, parsed)."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True, help="write the JSON report here")
    ap.add_argument("--out-dir", default=None,
                    help="artifacts dir for sample evaluation "
                         "(default: <out>.artifacts)")
    ap.add_argument("--tokenizer-vocab", default=None)
    ap.add_argument("--vae-ckpt", default=None)
    ap.add_argument("--dit-ckpt", default=None)
    ap.add_argument("--preset", default="tiny",
                    help="preset giving clip geometry for VAE measurement")
    ap.add_argument("--max-clips", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-psnr", type=float, default=None,
                    help="fail the run when VAE reconstruction PSNR is lower")
    ap.add_argument("--compare", default=None,
                    help="previous run_evaluation report to diff against")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    steps: dict[str, dict] = {}
    t_start = time.perf_counter()

    # 1 — dataset health
    ds = check_dataset(args.data_dir)
    steps["dataset"] = {
        "status": "fail" if ds.errors else ("warn" if ds.warnings else "pass"),
        "clips": ds.manifest_entries if hasattr(ds, "manifest_entries") else None,
        "errors": len(ds.errors), "warnings": len(ds.warnings),
        "report": ds.to_dict(),
    }

    # 2 — caption audit
    entries: list[dict] = []
    try:
        entries = _read_manifest_strict(args.data_dir)
        audit = audit_captions(entries)
        steps["captions"] = {
            "status": "pass" if audit.ok else "fail",
            "audit": audit.to_dict(),
        }
        clip_count = len(entries)
    except Exception as exc:  # noqa: BLE001
        steps["captions"] = {"status": "fail", "error": str(exc)}
        clip_count = None

    # 3 — tokenizer coverage (+ same-corpus word-vs-BPE comparison)
    if args.tokenizer_vocab:
        tok = load_tokenizer(args.tokenizer_vocab)
        captions = [str(e.get("caption", "")) for e in entries]
        info = {"vocab_size": len(tok),
                "kind": type(tok).__name__,
                "oov_rate": round(tok.oov_rate(captions), 4)}
        if hasattr(tok, "unk_token_rate"):
            info["unk_token_rate"] = round(tok.unk_token_rate(captions), 4)
        if type(tok).__name__ == "CaptionTokenizer":
            # Same corpus, same vocab budget: what would BPE give?
            total_words = sum(len(c.split()) for c in captions)
            if 0 < total_words <= 200_000:
                from vidgen.text.bpe import BPETokenizer

                bpe = BPETokenizer.train(captions, vocab_size=len(tok))
                info["bpe_comparison"] = {
                    "vocab_size": len(bpe),
                    "oov_rate": round(bpe.oov_rate(captions), 4),
                    "unk_token_rate": round(bpe.unk_token_rate(captions), 4),
                    "note": "coverage only — BPE is not thereby proven "
                            "better for training; that needs a real run",
                }
        steps["tokenizer"] = {"status": "pass", **info}
    else:
        steps["tokenizer"] = {"status": "skipped",
                              "reason": "no --tokenizer-vocab given"}

    # 3b — training gradient check: one real tiny DiT step on CPU.
    # Proves the training path is alive (finite loss, gradients flow,
    # parameters actually move) independently of any checkpoint.
    try:
        import torch as _torch

        from vidgen.train import DiTTrainer as _DiTTrainer

        _tr = _DiTTrainer(preset="tiny", batch_size=1, device="cpu")
        _before = [p.detach().clone() for p in
                   list(_tr.dit.parameters())[:4]]
        _rec = _tr.train_step()
        _grads = [float(p.grad.abs().sum()) if p.grad is not None else 0.0
                  for p in _tr.dit.parameters()]
        _moved = any(not _torch.equal(a, b) for a, b in zip(
            _before, list(_tr.dit.parameters())[:4]))
        _ok = (bool(np.isfinite(_rec["loss"])) and sum(_grads) > 0
               and _moved)
        steps["gradient_check"] = {
            "status": "pass" if _ok else "fail",
            "loss": _rec["loss"],
            "nonzero_grad_params": int(sum(1 for g in _grads if g > 0)),
            "total_params_checked": len(_grads),
            "parameters_moved": _moved,
        }
    except Exception as exc:  # noqa: BLE001
        steps["gradient_check"] = {"status": "fail", "error": str(exc)}

    # 4 — split leakage
    if os.path.isfile(os.path.join(args.data_dir, "manifest_train.jsonl")):
        result = verify_splits(args.data_dir)
        steps["splits"] = {
            "status": "pass" if result["ok"] else "fail", **result}
    else:
        steps["splits"] = {"status": "warn",
                           "reason": "no split manifests; run helper `split`"}

    # 5 — VAE reconstruction (subprocess: the existing measured tool)
    if args.vae_ckpt:
        latent_out = (args.out_dir or (args.out + ".artifacts")) + "/latent_report.json"
        os.makedirs(os.path.dirname(latent_out), exist_ok=True)
        cmd = [sys.executable,
               os.path.join(repo, "scripts", "measure_latents.py"),
               "--vae-ckpt", args.vae_ckpt, "--preset", args.preset,
               "--data-dir", args.data_dir,
               "--max-clips", str(args.max_clips),
               "--device", args.device, "--out", latent_out]
        if args.min_psnr is not None:
            cmd += ["--min-psnr", str(args.min_psnr)]
        rc, _proc = _run_json_step(cmd)
        latent = None
        if os.path.isfile(latent_out):
            with open(latent_out, encoding="utf-8") as f:
                latent = json.load(f)
        steps["vae"] = {
            "status": "pass" if rc == 0 else "fail",
            "report": latent, "report_path": latent_out,
        }
    else:
        steps["vae"] = {"status": "skipped", "reason": "no --vae-ckpt given"}

    # 6 — DiT sample evaluation (subprocess: fixed prompts/seeds)
    if args.dit_ckpt:
        eval_dir = (args.out_dir or (args.out + ".artifacts")) + "/dit_eval"
        cmd = [sys.executable,
               os.path.join(repo, "scripts", "evaluate_checkpoint.py"),
               "--dit-ckpt", args.dit_ckpt,
               "--train-dir", args.data_dir,
               "--device", args.device, "--out-dir", eval_dir]
        if args.vae_ckpt:
            cmd += ["--vae-ckpt", args.vae_ckpt]
        rc, _proc = _run_json_step(cmd)
        eval_report = None
        rp = os.path.join(eval_dir, "report.json")
        if os.path.isfile(rp):
            with open(rp, encoding="utf-8") as f:
                eval_report = json.load(f)
        steps["dit_samples"] = {
            "status": "pass" if rc == 0 else "fail",
            "report": eval_report, "report_path": rp,
        }
    else:
        steps["dit_samples"] = {"status": "skipped",
                                "reason": "no --dit-ckpt given"}

    import torch

    report = {
        "tool": "run_evaluation.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": _git_commit(repo),
            "seed": args.seed,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "dataset": os.path.abspath(args.data_dir),
            "dataset_clips": clip_count,
            "vae_ckpt": args.vae_ckpt,
            "dit_ckpt": args.dit_ckpt,
            "tokenizer_vocab": args.tokenizer_vocab,
        },
        "steps": steps,
        "runtime_seconds": round(time.perf_counter() - t_start, 2),
        "peak_gpu_mem_mb": (round(torch.cuda.max_memory_allocated()
                                  / (1024 * 1024), 1)
                            if torch.cuda.is_available() else None),
        "overall_status": _worst([s["status"] for s in steps.values()]),
    }

    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            old = json.load(f)
        deltas = {}

        def dig(d, *keys):
            cur = d
            for k in keys:
                if not isinstance(cur, dict) or k not in cur:
                    return None
                cur = cur[k]
            return cur

        for label, keys in (
            ("vae_psnr_db", ("steps", "vae", "report", "recon_psnr_db")),
            ("vae_latent_std", ("steps", "vae", "report", "latent_std")),
            ("captions_duplicated",
             ("steps", "captions", "audit", "duplicated")),
            ("dataset_errors", ("steps", "dataset", "errors")),
        ):
            new_v, old_v = dig(report, *keys), dig(old, *keys)
            if isinstance(new_v, (int, float)) and isinstance(old_v, (int, float)):
                deltas[label] = {"from": old_v, "to": new_v,
                                 "delta": round(new_v - old_v, 4)}
        report["compare"] = {"against": args.compare, "deltas": deltas}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    for name, step in steps.items():
        print(f"  {name:12s} {step['status']}")
    print(f"Overall: {report['overall_status']}  Report: {args.out}")
    return 1 if report["overall_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
