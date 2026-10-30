#!/usr/bin/env python3
"""Train DepthBrain, Vidgin's own depth estimator (vidgen/depthbrain.py).

DepthBrain is to replace MiDaS (a borrowed pretrained model) in prep. Its
weights start random and are trained here, from datasets that already have
MiDaS depth maps (Rounds 1–4; ``prepare_clips.py --depth`` or
``label_depth.py``) — MiDaS is the teacher that makes the labels, the way
Gemini makes our captions.

A fixed slice of the clips (``--val-pct``, chosen by a hash of each clip's
source video — or folder+file when the manifest has no ``source_url`` — so
it never moves between runs and one video never lands on both sides) is
held out and DepthBrain's
agreement with the teacher is measured on it every ``--eval-every`` steps
(``mae``, ``order``, ``flicker``, ``pearson``; see
``vidgen.depthbrain.depth_metrics``).
At the end, ALL held-out clips are measured and the agreement check
(``vidgen.depthbrain.agreement_check``) is run and recorded inside
``depthbrain.pt`` and in ``<ckpt-dir>/depthbrain_report.json``. Prep and
label_depth refuse a DepthBrain that did not pass it.

Usage:
  python scripts/train_depthbrain.py --data /data/round1 /data/round2 \\
      --steps 20000 --batch-size 8 --ckpt-dir checkpoints/depthbrain \\
      --amp fp16 --log checkpoints/depthbrain/train.jsonl
  # resume (bit-identical continuation):
  python scripts/train_depthbrain.py --resume checkpoints/depthbrain/latest.pt \\
      --data /data/round1 /data/round2 --steps 40000
  # one command for a fresh start AND every later session (cloud runs):
  # --resume auto continues <ckpt-dir>/latest.pt when it exists;
  # --session-steps stops this session early (checkpoint saved)
  python scripts/train_depthbrain.py --resume auto --ckpt-dir <drive>/depthbrain \\
      --data /data/round1 --steps 20000 --save-every 500 --session-steps 3000
  # check the data before a long run (no training): counts must match
  python scripts/train_depthbrain.py --data /data/round1 --dry-run
  # one-page summary of a finished run
  python scripts/train_depthbrain.py --summary checkpoints/depthbrain/depthbrain_report.json
  # once its report says "passed": true, label new data with it:
  python scripts/prepare_clips.py ... --depth --depth-source depthbrain \\
      --depthbrain-ckpt checkpoints/depthbrain/depthbrain.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from vidgen.checkpoint import load_checkpoint, save_checkpoint
from vidgen.config import resolve_device
from vidgen.data.clips import _load_manifest
from vidgen.depthbrain import (AGREEMENT_GATE, CHECKPOINT_KIND, DepthBrain,
                               DepthBrainConfig, agreement_check, count_params,
                               depth_loss, depth_metrics)
from vidgen.logging import StructuredLogger
from vidgen.seed import seed_everything
from vidgen.train_utils import (ModelEMA, NonFiniteGuard, amp_context,
                                build_lr_scheduler, guarded_optimizer_step,
                                make_grad_scaler, validate_train_options)


def split_key(data_dir: str, rec: dict) -> str:
    """What decides a clip's side of the split: its source video when the
    manifest records one (the same video in two rounds must not be in
    training AND in the held-out check), else dataset folder name + file.
    Renaming a folder between resumes changes the fallback key — keep
    folder names fixed for a run."""
    url = rec.get("source_url")
    if url:
        return f"url:{url}"
    return f"{os.path.basename(os.path.normpath(data_dir))}/{rec['file']}"


def is_val(key: str, val_pct: float) -> bool:
    """Held-out membership by a hash of :func:`split_key`: stable across
    runs, machines and manifest order."""
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 10000) < val_pct * 100


def collect_items(data_dirs: list[str], val_pct: float):
    """(train, val, teachers): clips with a depth map, split by hash.

    ``teachers`` counts which estimator made the maps (``depth_source`` in
    the manifest; maps from before it was recorded were MiDaS's)."""
    train, val, teachers = [], [], {}
    for d in data_dirs:
        for rec in _load_manifest(d):
            name = rec.get("depth")
            if not name or not os.path.isfile(os.path.join(d, name)):
                continue
            teacher = rec.get("depth_source") or "midas_small"
            teachers[teacher] = teachers.get(teacher, 0) + 1
            item = (d, rec["file"], name)
            (val if is_val(split_key(d, rec), val_pct) else train).append(item)
    return train, val, teachers


def audit_data(data_dirs: list[str], val_pct: float,
               min_val_clips: int | None = None) -> dict:
    """What the trainer would read, without training (``--dry-run``).

    Per dataset folder: manifest lines, clips present on disk, depth maps
    present on disk, and the train / held-out split. ``ok`` is False when
    clips, depth maps and manifest lines differ (a half-staged copy, a
    missing map) or when fewer clips are held out than the agreement
    check needs (``min_val_clips``, default the gate's)."""
    need = AGREEMENT_GATE["min_val_clips"] if min_val_clips is None \
        else min_val_clips
    rows, problems = [], []
    train, val, teachers = collect_items(data_dirs, val_pct)
    for d in data_dirs:
        recs = _load_manifest(d)
        clips = sum(os.path.isfile(os.path.join(d, r["file"])) for r in recs)
        maps = sum(bool(r.get("depth")) and
                   os.path.isfile(os.path.join(d, r["depth"])) for r in recs)
        row = {"data": d, "manifest_lines": len(recs), "clips": clips,
               "depth_maps": maps,
               "train": sum(i[0] == d for i in train),
               "val": sum(i[0] == d for i in val)}
        if not len(recs) == clips == maps:
            problems.append(f"{d}: manifest lines {len(recs)}, clips {clips}, "
                            f"depth maps {maps} — must all be equal")
        rows.append(row)
    if len(val) < need:
        problems.append(f"only {len(val)} held-out clips at --val-pct "
                        f"{val_pct} (the agreement check needs >= {need}); "
                        "raise --val-pct")
    return {"datasets": rows, "train": len(train), "val": len(val),
            "teachers": teachers, "val_pct": val_pct,
            "ok": not problems, "problems": problems}


def resolve_resume(resume: str | None, ckpt_dir: str | None) -> str | None:
    """``--resume auto``: ``<ckpt-dir>/latest.pt`` when it exists (the
    newest checkpoint — every save overwrites it atomically), else None =
    start fresh. Any other value is a checkpoint path."""
    if resume != "auto":
        return resume
    path = os.path.join(ckpt_dir or "checkpoints/depthbrain", "latest.pt")
    return path if os.path.isfile(path) else None


def format_summary(report: dict) -> str:
    """The one-page result of a run, from ``depthbrain_report.json``."""
    ag = report.get("agreement") or {}
    m, gate = ag.get("measured") or {}, ag.get("gate") or {}

    def num(v, fmt):
        return "n/a" if v is None else format(v, fmt)

    per = [r for r in (ag.get("pearson_per_clip") or {}).values()
           if r is not None]
    lines = [
        "DepthBrain — training summary",
        "=" * 40,
        f"steps trained:      {report.get('step')}",
        f"final loss:         {num(report.get('final_loss'), '.4f')} "
        f"(mean of the last {report.get('final_loss_steps', 0)} steps)",
        f"params:             {report.get('params')}",
        f"clips:              {report.get('train_clips')} train / "
        f"{report.get('val_clips')} held out",
        f"teachers:           {report.get('teachers')}",
        "",
        "agreement with MiDaS on held-out clips (gate):",
        f"  held-out clips    {num(m.get('val_clips'), 'd')}   "
        f"(need >= {gate.get('min_val_clips')})",
        f"  order             {num(m.get('val_order'), '.3f')}   "
        f"(need >= {gate.get('min_order')})",
        f"  mae               {num(m.get('val_mae'), '.4f')}  "
        f"(need <= {gate.get('max_mae')})",
        f"  flicker           {num(m.get('val_flicker'), '.4f')}  "
        f"(need <= {gate.get('max_flicker')})",
        f"  pearson (mean)    {num(m.get('val_pearson'), '.3f')}   "
        f"(need >= {gate.get('min_pearson')})",
    ]
    if per:
        lines.append(f"  pearson per clip  min {min(per):.3f}  "
                     f"max {max(per):.3f}  ({len(per)} clips)")
    lines += ["", "RESULT: " + ("PASS" if ag.get("passed") else "FAIL")]
    lines += [f"  - {r}" for r in ag.get("reasons") or []]
    if not ag:
        lines.append("  - no agreement check recorded (run not finished)")
    return "\n".join(lines)


def load_pair(item, size: int, frames: int, start: int, stride: int,
              flip: bool):
    """One clip window: rgb [k, 3, S, S] in [-1, 1], depth [k, S, S] in [0, 1]."""
    d, file, dname = item
    clip = np.load(os.path.join(d, file))
    dep = np.load(os.path.join(d, dname))
    if clip.dtype != np.uint8 or clip.ndim != 4 or dep.shape != clip.shape[:3]:
        raise ValueError(f"{os.path.join(d, file)}: clip {clip.shape} and "
                         f"depth {dep.shape} do not match")
    idx = [min(start + i * stride, clip.shape[0] - 1) for i in range(frames)]
    x = torch.from_numpy(clip[idx]).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    y = torch.from_numpy(dep[idx].astype(np.float32) / 255.0).unsqueeze(1)
    if x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode="area")
        y = F.interpolate(y, size=(size, size), mode="area")
    if flip:
        x, y = x.flip(-1), y.flip(-1)
    return x, y[:, 0]


class DepthBrainTrainer:
    def __init__(self, data: list[str], ckpt_dir: str = "checkpoints/depthbrain",
                 batch_size: int = 8, frames: int = 4, size: int = 64,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 total_steps: int | None = None, warmup_steps: int = 200,
                 lr_schedule: str = "cosine", ema_decay: float | None = 0.999,
                 amp: str = "none", val_pct: float = 5.0, seed: int = 0,
                 device: str = "auto", base: int = 64, levels: int = 4,
                 grad_weight: float = 0.5, log_path: str | None = None):
        validate_train_options(1, amp)
        if frames < 1:
            raise ValueError(f"frames must be >= 1, got {frames}")
        self.data = list(data)
        self.ckpt_dir = ckpt_dir
        self.batch_size, self.frames, self.size = batch_size, frames, size
        self.amp, self.val_pct, self.seed = amp, val_pct, seed
        self.grad_weight = grad_weight
        self.device = torch.device(resolve_device(device))
        seed_everything(seed)
        self.train_items, self.val_items, self.teachers = \
            collect_items(self.data, val_pct)
        if not self.train_items:
            raise ValueError(
                f"no clips with depth maps in {self.data} — DepthBrain "
                "learns from existing maps; label a dataset first with "
                "scripts/label_depth.py (MiDaS is its default) or "
                "prepare_clips.py --depth")
        own = sum(n for t, n in self.teachers.items()
                  if t.startswith("depthbrain"))
        if own:
            print(f"WARNING: {own} of the maps were made by DepthBrain "
                  "itself — learning from its own output teaches it "
                  "nothing new; train on teacher (MiDaS) maps",
                  file=sys.stderr)
        self.cfg = DepthBrainConfig(base=base, levels=levels, train_size=size)
        self.model = DepthBrain(self.cfg).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
        self.sched = build_lr_scheduler(
            self.opt, warmup_steps if total_steps is None
            else min(warmup_steps, max(total_steps - 1, 0)),
            total_steps, lr_schedule if total_steps else "none")
        self.scaler = make_grad_scaler(self.device, amp)
        self.guard = NonFiniteGuard()
        self.ema = (ModelEMA({"depthbrain": self.model}, ema_decay)
                    if ema_decay is not None else None)
        self.train_config = dict(
            batch_size=batch_size, frames=frames, size=size, lr=lr,
            weight_decay=weight_decay, total_steps=total_steps,
            warmup_steps=warmup_steps, lr_schedule=lr_schedule,
            ema_decay=ema_decay, amp=amp, val_pct=val_pct, seed=seed,
            grad_weight=grad_weight, data=self.data)
        self.step = 0
        self.epoch, self.pos = 0, 0
        self.perm = self._perm()
        self.last_eval: dict | None = None
        self.agreement: dict | None = None
        self.recent_losses: list[float] = []   # for the report's final loss
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)),
                        exist_ok=True)
        self.logger = StructuredLogger(log_path) if log_path else None

    # ------------------------------------------------------------ data
    def _perm(self) -> list[int]:
        g = random.Random(self.seed * 1_000_003 + self.epoch)
        p = list(range(len(self.train_items)))
        g.shuffle(p)
        return p

    def _next_batch(self):
        xs, ys = [], []
        for _ in range(self.batch_size):
            if self.pos >= len(self.perm):
                self.epoch, self.pos = self.epoch + 1, 0
                self.perm = self._perm()
            item = self.train_items[self.perm[self.pos]]
            self.pos += 1
            t = np.load(os.path.join(item[0], item[2]), mmap_mode="r").shape[0]
            stride = random.choice([s for s in (1, 2, 4)
                                    if (self.frames - 1) * s < t] or [1])
            start = random.randrange(max(1, t - (self.frames - 1) * stride))
            x, y = load_pair(item, self.size, self.frames, start, stride,
                             flip=random.random() < 0.5)
            xs.append(x)
            ys.append(y)
        return torch.stack(xs).to(self.device), torch.stack(ys).to(self.device)

    # ------------------------------------------------------------ train
    def train_step(self) -> dict:
        self.model.train()
        x, y = self._next_batch()                     # [B,k,3,S,S], [B,k,S,S]
        b, k = y.shape[:2]
        with amp_context(self.device, self.amp):
            pred = self.model(x.flatten(0, 1))
        parts = depth_loss(pred.float().reshape(b, k, *pred.shape[-2:]), y,
                           grad_weight=self.grad_weight)
        loss = parts["loss"]
        loss_value = float(loss.detach())
        self.opt.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
        else:
            loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0))
        outcome = guarded_optimizer_step(self.opt, self.scaler, self.guard,
                                         loss_value, gn)
        if outcome == "stepped":
            if self.sched is not None:
                self.sched.step()
            if self.ema is not None:
                self.ema.update({"depthbrain": self.model})
        self.step += 1
        self.agreement = None   # a check is only valid for the weights it measured
        self.recent_losses = (self.recent_losses + [loss_value])[-100:]
        rec = {"step": self.step, "loss": loss_value, "ssi": parts["ssi"],
               "grad": parts["grad"], "grad_norm": gn, "outcome": outcome,
               "lr": self.opt.param_groups[0]["lr"]}
        return rec

    @torch.no_grad()
    def evaluate(self, max_clips: int | None = 64,
                 per_clip: bool = False) -> dict:
        """Agreement with the teacher on the held-out clips (EMA weights
        when kept; ``max_clips=None`` = all of them). Fixed frames, no
        flip, RNG untouched. ``per_clip`` adds each clip's Pearson
        (``val_pearson_per_clip``, keyed by dataset folder/file)."""
        items = self.val_items[:max_clips]
        if not items:
            return {"val_clips": 0}
        model = self.model
        if self.ema is not None:
            model = DepthBrain(self.cfg).to(self.device)
            model.load_state_dict(self.ema.shadow["depthbrain"])
        was_training = self.model.training
        model.eval()
        preds, tgts = [], []
        for item in items:
            t = np.load(os.path.join(item[0], item[2]), mmap_mode="r").shape[0]
            stride = max(1, (t - 1) // max(self.frames - 1, 1))
            x, y = load_pair(item, self.size, self.frames, 0, stride, False)
            preds.append(model(x.to(self.device)).float().cpu())
            tgts.append(y)
        self.model.train(was_training)
        m = depth_metrics(torch.stack(preds), torch.stack(tgts))
        out = {"eval": self.step, "val_mae": m["mae"], "val_order": m["order"],
               "val_flicker": m["flicker"], "val_pearson": m["pearson"],
               "val_clips": m["clips"],
               "weights": "ema" if self.ema is not None else "raw"}
        if per_clip:
            out["val_pearson_per_clip"] = {
                f"{os.path.basename(os.path.normpath(it[0]))}/{it[1]}": r
                for it, r in zip(items, m["pearson_per_clip"])}
        self.last_eval = out
        return out

    def final_check(self, gate: dict | None = None) -> dict:
        """Measure ALL held-out clips and run the agreement check; the
        result is stored in the checkpoint (``agreement``) that prep reads."""
        ev = self.evaluate(max_clips=None, per_clip=True)
        self.agreement = agreement_check(ev, gate, step=self.step)
        self.agreement["pearson_per_clip"] = ev.get("val_pearson_per_clip")
        return self.agreement

    # ------------------------------------------------------------ state
    def state_dict(self) -> dict:
        return {
            "kind": CHECKPOINT_KIND, "config": self.cfg.to_dict(),
            "model": self.model.state_dict(), "opt": self.opt.state_dict(),
            "sched": self.sched.state_dict() if self.sched else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "ema": self.ema.state_dict() if self.ema else None,
            "step": self.step, "epoch": self.epoch, "pos": self.pos,
            "guard_bad_steps": self.guard.bad_steps,
            "train_config": self.train_config, "teachers": self.teachers,
            "last_eval": self.last_eval, "agreement": self.agreement,
            "recent_losses": self.recent_losses,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        save_checkpoint(path, self.state_dict())

    @classmethod
    def resume(cls, path: str, data: list[str] | None = None, **overrides):
        payload = load_checkpoint(path)
        if payload.get("kind") != CHECKPOINT_KIND:
            raise ValueError(f"{path}: not a DepthBrain checkpoint")
        tc = dict(payload["train_config"])
        vp = overrides.get("val_pct")
        if vp is not None and float(vp) != float(tc["val_pct"]):
            raise ValueError(
                f"--val-pct {vp} differs from the run's {tc['val_pct']}: that "
                "would move clips it already trained on into the held-out "
                "check. Keep the original value (or start a fresh run).")
        cfg = payload["config"]
        tc.update(base=cfg["base"], levels=cfg["levels"])
        tc["ckpt_dir"] = os.path.dirname(os.path.abspath(path))
        tc.update({k: v for k, v in overrides.items() if v is not None})
        tc["data"] = data or tc["data"]
        tr = cls(**tc)
        tr.model.load_state_dict(payload["model"])
        tr.opt.load_state_dict(payload["opt"])
        if tr.sched is not None and payload.get("sched"):
            tr.sched.load_state_dict(payload["sched"])
        if tr.scaler is not None and payload.get("scaler"):
            tr.scaler.load_state_dict(payload["scaler"])
        if tr.ema is not None and payload.get("ema"):
            tr.ema.load_state_dict(payload["ema"])
        tr.step, tr.epoch, tr.pos = (int(payload["step"]),
                                     int(payload["epoch"]), int(payload["pos"]))
        tr.guard.bad_steps = int(payload.get("guard_bad_steps", 0))
        tr.perm = tr._perm()
        tr.last_eval = payload.get("last_eval")
        tr.agreement = payload.get("agreement")
        tr.recent_losses = list(payload.get("recent_losses") or [])
        load_checkpoint(path, restore_rng=True)   # continue the same streams
        return tr

    def write_report(self) -> str:
        """``depthbrain_report.json``: what it learned from and how well it
        agrees with that teacher on clips it never trained on."""
        report = {
            "tool": "train_depthbrain.py", "step": self.step,
            "params": count_params(self.model), "config": self.cfg.to_dict(),
            "train_clips": len(self.train_items),
            "val_clips": len(self.val_items), "teachers": self.teachers,
            "val": self.last_eval, "agreement": self.agreement,
            "final_loss": (float(np.mean(self.recent_losses))
                           if self.recent_losses else None),
            "final_loss_steps": len(self.recent_losses),
            "note": ("These numbers measure agreement with the teacher's "
                     "(MiDaS) maps on held-out clips, not true depth. order "
                     "is the share of near/far pixel pairs put in the "
                     "teacher's order (0.5 = chance, 1.0 = identical). Prep "
                     "uses this DepthBrain only if agreement.passed."),
        }
        path = os.path.join(self.ckpt_dir, "depthbrain_report.json")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", nargs="+", default=None,
                    help="prepared dataset folder(s) with depth maps")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to continue, or 'auto' = <ckpt-dir>/"
                         "latest.pt if it exists, else start fresh")
    ap.add_argument("--steps", type=int, default=20000,
                    help="train until this step (total, across resumes)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--frames", type=int, default=None,
                    help="frames per clip per step (shared scale over time)")
    ap.add_argument("--size", type=int, default=None,
                    help="training size in pixels (inference resizes to it)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lr-schedule", choices=["none", "cosine", "linear"],
                    default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--ema-decay", type=float, default=None,
                    help="EMA decay (default 0.999; prep uses the EMA)")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--amp", choices=["none", "bf16", "fp16"], default=None)
    ap.add_argument("--val-pct", type=float, default=None,
                    help="percent of clips held out (default 5)")
    ap.add_argument("--base", type=int, default=None,
                    help="channels at full resolution (default 64, ~9.3M "
                         "params; 48 ~5.3M)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--log", default=None, help="JSONL log path")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--session-steps", type=int, default=None,
                    help="stop THIS session after this many steps (latest.pt "
                         "saved); a later --resume continues to --steps")
    ap.add_argument("--dry-run", action="store_true",
                    help="check the data (clips == depth maps == manifest "
                         "lines, enough held-out clips) and exit; no training")
    ap.add_argument("--summary", metavar="REPORT",
                    help="print the one-page summary of a depthbrain_report"
                         ".json and exit")
    args = ap.parse_args(argv)

    if args.summary:
        with open(args.summary, encoding="utf-8") as fh:
            print(format_summary(json.load(fh)))
        return
    if args.dry_run:
        if not args.data:
            ap.error("--dry-run needs --data")
        audit = audit_data(args.data, 5.0 if args.val_pct is None
                           else args.val_pct)
        for row in audit["datasets"]:
            print(f"{row['data']}: manifest lines {row['manifest_lines']}, "
                  f"clips {row['clips']}, depth maps {row['depth_maps']} -> "
                  f"{row['train']} train / {row['val']} held out")
        print(f"total: {audit['train']} train / {audit['val']} held out "
              f"(--val-pct {audit['val_pct']}), teachers {audit['teachers']}")
        for p in audit["problems"]:
            print(f"PROBLEM: {p}")
        print("DRY RUN: " + ("OK — ready to train" if audit["ok"]
                             else "FAILED — fix the data before training"))
        sys.exit(0 if audit["ok"] else 1)
    if args.session_steps is not None and args.session_steps < 1:
        ap.error("--session-steps must be >= 1")
    if args.resume == "auto":
        args.resume = resolve_resume("auto", args.ckpt_dir)
        print("resume auto: " + (f"continuing {args.resume}" if args.resume
                                 else "no latest.pt yet — starting fresh"))

    opts = dict(batch_size=args.batch_size, frames=args.frames, size=args.size,
                lr=args.lr, lr_schedule=args.lr_schedule,
                warmup_steps=args.warmup_steps, amp=args.amp,
                val_pct=args.val_pct, seed=args.seed, device=args.device,
                ckpt_dir=args.ckpt_dir, base=args.base, log_path=args.log)
    if args.no_ema:
        opts["ema_decay"] = None
    elif args.ema_decay is not None:
        opts["ema_decay"] = args.ema_decay
    if args.resume:
        tr = DepthBrainTrainer.resume(args.resume, data=args.data, **opts)
        horizon = tr.train_config.get("total_steps")
        if horizon and args.steps > horizon and tr.sched is not None:
            print(f"WARNING: the LR schedule was planned for {horizon} steps; "
                  f"steps past it run at the schedule's floor LR",
                  file=sys.stderr)
    else:
        if not args.data:
            ap.error("--data is required (or --resume)")
        opts["total_steps"] = args.steps
        tr = DepthBrainTrainer(data=args.data,
                               **{k: v for k, v in opts.items()
                                  if v is not None or k == "ema_decay"})
    print(f"DepthBrain: {count_params(tr.model) / 1e6:.2f}M params, "
          f"{len(tr.train_items)} train / {len(tr.val_items)} val clips, "
          f"teachers {tr.teachers}, device {tr.device}")
    if args.resume:
        print(f"resumed at step {tr.step} (target {args.steps})")
    if len(tr.val_items) < AGREEMENT_GATE["min_val_clips"]:
        print(f"WARNING: only {len(tr.val_items)} held-out clips; the "
              f"agreement check needs >= {AGREEMENT_GATE['min_val_clips']} "
              "and will fail — raise --val-pct (fresh runs only)",
              file=sys.stderr)
    log = tr.logger.log if tr.logger else (lambda r, echo=None: None)
    session_end = (None if args.session_steps is None
                   else tr.step + args.session_steps)
    while tr.step < args.steps:
        if session_end is not None and tr.step >= session_end:
            tr.save(os.path.join(tr.ckpt_dir, "latest.pt"))
            print(f"session stopped at step {tr.step} of {args.steps}; "
                  f"checkpoint {os.path.join(tr.ckpt_dir, 'latest.pt')} — "
                  "run again with --resume to continue")
            if tr.logger:
                tr.logger.close()
            return
        rec = tr.train_step()
        if tr.step % args.log_every == 0 or tr.step == args.steps:
            print(f"step {tr.step}: loss {rec['loss']:.4f} "
                  f"(ssi {rec['ssi']:.4f}, grad {rec['grad']:.4f})")
            log(rec, echo=False)
        if tr.step % args.eval_every == 0 or tr.step == args.steps:
            ev = tr.evaluate()
            if ev.get("val_clips"):
                print(f"  val: mae {ev['val_mae']:.4f}  order "
                      f"{ev['val_order']:.3f}  flicker {ev['val_flicker']:.4f}"
                      f"  pearson {ev['val_pearson']:.3f}")
            log(ev, echo=False)
        if tr.step % args.save_every == 0 or tr.step == args.steps:
            tr.save(os.path.join(tr.ckpt_dir, "latest.pt"))
    check = tr.final_check()
    final = os.path.join(tr.ckpt_dir, "depthbrain.pt")
    tr.save(final)
    print(f"DepthBrain checkpoint: {final}")
    if check["passed"]:
        print("agreement check vs MiDaS: PASSED — prep may use it "
              "(--depth-source depthbrain)")
    else:
        print("agreement check vs MiDaS: NOT passed — prep will refuse it: "
              + "; ".join(check["reasons"]))
    print(f"report:                {tr.write_report()}")
    if tr.logger:
        tr.logger.close()


if __name__ == "__main__":
    main()
