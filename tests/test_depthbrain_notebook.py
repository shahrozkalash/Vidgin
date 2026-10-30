"""DepthBrain cloud training flow (notebooks/train_depthbrain.ipynb).

The notebook only runs repo commands. These tests run the same commands, in
the same order, on synthetic data on CPU: stage -> dry run -> first session
(stops early, checkpoint saved) -> resume from the newest checkpoint ->
summary; and they check that every flag the notebook passes exists.
"""
import ast
import json
import os
import re
import shutil
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))

import stage_depth_data
import train_depthbrain
from test_depthbrain import _dataset

from vidgen.depthbrain import AGREEMENT_GATE, load_depthbrain

ROOT = os.path.join(os.path.dirname(__file__), "..")
NOTEBOOK = os.path.join(ROOT, "notebooks", "train_depthbrain.ipynb")
TINY = ["--batch-size", "2", "--frames", "2", "--size", "32", "--base", "8",
        "--device", "cpu", "--eval-every", "1000", "--log-every", "1000"]


# ------------------------------------------------------------------ staging
def test_stage_copies_only_what_the_trainer_reads(tmp_path):
    src = _dataset(tmp_path / "drive" / "round1", n=5)
    (tmp_path / "drive" / "round1" / "video.mp4").write_bytes(b"x" * 100)
    (tmp_path / "drive" / "round1" / "manifest.jsonl.bak").write_text("old")
    out = stage_depth_data.stage([src], str(tmp_path / "local"), quiet=True)
    staged = out["dirs"][0]
    assert os.path.basename(staged) == "round1"   # the split key keeps working
    assert sorted(os.listdir(staged)) == sorted(
        ["manifest.jsonl"] + [f"clip_{i:06d}{s}.npy" for i in range(1, 6)
                              for s in ("", "_depth")])
    assert open(os.path.join(staged, "manifest.jsonl"), "rb").read() == \
        open(os.path.join(src, "manifest.jsonl"), "rb").read()
    assert json.load(open(tmp_path / "local" / "staged.json"))["dirs"] == [staged]
    again = stage_depth_data.stage([src], str(tmp_path / "local"), quiet=True)
    assert again["datasets"][0]["files_copied"] == 0
    assert again["datasets"][0]["files_already_staged"] == 10


def test_stage_names_same_named_folders_apart(tmp_path):
    a = _dataset(tmp_path / "round1" / "packed", n=2)
    b = _dataset(tmp_path / "round2" / "packed", n=2, seed=1)
    out = stage_depth_data.stage([a, b], str(tmp_path / "local"), quiet=True)
    assert [os.path.basename(d) for d in out["dirs"]] == \
        ["round1__packed", "round2__packed"]


# ------------------------------------------------------------------ dry run
def test_dry_run_counts_and_refuses_mismatches(tmp_path, capsys):
    data = _dataset(tmp_path / "d", n=70)
    audit = train_depthbrain.audit_data([data], 95.0)
    row = audit["datasets"][0]
    assert row["manifest_lines"] == row["clips"] == row["depth_maps"] == 70
    assert row["train"] + row["val"] == 70 and audit["val"] >= 50
    assert audit["ok"], audit["problems"]
    with pytest.raises(SystemExit) as e:
        train_depthbrain.main(["--data", data, "--val-pct", "95", "--dry-run"])
    assert e.value.code == 0
    assert "DRY RUN: OK" in capsys.readouterr().out
    assert not os.path.exists(tmp_path / "checkpoints")   # nothing trained

    # Too few held out for the agreement check (50).
    low = train_depthbrain.audit_data([data], 5.0)
    assert not low["ok"] and "held-out clips" in low["problems"][0]
    # A missing depth map and a missing clip: counts differ -> refused.
    os.remove(os.path.join(data, "clip_000003_depth.npy"))
    os.remove(os.path.join(data, "clip_000004.npy"))
    bad = train_depthbrain.audit_data([data], 95.0)
    assert not bad["ok"]
    assert "manifest lines 70, clips 69, depth maps 69" in bad["problems"][0]
    with pytest.raises(SystemExit) as e:
        train_depthbrain.main(["--data", data, "--val-pct", "95", "--dry-run"])
    assert e.value.code == 1
    assert "DRY RUN: FAILED" in capsys.readouterr().out


def test_resume_auto_finds_the_newest_checkpoint(tmp_path):
    assert train_depthbrain.resolve_resume("auto", str(tmp_path)) is None
    (tmp_path / "latest.pt").write_bytes(b"")
    assert train_depthbrain.resolve_resume("auto", str(tmp_path)) == \
        os.path.join(str(tmp_path), "latest.pt")
    assert train_depthbrain.resolve_resume("x.pt", str(tmp_path)) == "x.pt"
    assert train_depthbrain.resolve_resume(None, str(tmp_path)) is None


# -------------------------------------------------- the notebook's sequence
def test_notebook_flow_end_to_end(tmp_path, capsys):
    """stage -> dry run -> session 1 (stops at step 2, checkpoint every 2)
    -> resume auto to step 4 -> summary. The resumed run ends with the same
    weights as one uninterrupted run."""
    drive = [_dataset(tmp_path / "drive" / f"round{i}", n=40, seed=i)
             for i in (1, 2)]
    stage_depth_data.main([*drive, "--dst", str(tmp_path / "local")])
    dirs = json.load(open(tmp_path / "local" / "staged.json"))["dirs"]

    with pytest.raises(SystemExit) as e:
        train_depthbrain.main(["--data", *dirs, "--val-pct", "70", "--dry-run"])
    assert e.value.code == 0

    ck = str(tmp_path / "drive" / "ckpt")
    train = ["--data", *dirs, "--resume", "auto", "--ckpt-dir", ck,
             "--steps", "4", "--save-every", "2", "--val-pct", "70",
             "--log", os.path.join(ck, "train.jsonl"), *TINY]
    capsys.readouterr()
    train_depthbrain.main(train + ["--session-steps", "2"])
    out = capsys.readouterr().out
    assert "no latest.pt yet" in out and "session stopped at step 2" in out
    assert os.path.isfile(os.path.join(ck, "latest.pt"))
    assert not os.path.exists(os.path.join(ck, "depthbrain.pt"))

    train_depthbrain.main(train)
    out = capsys.readouterr().out
    assert "resumed at step 2 (target 4)" in out
    resumed = load_depthbrain(os.path.join(ck, "depthbrain.pt"), use_ema=False)
    assert resumed.step == 4

    straight = str(tmp_path / "straight")
    train_depthbrain.main(["--data", *dirs, "--ckpt-dir", straight,
                           "--steps", "4", "--val-pct", "70", *TINY])
    ref = load_depthbrain(os.path.join(straight, "depthbrain.pt"), use_ema=False)
    for k, v in ref.state_dict().items():
        assert torch.equal(v, resumed.state_dict()[k]), k

    capsys.readouterr()
    train_depthbrain.main(["--summary", os.path.join(ck, "depthbrain_report.json")])
    text = capsys.readouterr().out
    for want in ("steps trained:      4", "final loss:", "held-out clips",
                 "order", "mae", "flicker", "pearson (mean)",
                 "pearson per clip", "RESULT: FAIL"):
        assert want in text, want
    report = json.load(open(os.path.join(ck, "depthbrain_report.json")))
    assert report["final_loss_steps"] == 4 and report["final_loss"] > 0
    assert report["agreement"]["measured"]["val_clips"] >= 50


def test_summary_shows_pass_with_every_metric():
    report = {"step": 20000, "final_loss": 0.1234, "final_loss_steps": 100,
              "params": 9337729, "train_clips": 900, "val_clips": 60,
              "teachers": {"midas_small": 960},
              "agreement": {"passed": True, "reasons": [], "step": 20000,
                            "gate": AGREEMENT_GATE,
                            "measured": {"val_clips": 60, "val_order": 0.91,
                                         "val_mae": 0.05, "val_flicker": 0.02,
                                         "val_pearson": 0.93},
                            "pearson_per_clip": {"r1/a.npy": 0.9,
                                                 "r1/b.npy": 0.96}}}
    text = train_depthbrain.format_summary(report)
    assert "RESULT: PASS" in text and "0.930" in text and "(need >= 0.9)" in text
    assert "min 0.900  max 0.960  (2 clips)" in text
    assert "RESULT: FAIL" in train_depthbrain.format_summary({"step": 5})


# ------------------------------------------------------------ the notebook
def _cells():
    nb = json.load(open(NOTEBOOK, encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _flags(parser_main, capsys):
    with pytest.raises(SystemExit):
        parser_main(["--help"])
    return set(re.findall(r"--[a-z][a-z-]+", capsys.readouterr().out))


def test_notebook_only_runs_the_repo_commands_with_real_flags(capsys):
    cells = _cells()
    for src in cells:
        ast.parse(src)                                # valid Python
    text = "\n".join(cells)
    assert "!python" not in text and "%run" not in text
    trainer = _flags(train_depthbrain.main, capsys)
    stager = _flags(stage_depth_data.main, capsys)
    used = {"train_depthbrain.py": set(), "stage_depth_data.py": set()}
    for src in cells:
        for script in used:
            if script in src:
                used[script] |= set(re.findall(r'"(--[a-z][a-z-]+)"', src))
    assert used["train_depthbrain.py"] and used["stage_depth_data.py"]
    assert used["train_depthbrain.py"] <= trainer, used["train_depthbrain.py"] - trainer
    assert used["stage_depth_data.py"] <= stager
    for need in ("--dry-run", "--resume", "--save-every", "--session-steps",
                 "--summary", "--ckpt-dir", "--val-pct"):
        assert need in used["train_depthbrain.py"], need
    assert '"--resume", "auto"' in text
    assert re.search(r"^SAVE_EVERY = 500\b", text, re.M)
    assert "requirements.txt" in text and "git" in text
    # the notebook never downloads a model or calls an AI service
    for banned in ("intel-isl", "torch.hub", "from_pretrained", "genai",
                   "hf_hub", "wget", "curl"):
        assert banned not in text.lower(), banned


def test_notebook_steps_run_in_order():
    text = "\n".join(_cells())
    order = [text.index(s) for s in ("drive.mount", "requirements.txt",
                                     "stage_depth_data.py", '"--dry-run"',
                                     '"--session-steps"', "run(TRAIN)",
                                     '"--summary"')]
    assert order == sorted(order)
