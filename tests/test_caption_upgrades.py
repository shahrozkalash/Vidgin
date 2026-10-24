"""Upgrade tests for the caption pipeline (honesty, sampling, dry-run,
reports) added on top of Studio commit 0c234aa.

All offline: no API key, no network, no downloaded model.
"""
import json
import os
import sys
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import caption_clips
from caption_clips import (
    caption_dataset,
    normalize_frame,
    resolve_local_model_source,
    sample_frames,
    validate_caption_detailed,
)


def _clip(t=33, h=16, w=16, seed=0, gray=False, floating=False):
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 255, size=(t, h, w) if gray else (t, h, w, 3))
    if floating:
        arr = arr.astype(np.float32) / 255.0
    return arr


def _dataset(tmp_path, n=2, shape=(17, 64, 64, 3)):
    rng = np.random.RandomState(7)
    entries = []
    for i in range(1, n + 1):
        name = f"clip_{i:06d}.npy"
        np.save(str(tmp_path / name), rng.randint(0, 255, size=shape).astype(np.uint8))
        entries.append({"file": name, "caption": f"weak {i}", "depth": True})
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in entries))
    return str(tmp_path)


# ------------------------------------------------------------ frame sampling
def test_sample_frames_counts_and_order():
    clip = _clip(t=33)
    frames = sample_frames(clip, 3)
    assert 1 <= len(frames) <= 3
    assert frames[0].shape == (16, 16, 3) and frames[0].dtype == np.uint8
    five = sample_frames(clip, 5)
    assert 1 <= len(five) <= 5


def test_sample_frames_one_and_two_frame_clips():
    assert len(sample_frames(_clip(t=1), 3)) == 1
    two = sample_frames(_clip(t=2), 3)
    assert 1 <= len(two) <= 2


def test_sample_frames_static_clip_collapses():
    still = np.zeros((33, 16, 16, 3), dtype=np.uint8)
    frames = sample_frames(still, 5)
    assert len(frames) == 1  # identical frames are not re-sent


def test_sample_frames_grayscale_and_float():
    gray = sample_frames(_clip(t=9, gray=True), 3)
    assert gray[0].shape == (16, 16, 3)
    fl = sample_frames(_clip(t=9, floating=True), 3)
    assert fl[0].dtype == np.uint8


def test_normalize_frame_variants():
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    assert normalize_frame(rgba).shape == (8, 8, 3)
    g = np.zeros((8, 8), dtype=np.uint8)
    assert normalize_frame(g).shape == (8, 8, 3)
    try:
        normalize_frame(np.zeros((0,), dtype=np.uint8))
        assert False, "empty frame must raise"
    except ValueError:
        pass


# ------------------------------------------------------- layered validation
def test_validation_hard_failures():
    assert not validate_caption_detailed("Here is the caption for this scene of a busy street market today")[0]
    assert not validate_caption_detailed("As an AI language model I cannot view the images you provided today")[0]
    assert not validate_caption_detailed("**A bold claim** about a street scene with people walking around")[0]
    assert not validate_caption_detailed("A dog runs fast. A dog runs fast.")[0]
    assert not validate_caption_detailed('{"caption": "a dog runs across the yard quickly today"}')[0]


def test_validation_warnings_not_failures():
    ok, _r, warns = validate_caption_detailed("A video of something happening in a place somewhere")
    assert ok is True
    assert any("generic" in w for w in warns)


# ------------------------------------------------------------- local source
def test_resolve_local_model_source(monkeypatch):
    monkeypatch.delenv("VIDGIN_LOCAL_CAPTION_MODEL", raising=False)
    assert resolve_local_model_source() == caption_clips.LOCAL_MODEL_NAME
    monkeypatch.setenv("VIDGIN_LOCAL_CAPTION_MODEL", "/tmp/from-env")
    assert resolve_local_model_source() == "/tmp/from-env"
    assert resolve_local_model_source("/tmp/explicit") == "/tmp/explicit"


# ------------------------------------------------------------------ dry run
def test_dry_run_writes_nothing(tmp_path):
    data_dir = _dataset(tmp_path, n=2)
    manifest_before = (tmp_path / "manifest.jsonl").read_bytes()
    mock_local = MagicMock(side_effect=[
        "a surfer riding a tall ocean wave at sunrise",
        "a deer standing quietly in a misty autumn forest",
    ])
    processed, skipped, failed = caption_dataset(
        data_dir=data_dir, backend="local", pace_seconds=0.0,
        local_captioner=mock_local, dry_run=True)
    assert processed == 2 and failed == 0
    assert (tmp_path / "manifest.jsonl").read_bytes() == manifest_before
    assert not (tmp_path / "caption_failures.log").exists()


# ------------------------------------------------------- report / review
def test_report_and_review_export(tmp_path):
    data_dir = _dataset(tmp_path, n=2)
    report_path = str(tmp_path / "report.json")
    review_path = str(tmp_path / "review.jsonl")
    same = "a surfer riding a tall ocean wave at sunrise"
    mock_local = MagicMock(side_effect=[same, same])
    processed, _skipped, failed = caption_dataset(
        data_dir=data_dir, backend="local", pace_seconds=0.0,
        local_captioner=mock_local,
        report_path=report_path, review_export_path=review_path)
    assert processed == 2 and failed == 0
    report = json.loads(open(report_path).read())
    assert report["processed"] == 2
    assert report["by_backend"]["local"]["processed"] == 2
    assert any("duplicate" in w["warning"] for w in report["warnings"])
    review = [json.loads(l) for l in open(review_path) if l.strip()]
    assert any("duplicate" in " ".join(r["issues"]) for r in review)
    # Manifest entries gained no unexpected keys beyond the known schema.
    entries = [json.loads(l) for l in open(os.path.join(data_dir, "manifest.jsonl"))]
    for e in entries:
        assert set(e.keys()) <= {"file", "caption", "caption_weak",
                                 "caption_source", "depth"}
        assert e["depth"] is True  # unrelated key preserved
