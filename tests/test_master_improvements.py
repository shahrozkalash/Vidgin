"""Tests for the master improvement programme (2026-10-22, second prompt).

Covers: quality resize + configurable filters, caption audit,
perceptual near-duplicates, split verification, the from-scratch BPE
tokenizer (+ checkpoint payload dispatch), atomic checkpoint saves,
and the VAE recon-weight term.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidgen.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from vidgen.data.dataset_tools import (  # noqa: E402
    audit_captions,
    clip_signature,
    find_near_duplicates,
    make_splits,
    verify_splits,
)
from vidgen.data.filters import brightness, passes_quality  # noqa: E402
from vidgen.text.bpe import BPETokenizer  # noqa: E402
from vidgen.text.tokenizer import tokenizer_from_dict  # noqa: E402
from vidgen.vae.model import vae_loss  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ resize
def test_resize_area_differs_from_nearest_and_keeps_shape():
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import prepare_clips as pc

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    area = pc.resize_frame(frame, 32, "area")
    near = pc.resize_frame(frame, 32, "nearest")
    assert area.shape == (32, 32, 3) and area.dtype == np.uint8
    assert near.shape == (32, 32, 3)
    assert np.array_equal(near, pc.resize_nearest(frame, 32))
    same = pc.resize_frame(frame, 64, "area")
    assert np.array_equal(same, frame)
    with pytest.raises(ValueError):
        pc.resize_frame(frame, 32, "cubic-ish")


# ----------------------------------------------------------------- filters
def _norm_clip(value: float, frames: int = 5, size: int = 16) -> np.ndarray:
    clip = np.full((3, frames, size, size), value, dtype=np.float32)
    return clip


def test_passes_quality_defaults_unchanged_and_thresholds_configurable():
    rng = np.random.default_rng(1)
    clip = rng.uniform(-1, 1, (3, 5, 16, 16)).astype(np.float32)
    default = passes_quality(clip)
    explicit = passes_quality(clip, {})
    assert default["pass"] == explicit["pass"]
    assert "brightness" in default and "brightness_ok" in default
    # An all-white clip fails ONLY when a brightness ceiling is set.
    white = _norm_clip(1.0)
    assert passes_quality(white)["brightness_ok"]
    capped = passes_quality(white, {"brightness_max": 100.0})
    assert not capped["brightness_ok"] and not capped["pass"]
    assert brightness(white) == pytest.approx(255.0, abs=1.0)
    with pytest.raises(ValueError):
        passes_quality(clip, {"nonsense": 1.0})


# ------------------------------------------------------------ caption audit
def test_audit_captions_flags_mechanical_failures():
    entries = [
        {"file": "a.npy", "caption": "city and people",
         "caption_source": "search_query"},
        {"file": "b.npy", "caption": "city and people",
         "caption_source": "search_query"},
        {"file": "c.npy", "caption": ""},
        {"file": "d.npy"},
        {"file": "e.npy",
         "caption": "A woman in a red coat crosses the busy street",
         "caption_source": "gemini"},
    ]
    audit = audit_captions(entries)
    assert audit.total == 5
    assert audit.missing == 1 and audit.empty == 1
    assert audit.duplicated == 2 and audit.generic == 2
    assert audit.sources["gemini"] == 1
    assert not audit.ok


# ------------------------------------------------------- near duplicates
def _write_dataset(tmp_path, clips: dict[str, np.ndarray],
                   captions: dict[str, str] | None = None):
    d = str(tmp_path)
    lines = []
    for name, arr in clips.items():
        np.save(os.path.join(d, name), arr)
        lines.append({"file": name,
                      "caption": (captions or {}).get(name, "a caption here"),
                      "source_url": f"https://example.com/{name}"})
    with open(os.path.join(d, "manifest.jsonl"), "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return d


def test_near_duplicates_find_identical_but_not_different(tmp_path):
    rng = np.random.default_rng(2)
    base = rng.integers(0, 255, (17, 32, 32, 3), dtype=np.uint8)
    other = rng.integers(0, 255, (17, 32, 32, 3), dtype=np.uint8)
    d = _write_dataset(tmp_path, {
        "clip_000001.npy": base,
        "clip_000002.npy": base.copy(),
        "clip_000003.npy": other,
    })
    pairs = find_near_duplicates(d, threshold=0.01)
    assert len(pairs) == 1
    assert pairs[0][0] == "clip_000001.npy" and pairs[0][2] == 0.0
    sig = clip_signature(base)
    assert sig.shape == (256,) and 0.0 <= sig.min() and sig.max() <= 1.0


# --------------------------------------------------------- split verify
def test_verify_splits_ok_then_detects_leak(tmp_path):
    rng = np.random.default_rng(3)
    clips = {}
    for i in range(20):
        clips[f"clip_{i:06d}.npy"] = rng.integers(
            0, 255, (5, 8, 8, 3), dtype=np.uint8)
    d = _write_dataset(tmp_path, clips)
    make_splits(d, ratios=(0.6, 0.2, 0.2), seed=7)
    result = verify_splits(d)
    assert result["ok"] and result["leaked_groups"] == []
    assert sum(result["counts"].values()) == 20
    # Tamper: copy a train entry into the test manifest -> its group leaks.
    with open(os.path.join(d, "manifest_train.jsonl")) as f:
        first = f.readline()
    with open(os.path.join(d, "manifest_test.jsonl"), "a") as f:
        f.write(first)
    result = verify_splits(d)
    assert not result["ok"] and result["leaked_groups"]


# --------------------------------------------------------------------- BPE
CAPTIONS = [
    "a man walking down a busy city street",
    "people walk across the street in the city",
    "a woman walks her dog in the park",
    "cars drive through city streets at night",
    "a man walking his dog down the street",
    "the dog runs through the park",
] * 6


def test_bpe_deterministic_and_roundtrips():
    a = BPETokenizer.train(CAPTIONS, vocab_size=200)
    b = BPETokenizer.train(list(reversed(CAPTIONS)), vocab_size=200)
    assert a == b  # same corpus multiset -> same tokenizer, any order
    ids = a.encode("A man walking down the street!")
    assert ids[0] == 2 and ids[-1] == 3
    assert a.decode(ids) == "a man walking down the street"
    assert a.encode("") == [2, 3]
    assert max(a.encode(" ".join(CAPTIONS))) < len(a)


def test_bpe_unseen_words_split_instead_of_collapsing():
    tok = BPETokenizer.train(CAPTIONS, vocab_size=300)
    units = tok.tokenize("skateboarding")
    assert len(units) > 1  # split into subwords, not one <unk>
    assert tok.tokenize("walkking")[0] == "walk"  # typo keeps the stem
    assert tok.oov_rate(CAPTIONS) == 0.0


def test_bpe_serialization_and_dispatcher():
    tok = BPETokenizer.train(CAPTIONS, vocab_size=150)
    back = tokenizer_from_dict(tok.to_dict())
    assert isinstance(back, BPETokenizer) and back == tok
    with pytest.raises(ValueError):
        tokenizer_from_dict({"format": "something-else"})


def test_bpe_trains_with_dit_and_roundtrips_checkpoint(tmp_path):
    from vidgen.train import DiTTrainer

    tok = BPETokenizer.train(CAPTIONS, vocab_size=150)
    tr = DiTTrainer(preset="tiny", batch_size=1,
                    device="cpu", tokenizer=tok)
    rec = tr.train_step()
    assert np.isfinite(rec["loss"])
    ckpt = str(tmp_path / "dit_bpe.pt")
    tr.save(ckpt)
    payload = load_checkpoint(ckpt)
    assert payload["tokenizer"]["format"] == "vidgin-bpe-tokenizer-v1"
    tr2 = DiTTrainer.resume(ckpt, device="cpu")
    assert isinstance(tr2.tokenizer, BPETokenizer)
    assert tr2.tokenizer == tok


# ------------------------------------------------------- atomic checkpoint
def test_save_checkpoint_is_atomic_and_complete(tmp_path):
    path = str(tmp_path / "ck.pt")
    save_checkpoint(path, {"kind": "test", "step": 3})
    assert os.path.isfile(path)
    assert not os.path.exists(path + ".tmp")
    payload = load_checkpoint(path)
    assert payload["step"] == 3 and "rng" in payload


# -------------------------------------------------------------- vae weights
def test_vae_loss_recon_weight_scales_only_recon():
    recon = torch.zeros(2, 3)
    x = torch.ones(2, 3)
    kl = torch.tensor(2.0)
    total1, r, k = vae_loss(recon, x, kl, kl_weight=0.5, recon_weight=1.0)
    total2, _, _ = vae_loss(recon, x, kl, kl_weight=0.5, recon_weight=3.0)
    assert float(r) == pytest.approx(1.0)
    assert float(total1) == pytest.approx(1.0 + 1.0)
    assert float(total2) == pytest.approx(3.0 + 1.0)


# ------------------------------------------------- download integrity fix
class _FakeResponse:
    def __init__(self, body: bytes, declared: int | None = None):
        self._body = body
        self.headers = {"Content-Length": str(
            declared if declared is not None else len(body))}

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            out, self._body = self._body, b""
            return out
        out, self._body = self._body[:n], self._body[n:]
        return out

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_rejects_truncated_body(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import prepare_clips as pc

    monkeypatch.setattr(pc.time, "sleep", lambda s: None)
    # Server declares 10 bytes but the connection drops after 5.
    monkeypatch.setattr(
        pc.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeResponse(b"12345", declared=10))
    dest = str(tmp_path / "v.mp4")
    with pytest.raises(RuntimeError):
        pc.download("https://example.com/v.mp4", dest)


def test_download_accepts_complete_body(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import prepare_clips as pc

    monkeypatch.setattr(
        pc.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeResponse(b"0123456789"))
    dest = str(tmp_path / "v.mp4")
    pc.download("https://example.com/v.mp4", dest)
    with open(dest, "rb") as f:
        assert f.read() == b"0123456789"


# --------------------------------------------------------- eval workflow
def test_run_evaluation_smoke(tmp_path):
    rng = np.random.default_rng(4)
    clips = {f"clip_{i:06d}.npy": rng.integers(0, 255, (17, 16, 16, 3),
                                               dtype=np.uint8)
             for i in range(6)}
    ds_dir = str(tmp_path / "ds")
    os.makedirs(ds_dir, exist_ok=True)
    d = _write_dataset(ds_dir, clips)
    out = str(tmp_path / "report.json")
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "run_evaluation.py"),
         "--data-dir", d, "--out", out],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode in (0, 1), proc.stderr[-2000:]
    report = json.load(open(out))
    assert report["provenance"]["dataset_clips"] == 6
    assert report["steps"]["dataset"]["status"] in ("pass", "warn", "fail")
    assert report["steps"]["tokenizer"]["status"] == "skipped"
    assert report["overall_status"] in ("pass", "warn", "fail")
