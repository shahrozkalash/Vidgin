"""FolderClipDataset (real clips) + prepare_clips.py helpers.

Sample format must match MovingMNISTDataset exactly: a tuple
(tensor, caption) with tensor float32 [3, T, H, W] in [-1, 1] and caption a str.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from vidgen.data.clips import FolderClipDataset
from vidgen.data.moving_mnist import MovingMNISTDataset
from vidgen.train import DiTTrainer
from vidgen.train_vae import VAETrainer

from prepare_clips import (
    build_clip,
    caption_from_filename,
    center_crop_square,
    choose_video_file,
    content_hash,
    pexels_caption,
    resize_nearest,
    select_frame_indices,
)


def _write_dataset(tmp_path, n=6, captions=None):
    rng = np.random.RandomState(42)
    for i in range(n):
        clip = rng.randint(0, 256, size=(17, 64, 64, 3), dtype=np.uint8)
        np.save(tmp_path / f"clip_{i + 1:06d}.npy", clip)
    with open(tmp_path / "manifest.jsonl", "w", encoding="utf-8") as f:
        for i in range(n):
            cap = (captions or [f"caption {i}"])[i] if captions else f"caption {i}"
            f.write(json.dumps({"file": f"clip_{i + 1:06d}.npy",
                                "caption": cap}) + "\n")
    return str(tmp_path)


def test_folder_clip_dataset_format_matches_moving_mnist(tmp_path):
    data_dir = _write_dataset(tmp_path)
    real = FolderClipDataset(data_dir)
    synth = MovingMNISTDataset(num_clips=8)

    assert len(real) == 6

    t_real, cap_real = real[0]
    t_synth, cap_synth = synth[0]
    # same container type and keys
    assert isinstance(t_real, torch.Tensor) and isinstance(t_synth, torch.Tensor)
    assert isinstance(cap_real, str) and isinstance(cap_synth, str)
    # same shapes / dtype / value range
    assert t_real.shape == t_synth.shape == (3, 17, 64, 64)
    assert t_real.dtype == t_synth.dtype == torch.float32
    assert t_real.min() >= -1.0 and t_real.max() <= 1.0
    assert t_synth.min() >= -1.0 and t_synth.max() <= 1.0
    # caption passthrough from the manifest
    assert cap_real == "caption 0"
    # deterministic indexing
    a, _ = real[3]
    b, _ = real[3]
    assert torch.equal(a, b)


def test_folder_clip_dataset_max_clips_and_errors(tmp_path):
    data_dir = _write_dataset(tmp_path)
    assert len(FolderClipDataset(data_dir, max_clips=2)) == 2
    assert len(FolderClipDataset(data_dir, max_clips=100)) == 6

    import pytest

    with pytest.raises(FileNotFoundError):
        FolderClipDataset(str(tmp_path / "nope"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        FolderClipDataset(str(empty))  # no manifest.jsonl
    # manifest references a missing file -> clear error on access
    with open(empty / "manifest.jsonl", "w") as f:
        f.write(json.dumps({"file": "ghost.npy", "caption": "x"}) + "\n")
    ds = FolderClipDataset(str(empty))
    with pytest.raises(FileNotFoundError):
        ds[0]


def test_trainers_accept_data_dir(tmp_path):
    data_dir = _write_dataset(tmp_path)
    vae_tr = VAETrainer(preset="tiny", seed=0, batch_size=2, num_clips=8,
                        data_dir=data_dir)
    rec = vae_tr.train_step()
    assert np.isfinite(rec["loss"])

    dit_tr = DiTTrainer(preset="tiny", seed=0, batch_size=2, num_clips=8,
                        data_dir=data_dir)
    rec = dit_tr.train_step()
    assert np.isfinite(rec["loss"])

    # default path is untouched: synthetic data still works
    vae_synth = VAETrainer(preset="tiny", seed=0, batch_size=2, num_clips=8)
    assert np.isfinite(vae_synth.train_step()["loss"])


def test_caption_from_filename():
    assert caption_from_filename(
        "https://example.com/videos/city_traffic-01.mp4?x=1") == "city traffic 01"
    assert caption_from_filename("https://cdn/x/VID_2024.MOV") == "VID 2024"
    assert caption_from_filename("https://cdn/x/") == "video clip"


def test_content_hash_dedupe():
    rng = np.random.RandomState(0)
    a = rng.randint(0, 256, size=(17, 64, 64, 3), dtype=np.uint8)
    b = a.copy()
    c = rng.randint(0, 256, size=(17, 64, 64, 3), dtype=np.uint8)
    assert content_hash(a) == content_hash(b)
    assert content_hash(a) != content_hash(c)


def test_select_frame_indices():
    assert select_frame_indices(10, 17) is None  # too short
    idx = select_frame_indices(100, 17)
    assert len(idx) == 17
    diffs = np.diff(idx)
    assert (diffs == diffs[0]).all()  # even stride
    assert idx[0] >= 0 and idx[-1] < 100  # centered inside
    assert idx[0] > 0 and idx[-1] < 99  # actually centered, not edge-hugging


def test_crop_and_resize():
    f = np.zeros((90, 160, 3), np.uint8)
    c = center_crop_square(f)
    assert c.shape == (90, 90, 3)
    r = resize_nearest(c, 64)
    assert r.shape == (64, 64, 3)


def test_choose_video_file():
    files = [{"file_type": "video/mp4", "width": w, "height": w // 2, "link": f"u{w}"}
             for w in (1920, 1280, 640, 480, 320)]
    assert choose_video_file(files)["width"] == 480
    small = [f for f in files if f["width"] < 480]
    assert choose_video_file(small)["width"] == 320
    assert choose_video_file([]) is None


def test_pexels_caption():
    assert pexels_caption({"id": 1}, "crowd crossing street") == "crowd crossing street"
    assert pexels_caption({}, "") == "stock video clip"


def _moving_gradient(n=60, s=96):
    yy, xx = np.mgrid[0:s, 0:s]
    grad = ((xx / s) * 200).astype(np.uint8)
    out = []
    for t in range(n):
        f = np.stack([grad, grad, grad], axis=-1).copy()
        x0 = (t * 3) % (s - 20)
        f[20:40, x0:x0 + 20] = 255
        out.append(f)
    return np.stack(out)


def test_build_clip_filters():
    clip, reason = build_clip(_moving_gradient(), 17, 64)
    assert reason is None and clip is not None
    assert clip.shape == (17, 64, 64, 3) and clip.dtype == np.uint8

    dark = np.zeros((30, 64, 64, 3), np.uint8)
    assert build_clip(dark, 17, 64) == (None, "too_dark")
    short = np.zeros((10, 64, 64, 3), np.uint8)
    assert build_clip(short, 17, 64) == (None, "too_short")
    static = np.stack([_moving_gradient()[0]] * 60)
    assert build_clip(static, 17, 64)[1] == "quality_fail(motion)"
