"""Tests for 3D depth conditioning (Dawood's idea, vidgen/depth.py).

Covered: config gate + checkpoint-config roundtrip, latent-grid alignment,
the DiT depth stem (zero-init invariance at birth, real signal after the
stem wakes, gradient flow, misuse errors), FolderClipDataset.get_depth,
trainer wiring (a depth train step + resume keeping the mode), and the
sampler's depth path. The MiDaS estimator itself is not exercised (it
downloads weights); only its input validation is.
"""

import json
import os

import numpy as np
import pytest
import torch

from vidgen.config import VidgenConfig, get_preset
from vidgen.data.clips import FolderClipDataset
from vidgen.depth import (depth_frame_indices, depth_to_latent_grid,
                          estimate_depth, zeros_latent_depth)
from vidgen.dit.model import DiT, count_params
from vidgen.sampler import sample_latents
from vidgen.text.encoder import TinyTextEncoder
from vidgen.train import DiTTrainer


def _cfg(use_depth: bool) -> VidgenConfig:
    cfg = get_preset("tiny")
    cfg.use_depth = use_depth
    return cfg


def _inputs(cfg, b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    latent = torch.randn(b, cfg.latent_channels, cfg.latent_t,
                         cfg.latent_h, cfg.latent_w, generator=g)
    t = torch.rand(b, generator=g)
    enc = TinyTextEncoder(d_t=cfg.text_dim, max_len=cfg.text_max_len)
    text, mask = enc(["busy city street", "people walking"])
    return latent, t, text, mask


# ------------------------------------------------------------- config
def test_depth_defaults_off_everywhere():
    for name in ("tiny", "small", "vidgen480p", "vidgen14b"):
        assert get_preset(name).use_depth is False


def test_config_roundtrip_keeps_use_depth():
    cfg = _cfg(True)
    back = VidgenConfig.from_dict(cfg.to_dict())
    assert back.use_depth is True
    old = cfg.to_dict()
    del old["use_depth"]                      # pre-depth-era checkpoint dict
    assert VidgenConfig.from_dict(old).use_depth is False


# ------------------------------------------------------------- grid math
def test_depth_frame_indices_match_causal_vae():
    assert depth_frame_indices(17, 5) == [0, 1, 5, 9, 13]
    assert depth_frame_indices(33, 9) == [0, 1, 5, 9, 13, 17, 21, 25, 29]
    assert depth_frame_indices(81, 21)[-1] == 77
    with pytest.raises(ValueError):
        depth_frame_indices(16, 5)


def test_depth_to_latent_grid_shape_and_range():
    d = torch.rand(2, 1, 17, 64, 64)
    g = depth_to_latent_grid(d, 5, 8, 8)
    assert g.shape == (2, 1, 5, 8, 8)
    assert 0.0 <= float(g.min()) and float(g.max()) <= 1.0
    const = depth_to_latent_grid(torch.full((1, 1, 33, 32, 48), 0.4), 9, 4, 6)
    assert const.shape == (1, 1, 9, 4, 6)
    assert torch.allclose(const, torch.full_like(const, 0.4))
    with pytest.raises(ValueError):
        depth_to_latent_grid(torch.rand(2, 3, 17, 64, 64), 5, 8, 8)


# ------------------------------------------------------------- DiT stem
def test_depth_model_has_stem_plain_model_does_not():
    plain = DiT(_cfg(False))
    deep = DiT(_cfg(True))
    assert plain.depth_stem is None
    assert deep.depth_stem is not None
    assert count_params(deep) > count_params(plain)


def test_depth_stem_zero_init_invariance():
    """At birth the stem is zero-init, so ANY map (or none) must give the
    plain model's exact output — depth can only start to matter as
    training teaches the stem."""
    cfg = _cfg(True)
    model = DiT(cfg).eval()
    latent, t, text, mask = _inputs(cfg)
    g = torch.Generator().manual_seed(3)
    rand_depth = torch.rand(2, 1, cfg.latent_t, cfg.latent_h, cfg.latent_w,
                            generator=g)
    with torch.no_grad():
        base = model(latent, t, text, mask)
        assert torch.equal(base, model(latent, t, text, mask, depth=None))
        assert torch.equal(base, model(
            latent, t, text, mask,
            depth=zeros_latent_depth(cfg, 2, latent.device)))
        assert torch.equal(base, model(latent, t, text, mask, depth=rand_depth))


def test_depth_matters_once_stem_wakes():
    cfg = _cfg(True)
    model = DiT(cfg).eval()
    with torch.no_grad():
        model.depth_stem.weight.normal_(0, 0.5)
    latent, t, text, mask = _inputs(cfg)
    zeros = zeros_latent_depth(cfg, 2, latent.device)
    ones = torch.ones_like(zeros)
    with torch.no_grad():
        a = model(latent, t, text, mask, depth=zeros)
        b = model(latent, t, text, mask, depth=ones)
        c = model(latent, t, text, mask)          # None == zeros condition
    assert not torch.allclose(a, b)
    assert torch.equal(a, c)


def test_depth_gradient_flows_to_stem():
    cfg = _cfg(True)
    model = DiT(cfg)
    latent, t, text, mask = _inputs(cfg)
    depth = torch.rand(2, 1, cfg.latent_t, cfg.latent_h, cfg.latent_w)
    loss = model(latent, t, text, mask, depth=depth).pow(2).mean()
    loss.backward()
    grad = model.depth_stem.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0


def test_depth_misuse_is_a_clear_error():
    cfg = _cfg(False)
    plain = DiT(cfg).eval()
    latent, t, text, mask = _inputs(cfg)
    with pytest.raises(ValueError):
        plain(latent, t, text, mask,
              depth=zeros_latent_depth(_cfg(True), 2, latent.device))
    deep = DiT(_cfg(True)).eval()
    with pytest.raises(ValueError):
        deep(latent, t, text, mask, depth=torch.rand(2, 1, 4, 8, 8))


# ------------------------------------------------------------- dataset
def _write_clip_dir(tmp_path, with_depth: bool):
    t, h, w = 17, 64, 64
    clip = (np.random.default_rng(0).random((t, h, w, 3)) * 255).astype(np.uint8)
    np.save(tmp_path / "clip_000001.npy", clip)
    rec = {"file": "clip_000001.npy", "caption": "city street"}
    if with_depth:
        dmap = (np.random.default_rng(1).random((t, h, w)) * 255).astype(np.uint8)
        np.save(tmp_path / "clip_000001_depth.npy", dmap)
        rec["depth"] = "clip_000001_depth.npy"
    with open(tmp_path / "manifest.jsonl", "w") as f:
        f.write(json.dumps(rec) + "\n")
    return str(tmp_path)


def test_dataset_get_depth(tmp_path):
    ds = FolderClipDataset(_write_clip_dir(tmp_path, with_depth=True))
    d = ds.get_depth(0)
    assert d is not None and d.shape == (1, 17, 64, 64)
    assert 0.0 <= float(d.min()) and float(d.max()) <= 1.0
    clip, cap = ds[0]                       # sample contract unchanged
    assert clip.shape == (3, 17, 64, 64) and cap == "city street"


def test_dataset_get_depth_absent(tmp_path):
    ds = FolderClipDataset(_write_clip_dir(tmp_path, with_depth=False))
    assert ds.get_depth(0) is None


# ------------------------------------------------------------- trainer
def test_trainer_depth_step_and_resume(tmp_path):
    ck = str(tmp_path / "ck")
    tr = DiTTrainer(preset="tiny", batch_size=2, num_clips=8,
                    ckpt_dir=ck, use_depth=True)
    assert tr.cfg.use_depth is True and tr.dit.depth_stem is not None
    rec = tr.train_step()
    assert np.isfinite(rec["loss"])
    path = os.path.join(ck, "depth_ckpt.pt")
    tr.save(path)
    tr2 = DiTTrainer.resume(path, batch_size=2, num_clips=8)
    assert tr2.cfg.use_depth is True and tr2.dit.depth_stem is not None
    assert np.isfinite(tr2.train_step()["loss"])


def test_trainer_plain_resume_stays_plain(tmp_path):
    ck = str(tmp_path / "ck")
    tr = DiTTrainer(preset="tiny", batch_size=2, num_clips=8, ckpt_dir=ck)
    tr.train_step()
    path = os.path.join(ck, "plain_ckpt.pt")
    tr.save(path)
    tr2 = DiTTrainer.resume(path, batch_size=2, num_clips=8)
    assert tr2.cfg.use_depth is False and tr2.dit.depth_stem is None


# ------------------------------------------------------------- sampler
def test_sampler_accepts_depth():
    cfg = _cfg(True)
    model = DiT(cfg).eval()
    enc = TinyTextEncoder(d_t=cfg.text_dim, max_len=cfg.text_max_len)
    text, mask = enc(["a street"])
    null_text, null_mask = enc.null_embedding()
    shape = (1, cfg.latent_channels, cfg.latent_t, cfg.latent_h, cfg.latent_w)
    depth = zeros_latent_depth(cfg, 1, torch.device("cpu"))
    z = sample_latents(model, shape, text, mask, null_text, null_mask,
                       num_steps=2, seed=1, depth=depth)
    assert z.shape == shape and torch.isfinite(z).all()


# ------------------------------------------------------------- estimator
def test_estimate_depth_validates_input_without_network():
    with pytest.raises(ValueError):
        estimate_depth(np.zeros((17, 64, 64, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        estimate_depth(np.zeros((64, 64, 3), dtype=np.uint8))
