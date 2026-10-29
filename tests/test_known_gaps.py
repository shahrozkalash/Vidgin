"""Known-gaps pass (2026-10-29): safe loading on any torch, latent-cache
RAM, exact I2V first frame, VAE EMA, pooled EMA, and the in-training
quality probe."""
import json
import os

import numpy as np
import pytest
import torch

from vidgen.checkpoint import load_checkpoint, save_checkpoint, vae_weights
from vidgen.config import get_preset
from vidgen.i2v import pin_first_frame
from vidgen.pool import merge_payloads
from vidgen.probe import DiTProbe, VAEProbe, format_probe
from vidgen.seed import get_rng_states, set_rng_states
from vidgen.train import DiTTrainer
from vidgen.train_vae import VAETrainer

KW = dict(preset="tiny", seed=0, batch_size=2, num_clips=8)


# ------------------------------------------------- safe loading, any torch
def test_new_checkpoints_need_no_allowlist(tmp_path):
    """Plain weights_only loading (what torch < 2.6 offers) opens them."""
    tr = DiTTrainer(ckpt_dir=str(tmp_path), ema_decay=0.9, **KW)
    tr.train_step()
    path = str(tmp_path / "c.pt")
    tr.save(path)
    payload = torch.load(path, weights_only=True)   # no safe_globals
    assert payload["kind"] == "dit"
    assert isinstance(payload["rng"]["numpy"][1], torch.Tensor)


def test_loader_falls_back_on_torch_without_safe_globals(tmp_path, monkeypatch):
    """torch < 2.5 has no safe_globals context manager: plain
    weights_only loading must still open a current checkpoint."""
    tr = DiTTrainer(ckpt_dir=str(tmp_path), **KW)
    path = str(tmp_path / "c.pt")
    tr.save(path)
    monkeypatch.delattr(torch.serialization, "safe_globals", raising=False)
    monkeypatch.delattr(torch.serialization, "add_safe_globals", raising=False)
    payload = load_checkpoint(path, restore_rng=True)
    assert payload["kind"] == "dit"


def test_numpy_rng_roundtrips_through_tensor_format():
    np.random.seed(7)
    state = get_rng_states()
    expect = np.random.rand(3)
    np.random.seed(99)
    set_rng_states(state)
    np.testing.assert_array_equal(np.random.rand(3), expect)


def test_legacy_ndarray_rng_still_restores():
    np.random.seed(7)
    state = get_rng_states()
    state["numpy"] = np.random.get_state()          # pre-2026-10-29 format
    expect = np.random.rand(3)
    np.random.seed(99)
    set_rng_states(state)
    np.testing.assert_array_equal(np.random.rand(3), expect)


# ----------------------------------------------------------- latent cache
def test_latent_cache_bytes_match_the_real_cost(tmp_path):
    tr = DiTTrainer(ckpt_dir=str(tmp_path), cache_latents=True, **KW)
    cfg = tr.cfg
    per = 4 * cfg.latent_channels * cfg.latent_t * cfg.latent_h * cfg.latent_w
    assert tr.latent_cache_bytes() == per * len(tr.dataset)
    small = get_preset("small")
    # The figure the docs quote: 576 KB per clip at the Small preset, fp32.
    assert 4 * small.latent_channels * small.latent_t * small.latent_h \
        * small.latent_w == 576 * 1024


def test_fp16_latent_cache_halves_ram_and_trains(tmp_path):
    tr = DiTTrainer(ckpt_dir=str(tmp_path), cache_latents=True,
                    cache_dtype="fp16", **KW)
    full = DiTTrainer(ckpt_dir=str(tmp_path), cache_latents=True, **KW)
    assert tr.latent_cache_bytes() * 2 == full.latent_cache_bytes()
    mu16, _ = tr._encode_batch([0, 1])
    mu32, _ = full._encode_batch([0, 1])
    assert mu16.dtype == torch.float32               # training stays fp32
    assert tr._latent_cache[0][0].dtype == torch.float16
    torch.testing.assert_close(mu16, mu32, atol=2e-3, rtol=2e-3)
    for _ in range(4):
        rec = tr.train_step()
    assert np.isfinite(rec["loss"])
    with pytest.raises(ValueError, match="cache dtype"):
        DiTTrainer(ckpt_dir=str(tmp_path), cache_dtype="int8", **KW)


# ------------------------------------------------------ exact first frame
def test_pin_first_frame_is_exact_and_fades():
    rng = np.random.default_rng(0)
    frames = rng.uniform(-0.5, 0.5, (6, 8, 8, 3)).astype(np.float32)
    image = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    out = pin_first_frame(frames, image, fade_frames=2)
    exact = image.astype(np.float32) / 127.5 - 1
    np.testing.assert_array_equal(out[0], exact)
    residual = exact - frames[0]
    np.testing.assert_allclose(out[1], np.clip(frames[1] + residual * 2 / 3, -1, 1),
                               atol=1e-6)
    np.testing.assert_allclose(out[2], np.clip(frames[2] + residual / 3, -1, 1),
                               atol=1e-6)
    np.testing.assert_array_equal(out[3:], frames[3:])
    only0 = pin_first_frame(frames, image, fade_frames=0)
    np.testing.assert_array_equal(only0[1:], frames[1:])
    with pytest.raises(ValueError):
        pin_first_frame(frames, image[:4], 2)


# ---------------------------------------------------------------- VAE EMA
VKW = dict(preset="tiny", seed=0, batch_size=2, num_clips=8)


def test_vae_ema_does_not_change_trajectory(tmp_path):
    plain = VAETrainer(ckpt_dir=str(tmp_path), **VKW)
    losses = [plain.train_step()["loss"] for _ in range(3)]
    ema = VAETrainer(ckpt_dir=str(tmp_path), ema_decay=0.9, **VKW)
    assert [ema.train_step()["loss"] for _ in range(3)] == losses
    assert ema.ema.updates == 3


def test_vae_ema_resume_is_bit_for_bit(tmp_path):
    a = VAETrainer(ckpt_dir=str(tmp_path), ema_decay=0.9, **VKW)
    for _ in range(4):
        a.train_step()
    b = VAETrainer(ckpt_dir=str(tmp_path), ema_decay=0.9, **VKW)
    for _ in range(2):
        b.train_step()
    path = str(tmp_path / "v.pt")
    b.save(path)
    c = VAETrainer.resume(path, ckpt_dir=str(tmp_path), seed=0,
                          batch_size=2, num_clips=8)   # decay from checkpoint
    assert c.ema is not None and c.ema.updates == 2
    for _ in range(2):
        c.train_step()
    for k, v in a.ema.state_dict()["vae"].items():
        assert torch.equal(v, c.ema.state_dict()["vae"][k]), k


def test_dit_trainer_uses_the_vaes_ema_weights(tmp_path):
    v = VAETrainer(ckpt_dir=str(tmp_path), ema_decay=0.5, **VKW)
    for _ in range(2):
        v.train_step()
    path = str(tmp_path / "vae.pt")
    v.save(path)
    payload = load_checkpoint(path)
    ema_sd = vae_weights(payload)
    assert ema_sd is payload["ema"]["vae"]
    assert vae_weights(payload, use_ema=False) is payload["vae"]
    dit = DiTTrainer(ckpt_dir=str(tmp_path), vae_ckpt=path, **KW)
    key = "encoder.conv_in.conv.weight"
    key = key if key in ema_sd else next(
        k for k in ema_sd if ema_sd[k].is_floating_point() and ema_sd[k].ndim > 1)
    assert torch.equal(dit.vae.state_dict()[key], ema_sd[key])
    assert not torch.equal(ema_sd[key], payload["vae"][key])


# ------------------------------------------------------------- pooled EMA
def _worker(tmp_path, name, shard, cls=DiTTrainer, **extra):
    tr = cls(ckpt_dir=str(tmp_path / name), data_shard=shard,
             ema_decay=0.9, **extra)
    for _ in range(2):
        tr.train_step()
    return tr.state_dict()


def test_pool_merges_ema_and_next_round_continues_it(tmp_path):
    w1 = _worker(tmp_path, "w1", (1, 2), **KW)
    w2 = _worker(tmp_path, "w2", (2, 2), **KW)
    merged = merge_payloads([w1, w2])
    assert merged["pool"]["ema"] is True
    k = "proj_in.weight"
    torch.testing.assert_close(
        merged["ema"]["dit"][k],
        (w1["ema"]["dit"][k] + w2["ema"]["dit"][k]) / 2)
    assert merged["ema"]["updates"] == 2
    path = str(tmp_path / "pooled.pt")
    torch.save(merged, path)
    nxt = DiTTrainer(ckpt_dir=str(tmp_path / "r2"), ema_decay=0.99, **KW)
    nxt.load_dit_weights(path)
    assert nxt.ema.updates == 2 and nxt.ema.decay == 0.99   # run's own decay
    assert torch.equal(nxt.ema.shadow["dit"][k], merged["ema"]["dit"][k])


def test_pool_drops_ema_unless_every_worker_has_one(tmp_path):
    w1 = _worker(tmp_path, "w1", (1, 2), **KW)
    w2 = _worker(tmp_path, "w2", (2, 2), **KW)
    del w2["ema"]
    merged = merge_payloads([w1, w2])
    assert "ema" not in merged and merged["pool"]["ema"] is False
    path = str(tmp_path / "pooled.pt")
    torch.save(merged, path)
    nxt = DiTTrainer(ckpt_dir=str(tmp_path / "r2"), ema_decay=0.9, **KW)
    nxt.load_dit_weights(path)
    assert nxt.ema.updates == 0                       # restarted from donor


def test_vae_pool_round_continues_ema(tmp_path):
    w1 = _worker(tmp_path, "w1", (1, 2), cls=VAETrainer, **VKW)
    w2 = _worker(tmp_path, "w2", (2, 2), cls=VAETrainer, **VKW)
    path = str(tmp_path / "pooled.pt")
    torch.save(merge_payloads([w1, w2]), path)
    nxt = VAETrainer(ckpt_dir=str(tmp_path / "r2"), ema_decay=0.9, **VKW)
    nxt.load_vae_weights(path)
    assert nxt.ema.updates == 2


# ------------------------------------------------------------------ probe
def test_dit_probe_never_changes_training(tmp_path):
    plain = DiTTrainer(ckpt_dir=str(tmp_path), **KW)
    losses = [plain.train_step()["loss"] for _ in range(4)]
    tr = DiTTrainer(ckpt_dir=str(tmp_path), ema_decay=0.9, **KW)
    probe = DiTProbe(tr, num_clips=2, num_samples=1, sample_steps=2,
                     out_dir=str(tmp_path / "probe"))
    got = []
    for _ in range(4):
        got.append(tr.train_step()["loss"])
        if tr.step % 2 == 0:
            rec = probe.run(tr.step)
    assert got == losses
    assert rec["probe"] == 4 and "val_loss_change" in rec
    assert "val_loss_ema" in rec and len(rec["val_loss_by_t"]) == 5
    assert rec["samples"]["weights"] == "ema"
    assert tr.dit.training and tr.text_enc.training     # mode restored
    assert os.listdir(str(tmp_path / "probe" / "step_000004"))
    assert "val_loss=" in format_probe(rec)
    json.dumps(rec)                                     # loggable


def test_dit_probe_is_deterministic(tmp_path):
    tr = DiTTrainer(ckpt_dir=str(tmp_path), **KW)
    probe = DiTProbe(tr, num_clips=2, num_samples=0)
    a = probe.run(0)
    b = probe.run(0)
    assert a["val_loss"] == b["val_loss"] and b["val_loss_change"] == 0
    assert "samples" not in a


def test_dit_probe_prefers_the_val_split(tmp_path):
    from vidgen.data.clips import FolderClipDataset  # noqa: F401
    cfg = get_preset("tiny")
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    for i in range(4):
        clip = np.random.default_rng(i).integers(
            0, 256, (cfg.frames, cfg.height, cfg.width, 3), dtype=np.uint8)
        np.save(str(data / f"c{i}.npy"), clip)
        rows.append({"file": f"c{i}.npy", "caption": f"clip {i}"})
    for name, part in (("manifest.jsonl", rows), ("manifest_train.jsonl", rows[:3]),
                       ("manifest_val.jsonl", rows[3:])):
        with open(data / name, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in part)
    tr = DiTTrainer(ckpt_dir=str(tmp_path), data_dir=str(data),
                    data_split="train", preset="tiny", seed=0, batch_size=1)
    probe = DiTProbe(tr, data_dir=str(data), num_clips=4, num_samples=0)
    assert probe.split == "val" and probe.captions == ["clip 3"]


def test_vae_probe_measures_and_never_changes_training(tmp_path):
    plain = VAETrainer(ckpt_dir=str(tmp_path), **VKW)
    losses = [plain.train_step()["loss"] for _ in range(3)]
    tr = VAETrainer(ckpt_dir=str(tmp_path), ema_decay=0.9, **VKW)
    probe = VAEProbe(tr, num_clips=2)
    got = []
    for _ in range(3):
        got.append(tr.train_step()["loss"])
        rec = probe.run(tr.step)
    assert got == losses
    assert rec["psnr"] > 0 and "recon_mse_ema" in rec and "psnr_change" in rec
    assert tr.vae.training
