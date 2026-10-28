"""Improvement pass (2026-10-28): EMA weights, latent caching, per-block
modulation, safe checkpoint loading, and side-effect-free loads."""
import os
import random

import pytest
import torch

from vidgen.checkpoint import (TRUST_ENV, inference_weights, load_checkpoint,
                               save_checkpoint)
from vidgen.config import get_preset
from vidgen.dit.model import DiT
from vidgen.train import DiTTrainer
from vidgen.train_utils import ModelEMA
from vidgen.vae.model import CausalVAE

KW = dict(preset="tiny", seed=0, batch_size=2, num_clips=8)


# ------------------------------------------------------------------- EMA
def test_ema_does_not_change_training_trajectory(tmp_path):
    # Sequential, not interleaved: trainers share the global RNG streams.
    plain = DiTTrainer(ckpt_dir=str(tmp_path), **KW)
    plain_losses = [plain.train_step()["loss"] for _ in range(3)]
    ema = DiTTrainer(ckpt_dir=str(tmp_path), ema_decay=0.9, **KW)
    assert [ema.train_step()["loss"] for _ in range(3)] == plain_losses


def test_ema_tracks_weights_and_is_used_for_inference(tmp_path):
    tr = DiTTrainer(ckpt_dir=str(tmp_path), ema_decay=0.9, **KW)
    init = {k: v.clone() for k, v in tr.dit.state_dict().items()}
    for _ in range(3):
        tr.train_step()
    assert tr.ema.updates == 3
    payload = tr.state_dict()
    dit_sd, _, used = inference_weights(payload)
    assert used
    key = "proj_in.weight"
    # The average sits strictly between the init and the current weights.
    assert not torch.equal(dit_sd[key], init[key])
    assert not torch.equal(dit_sd[key], tr.dit.state_dict()[key])
    raw_sd, _, used = inference_weights(payload, use_ema=False)
    assert not used and raw_sd is payload["dit"]
    # A DiT loads the EMA weights strictly.
    DiT(tr.cfg).load_state_dict(dit_sd)


def test_ema_resume_is_bit_for_bit(tmp_path):
    kw = dict(ckpt_dir=str(tmp_path), **KW)
    a = DiTTrainer(ema_decay=0.9, **kw)
    for _ in range(4):
        a.train_step()
    b = DiTTrainer(ema_decay=0.9, **kw)
    for _ in range(2):
        b.train_step()
    ckpt = str(tmp_path / "ema.pt")
    b.save(ckpt)
    # No ema_decay passed: it is restored from the checkpoint's train_config.
    c = DiTTrainer.resume(ckpt, **kw)
    assert c.ema is not None and c.ema.updates == 2
    for _ in range(2):
        c.train_step()
    for k, v in a.ema.state_dict()["dit"].items():
        assert torch.equal(v, c.ema.state_dict()["dit"][k]), k


def test_ema_rejects_bad_decay():
    with pytest.raises(ValueError):
        ModelEMA({"m": torch.nn.Linear(2, 2)}, 1.0)


# --------------------------------------------------------- latent caching
def test_latent_cache_skips_dataset_after_first_epoch(tmp_path):
    tr = DiTTrainer(ckpt_dir=str(tmp_path), cache_latents=True,
                    preset="tiny", seed=0, batch_size=2, num_clips=4)
    calls = []
    real = tr.dataset.__class__.__getitem__
    tr.dataset.__class__.__getitem__ = \
        lambda self, i: (calls.append(i), real(self, i))[1]
    try:
        for _ in range(2):          # epoch 1: every clip encoded once
            tr.train_step()
        assert sorted(calls) == [0, 1, 2, 3]
        for _ in range(4):          # epochs 2-3: all cache hits
            rec = tr.train_step()
        assert len(calls) == 4
    finally:
        tr.dataset.__class__.__getitem__ = real
    assert torch.isfinite(torch.tensor(rec["loss"]))


def test_cached_latents_match_batched_encoding(tmp_path):
    cached = DiTTrainer(ckpt_dir=str(tmp_path), cache_latents=True, **KW)
    plain = DiTTrainer(ckpt_dir=str(tmp_path), **KW)
    idx = [0, 3, 5]
    mu_c, cap_c = cached._encode_batch(idx)
    mu_p, cap_p = plain._encode_batch(idx)
    assert cap_c == cap_p
    torch.testing.assert_close(mu_c, mu_p, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------- per-block modulation
def test_per_block_mod_starts_identical_to_shared():
    torch.manual_seed(0)
    cfg = get_preset("tiny")
    shared = DiT(cfg).eval()
    torch.nn.init.normal_(shared.adaLN_shared.weight, std=0.02)
    cfg_pb = get_preset("tiny")
    cfg_pb.per_block_mod = True
    pb = DiT(cfg_pb).eval()
    missing = pb.load_state_dict(shared.state_dict(), strict=False).missing_keys
    assert missing == ["block_mod"]
    lat = torch.randn(1, cfg.latent_channels, cfg.latent_t, cfg.latent_h,
                      cfg.latent_w)
    t, text = torch.rand(1), torch.randn(1, 4, cfg.text_dim)
    mask = torch.ones(1, 4, dtype=torch.bool)
    with torch.no_grad():
        assert torch.equal(shared(lat, t, text, mask), pb(lat, t, text, mask))


def test_per_block_mod_trains_and_resumes(tmp_path):
    kw = dict(ckpt_dir=str(tmp_path), **KW)
    tr = DiTTrainer(per_block_mod=True, **kw)
    for _ in range(2):
        tr.train_step()
    assert tr.dit.block_mod.abs().sum() > 0   # it receives gradients
    ckpt = str(tmp_path / "pb.pt")
    tr.save(ckpt)
    res = DiTTrainer.resume(ckpt, **kw)       # config, not a flag, decides
    assert res.cfg.per_block_mod and res.dit.block_mod is not None


def test_grow_refuses_per_block_mod(tmp_path):
    from vidgen.grow import grow_payload

    tr = DiTTrainer(per_block_mod=True, ckpt_dir=str(tmp_path), **KW)
    with pytest.raises(ValueError, match="per-block-mod"):
        grow_payload(tr.state_dict(), "small")


# ----------------------------------------------------------- safe loading
class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_load_refuses_arbitrary_pickles(tmp_path, monkeypatch):
    path = str(tmp_path / "evil.pt")
    torch.save({"kind": "dit", "x": _Evil()}, path)
    monkeypatch.delenv(TRUST_ENV, raising=False)
    with pytest.raises(ValueError, match=TRUST_ENV):
        load_checkpoint(path)


class _Legacy:
    pass


def test_trust_env_allows_legacy_objects(tmp_path, monkeypatch):
    path = str(tmp_path / "legacy.pt")
    torch.save({"kind": "x", "obj": _Legacy()}, path)
    monkeypatch.delenv(TRUST_ENV, raising=False)
    with pytest.raises(ValueError):
        load_checkpoint(path)
    monkeypatch.setenv(TRUST_ENV, "1")
    assert isinstance(load_checkpoint(path)["obj"], _Legacy)


def test_trainer_checkpoints_load_safely(tmp_path):
    tr = DiTTrainer(ckpt_dir=str(tmp_path), ema_decay=0.9,
                    warmup_steps=1, lr_schedule="cosine", total_steps=4, **KW)
    tr.train_step()
    path = str(tmp_path / "c.pt")
    tr.save(path)
    payload = load_checkpoint(path)            # weights_only, no env var
    assert payload["kind"] == "dit" and "ema" in payload and "sched" in payload


# ------------------------------------------------------ RNG side effects
def test_load_checkpoint_leaves_rng_alone_unless_asked(tmp_path):
    path = str(tmp_path / "r.pt")
    torch.manual_seed(1)
    random.seed(1)
    save_checkpoint(path, {"kind": "x"})
    torch.manual_seed(2)
    random.seed(2)
    expect_t, expect_py = torch.rand(1), random.random()
    torch.manual_seed(2)
    random.seed(2)
    load_checkpoint(path)                       # no side effect
    assert torch.equal(torch.rand(1), expect_t)
    assert random.random() == expect_py
    load_checkpoint(path, restore_rng=True)     # resume semantics
    restored = torch.rand(1)
    torch.manual_seed(1)                        # the state that was saved
    assert torch.equal(restored, torch.rand(1))


def test_seed_matters_with_a_vae_checkpoint(tmp_path):
    """The VAE load used to overwrite the seeded RNG before the DiT was
    built, so every --seed produced the identical DiT init and noise."""
    vae_path = str(tmp_path / "vae.pt")
    save_checkpoint(vae_path, {"kind": "vae", "vae": CausalVAE().state_dict()})
    a = DiTTrainer(preset="tiny", seed=0, batch_size=1, num_clips=4,
                   vae_ckpt=vae_path, ckpt_dir=str(tmp_path))
    b = DiTTrainer(preset="tiny", seed=1, batch_size=1, num_clips=4,
                   vae_ckpt=vae_path, ckpt_dir=str(tmp_path))
    assert not torch.equal(a.dit.proj_in.weight, b.dit.proj_in.weight)
