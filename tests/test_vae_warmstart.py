"""VAE weights-only warm start (Stage 2: Small VAE from Round-1 tiny VAE).

Warm start copies only the weights — optimizer, step counter, loss
history and RNG stay fresh (it is a new run, not a resume). Donor and
target must share the same VAE shape: while every preset's VAE was
16/32 a tiny donor initialized the Small VAE directly; since the
2026-10-31 growth, Small's VAE is 32/64, so a 16/32 donor is first
transplanted with vidgen.vae.grow (the flow tested below).
"""

import math
import os

import pytest
import torch

from vidgen.train_vae import VAETrainer


@pytest.fixture(scope="module")
def donor_ckpt(tmp_path_factory):
    """One shared donor checkpoint for the whole module.

    A full VAE checkpoint is ~43 MB (weights + AdamW state); saving one per
    test would fill small /tmp filesystems when the suite runs together, so
    the donor is built once and treated as read-only by every test.
    """
    tmpdir = tmp_path_factory.mktemp("vae_donor")
    donor = VAETrainer(preset="tiny", seed=0, num_clips=8, batch_size=2,
                       ckpt_dir=None)
    for _ in range(2):
        donor.train_step()
    path = str(tmpdir / "donor.pt")
    donor.save(path)
    params = {k: v.detach().clone() for k, v in donor.vae.state_dict().items()}
    return path, params


def _params(tr):
    return {k: v.detach().clone() for k, v in tr.vae.state_dict().items()}


def test_warmstart_across_vae_shapes_is_refused(donor_ckpt):
    # The donor is 16/32; the small preset's VAE is now 32/64 (grown).
    # A direct strict load must refuse — silently pouring the donor
    # into the wrong geometry is exactly what growth exists to avoid.
    path, _ = donor_ckpt
    tr = VAETrainer(preset="small", seed=7, num_clips=4, batch_size=2,
                    ckpt_dir=None)
    with pytest.raises(RuntimeError):
        tr.load_vae_weights(path)


def test_warmstart_copies_grown_donor_weights(tmp_path, donor_ckpt):
    # The post-growth flow: transplant the 16/32 donor into the small
    # preset's 32/64 geometry, then warm-start from the grown file.
    from vidgen.pool import load_payload_file
    from vidgen.vae.grow import grow_vae_payload

    path, _ = donor_ckpt
    grown = grow_vae_payload(load_payload_file(path), "small")
    grown_path = str(tmp_path / "grown_donor.pt")
    torch.save(grown, grown_path)
    grown_params = grown["vae"]

    tr = VAETrainer(preset="small", seed=7, num_clips=4, batch_size=2,
                    ckpt_dir=None)
    before = _params(tr)
    tr.load_vae_weights(grown_path)
    after = _params(tr)
    assert set(after) == set(grown_params)
    for k in after:
        assert torch.equal(after[k], grown_params[k]), \
            f"weight {k} not copied"
    assert any(not torch.equal(after[k], before[k]) for k in after), \
        "weights unchanged by warm start"
    # fresh run, not a resume
    assert tr.step == 0
    assert tr.loss_history == []
    assert tr.init_from == grown_path
    assert all(len(st) == 0 for st in tr.opt.state.values()), \
        "optimizer must stay fresh after warm start"


def test_warmstart_does_not_inherit_donor_rng(donor_ckpt):
    path, _ = donor_ckpt
    donor_rng = torch.load(path, map_location="cpu",
                           weights_only=False)["rng"]["torch"]
    tr = VAETrainer(preset="tiny", seed=5, num_clips=8, batch_size=2,
                    ckpt_dir=None)
    tr.load_vae_weights(path)
    assert not torch.equal(torch.get_rng_state(), donor_rng), \
        "warm-started run must not continue the donor's RNG stream"


def test_warmstart_is_deterministic(donor_ckpt):
    # Same seed + same donor must follow the same trajectory. Exact
    # bit-equality is not asserted: this CPU's conv backward pass is itself
    # only deterministic to ~1e-6 (two fresh same-seed trainers already
    # differ at that level), so the bar is tight closeness, which still
    # catches any real divergence (e.g. different batches).
    path, _ = donor_ckpt

    def _mk():
        return VAETrainer(preset="tiny", seed=5, num_clips=8, batch_size=2,
                          ckpt_dir=None)

    a, b = _mk(), _mk()
    a.load_vae_weights(path)
    b.load_vae_weights(path)
    ra, rb = a.train_step(), b.train_step()
    assert math.isclose(ra["loss"], rb["loss"], rel_tol=1e-4), \
        f"same seed + same donor diverged: {ra['loss']} vs {rb['loss']}"


def test_warmstart_rejects_non_vae_checkpoint(tmp_path, donor_ckpt):
    bad = os.path.join(str(tmp_path), "bad.pt")
    torch.save({"kind": "dit"}, bad)
    tr = VAETrainer(preset="tiny", seed=0, num_clips=8, ckpt_dir=None)
    with pytest.raises(ValueError, match="not a VAE checkpoint"):
        tr.load_vae_weights(bad)


def test_warmstart_rejects_shape_mismatch(tmp_path, donor_ckpt):
    path, _ = donor_ckpt
    payload = torch.load(path, map_location="cpu", weights_only=False)
    k = next(iter(payload["vae"]))
    payload["vae"][k] = payload["vae"][k][:1]  # break one tensor's shape
    bad = os.path.join(str(tmp_path), "mismatch.pt")
    torch.save(payload, bad)
    tr = VAETrainer(preset="tiny", seed=0, num_clips=8, ckpt_dir=None)
    with pytest.raises(RuntimeError):
        tr.load_vae_weights(bad)


def test_warmstart_then_trains_and_checkpoints(tmp_path, donor_ckpt):
    path, _ = donor_ckpt
    ckpt_dir = os.path.join(str(tmp_path), "ckpts")
    tr = VAETrainer(preset="tiny", seed=3, num_clips=8, batch_size=2,
                    ckpt_dir=ckpt_dir)
    tr.load_vae_weights(path)
    rec = tr.train_step()
    assert math.isfinite(rec["loss"])
    out = os.path.join(ckpt_dir, "step_000001.pt")
    tr.save(out)
    assert os.path.exists(out)
    # the saved checkpoint is a normal resume-from-here VAE checkpoint
    resumed = VAETrainer.resume(out, ckpt_dir=None)
    assert resumed.step == 1
