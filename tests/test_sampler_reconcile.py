"""Sampler reconciliation tests.

Incident (Stage 2 VAE, found 2026-10-28): checkpoints saved on a
4,203-clip dataset carried BucketSampler(num_items=4203) through every
resume, because resume restored the sampler wholesale — on a 6,339-clip
dataset the VAE silently trained on only the first 4,203 manifest
entries for 10,000 steps. Resume must reconcile the sampler against the
CURRENT dataset (num_items + batch size), regenerating the permutation
and restarting the epoch, and say so loudly. Matching resumes must stay
bit-for-bit identical.
"""
import os

import pytest
import torch

from vidgen.data.buckets import BucketSampler

REAL_CKPT = "/home/hatch/workspace/tmp/stage2_vae_final/step_012500.pt"


def _saved_state(num_items=5, batch_size=2, seed=2, batches=2):
    s = BucketSampler(num_items, batch_size, seed=seed)
    for _ in range(batches):
        s.next_batch()
    return s.state_dict()


def _drain_epoch(sampler):
    """Indices served before the sampler rolls into the next epoch."""
    seen = []
    while True:
        epoch = sampler.epoch
        batch = sampler.next_batch()
        if sampler.epoch != epoch:
            break
        if batch:
            seen.extend(batch)
    return seen


def _expected_coverage(num_items, batch_size):
    # drop_last=True: the shuffled remainder tail (< batch_size) is held
    # back each epoch; every served index is distinct.
    return num_items - (num_items % batch_size)


def test_resume_reconciles_sampler(capsys):
    saved = _saved_state(num_items=5, batch_size=2)
    fresh = BucketSampler(12, 2, seed=2)
    fresh.load_state_dict(saved)
    assert fresh.reconcile(12, 2) is True
    out = capsys.readouterr().out
    assert "SAMPLER RECONCILED" in out
    assert "num_items=5" in out and "num_items=12" in out
    assert len(fresh.perm) == 12
    seen = _drain_epoch(fresh)
    assert len(seen) == _expected_coverage(12, 2)
    assert len(set(seen)) == len(seen)
    assert all(0 <= i < 12 for i in seen)


def test_resume_smaller_dataset(capsys):
    saved = _saved_state(num_items=12, batch_size=2)
    fresh = BucketSampler(5, 2, seed=2)
    fresh.load_state_dict(saved)
    assert fresh.reconcile(5, 2) is True
    assert "SAMPLER RECONCILED" in capsys.readouterr().out
    assert len(fresh.perm) == 5
    seen = _drain_epoch(fresh)
    assert len(seen) == _expected_coverage(5, 2)
    assert all(0 <= i < 5 for i in seen)


def test_resume_batch_size_wins(capsys):
    saved = _saved_state(num_items=12, batch_size=2)
    fresh = BucketSampler(12, 4, seed=2)
    fresh.load_state_dict(saved)
    assert fresh.reconcile(12, 4) is True
    out = capsys.readouterr().out
    assert "batch_size=2" in out and "batch_size=4" in out
    assert fresh.batch_size == 4
    assert len(fresh.next_batch()) == 4
    assert len(fresh.next_batch()) == 4


def test_fresh_run_unchanged(capsys):
    # Deterministic fresh sequences (pre-existing sampler behavior).
    a = BucketSampler(10, 2, seed=7)
    b = BucketSampler(10, 2, seed=7)
    assert [a.next_batch() for _ in range(12)] == \
           [b.next_batch() for _ in range(12)]
    # Reconcile with matching values is a strict no-op: resume stays
    # bit-for-bit reproducible, nothing printed.
    s = BucketSampler(10, 2, seed=7)
    s.next_batch()
    before = s.state_dict()
    assert s.reconcile(10, 2) is False
    after = s.state_dict()
    assert after["num_items"] == before["num_items"]
    assert after["cursor"] == before["cursor"]
    assert after["epoch"] == before["epoch"]
    assert torch.equal(after["perm"], before["perm"])
    assert "SAMPLER RECONCILED" not in capsys.readouterr().out


@pytest.mark.skipif(not os.path.exists(REAL_CKPT),
                    reason="real Stage 2 checkpoint not on this machine")
def test_old_checkpoint_loads(capsys):
    payload = torch.load(REAL_CKPT, map_location="cpu", weights_only=False)
    saved = payload["sampler"]
    assert saved["num_items"] == 4203  # the stale incident values
    fresh = BucketSampler(8182, 4, seed=2)
    fresh.load_state_dict(saved)
    assert fresh.reconcile(8182, 4) is True
    out = capsys.readouterr().out
    assert "num_items=4203" in out and "num_items=8182" in out
    assert fresh.num_items == 8182 and fresh.batch_size == 4
    assert len(fresh.perm) == 8182
    seen = _drain_epoch(fresh)
    assert len(seen) == _expected_coverage(8182, 4)
    assert len(set(seen)) == len(seen)
    assert all(0 <= i < 8182 for i in seen)
