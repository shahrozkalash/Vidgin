"""Weight-averaging pool (vidgen/pool.py): shard math, merge math,
and the trainers' pool-round plumbing (--data-shard, --init-weights).

The merge-math tests use hand-checkable tensors so the averaging and the
DiLoCo outer step are verified exactly, not approximately. The trainer
tests run the real tiny-preset trainers for a couple of steps on CPU.
"""

import json
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pool_status  # noqa: E402
from vidgen.data.clips import FolderClipDataset  # noqa: E402
from vidgen.data.moving_mnist import MovingMNISTDataset  # noqa: E402
from vidgen.pool import (  # noqa: E402
    load_payload_file,
    merge_payloads,
    parse_data_shard,
    shard_positions,
)
from vidgen.train import DiTTrainer  # noqa: E402
from vidgen.train_vae import VAETrainer  # noqa: E402


# ------------------------------------------------------------------ shards
def test_parse_data_shard():
    assert parse_data_shard("1/2") == (1, 2)
    assert parse_data_shard(" 4/4 ") == (4, 4)
    assert parse_data_shard((2, 3)) == (2, 3)
    for bad in ["0/2", "3/2", "1/0", "a/b", "1-2", "1/2/3", "", "2"]:
        with pytest.raises(ValueError):
            parse_data_shard(bad)


def test_shard_positions_disjoint_and_complete():
    for n in (1, 2, 3, 4, 7):
        shards = [shard_positions(50, (k, n)) for k in range(1, n + 1)]
        flat = [i for s in shards for i in s]
        assert sorted(flat) == list(range(50))  # union = everything
        assert len(flat) == len(set(flat))      # and pairwise disjoint


@pytest.fixture()
def clip_dir(tmp_path):
    """3 clips whose manifest order is deliberately NOT file order."""
    d = str(tmp_path / "clips")
    os.makedirs(d)
    for name in ("clip_000001.npy", "clip_000002.npy", "clip_000003.npy"):
        np.save(os.path.join(d, name),
                np.zeros((17, 64, 64, 3), dtype=np.uint8))
    with open(os.path.join(d, "manifest.jsonl"), "w") as f:
        for name in ("clip_000003.npy", "clip_000001.npy", "clip_000002.npy"):
            f.write(json.dumps({"file": name, "caption": name}) + "\n")
    return d


def test_folder_shard_sorts_by_file_and_partitions(clip_dir):
    kw = dict(frames=17, height=64, width=64)
    full = FolderClipDataset(clip_dir, **kw)
    # No flag: manifest order preserved exactly (today's behavior).
    assert [e["file"] for e in full.entries] == [
        "clip_000003.npy", "clip_000001.npy", "clip_000002.npy"]
    a = FolderClipDataset(clip_dir, shard=(1, 2), **kw)
    b = FolderClipDataset(clip_dir, shard=(2, 2), **kw)
    # Sorted by file: positions 0,1,2 -> shards take % 2 == 0 / == 1.
    assert [e["file"] for e in a.entries] == [
        "clip_000001.npy", "clip_000003.npy"]
    assert [e["file"] for e in b.entries] == ["clip_000002.npy"]
    union = sorted(e["file"] for e in a.entries + b.entries)
    assert union == ["clip_000001.npy", "clip_000002.npy",
                     "clip_000003.npy"]


def test_mnist_shard_maps_to_global_indices():
    kw = dict(num_clips=10, height=64, width=64, frames=17, base_seed=3)
    full = MovingMNISTDataset(**kw)
    a = MovingMNISTDataset(shard=(1, 2), **kw)
    b = MovingMNISTDataset(shard=(2, 2), **kw)
    assert len(a) == len(b) == 5
    for j in range(5):
        ta, ca = a[j]
        tf, cf = full[2 * j]
        assert torch.equal(ta, tf) and ca == cf
        tb, cb = b[j]
        tf2, cf2 = full[2 * j + 1]
        assert torch.equal(tb, tf2) and cb == cf2


# ------------------------------------------------------------------ merging
def _dit_payload(step, w, e, cnt=7, config=None):
    return {
        "kind": "dit", "preset": "tiny", "step": step,
        "dit": {"w": torch.tensor(w, dtype=torch.float32),
                "cnt": torch.tensor([cnt], dtype=torch.int64)},
        "text_enc": {"e": torch.tensor(e, dtype=torch.float32)},
        "opt": {"state": {}, "param_groups": []},
        "config": {"name": "tiny"} if config is None else config,
    }


def _vae_payload(step, w):
    return {
        "kind": "vae", "preset": "tiny", "step": step,
        "vae": {"w": torch.tensor(w, dtype=torch.float32)},
        "opt": {"state": {}, "param_groups": []},
        "config": {"name": "tiny"},
    }


def test_merge_mean_is_exact_and_buffers_come_from_first():
    a = _dit_payload(24, [1.0, 3.0], [[2.0, 4.0]], cnt=7)
    b = _dit_payload(24, [3.0, 5.0], [[4.0, 8.0]], cnt=9)
    out = merge_payloads([a, b], input_names=["a.pt", "b.pt"])
    assert torch.equal(out["dit"]["w"], torch.tensor([2.0, 4.0]))
    assert torch.equal(out["text_enc"]["e"], torch.tensor([[3.0, 6.0]]))
    # integer buffer: copied from the first input, never averaged
    assert torch.equal(out["dit"]["cnt"], torch.tensor([7]))
    assert out["step"] == 24 and out["kind"] == "dit"
    assert out["preset"] == "tiny" and out["config"] == {"name": "tiny"}
    # weights-only artifact: no optimizer / sampler / RNG state
    assert "opt" not in out and "sampler" not in out and "rng" not in out
    assert out["pool"]["num_workers"] == 2
    assert out["pool"]["inputs"] == ["a.pt", "b.pt"]
    assert out["pool"]["base"] is None and out["pool"]["outer_lr"] == 1.0


def test_merge_vae_payload():
    out = merge_payloads([_vae_payload(10, [0.0, 2.0]),
                          _vae_payload(10, [2.0, 4.0])])
    assert torch.equal(out["vae"]["w"], torch.tensor([1.0, 3.0]))
    assert "dit" not in out


def test_merge_outer_step_against_base():
    base = _dit_payload(0, [0.0, 0.0], [[0.0, 0.0]])
    a = _dit_payload(24, [1.0, 3.0], [[2.0, 4.0]])
    b = _dit_payload(24, [3.0, 5.0], [[4.0, 8.0]])
    out = merge_payloads([a, b], base=base, outer_lr=0.5,
                         base_name="base.pt")
    # mean = [2,4] / [[3,6]]; base + 0.5 * (mean - base)
    assert torch.equal(out["dit"]["w"], torch.tensor([1.0, 2.0]))
    assert torch.equal(out["text_enc"]["e"], torch.tensor([[1.5, 3.0]]))
    assert out["pool"]["base"] == "base.pt"
    assert out["pool"]["base_step"] == 0


def test_merge_rejects_mismatches():
    a = _dit_payload(24, [1.0], [[1.0]])
    with pytest.raises(ValueError, match="step mismatch"):
        merge_payloads([a, _dit_payload(25, [1.0], [[1.0]])])
    with pytest.raises(ValueError, match="different checkpoint kinds"):
        merge_payloads([a, _vae_payload(24, [1.0])])
    bad_shape = _dit_payload(24, [1.0, 2.0], [[1.0]])
    with pytest.raises(ValueError, match="shape mismatch"):
        merge_payloads([a, bad_shape])
    bad_cfg = _dit_payload(24, [1.0], [[1.0]], config={"name": "small"})
    with pytest.raises(ValueError, match="config mismatch"):
        merge_payloads([a, bad_cfg])
    with pytest.raises(ValueError, match="at least 2"):
        merge_payloads([a])
    with pytest.raises(ValueError, match="outer-lr"):
        merge_payloads([a, _dit_payload(24, [2.0], [[2.0]])],
                       outer_lr=0.5)


# ------------------------------------------------------- real trainer rounds
@pytest.fixture(scope="module")
def dit_round(tmp_path_factory):
    """Donor -> two shard workers (2 steps each) -> merged checkpoint."""
    tmp = tmp_path_factory.mktemp("dit_pool")
    donor = DiTTrainer(preset="tiny", seed=0, num_clips=8, batch_size=2,
                       ckpt_dir=None)
    init_path = str(tmp / "init.pt")
    donor.save(init_path)

    workers = []
    for shard in ((1, 2), (2, 2)):
        tr = DiTTrainer(preset="tiny", seed=0, num_clips=8, batch_size=2,
                        ckpt_dir=None, data_shard=shard)
        tr.load_dit_weights(init_path)
        assert tr.step == 0  # plain donor: fresh numbering
        for _ in range(2):
            rec = tr.train_step()
            assert math.isfinite(rec["loss"])
        workers.append(tr)
    merged = merge_payloads([w.state_dict() for w in workers],
                            input_names=["w1.pt", "w2.pt"])
    merged_path = str(tmp / "merged.pt")
    torch.save(merged, merged_path)
    return {"init": init_path, "merged": merged_path,
            "merged_payload": merged, "workers": workers}


def test_dit_pool_workers_actually_diverged(dit_round):
    wa, wb = dit_round["workers"]
    pa = {k: v.clone() for k, v in wa.dit.state_dict().items()}
    pb = {k: v.clone() for k, v in wb.dit.state_dict().items()}
    assert any(not torch.equal(pa[k], pb[k]) for k in pa), \
        "workers on different data shards must end with different weights"
    merged_w = dit_round["merged_payload"]["dit"]
    assert any(not torch.equal(merged_w[k], pa[k]) for k in pa)
    assert any(not torch.equal(merged_w[k], pb[k]) for k in pb)


def test_dit_merged_loads_via_init_weights_and_continues_numbering(dit_round):
    tr = DiTTrainer(preset="tiny", seed=1, num_clips=8, batch_size=2,
                    ckpt_dir=None)
    tr.load_dit_weights(dit_round["merged"])
    assert tr.step == 2  # pooled checkpoint donates its stored step
    assert tr.init_from == dit_round["merged"]
    assert all(len(st) == 0 for st in tr.opt.state.values()), \
        "optimizer must stay fresh after weights-only init"
    merged_w = dit_round["merged_payload"]["dit"]
    for k, v in tr.dit.state_dict().items():
        assert torch.equal(v, merged_w[k]), f"merged weight {k} not loaded"
    rec = tr.train_step()
    assert math.isfinite(rec["loss"]) and tr.step == 3


def test_dit_init_weights_start_step_override_and_rejects_vae(tmp_path,
                                                              dit_round):
    tr = DiTTrainer(preset="tiny", seed=0, num_clips=8, batch_size=2,
                    ckpt_dir=None)
    tr.load_dit_weights(dit_round["merged"], start_step=100)
    assert tr.step == 100
    bad = str(tmp_path / "bad.pt")
    torch.save({"kind": "vae", "vae": {}}, bad)
    with pytest.raises(ValueError, match="not a DiT checkpoint"):
        tr.load_dit_weights(bad)


def test_vae_pool_round_step_carryover(tmp_path):
    paths = []
    for shard in ((1, 2), (2, 2)):
        tr = VAETrainer(preset="tiny", seed=0, num_clips=8, batch_size=2,
                        ckpt_dir=None, data_shard=shard)
        tr.train_step()
        tr.train_step()
        p = str(tmp_path / f"vae_w{shard[0]}.pt")
        tr.save(p)
        paths.append(p)
    merged = merge_payloads([load_payload_file(p) for p in paths])
    merged_path = str(tmp_path / "vae_merged.pt")
    torch.save(merged, merged_path)
    nxt = VAETrainer(preset="tiny", seed=0, num_clips=8, batch_size=2,
                     ckpt_dir=None)
    nxt.load_vae_weights(merged_path)
    assert nxt.step == 2  # pooled VAE checkpoint donates its step too
    assert math.isfinite(nxt.train_step()["loss"])


# ---------------------------------------------------------------- status
def test_pool_status_groups_merge_ready(tmp_path, capsys):
    for name, step in (("w1.pt", 24), ("w2.pt", 24), ("w3.pt", 30)):
        torch.save(_dit_payload(step, [1.0], [[1.0]]),
                   str(tmp_path / name))
    rc = pool_status.main([str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "w1.pt: kind=dit preset=tiny step=24" in out
    assert "w3.pt: kind=dit preset=tiny step=30" in out
    assert "merge-ready" in out
    assert "average_checkpoints.py" in out
