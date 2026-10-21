"""Multi-GPU (DDP) support: helpers, sharding/seeding, and proof that
single-process training is unchanged.

The real 2-process torchrun smoke lives outside pytest (see
PYTHON_IMPLEMENTATION.md); these tests cover everything provable in one
process, including a *simulated* 2-rank world (explicit DistInfo, no
process group): disjoint data slices and divergent-but-deterministic
per-rank RNG streams.
"""

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from vidgen.distributed import (
    DistInfo,
    current_dist_info,
    env_dist_info,
    init_distributed,
    is_main_rank,
    plain_state_dict,
    rank_seed,
    shard_batch,
    unwrap_model,
    wrap_ddp,
)
from vidgen.train import DiTTrainer
from vidgen.train_vae import VAETrainer

FAKE2_RANK0 = DistInfo(rank=0, world_size=2, local_rank=0)
FAKE2_RANK1 = DistInfo(rank=1, world_size=2, local_rank=1)


# ------------------------------------------------------------ helper units
def test_env_fallback_is_single_process(monkeypatch):
    for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(var, raising=False)
    info = env_dist_info()
    assert (info.rank, info.world_size, info.local_rank) == (0, 1, 0)
    assert not info.is_distributed and info.is_main
    assert current_dist_info().world_size == 1
    assert is_main_rank()


def test_init_distributed_noop_without_torchrun(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    info = init_distributed("cpu")
    assert info.world_size == 1 and not info.is_distributed
    assert not (torch.distributed.is_available()
                and torch.distributed.is_initialized())


def test_env_parsing_under_torchrun(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    info = env_dist_info()
    assert (info.rank, info.world_size, info.local_rank) == (1, 2, 1)
    assert info.is_distributed and not info.is_main


def test_rank_seed_diverges_and_rank0_is_base():
    assert rank_seed(0, 0) == 0
    assert rank_seed(42, 0) == 42
    assert rank_seed(42, 1) != rank_seed(42, 0)
    assert rank_seed(42, 1) != rank_seed(43, 0)  # no neighbour collision


def test_shard_batch_disjoint_and_complete():
    global_batch = list(range(8))
    slices = [shard_batch(global_batch, 2, r) for r in range(4)]
    assert all(len(s) == 2 for s in slices)
    flat = [i for s in slices for i in s]
    assert sorted(flat) == global_batch and len(set(flat)) == 8  # disjoint


def test_unwrap_and_plain_state_dict_on_plain_model():
    model = torch.nn.Linear(4, 3)
    assert unwrap_model(model) is model
    sd = plain_state_dict(model)
    assert set(sd) == set(model.state_dict())
    assert not any(k.startswith("module.") for k in sd)


def test_wrap_ddp_noop_single_process():
    model = torch.nn.Linear(4, 3)
    wrapped = wrap_ddp(model, DistInfo(), torch.device("cpu"))
    assert wrapped is model and not isinstance(wrapped, DDP)


def test_wrap_ddp_noop_for_simulated_world_without_group():
    # A DistInfo claiming world_size=2 but with no initialized process
    # group (the unit-test simulation) must not try to build a real DDP.
    model = torch.nn.Linear(4, 3)
    assert wrap_ddp(model, FAKE2_RANK1, torch.device("cpu")) is model


# ------------------------------------------- single-process behavior intact
def test_single_process_trainer_is_unwrapped_and_unsharded(tmp_path):
    tr = DiTTrainer(preset="tiny", seed=0, batch_size=2, num_clips=8,
                    ckpt_dir=str(tmp_path))
    assert tr.dist_info.world_size == 1 and tr.is_main_rank
    assert tr.global_batch_size == 2
    assert tr.sampler.batch_size == 2  # exactly the pre-DDP construction
    assert not isinstance(tr.dit, DDP) and not isinstance(tr.text_enc, DDP)
    rec = tr.train_step()
    assert rec["loss"] == rec["loss"] and abs(rec["loss"]) < 1e9  # finite
    sd = tr.state_dict()
    assert not any(k.startswith("module.")
                   for k in list(sd["dit"]) + list(sd["text_enc"]))


# ------------------------------------------- simulated 2-rank data + noise
def _dit_trainer(rank_info, tmp_path):
    return DiTTrainer(preset="tiny", seed=0, batch_size=2, num_clips=16,
                      ckpt_dir=str(tmp_path), dist_info=rank_info)


def test_simulated_ranks_get_disjoint_clips(tmp_path):
    tr0 = _dit_trainer(FAKE2_RANK0, tmp_path / "r0")
    tr1 = _dit_trainer(FAKE2_RANK1, tmp_path / "r1")
    assert tr0.global_batch_size == 4 and tr0.sampler.batch_size == 4
    b0, b1 = tr0._next_batch_idx(), tr1._next_batch_idx()
    assert len(b0) == len(b1) == 2
    assert not set(b0) & set(b1)                      # disjoint clips
    assert sorted(b0 + b1) == sorted(tr0.sampler.state_dict()["perm"][:4].tolist())


def test_simulated_ranks_rng_diverges_but_is_deterministic(tmp_path):
    # The RNG is process-global, so each rank's first draw must be taken
    # immediately after that rank's trainer is built (construction ends
    # by seeding the per-rank stream).
    _dit_trainer(FAKE2_RANK0, tmp_path / "r0")
    draw0 = torch.randn(4)
    _dit_trainer(FAKE2_RANK1, tmp_path / "r1")
    draw1 = torch.randn(4)
    assert not torch.equal(draw0, draw1)               # noise streams differ
    # Rebuilding the same ranks reproduces the same streams exactly.
    _dit_trainer(FAKE2_RANK0, tmp_path / "r0b")
    assert torch.equal(torch.randn(4), draw0)
    _dit_trainer(FAKE2_RANK1, tmp_path / "r1b")
    assert torch.equal(torch.randn(4), draw1)


def test_simulated_rank1_writes_no_checkpoint(tmp_path):
    tr1 = _dit_trainer(FAKE2_RANK1, tmp_path / "r1")
    target = tmp_path / "r1" / "latest.pt"
    tr1.save(str(target))
    assert not target.exists()                         # rank 0 writes, only


def test_vae_trainer_simulated_sharding(tmp_path):
    tr = VAETrainer(preset="tiny", seed=0, batch_size=2, num_clips=16,
                    ckpt_dir=str(tmp_path), dist_info=FAKE2_RANK1)
    assert tr.global_batch_size == 4 and tr.sampler.batch_size == 4
    assert len(tr._next_batch_idx()) == 2
    assert not isinstance(tr.vae, DDP)
    rec = tr.train_step()
    assert abs(rec["loss"]) < 1e9
