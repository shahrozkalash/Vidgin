"""FSDP (--fsdp): a real 2-process gloo world on CPU, no torchrun needed.

Proves the sharded DiT trains to the same weights as DDP (to float
rounding), that its checkpoints are plain (they load on one process),
and that an FSDP run resumes from either kind of checkpoint.
"""
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vidgen.checkpoint import load_checkpoint

WORLD = 2
KW = dict(preset="tiny", seed=0, batch_size=1, num_clips=8)

pytestmark = pytest.mark.skipif(
    not dist.is_available(), reason="torch.distributed unavailable")


def _worker(rank: int, store: str, out: str, fsdp: bool, resume: str | None):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(WORLD),
                      LOCAL_RANK=str(rank))
    dist.init_process_group("gloo", init_method="file:///" + store.lstrip("/"),
                            rank=rank, world_size=WORLD)
    try:
        from vidgen.train import DiTTrainer

        if resume:
            tr = DiTTrainer.resume(resume, ckpt_dir=out, fsdp=fsdp,
                                   seed=0, batch_size=1, num_clips=8)
        else:
            tr = DiTTrainer(ckpt_dir=out, fsdp=fsdp, ema_decay=0.9, **KW)
        assert tr.fsdp == fsdp
        for _ in range(2):
            tr.train_step()
        tr.save(os.path.join(out, "latest.pt"))   # collective under FSDP
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run(tmp_path, name: str, fsdp: bool, resume: str | None = None) -> str:
    out = str(tmp_path / name)
    os.makedirs(out, exist_ok=True)
    store = str(tmp_path / f"{name}.store").replace("\\", "/")
    mp.spawn(_worker, args=(store, out, fsdp, resume), nprocs=WORLD, join=True)
    return os.path.join(out, "latest.pt")


def _max_diff(a: dict, b: dict) -> float:
    assert a.keys() == b.keys()
    return max((a[k].float() - b[k].float()).abs().max().item() for k in a)


def test_fsdp_matches_ddp_and_checkpoints_stay_plain(tmp_path):
    ddp = load_checkpoint(_run(tmp_path, "ddp", fsdp=False))
    fsdp_path = _run(tmp_path, "fsdp", fsdp=True)
    fsdp = load_checkpoint(fsdp_path)
    assert _max_diff(fsdp["dit"], ddp["dit"]) < 1e-5
    assert _max_diff(fsdp["text_enc"], ddp["text_enc"]) < 1e-5
    assert _max_diff(fsdp["ema"]["dit"], ddp["ema"]["dit"]) < 1e-5
    # Optimizer state in the ordinary integer-id format, same as DDP's.
    assert fsdp["opt"]["param_groups"][0]["params"] == \
        ddp["opt"]["param_groups"][0]["params"]
    for i, s in ddp["opt"]["state"].items():
        assert (fsdp["opt"]["state"][i]["exp_avg"]
                - s["exp_avg"]).abs().max() < 1e-5

    # Plain: a single process resumes the FSDP checkpoint.
    from vidgen.train import DiTTrainer
    single = DiTTrainer.resume(fsdp_path, ckpt_dir=str(tmp_path / "one"),
                               seed=0, batch_size=2, num_clips=8)
    assert single.step == 2 and not single.fsdp
    single.train_step()

    # FSDP resumes an FSDP checkpoint and a DDP one identically.
    a = load_checkpoint(_run(tmp_path, "res_f", True, resume=fsdp_path))
    b = load_checkpoint(_run(tmp_path, "res_d", True,
                             resume=str(tmp_path / "ddp" / "latest.pt")))
    assert a["step"] == b["step"] == 4
    assert _max_diff(a["dit"], b["dit"]) < 1e-5


def test_mixed_clip_grad_norm_matches_torch_on_plain_params():
    from vidgen.distributed import clip_grad_norm

    torch.manual_seed(0)
    a = [torch.nn.Parameter(torch.randn(5)) for _ in range(3)]
    b = [torch.nn.Parameter(p.detach().clone()) for p in a]
    for p, q in zip(a, b):
        p.grad = torch.randn(5) * 10
        q.grad = p.grad.clone()
    n1 = clip_grad_norm(a, 1.0)
    n2 = torch.nn.utils.clip_grad_norm_(b, 1.0)
    assert torch.allclose(n1, n2)
    for p, q in zip(a, b):
        assert torch.equal(p.grad, q.grad)


def test_fsdp_without_a_process_group_trains_unsharded(tmp_path, capsys):
    from vidgen.train import DiTTrainer

    tr = DiTTrainer(ckpt_dir=str(tmp_path), fsdp=True, **KW)
    assert not tr.fsdp
    tr.train_step()
    assert "--fsdp needs a torchrun launch" in capsys.readouterr().err
