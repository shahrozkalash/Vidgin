"""Multi-GPU support: DistributedDataParallel (DDP) helpers.

Design contract (see PYTHON_IMPLEMENTATION.md, "Multi-GPU training"):

- **Opt-in by launcher.** Nothing changes unless the process is started
  with ``torchrun`` (which sets ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK``
  in the environment). Without torchrun, every helper here is a no-op and
  the trainers behave byte-identically to the original single-GPU code.
- **Data parallel by default.** Each rank holds a FULL copy of the model
  and trains on a disjoint slice of each global batch; gradients are
  all-reduced by DDP so the copies stay in lock-step. DDP does NOT make a
  model that exceeds one GPU's memory fit.
- **FSDP for big models (opt-in, DiT only: ``--fsdp``).** ``wrap_fsdp``
  shards the DiT's parameters, gradients and optimizer state across the
  ranks (PyTorch FSDP2 ``fully_shard``, one unit per transformer block),
  so each GPU holds ~1/N of them — the path for the 9B/14B presets. The
  text encoder stays replicated (its gradients are synced explicitly, as
  under DDP). Checkpoints are gathered to rank 0 in the SAME plain format
  as DDP/single-GPU ones (``full_model_state`` / ``full_optim_state``),
  so an FSDP checkpoint resumes on one GPU and vice versa.
- **Plain checkpoints.** Checkpoints are always written from the
  UNWRAPPED model, so their keys never carry a ``module.`` prefix and a
  checkpoint trained on N GPUs loads into a plain single-GPU model (and
  vice versa). Only rank 0 writes checkpoints or logs.
- **Per-rank divergence, deterministically.** All ranks build identical
  model weights (same base seed). Afterwards each rank re-seeds its RNG
  stream with ``rank_seed(base_seed, rank)`` so flow noise, timestep
  draws, caption/depth dropout and VAE reparameterization noise differ
  per rank, while data order is sharded from ONE shared sampler sequence
  (see ``shard_batch`` and the trainers), so a distributed run resumed
  with the same world size and seed reproduces itself.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Per-rank RNG streams are derived as base_seed + rank * RANK_SEED_STRIDE.
# A large odd stride keeps neighbouring base seeds from colliding with a
# neighbouring rank's stream for any realistic seed range.
RANK_SEED_STRIDE = 1_000_003


def rank_seed(base_seed: int, rank: int) -> int:
    """Deterministic per-rank seed: rank 0 keeps the base seed itself."""
    return int(base_seed) + int(rank) * RANK_SEED_STRIDE


def shard_batch(global_batch: list, per_rank_batch: int, rank: int) -> list:
    """Rank ``r``'s disjoint slice of one shared global batch.

    The trainers draw ONE global batch of ``per_rank_batch * world_size``
    indices from a sampler that is identically seeded on every rank, then
    each rank keeps only its own contiguous slice. Slices are disjoint,
    their union is the global batch, and the sampler state stays identical
    on all ranks — so rank 0's saved sampler state is complete and
    authoritative for resume.
    """
    start = int(rank) * int(per_rank_batch)
    return list(global_batch[start:start + int(per_rank_batch)])


@dataclass
class DistInfo:
    """Snapshot of this process's place in the (possible) process group."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str | None = None

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def env_dist_info() -> DistInfo:
    """Parse torchrun's environment WITHOUT initializing anything.

    ``WORLD_SIZE`` unset or 1 means single-process: rank 0 of 1.
    """
    try:
        world = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world = 1
    if world <= 1:
        return DistInfo(rank=0, world_size=1, local_rank=0)
    return DistInfo(
        rank=int(os.environ.get("RANK", "0")),
        world_size=world,
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
    )


def current_dist_info() -> DistInfo:
    """DistInfo for the live process group if initialized, else the env."""
    if dist.is_available() and dist.is_initialized():
        return DistInfo(rank=dist.get_rank(), world_size=dist.get_world_size(),
                        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
                        backend=dist.get_backend())
    return env_dist_info()


def init_distributed(prefer: str = "cpu") -> DistInfo:
    """Initialize the process group when launched under ``torchrun``.

    No-op (returns the single-process DistInfo) when ``WORLD_SIZE`` is
    unset or 1. Backend is ``nccl`` when the run targets CUDA (per the
    ``prefer`` device string and CUDA availability), else ``gloo`` — so
    the same code path works for a CPU ``torchrun`` smoke. Idempotent:
    calling it when a group already exists returns the current info.
    The device hint only selects backend/local CUDA device; the trainers
    make the final per-rank device choice from the returned info.
    """
    info = env_dist_info()
    if not info.is_distributed:
        return info
    if dist.is_available() and dist.is_initialized():
        return current_dist_info()
    want_cuda = prefer != "cpu" and torch.cuda.is_available()
    backend = "nccl" if want_cuda else "gloo"
    if want_cuda:
        torch.cuda.set_device(info.local_rank)
    dist.init_process_group(backend=backend)
    return DistInfo(rank=dist.get_rank(), world_size=dist.get_world_size(),
                    local_rank=info.local_rank, backend=backend)


def is_distributed() -> bool:
    return current_dist_info().is_distributed


def is_main_rank(info: DistInfo | None = None) -> bool:
    return (info or current_dist_info()).is_main


def world_size(info: DistInfo | None = None) -> int:
    return (info or current_dist_info()).world_size


def rank(info: DistInfo | None = None) -> int:
    return (info or current_dist_info()).rank


def resolve_rank_device(prefer: str, info: DistInfo) -> torch.device:
    """Device for this rank: ``cuda:<local_rank>`` in a CUDA run, else the
    caller's device unchanged. Single-process runs are untouched."""
    if info.is_distributed and prefer.startswith("cuda"):
        return torch.device(f"cuda:{info.local_rank}")
    return torch.device(prefer)


def barrier(info: DistInfo | None = None) -> None:
    """Synchronize all ranks; a no-op unless a real group is initialized."""
    info = info or current_dist_info()
    if info.is_distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def sync_gradients(module: torch.nn.Module, info: DistInfo | None = None) -> None:
    """All-reduce + average a NON-DDP module's gradients in place.

    Used for the DiT trainer's text encoder: it is deliberately kept
    unwrapped (DDP only permits ``forward`` through the wrapper, and the
    trainer also calls ``null_embedding()``; its parameters are also used
    outside the wrapped forward), so its gradients are synchronized
    explicitly after ``backward()`` instead. Every rank runs the same
    code path, so the set of parameters carrying grads is identical on
    all ranks and the collectives match up. No-op without a live group.
    """
    info = info or current_dist_info()
    if not info.is_distributed:
        return
    if not (dist.is_available() and dist.is_initialized()):
        return
    for p in module.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(info.world_size)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Strip a DDP wrapper (``model.module``); identity for plain models."""
    if isinstance(model, DDP):
        return model.module
    return model


def plain_state_dict(model: torch.nn.Module) -> dict:
    """The unwrapped model's state_dict — no ``module.`` prefix, ever."""
    return unwrap_model(model).state_dict()


def is_fsdp(model: torch.nn.Module) -> bool:
    """True for a module sharded by :func:`wrap_fsdp`."""
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:  # torch without FSDP2
        return False
    return isinstance(model, FSDPModule)


def is_dtensor(t) -> bool:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    return isinstance(t, DTensor)


def wrap_fsdp(model: torch.nn.Module, info: DistInfo, device: torch.device,
              param_dtype: torch.dtype | None = None) -> torch.nn.Module:
    """Shard ``model`` with FSDP2 when a real process group is live.

    Each block in ``model.blocks`` becomes its own FSDP unit (its
    parameters are all-gathered just for its forward/backward, then
    freed), and the root unit holds the rest (embeddings, modulation,
    head). ``param_dtype`` (e.g. bf16) makes the all-gathered compute
    copies low precision while the sharded master weights and the
    gradient reduction stay fp32. Sharding replaces the module's
    parameters, so optimizers must be built AFTER this call. Returns the
    same module object; a no-op without a live group (single process,
    simulated worlds in unit tests).
    """
    if not info.is_distributed:
        return model
    if not (dist.is_available() and dist.is_initialized()):
        return model
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    # The DiT returns a reshaped VIEW of its head output; FSDP2 warns that
    # an in-place op on it would skip the backward all-gather. Nothing
    # here ever modifies the prediction in place (flow_loss, the sampler
    # and the probe are all out-of-place), so the warning is noise.
    warnings.filterwarnings("ignore", message=r"FSDP2-wrapped module .* "
                            r"returned a view tensor")
    mesh = init_device_mesh(device.type, (info.world_size,))
    mp = MixedPrecisionPolicy(param_dtype=param_dtype,
                              reduce_dtype=torch.float32)
    for blk in getattr(model, "blocks", []):
        fully_shard(blk, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)
    return model


def _full_options():
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    return StateDictOptions(full_state_dict=True, cpu_offload=True)


def full_model_state(model: torch.nn.Module) -> dict:
    """Plain (unsharded, CPU) state dict of a possibly-FSDP module.

    For an FSDP module this is a COLLECTIVE: every rank must call it;
    only rank 0 receives the tensors (others get an empty dict). Plain
    and DDP modules return their ordinary unwrapped state dict.
    """
    if not is_fsdp(model):
        return plain_state_dict(model)
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    return get_model_state_dict(model, options=_full_options())


def load_full_model_state(model: torch.nn.Module, state: dict) -> None:
    """Strict load of a plain state dict into a possibly-FSDP module
    (every rank passes the full dict; each keeps its own shard)."""
    if not is_fsdp(model):
        unwrap_model(model).load_state_dict(state)
        return
    from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                         set_model_state_dict)

    set_model_state_dict(model, state,
                         options=StateDictOptions(full_state_dict=True))


class _Joint(torch.nn.Module):
    """Non-owning container naming the optimizer's modules for the
    distributed state-dict API (parameter FQNs ``<name>.<param>``)."""

    def __init__(self, modules: dict):
        super().__init__()
        for name, m in modules.items():
            self.add_module(name, m)


def _opt_fqns(joint: torch.nn.Module, optimizer) -> list[str]:
    """FQN of every optimizer parameter, in the optimizer's own order."""
    names = {id(p): n for n, p in joint.named_parameters()}
    return [names[id(p)] for g in optimizer.param_groups for p in g["params"]]


def full_optim_state(modules: dict, optimizer) -> dict:
    """The optimizer's state in the ordinary ``Optimizer.state_dict()``
    format (integer parameter ids), gathered from FSDP shards.

    ``modules`` maps names to the modules the optimizer covers (some may
    be FSDP-sharded, some plain). Collective across ranks; rank 0 gets
    the full state. Without FSDP this is just ``optimizer.state_dict()``.
    """
    if not any(is_fsdp(m) for m in modules.values()):
        return optimizer.state_dict()
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    joint = _Joint(modules)
    fqns = _opt_fqns(joint, optimizer)
    osd = get_optimizer_state_dict(joint, optimizer, options=_full_options())
    index = {n: i for i, n in enumerate(fqns)}
    groups = []
    for g in osd.get("param_groups", []):
        g = dict(g)
        g["params"] = [index[n] for n in g["params"]]
        groups.append(g)
    state = {index[n]: s for n, s in osd.get("state", {}).items()}
    return {"state": state, "param_groups": groups}


def load_full_optim_state(modules: dict, optimizer, state: dict) -> None:
    """Load an ordinary ``Optimizer.state_dict()`` into an optimizer whose
    parameters may be FSDP-sharded (inverse of :func:`full_optim_state`)."""
    if not any(is_fsdp(m) for m in modules.values()):
        optimizer.load_state_dict(state)
        return
    from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                         set_optimizer_state_dict)

    joint = _Joint(modules)
    fqns = _opt_fqns(joint, optimizer)
    groups = []
    for g in state["param_groups"]:
        g = dict(g)
        g["params"] = [fqns[i] for i in g["params"]]
        groups.append(g)
    osd = {"state": {fqns[int(i)]: s for i, s in state["state"].items()},
           "param_groups": groups}
    set_optimizer_state_dict(joint, optimizer, osd,
                             options=StateDictOptions(full_state_dict=True))


def clip_grad_norm(params, max_norm: float) -> torch.Tensor:
    """``clip_grad_norm_`` that also accepts a MIX of FSDP-sharded
    (DTensor) and plain parameters, which torch's version rejects.

    Plain parameters only: exactly ``torch.nn.utils.clip_grad_norm_``.
    Mixed: the sharded group's norm is reduced across ranks, combined
    with the replicated group's, and both are scaled by the same factor.
    Returns the total norm as a plain tensor.
    """
    params = [p for p in params if p.grad is not None]
    sharded = [p for p in params if is_dtensor(p.grad)]
    if not sharded:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    plain = [p for p in params if not is_dtensor(p.grad)]
    norm = torch.nn.utils.get_total_norm([p.grad for p in sharded])
    norm = norm.full_tensor() if is_dtensor(norm) else norm
    total_sq = norm.float() ** 2
    if plain:
        pn = torch.nn.utils.get_total_norm([p.grad for p in plain])
        total_sq = total_sq + pn.to(total_sq.device).float() ** 2
    total = total_sq.sqrt()
    # Same coefficient rule as torch's clip_grad_norm_ (applied as a
    # python float: it multiplies DTensor and plain gradients alike).
    coef = min(1.0, float(max_norm) / (float(total) + 1e-6))
    if coef < 1.0:
        for p in params:
            p.grad.mul_(coef)
    return total


def wrap_ddp(model: torch.nn.Module, info: DistInfo,
             device: torch.device) -> torch.nn.Module:
    """Wrap ``model`` in DDP when a real process group is live.

    Returns the model unchanged in single-process runs (and when handed a
    DistInfo for a world that has no initialized group, e.g. unit tests
    that exercise the sharding/seeding logic without torchrun). Gradient
    (activation) checkpointing uses the non-reentrant implementation in
    ``vidgen.dit.model``, which is the DDP-compatible variant.
    """
    if not info.is_distributed:
        return model
    if not (dist.is_available() and dist.is_initialized()):
        return model
    if device.type == "cuda":
        return DDP(model, device_ids=[device.index])
    return DDP(model)
