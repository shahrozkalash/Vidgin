"""Multi-GPU support: DistributedDataParallel (DDP) helpers.

Design contract (see PYTHON_IMPLEMENTATION.md, "Multi-GPU training"):

- **Opt-in by launcher.** Nothing changes unless the process is started
  with ``torchrun`` (which sets ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK``
  in the environment). Without torchrun, every helper here is a no-op and
  the trainers behave byte-identically to the original single-GPU code.
- **Data parallel only.** Each rank holds a FULL copy of the model and
  trains on a disjoint slice of each global batch; gradients are
  all-reduced by DDP so the copies stay in lock-step. The model itself is
  never sharded — DDP does NOT make a model that exceeds one GPU's memory
  fit (that would need FSDP/ZeRO, which is future work, not this module).
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
