"""VidGen Training & Scaling Package.

Exports:
- WanTrainer: Main flow matching trainer loop
- RectifiedFlowLoss: Flow matching objective with shifted logit-normal timesteps
- EMAModel: Exponential moving average of model parameters
- TrainingProfiler: Tokens/sec, MFU, and stall detector
- UlyssesSequenceParallel: DeepSpeed Ulysses context parallelism for ~30k tokens
- calculate_memory_footprint_gb: VRAM breakdown (FSDP2, activation checkpointing, 8-bit opt)
- compute_invariant_batch_schedule: Invariant global batch scaling across GPU counts
- ReshardingCoordinator: Topology-agnostic checkpoint resumption
"""

from vidgen.train.ema import EMAModel
from vidgen.train.losses import RectifiedFlowLoss, sample_logit_normal_timesteps
from vidgen.train.profiler import TrainingProfiler, calculate_flops_per_token, KNOWN_HARDWARE
from vidgen.train.parallel import (
    ParallelismConfig,
    UlyssesSequenceParallel,
    calculate_memory_footprint_gb,
    compute_invariant_batch_schedule,
    ReshardingCoordinator,
)
from vidgen.train.checkpoint import save_checkpoint, resume_checkpoint, get_rng_state, set_rng_state
from vidgen.train.train import WanTrainer

__all__ = [
    "WanTrainer",
    "RectifiedFlowLoss",
    "sample_logit_normal_timesteps",
    "EMAModel",
    "TrainingProfiler",
    "calculate_flops_per_token",
    "KNOWN_HARDWARE",
    "ParallelismConfig",
    "UlyssesSequenceParallel",
    "calculate_memory_footprint_gb",
    "compute_invariant_batch_schedule",
    "ReshardingCoordinator",
    "save_checkpoint",
    "resume_checkpoint",
    "get_rng_state",
    "set_rng_state",
]
