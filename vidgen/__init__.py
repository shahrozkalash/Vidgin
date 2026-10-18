"""
Vidgen: Open-source Video Latent Diffusion Engine
Modular, CPU-testable implementation for long-video causal diffusion.
"""

__version__ = "0.2.0"

from vidgen.config import (
    AttentionConfig,
    PresetConfig,
    LongVideoConfig,
    ChunkContinuityConfig,
    ConditioningConfig,
    TINY_PRESET,
    SMALL_PRESET,
)
from vidgen.attention import (
    SpatialAttention,
    TemporalAttention,
    FullSpacetimeAttention,
    DiTAttentionBlock,
    DecoupledDiTModel,
    estimate_attention_flops_and_memory,
    apply_rope_3d,
    get_rope_partition,
)
