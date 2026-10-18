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
    MemoryConfig,
    FeedbackConfig,
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
from vidgen.memory import (
    MemoryStore,
    MemoryEntry,
    StubTextVisualEncoder,
    GenerationMemoryHook,
    ReferenceAttachment,
    EpisodicMemoryBuffer,
    map_type_to_role,
)
from vidgen.feedback_widget import (
    show_and_rate,
    compare_and_rate,
    analyze_rater_hygiene,
    FakeWidgetBackend,
    CliBackend,
    get_active_ratings,
    read_feedback_records,
    undo_last_feedback,
)


