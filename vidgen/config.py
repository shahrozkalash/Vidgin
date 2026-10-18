"""
Vidgen Configuration Module
Defines configuration schemas and presets for long-video generation.
All new improvement features are guarded behind flags that default to OFF.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple


@dataclass
class PresetConfig:
    """Model architecture preset configuration."""
    name: str = "tiny"
    dim: int = 256
    depth: int = 6
    num_heads: int = 4
    head_dim: int = 64
    in_channels: int = 16
    patch_size: Tuple[int, int, int] = (1, 2, 2)  # (t, h, w)
    vae_stride: Tuple[int, int, int] = (4, 8, 8)   # (t, h, w)
    default_height: int = 64
    default_width: int = 64


TINY_PRESET = PresetConfig(
    name="tiny",
    dim=256,
    depth=6,
    num_heads=4,
    head_dim=64,
    in_channels=16,
    patch_size=(1, 2, 2),
    vae_stride=(4, 8, 8),
    default_height=64,
    default_width=64,
)

SMALL_PRESET = PresetConfig(
    name="small",
    dim=768,
    depth=12,
    num_heads=12,
    head_dim=64,
    in_channels=16,
    patch_size=(1, 2, 2),
    vae_stride=(4, 8, 8),
    default_height=256,
    default_width=256,
)


@dataclass
class ChunkContinuityConfig:
    """
    Batch 1: Smooth chunk-to-chunk continuity configuration.
    CRITICAL: Every new improvement feature defaults to OFF (False).
    """
    # 1. Chunk overlap (1 to 2 latent frames shared with previous chunk)
    enable_chunk_overlap: bool = False
    overlap_frames: int = 1  # 1 or 2 latent frames
    overlap_mode: str = "fixed"  # "fixed" (clean conditioning) or "blended"

    # 2. Linear cross-fade in overlap region
    enable_linear_crossfade: bool = False
    blend_width: int = 1  # Number of frames across which to blend

    # 3. RoPE re-indexing relative to window start
    enable_rope_reindexing: bool = False

    # 4. Motion carry-over from last N clean frames
    enable_motion_carryover: bool = False
    motion_carryover_frames: int = 2  # Condition on last N frames (velocity & trajectory)


@dataclass
class LongVideoConfig:
    """
    Complete Part 13 30-Second Video Generation Configuration.
    Target: 30s at 16 fps = 481 raw frames (4k+1 with k=120) -> 121 latent frames.
    """
    duration_seconds: float = 30.0
    fps: int = 16
    chunk_duration_seconds: float = 5.0
    preset: PresetConfig = field(default_factory=lambda: TINY_PRESET)
    continuity: ChunkContinuityConfig = field(default_factory=ChunkContinuityConfig)

    # Architectural parameters
    anchor_frames: int = 2
    window_chunks: int = 1

    @property
    def total_raw_frames(self) -> int:
        """Total raw frames satisfying T = 4k + 1."""
        target = int(round(self.duration_seconds * self.fps)) + 1
        # Round to nearest 4k + 1
        k = max(0, round((target - 1) / 4))
        return 4 * k + 1

    @property
    def total_latent_frames(self) -> int:
        """Latent temporal frames: (T - 1) // 4 + 1."""
        return (self.total_raw_frames - 1) // 4 + 1

    @property
    def chunk_raw_frames(self) -> int:
        """Raw frames per nominal chunk (e.g. 5s at 16 fps = 81 frames)."""
        target = int(round(self.chunk_duration_seconds * self.fps)) + 1
        k = max(0, round((target - 1) / 4))
        return 4 * k + 1

    @property
    def chunk_latent_frames(self) -> int:
        """Latent frames per nominal chunk: 21 latents for 81 raw frames."""
        return (self.chunk_raw_frames - 1) // 4 + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "total_raw_frames": self.total_raw_frames,
            "total_latent_frames": self.total_latent_frames,
            "chunk_latent_frames": self.chunk_latent_frames,
            "enable_chunk_overlap": self.continuity.enable_chunk_overlap,
            "overlap_frames": self.continuity.overlap_frames,
            "overlap_mode": self.continuity.overlap_mode,
            "enable_linear_crossfade": self.continuity.enable_linear_crossfade,
            "blend_width": self.continuity.blend_width,
            "enable_rope_reindexing": self.continuity.enable_rope_reindexing,
            "enable_motion_carryover": self.continuity.enable_motion_carryover,
            "motion_carryover_frames": self.continuity.motion_carryover_frames,
        }


@dataclass
class ConditioningConfig:
    """
    Shared Conditioning Configuration (Batch 1).
    Concatenates [mask (1 channel), conditioning latents (C channels)] to noisy latents.
    All new feature flags default to OFF (False).
    """
    enable_conditioning: bool = False
    in_channels: int = 16  # Latent channels C
    cond_channels: int = 17  # 1 channel mask + C channels conditioning latents = 17 for C=16
    total_model_channels: int = 33  # C + 1 + C = 33 channels
    enable_conditioning_noise: bool = False
    conditioning_noise_strength: float = 0.02
    loss_non_conditioned_only: bool = True


@dataclass
class ConditioningTrainingConfig:
    """
    Multi-mode conditioning training configuration with configurable sampling probabilities.
    All flags default to baseline-compatible values.
    """
    prob_none: float = 0.20             # Text-to-video (no condition, mask=0)
    prob_first_frame: float = 0.40      # Image-to-video (first frame conditioned)
    prob_first_and_last: float = 0.20   # First and last frame control
    prob_first_n_frames: float = 0.20   # Video extension (first N frames conditioned)
    extension_n_frames: int = 4         # N latent frames for extension training
    loss_non_conditioned_only: bool = True

