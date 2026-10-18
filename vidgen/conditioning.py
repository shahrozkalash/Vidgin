"""
Vidgen Shared Conditioning Framework (Batch 1)
Provides:
1. ConditioningSpec: Mapping pixel frame indices to latent indices with causal VAE documentation and error measurement.
2. ConditioningTensorBuilder: Concatenation of [noisy_latents (C), mask (1), cond_latents (C)] -> 2C+1 channels.
3. TrainingModeSampler: Multi-mode training (none, first_frame, first_and_last, first_n_frames) with configurable probabilities.
4. Masked Loss: Loss computed strictly on non-conditioned latent frames.
5. Optional small noise augmentation on conditioning latents.
"""

import math
import random
from typing import List, Dict, Tuple, Optional, Any, Union
from enum import Enum
from vidgen.config import ConditioningConfig, ConditioningTrainingConfig


class ConditioningMode(Enum):
    NONE = "none"                       # Text-to-video (no condition, mask=0)
    FIRST_FRAME = "first_frame"         # Image-to-video (frame 0 conditioned)
    FIRST_AND_LAST = "first_and_last"   # First and last frames conditioned
    FIRST_N_FRAMES = "first_n_frames"   # Video extension (frames 0..N-1 conditioned)


class ConditioningItem:
    """A single conditioning element specified by pixel frame index."""
    def __init__(self, pixel_idx: int, data: Any, is_clip: bool = False, clip_len: int = 1):
        self.pixel_idx = pixel_idx
        self.data = data
        self.is_clip = is_clip
        self.clip_len = clip_len


class ConditioningSpec:
    """
    Conditioning specification containing a list of (pixel frame index, image or clip).
    Maps pixel indices to latent indices under the causal 3D VAE stride (4x8x8).
    """

    def __init__(
        self,
        items: Optional[List[ConditioningItem]] = None,
        total_pixel_frames: int = 81,
        vae_temporal_stride: int = 4
    ):
        self.items: List[ConditioningItem] = items or []
        self.total_pixel_frames = total_pixel_frames
        self.vae_temporal_stride = vae_temporal_stride
        # Total latent frames for 4k + 1 formula: (T - 1) // 4 + 1
        self.total_latent_frames = max(1, (total_pixel_frames - 1) // vae_temporal_stride + 1)

    def add_first_frame(self, data: Any) -> "ConditioningSpec":
        """Add first-frame conditioning at pixel index 0."""
        self.items.append(ConditioningItem(pixel_idx=0, data=data, is_clip=False))
        return self

    def add_last_frame(self, data: Any) -> "ConditioningSpec":
        """Add last-frame conditioning at the final pixel index."""
        last_idx = max(0, self.total_pixel_frames - 1)
        self.items.append(ConditioningItem(pixel_idx=last_idx, data=data, is_clip=False))
        return self

    def add_first_n_frames(self, data: Any, n_pixel_frames: int) -> "ConditioningSpec":
        """Add prefix video clip conditioning for extension."""
        self.items.append(ConditioningItem(pixel_idx=0, data=data, is_clip=True, clip_len=n_pixel_frames))
        return self

    def map_to_latent_indices(self) -> Dict[int, Any]:
        """
        Maps pixel frame indices to latent temporal positions:
        - First frame (p=0): maps to latent 0.
          Because the causal VAE encodes the first frame alone without temporal context,
          first-frame conditioning is exact.
        - Last frame (p=T-1): maps to latent index (T-1) // 4.
          Encoded as a single-frame clip (T=1) placed at the last latent position.

        APPROXIMATION DOCUMENTATION:
        In a causal 3D VAE, receptive field flows strictly from past to present (t' <= t).
        When the last frame of a video is encoded in isolation as a single image (T=1),
        it lacks the temporal causal history from preceding frames. Consequently, its
        latent embedding has a small approximation delta compared to encoding the full clip.
        This difference is measured by `measure_causal_approximation_error` below.
        """
        latent_map: Dict[int, Any] = {}
        for item in self.items:
            if not item.is_clip:
                if item.pixel_idx == 0:
                    latent_idx = 0
                else:
                    latent_idx = min(item.pixel_idx // self.vae_temporal_stride, self.total_latent_frames - 1)
                latent_map[latent_idx] = item.data
            else:
                # Video clip prefix
                latent_count = max(1, (item.clip_len - 1) // self.vae_temporal_stride + 1)
                for l in range(min(latent_count, self.total_latent_frames)):
                    latent_map[l] = item.data
        return latent_map


def measure_causal_approximation_error(
    full_clip_latents: List[List[float]],
    isolated_single_frame_latent: List[float]
) -> Dict[str, float]:
    """
    Empirically measures the approximation error of encoding the last frame in isolation
    versus encoding it within the full continuous causal video.
    Returns L1, L2, and cosine similarity error metrics.
    """
    last_clip_latent = full_clip_latents[-1]
    dim = len(last_clip_latent)
    assert len(isolated_single_frame_latent) == dim, "Dimension mismatch"

    l1_diff = sum(abs(a - b) for a, b in zip(last_clip_latent, isolated_single_frame_latent)) / dim
    l2_diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(last_clip_latent, isolated_single_frame_latent)) / dim)

    norm_clip = math.sqrt(sum(a * a for a in last_clip_latent))
    norm_iso = math.sqrt(sum(b * b for b in isolated_single_frame_latent))
    if norm_clip > 1e-8 and norm_iso > 1e-8:
        dot = sum(a * b for a, b in zip(last_clip_latent, isolated_single_frame_latent))
        cosine_sim = dot / (norm_clip * norm_iso)
    else:
        cosine_sim = 1.0

    return {
        "l1_error": l1_diff,
        "l2_error": l2_diff,
        "cosine_similarity": cosine_sim,
        "approximation_note": "Causal VAE temporal context delta on isolated single frame",
    }


class ConditioningTensorBuilder:
    """
    Builds the 2C+1 channel model input tensor:
    Concatenates [noisy_latents (C), mask (1), cond_latents (C)].
    Supports CPU list-based representations and PyTorch tensors.
    """

    def __init__(self, config: Optional[ConditioningConfig] = None):
        self.config = config or ConditioningConfig()

    def build_concatenated_input_cpu(
        self,
        noisy_latents: List[List[List[float]]],  # Shape: [T, C, D] or [T, C]
        cond_map: Dict[int, List[List[float]]],  # Latent index -> clean latents [C, D]
        noise_rng: Optional[random.Random] = None
    ) -> Tuple[List[List[List[float]]], List[List[float]]]:
        """
        Builds concatenated input for CPU execution:
        Input channels: C (noisy) + 1 (mask) + C (cond) = 2C + 1.
        Returns:
            concatenated: [T, 2C + 1, D]
            mask: [T, 1]
        """
        T = len(noisy_latents)
        assert T > 0, "Empty noisy latents"
        C = len(noisy_latents[0])
        D = len(noisy_latents[0][0]) if isinstance(noisy_latents[0][0], list) else 1

        mask_t: List[List[float]] = []
        concatenated: List[List[List[float]]] = []

        rng = noise_rng or random.Random(42)

        for t in range(T):
            is_conditioned = t in cond_map
            mask_val = 1.0 if is_conditioned else 0.0
            mask_t.append([mask_val])

            # Channels slice 0..C-1: Noisy latents
            slice_noisy = noisy_latents[t]  # [C, D]

            # Channel C: Mask [1, D]
            if D > 1:
                slice_mask = [[mask_val] * D]
            else:
                slice_mask = [[mask_val]]

            # Channels C+1..2C: Conditioning clean latents
            if is_conditioned:
                clean_lat = cond_map[t]  # [C, D]
                if self.config.enable_conditioning_noise and self.config.conditioning_noise_strength > 0:
                    sigma = self.config.conditioning_noise_strength
                    # Add small noise perturbation
                    clean_lat = [
                        [val + rng.gauss(0, sigma) for val in ch_vec]
                        for ch_vec in clean_lat
                    ]
                slice_cond = clean_lat
            else:
                # All zeros for unconditioned frames
                if D > 1:
                    slice_cond = [[0.0] * D for _ in range(C)]
                else:
                    slice_cond = [[0.0] for _ in range(C)]

            # Concatenate along channel dimension: [C + 1 + C, D]
            frame_combined = slice_noisy + slice_mask + slice_cond
            concatenated.append(frame_combined)

        return concatenated, mask_t


class TrainingModeSampler:
    """
    Samples conditioning modes during training according to configured probability distributions:
    - none (text-to-video)
    - first_frame (image-to-video)
    - first_and_last (first and last frame control)
    - first_n_frames (video extension)
    """

    def __init__(self, config: Optional[ConditioningTrainingConfig] = None, seed: int = 42):
        self.config = config or ConditioningTrainingConfig()
        self.rng = random.Random(seed)

    def sample_mode(self) -> ConditioningMode:
        """Sample a training conditioning mode based on configured probabilities."""
        r = self.rng.random()
        p0 = self.config.prob_none
        p1 = p0 + self.config.prob_first_frame
        p2 = p1 + self.config.prob_first_and_last

        if r < p0:
            return ConditioningMode.NONE
        elif r < p1:
            return ConditioningMode.FIRST_FRAME
        elif r < p2:
            return ConditioningMode.FIRST_AND_LAST
        else:
            return ConditioningMode.FIRST_N_FRAMES

    def build_training_mask(
        self,
        total_latent_frames: int,
        mode: Optional[ConditioningMode] = None
    ) -> List[float]:
        """
        Builds a binary mask array of length total_latent_frames:
        1.0 = Conditioned (loss masked out if loss_non_conditioned_only=True)
        0.0 = Generated / Target for loss calculation
        """
        if mode is None:
            mode = self.sample_mode()

        mask = [0.0] * total_latent_frames

        if mode == ConditioningMode.NONE:
            pass  # Text-to-video: no conditioning
        elif mode == ConditioningMode.FIRST_FRAME:
            mask[0] = 1.0
        elif mode == ConditioningMode.FIRST_AND_LAST:
            mask[0] = 1.0
            if total_latent_frames > 1:
                mask[-1] = 1.0
        elif mode == ConditioningMode.FIRST_N_FRAMES:
            n = min(self.config.extension_n_frames, total_latent_frames - 1)
            for i in range(max(1, n)):
                mask[i] = 1.0

        return mask


def compute_masked_loss(
    v_pred: List[List[float]],      # Predicted velocity [T, D]
    v_target: List[List[float]],    # Ground truth velocity target [T, D]
    mask: List[float],              # Condition mask [T], 1.0 for conditioned
    loss_non_conditioned_only: bool = True
) -> Dict[str, float]:
    """
    Computes Rectified Flow loss:
    When loss_non_conditioned_only is True, computes loss strictly on non-conditioned frames (mask == 0).
    When False, computes loss across all frames.
    """
    T = len(v_pred)
    assert len(v_target) == T and len(mask) == T, "Length mismatch"
    dim = len(v_pred[0])

    loss_sum = 0.0
    weight_sum = 0.0
    total_l2 = 0.0

    for t in range(T):
        w = (1.0 - mask[t]) if loss_non_conditioned_only else 1.0
        frame_diff_sq = sum((v_pred[t][d] - v_target[t][d]) ** 2 for d in range(dim))
        total_l2 += frame_diff_sq
        if w > 0.0:
            loss_sum += w * frame_diff_sq
            weight_sum += w

    if weight_sum > 0.0:
        masked_loss = loss_sum / (weight_sum * dim)
    else:
        masked_loss = total_l2 / (T * dim)

    unmasked_loss = total_l2 / (T * dim)

    return {
        "loss": masked_loss,
        "unmasked_loss": unmasked_loss,
        "active_training_frames": weight_sum,
        "total_frames": float(T),
    }
