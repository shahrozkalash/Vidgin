"""
Vidgen Long Video Continuity Engine (Batch 1)
Implements:
1. Chunk overlap (fixed clean conditioning vs blended)
2. Linear cross-fade in overlap region with configurable blend width
3. RoPE re-indexing relative to window start with translation invariance
4. Motion carry-over from last N clean frames (velocity/acceleration continuity)
5. Seam metric calculation and oracle ground-truth verification
"""

import math
from typing import List, Tuple, Dict, Any, Optional
from vidgen.config import ChunkContinuityConfig, LongVideoConfig


class LatentFrame:
    """
    Representation of a single temporal latent frame.
    Supports linear vector space operations for CPU testing and verification.
    """
    def __init__(self, data: List[float]):
        self.data = [float(x) for x in data]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> float:
        return self.data[idx]

    def __add__(self, other: "LatentFrame") -> "LatentFrame":
        assert len(self.data) == len(other.data), "Dimension mismatch"
        return LatentFrame([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other: "LatentFrame") -> "LatentFrame":
        assert len(self.data) == len(other.data), "Dimension mismatch"
        return LatentFrame([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar: float) -> "LatentFrame":
        return LatentFrame([a * float(scalar) for a in self.data])

    def __rmul__(self, scalar: float) -> "LatentFrame":
        return self.__mul__(scalar)

    def l2_norm(self) -> float:
        return math.sqrt(sum(x * x for x in self.data))

    def diff_norm(self, other: "LatentFrame") -> float:
        assert len(self.data) == len(other.data), "Dimension mismatch"
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.data, other.data)))

    def to_list(self) -> List[float]:
        return list(self.data)


def compute_rope_frequencies(dim: int, base: float = 10000.0) -> List[float]:
    """Compute standard RoPE inverse frequencies."""
    freqs = []
    for i in range(0, dim, 2):
        freqs.append(1.0 / (base ** (i / dim)))
    return freqs


def apply_temporal_rope(
    vector: List[float],
    t: int,
    freqs: Optional[List[float]] = None
) -> List[float]:
    """
    Apply 1D temporal Rotary Positional Embedding (RoPE) at position t.
    Uses complex Givens rotation across pairs of coordinates.
    """
    dim = len(vector)
    assert dim % 2 == 0, "Dimension must be even for RoPE"
    if freqs is None:
        freqs = compute_rope_frequencies(dim)

    out = list(vector)
    for i in range(0, dim, 2):
        theta = t * freqs[i // 2]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        v0 = vector[i]
        v1 = vector[i + 1]
        out[i] = v0 * cos_t - v1 * sin_t
        out[i + 1] = v0 * sin_t + v1 * cos_t
    return out


def compute_rope_dot_product(
    q: List[float],
    k: List[float],
    t_q: int,
    t_k: int,
    freqs: Optional[List[float]] = None
) -> float:
    """
    Compute attention dot product <RoPE(q, t_q), RoPE(k, t_k)>.
    By RoPE mathematical identity, this depends strictly on the relative position (t_q - t_k).
    """
    q_rot = apply_temporal_rope(q, t_q, freqs)
    k_rot = apply_temporal_rope(k, t_k, freqs)
    return sum(a * b for a, b in zip(q_rot, k_rot))


class ChunkContinuityManager:
    """
    Manages chunk overlap, blending, RoPE re-indexing, and motion carry-over.
    Guarded by ChunkContinuityConfig where every flag defaults to OFF.
    """

    def __init__(self, config: Optional[ChunkContinuityConfig] = None):
        self.config = config or ChunkContinuityConfig()

    def linear_crossfade(
        self,
        prev_tail: List[LatentFrame],
        curr_head: List[LatentFrame],
        blend_width: int
    ) -> List[LatentFrame]:
        """
        Applies linear cross-fade in the overlap region.
        w_i = (i + 1) / (blend_width + 1)
        blended[i] = (1 - w_i) * prev_tail[i] + w_i * curr_head[i]
        """
        assert len(prev_tail) >= blend_width, "Previous tail shorter than blend_width"
        assert len(curr_head) >= blend_width, "Current head shorter than blend_width"

        blended = []
        for i in range(blend_width):
            w = (i + 1.0) / (blend_width + 1.0)
            frame = (1.0 - w) * prev_tail[i] + w * curr_head[i]
            blended.append(frame)
        return blended

    def stitch_two_chunks(
        self,
        prev_chunk: List[LatentFrame],
        curr_chunk: List[LatentFrame]
    ) -> List[LatentFrame]:
        """
        Stitches previous chunk and current chunk based on configuration.
        """
        if not self.config.enable_chunk_overlap:
            # Baseline: Concatenate without overlap
            return list(prev_chunk) + list(curr_chunk)

        overlap = min(self.config.overlap_frames, len(prev_chunk), len(curr_chunk))
        if overlap <= 0:
            return list(prev_chunk) + list(curr_chunk)

        # Region before overlap from prev_chunk
        prefix = prev_chunk[:-overlap]
        prev_overlap = prev_chunk[-overlap:]
        curr_overlap = curr_chunk[:overlap]
        curr_remainder = curr_chunk[overlap:]

        if self.config.enable_linear_crossfade or self.config.overlap_mode == "blended":
            blend_w = min(self.config.blend_width, overlap)
            blended_overlap = self.linear_crossfade(prev_overlap, curr_overlap, blend_w)
            # If overlap > blend_w, remainder of overlap uses curr_overlap
            full_overlap = blended_overlap + curr_overlap[blend_w:]
            return prefix + full_overlap + curr_remainder
        else:
            # Fixed clean conditioning mode: Keep previous clean chunk's tail
            return prefix + prev_overlap + curr_remainder

    def stitch_all_chunks(
        self,
        chunks: List[List[LatentFrame]]
    ) -> Tuple[List[LatentFrame], List[int]]:
        """
        Stitches an entire sequence of chunks into a single continuous video.
        Returns:
            (stitched_frames, chunk_boundary_indices)
        """
        if not chunks:
            return [], []
        if len(chunks) == 1:
            return list(chunks[0]), []

        stitched = list(chunks[0])
        boundaries: List[int] = []

        for i in range(1, len(chunks)):
            prev = stitched
            curr = chunks[i]
            boundary_idx = len(prev)
            if self.config.enable_chunk_overlap:
                overlap = min(self.config.overlap_frames, len(prev), len(curr))
                boundary_idx = len(prev) - overlap
                stitched = self.stitch_two_chunks(prev, curr)
            else:
                boundary_idx = len(prev)
                stitched = prev + curr
            boundaries.append(boundary_idx)

        return stitched, boundaries

    def reindex_rope_positions(
        self,
        window_start_t: int,
        chunk_local_positions: List[int],
        anchor_slots: int = 0
    ) -> List[int]:
        """
        RoPE re-indexing relative to window start.
        Ensures temporal positional indices never extrapolate beyond the trained window.
        t'_i = anchor_slots + (t_i - window_start_t)
        """
        if not self.config.enable_rope_reindexing:
            # Absolute coordinates (baseline)
            return [window_start_t + pos for pos in chunk_local_positions]

        return [anchor_slots + pos for pos in chunk_local_positions]

    def extract_motion_carryover(
        self,
        prev_chunk: List[LatentFrame]
    ) -> Dict[str, Any]:
        """
        Extracts motion condition (velocity & acceleration) from the last N clean frames.
        Enables continuing motion direction into the next chunk.
        """
        if not self.config.enable_motion_carryover:
            return {"enabled": False, "velocity": None, "extrapolated": None}

        n = min(self.config.motion_carryover_frames, len(prev_chunk))
        if n < 2:
            return {"enabled": False, "velocity": None, "extrapolated": None}

        recent = prev_chunk[-n:]
        # Velocity between last two frames
        velocity = recent[-1] - recent[-2]

        # Acceleration if n >= 3
        acceleration = None
        if n >= 3:
            v_prev = recent[-2] - recent[-3]
            acceleration = velocity - v_prev

        # Extrapolated next frame
        if acceleration is not None:
            extrapolated = recent[-1] + velocity + 0.5 * acceleration
        else:
            extrapolated = recent[-1] + velocity

        return {
            "enabled": True,
            "n_frames": n,
            "last_frame": recent[-1],
            "velocity": velocity,
            "acceleration": acceleration,
            "extrapolated": extrapolated,
        }


def compute_seam_metric(
    video: List[LatentFrame],
    chunk_boundaries: List[int]
) -> Dict[str, float]:
    """
    Computes the Seam Metric:
    difference across a chunk border divided by the average adjacent-frame difference inside a chunk.

    Seam Metric = (Mean Border Diff) / (Mean Intra-Chunk Diff)
    """
    if len(video) < 2 or not chunk_boundaries:
        return {"mean_seam_metric": 1.0, "mean_border_diff": 0.0, "mean_inside_diff": 0.0}

    boundary_set = set(chunk_boundaries)

    border_diffs: List[float] = []
    inside_diffs: List[float] = []

    for t in range(1, len(video)):
        diff = video[t].diff_norm(video[t - 1])
        if t in boundary_set:
            border_diffs.append(diff)
        else:
            inside_diffs.append(diff)

    mean_border_diff = sum(border_diffs) / len(border_diffs) if border_diffs else 0.0
    mean_inside_diff = sum(inside_diffs) / len(inside_diffs) if inside_diffs else 1e-8

    if mean_inside_diff < 1e-9:
        seam_metric = 1.0
    else:
        seam_metric = mean_border_diff / mean_inside_diff

    return {
        "mean_seam_metric": seam_metric,
        "mean_border_diff": mean_border_diff,
        "mean_inside_diff": mean_inside_diff,
        "max_border_diff": max(border_diffs) if border_diffs else 0.0,
    }
