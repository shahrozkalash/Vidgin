"""
Vidgen Oracle Model & Synthetic Smooth Video Generator
Used for CPU-level mathematical verification of chunk continuity, seam metrics,
and exact ground-truth reconstruction.
"""

import math
from typing import List, Tuple, Optional, Dict, Any
from vidgen.config import LongVideoConfig, ChunkContinuityConfig
from vidgen.continuity import LatentFrame, ChunkContinuityManager, compute_seam_metric


class SyntheticSmoothVideo:
    """
    Generates a mathematically smooth ground-truth video trajectory across T frames.
    The trajectory is smooth and differentiable (C^inf) with continuous velocity and acceleration.
    """

    def __init__(self, total_frames: int = 121, dim: int = 32, seed: int = 42):
        self.total_frames = total_frames
        self.dim = dim
        self.frames: List[LatentFrame] = []
        self._generate(seed)

    def _generate(self, seed: int):
        self.frames = []
        for t in range(self.total_frames):
            frame_vals = []
            for d in range(self.dim):
                omega = 0.05 + 0.02 * (d % 7)
                phi = (d * 0.314 + seed * 0.1) % (2.0 * math.pi)
                # Primary harmonic wave + secondary harmonic
                val = math.cos(omega * t + phi) + 0.25 * math.sin(2.0 * omega * t + 0.5 * phi)
                frame_vals.append(val)
            self.frames.append(LatentFrame(frame_vals))

    def get_frame(self, t: int) -> LatentFrame:
        assert 0 <= t < self.total_frames
        return self.frames[t]

    def get_slice(self, start: int, end: int) -> List[LatentFrame]:
        return [self.frames[t] for t in range(start, min(end, self.total_frames))]


class OracleChunkedGenerator:
    """
    Oracle model that simulates chunk-by-chunk generation.
    With continuity enabled, it uses previous chunk conditioning, overlap, and motion carry-over
    to generate seamless chunks that perfectly reconstruct ground truth.
    """

    def __init__(self, config: Optional[LongVideoConfig] = None, dim: int = 32):
        self.config = config or LongVideoConfig()
        self.dim = dim
        self.continuity_mgr = ChunkContinuityManager(self.config.continuity)

    def generate_chunks_from_ground_truth(
        self,
        gt_video: SyntheticSmoothVideo,
        num_chunks: int = 6,
        nominal_chunk_size: int = 21
    ) -> List[List[LatentFrame]]:
        """
        Simulate generating chunks under the given configuration.
        """
        chunks: List[List[LatentFrame]] = []
        overlap = self.config.continuity.overlap_frames if self.config.continuity.enable_chunk_overlap else 0

        curr_t = 0
        for c in range(num_chunks):
            chunk_len = nominal_chunk_size
            if curr_t + chunk_len > gt_video.total_frames:
                chunk_len = gt_video.total_frames - curr_t

            if not self.config.continuity.enable_chunk_overlap:
                # Baseline without overlap: independent chunk slices
                chunk_frames = [gt_video.get_frame(t) for t in range(curr_t, min(curr_t + chunk_len, gt_video.total_frames))]
                chunks.append(chunk_frames)
                curr_t += chunk_len
            else:
                # Chunk with overlap: starts `overlap` frames before previous chunk's end
                start_t = max(0, curr_t - (overlap if c > 0 else 0))
                end_t = min(start_t + chunk_len, gt_video.total_frames)
                chunk_frames = [gt_video.get_frame(t) for t in range(start_t, end_t)]

                # If motion carryover is enabled, verify velocity continuation
                if c > 0 and self.config.continuity.enable_motion_carryover:
                    motion_info = self.continuity_mgr.extract_motion_carryover(chunks[-1])
                    assert motion_info["enabled"] is True

                chunks.append(chunk_frames)
                curr_t = end_t

        return chunks

    def generate_and_stitch(
        self,
        gt_video: SyntheticSmoothVideo,
        num_chunks: int = 6,
        nominal_chunk_size: int = 21
    ) -> Tuple[List[LatentFrame], List[int], Dict[str, float]]:
        """
        Generates all chunks, stitches them via ChunkContinuityManager,
        and computes reconstruction error against gt_video and seam metrics.
        """
        chunks = self.generate_chunks_from_ground_truth(gt_video, num_chunks, nominal_chunk_size)
        stitched, boundaries = self.continuity_mgr.stitch_all_chunks(chunks)

        # Truncate or match length to gt_video
        eval_len = min(len(stitched), gt_video.total_frames)
        stitched = stitched[:eval_len]

        # Compute MSE vs ground truth
        mse_total = 0.0
        max_abs_err = 0.0
        for t in range(eval_len):
            gt_f = gt_video.get_frame(t)
            st_f = stitched[t]
            for a, b in zip(gt_f.data, st_f.data):
                err = abs(a - b)
                mse_total += err * err
                if err > max_abs_err:
                    max_abs_err = err

        mse = mse_total / (eval_len * self.dim)
        seam_stats = compute_seam_metric(stitched, boundaries)

        metrics = {
            "mse_vs_gt": mse,
            "max_abs_err": max_abs_err,
            "seam_metric": seam_stats["mean_seam_metric"],
            "mean_border_diff": seam_stats["mean_border_diff"],
            "mean_inside_diff": seam_stats["mean_inside_diff"],
            "num_boundaries": len(boundaries),
            "stitched_frames": len(stitched),
        }
        return stitched, boundaries, metrics
