"""Longer Video Generation via Chunked Overlap & Autoregressive Extension (Part 11).

Goal:
Overcome context window limits (e.g. 81 frames / ~33k tokens) to generate arbitrarily long videos
(e.g., 161, 241, 481+ frames) without temporal boundary flickering:
1. Overlapping Chunked Generation:
   - Partition full video timeline into overlapping temporal windows.
   - Example: Chunk size 81 frames, stride 65 frames -> 16 frames overlap (4 latent frames).
2. Latent Blend Modes:
   - Cosine temporal cross-fade: w(t) = cos^2(pi/2 * t / L) ensures C^1 continuity across boundaries.
   - Linear ramp blending.
3. Autoregressive I2V Chaining:
   - Generate chunk 0 via standard text-to-video.
   - For chunk k+1, extract the last frame of chunk k and pass it as the conditioning frame
     to the Image-to-Video DiT (Part 11 extension 2).
"""

from __future__ import annotations
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class ChunkedOverlapGenerator:
    """Manages long-form video generation by stitching temporal latent chunks with smooth blending."""

    def __init__(
        self,
        chunk_latent_frames: int = 21,  # e.g., 81 pixel frames = 21 latent frames (T = 4k + 1)
        overlap_latent_frames: int = 4,   # 16 pixel frames = 4 latent frames
        blend_mode: str = "cosine",       # "cosine", "linear", or "autoregressive"
    ):
        assert overlap_latent_frames < chunk_latent_frames, "Overlap must be smaller than chunk size"
        self.chunk_latent_frames = chunk_latent_frames
        self.overlap_latent_frames = overlap_latent_frames
        self.stride = chunk_latent_frames - overlap_latent_frames
        self.blend_mode = blend_mode

    def calculate_chunks_plan(self, total_latent_frames: int) -> List[Tuple[int, int]]:
        """Calculates (start_idx, end_idx) for each chunk along temporal dimension."""
        chunks = []
        start = 0
        while start < total_latent_frames:
            end = min(start + self.chunk_latent_frames, total_latent_frames)
            chunks.append((start, end))
            if end == total_latent_frames:
                break
            start += self.stride
        return chunks

    def compute_temporal_weights(self, overlap_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates blending weights for overlapping boundary region.

        Returns:
            w_fade_out: Weight going from 1.0 down to 0.0 for tail of chunk k [overlap_len]
            w_fade_in: Weight going from 0.0 up to 1.0 for head of chunk k+1 [overlap_len]
        """
        t = torch.linspace(0.0, 1.0, overlap_len, device=device)
        if self.blend_mode == "cosine":
            # Cosine squared smooth cross-fade: cos^2(pi/2 * t) + sin^2(pi/2 * t) = 1.0 exactly!
            w_fade_out = torch.cos(0.5 * math.pi * t) ** 2
            w_fade_in = torch.sin(0.5 * math.pi * t) ** 2
        else:
            # Linear ramp
            w_fade_out = 1.0 - t
            w_fade_in = t

        return w_fade_out, w_fade_in

    def stitch_latent_chunks(
        self,
        chunks: List[torch.Tensor],  # Each of shape [B, C, t_chunk, H, W]
        total_latent_frames: int,
    ) -> torch.Tensor:
        """Stitches overlapping latent chunks with seamless temporal weighting."""
        assert len(chunks) > 0, "No chunks provided"
        B, C, _, H, W = chunks[0].shape
        device = chunks[0].device
        dtype = chunks[0].dtype

        # Accumulator tensor and weight accumulator
        stitched = torch.zeros(B, C, total_latent_frames, H, W, device=device, dtype=dtype)
        weight_acc = torch.zeros(1, 1, total_latent_frames, 1, 1, device=device, dtype=dtype)

        chunks_plan = self.calculate_chunks_plan(total_latent_frames)

        for (start, end), chunk in zip(chunks_plan, chunks):
            actual_len = end - start
            chunk_slice = chunk[:, :, :actual_len, :, :]

            # Window weights for this chunk
            w = torch.ones(1, 1, actual_len, 1, 1, device=device, dtype=dtype)

            # Fade in head if not the very first chunk
            if start > 0 and self.overlap_latent_frames > 0:
                head_len = min(self.overlap_latent_frames, actual_len)
                _, w_in = self.compute_temporal_weights(head_len, device)
                w[:, :, :head_len, :, :] *= w_in.view(1, 1, head_len, 1, 1)

            # Fade out tail if not the very last chunk
            if end < total_latent_frames and self.overlap_latent_frames > 0:
                tail_len = min(self.overlap_latent_frames, actual_len)
                w_out, _ = self.compute_temporal_weights(tail_len, device)
                w[:, :, -tail_len:, :, :] *= w_out.view(1, 1, tail_len, 1, 1)

            stitched[:, :, start:end, :, :] += chunk_slice * w
            weight_acc[:, :, start:end, :, :] += w

        # Normalize by accumulated weights to guarantee unit sum everywhere
        weight_acc = torch.clamp(weight_acc, min=1e-5)
        stitched = stitched / weight_acc
        return stitched
