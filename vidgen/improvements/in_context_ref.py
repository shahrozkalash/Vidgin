"""In-Context Reference & Multi-Shot Video Conditioning (Part 12.5).

Target:
Overcome Wan's lack of reference conditioning and character drift across multi-shot sequences:
1. In-Context Reference Conditioning:
   - Rather than training heavy external cross-attention adapters (e.g. IP-Adapter),
     concatenate reference frame tokens directly into the input token sequence!
   - Assign reference tokens a distinct RoPE temporal index (e.g. t_ref = -1 or anchor position).
   - The same 3D self-attention transformer naturally attends to reference features across all layers.
2. Multi-Shot Boundary Tokens:
   - Insert special learned boundary tokens between consecutive narrative shots.
   - Resets temporal position embeddings while retaining character semantics in self-attention.
3. Identity Similarity Evaluation:
   - Computes cosine similarity between reference image token representations and generated video tokens.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class InContextReferenceDiT(nn.Module):
    """Wrapper that prepends reference image tokens with distinct RoPE time offsets."""

    def __init__(
        self,
        base_dit: nn.Module,
        hidden_dim: int = 384,
        enable_multi_shot_tokens: bool = True,
    ):
        super().__init__()
        self.base_dit = base_dit
        self.hidden_dim = hidden_dim
        self.enable_multi_shot_tokens = enable_multi_shot_tokens

        # Learned shot-boundary transition token
        if enable_multi_shot_tokens:
            self.shot_boundary_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        else:
            self.shot_boundary_token = None

    def prepend_reference_tokens(
        self,
        video_tokens: torch.Tensor,     # [B, N_video, D]
        reference_tokens: torch.Tensor, # [B, N_ref, D]
        insert_boundary: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        """Concatenates reference tokens at the head of the sequence.

        Returns:
            combined_tokens: [B, N_ref + (1) + N_video, D]
            ref_length: Number of conditioning prefix tokens
        """
        B, _, D = video_tokens.shape
        tokens_to_cat = [reference_tokens]

        if insert_boundary and self.shot_boundary_token is not None:
            boundary = self.shot_boundary_token.expand(B, 1, D)
            tokens_to_cat.append(boundary)

        tokens_to_cat.append(video_tokens)
        combined = torch.cat(tokens_to_cat, dim=1)
        ref_len = reference_tokens.shape[1] + (1 if insert_boundary and self.shot_boundary_token is not None else 0)
        return combined, ref_len

    @staticmethod
    def calculate_identity_similarity(
        ref_tokens: torch.Tensor,   # [B, N_ref, D]
        gen_tokens: torch.Tensor,   # [B, N_video, D]
    ) -> float:
        """Calculates cosine similarity between reference features and generated features."""
        # Mean pool spatially
        ref_mean = F.normalize(ref_tokens.mean(dim=1), dim=-1)  # [B, D]
        gen_mean = F.normalize(gen_tokens.mean(dim=1), dim=-1)  # [B, D]
        sim = (ref_mean * gen_mean).sum(dim=-1).mean().item()
        return float(sim)
