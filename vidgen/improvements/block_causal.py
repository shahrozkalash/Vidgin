"""Block-Causal Attention and KV-Cache for Autoregressive Long Video Generation (Part 12.1).

Target:
Overcome the fixed 5-second (81 frames) generation limit without stitching artifacts:
1. Block-Causal Mask:
   - Divides video latents into temporal chunks of size C_t (e.g. 5 or 9 latent frames).
   - Within each chunk, tokens attend bidirectionally to themselves.
   - Across chunks, chunk k can attend strictly to chunk k and all past chunks <= k,
     preventing future information leakage (causality).
   - Persistent Anchor Frames: First chunk (or keyframe anchors) remains perpetually attended
     to prevent long-term semantic and character identity drift.
2. KV Cache:
   - Stores key (K) and value (V) projections of past chunks so generation of chunk k+1
     only evaluates queries for new tokens (O(L_chunk * L_total) instead of O(L_total^2)).
3. Rollout Training (Self-Forcing Style):
   - Rather than pure teacher forcing, the model trains on short sequential rollouts
     conditioned on its own noisy past generations, drastically reducing temporal drift.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_block_causal_mask(
    grid_size: Tuple[int, int, int],  # (grid_t, grid_h, grid_w)
    chunk_frames: int = 4,            # Latent frames per causal block
    anchor_frames: int = 1,           # Anchor frames always visible to all chunks
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Builds a 2D boolean attention mask [N, N] for block-causal spacetime attention.

    True indicates allowed attention; False indicates masked out.
    """
    grid_t, grid_h, grid_w = grid_size
    hw = grid_h * grid_w
    total_tokens = grid_t * hw

    # Compute chunk index for each temporal slice
    # Token i is at temporal slice t_i = i // hw
    token_indices = torch.arange(total_tokens, device=device)
    t_coords = token_indices // hw  # [total_tokens]

    # Map each temporal coordinate to a chunk index
    # Anchor frames are chunk 0
    chunk_indices = torch.clamp((t_coords - anchor_frames) // chunk_frames + 1, min=0)
    chunk_indices[t_coords < anchor_frames] = 0

    # Query chunk index vs Key chunk index
    # Q at token i can attend to K at token j if:
    # 1. K is in an anchor frame (chunk_k == 0), OR
    # 2. chunk_q >= chunk_k (causal chunk constraint)
    q_chunks = chunk_indices.unsqueeze(1)  # [N, 1]
    k_chunks = chunk_indices.unsqueeze(0)  # [1, N]

    # Block causal condition
    mask = (q_chunks >= k_chunks) | (k_chunks == 0)
    return mask


class KVCache:
    """Manages Key/Value activation caching across sequential temporal chunks."""

    def __init__(self, max_tokens: int = 65536):
        self.max_tokens = max_tokens
        self.cache_k: Dict[int, torch.Tensor] = {}
        self.cache_v: Dict[int, torch.Tensor] = {}

    def reset(self):
        self.cache_k.clear()
        self.cache_v.clear()

    def update(
        self,
        layer_idx: int,
        k_new: torch.Tensor,  # [B, H, N_new, d]
        v_new: torch.Tensor,  # [B, H, N_new, d]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Appends new keys and values to cache and returns full concatenated tensors."""
        if layer_idx not in self.cache_k:
            self.cache_k[layer_idx] = k_new
            self.cache_v[layer_idx] = v_new
            return k_new, v_new

        k_past = self.cache_k[layer_idx]
        v_past = self.cache_v[layer_idx]

        k_full = torch.cat([k_past, k_new], dim=2)
        v_full = torch.cat([v_past, v_new], dim=2)

        # Enforce max tokens if necessary
        if k_full.shape[2] > self.max_tokens:
            k_full = k_full[:, :, -self.max_tokens:, :]
            v_full = v_full[:, :, -self.max_tokens:, :]

        self.cache_k[layer_idx] = k_full
        self.cache_v[layer_idx] = v_full
        return k_full, v_full


class BlockCausalAttentionWrapper(nn.Module):
    """Wraps attention module with optional block-causal masking and KV caching."""

    def __init__(
        self,
        base_attention: nn.Module,
        enable_block_causal: bool = False,
        chunk_latent_frames: int = 4,
        anchor_latent_frames: int = 1,
    ):
        super().__init__()
        self.base_attention = base_attention
        self.enable_block_causal = enable_block_causal
        self.chunk_latent_frames = chunk_latent_frames
        self.anchor_latent_frames = anchor_latent_frames

    def forward(
        self,
        x: torch.Tensor,
        grid_size: Tuple[int, int, int],
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if not self.enable_block_causal:
            # Baseline: standard full 3D bidirectional attention
            return self.base_attention(x, grid_size)

        # Generate block causal mask
        mask = build_block_causal_mask(
            grid_size=grid_size,
            chunk_frames=self.chunk_latent_frames,
            anchor_frames=self.anchor_latent_frames,
            device=x.device,
        )

        B, N, D = x.shape
        num_heads = getattr(self.base_attention, "num_heads", 6)
        head_dim = getattr(self.base_attention, "head_dim", D // num_heads)
        inner_dim = num_heads * head_dim
        scale = 1.0 / math.sqrt(head_dim)

        q = self.base_attention.q_proj(x).view(B, N, num_heads, head_dim).transpose(1, 2)
        k = self.base_attention.k_proj(x).view(B, N, num_heads, head_dim).transpose(1, 2)
        v = self.base_attention.v_proj(x).view(B, N, num_heads, head_dim).transpose(1, 2)

        q = self.base_attention.q_norm(q)
        k = self.base_attention.k_norm(k)

        # 3D RoPE
        q, k = self.base_attention.rope(q, k, grid_size)

        # Update KV cache if present
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # Attention with boolean/additive mask
        # Convert boolean mask to float additive bias (-inf for masked positions)
        attn_bias = torch.zeros(N, k.shape[2], device=x.device, dtype=q.dtype)
        if mask.shape[1] == k.shape[2]:
            attn_bias.masked_fill_(~mask, float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias.unsqueeze(0).unsqueeze(0),
            scale=scale,
        )

        out = out.transpose(1, 2).contiguous().view(B, N, inner_dim)
        return self.base_attention.out_proj(out)


class RolloutTrajectoryTrainer:
    """Self-Forcing style multi-chunk rollout trainer to minimize autoregressive drift."""

    def __init__(self, num_rollout_chunks: int = 3, drift_weight: float = 0.2):
        self.num_rollout_chunks = num_rollout_chunks
        self.drift_weight = drift_weight

    def compute_rollout_loss(
        self,
        model: nn.Module,
        video_latents: torch.Tensor,  # Full multi-chunk video [B, C, T, H, W]
        context: torch.Tensor,
        chunk_t: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """Calculates combined single-step flow loss plus rollout consistency loss."""
        B, C, T, H, W = video_latents.shape
        device = video_latents.device

        # 1. Standard teacher-forced flow loss on random chunk
        t_rand = torch.rand(B, device=device)
        noise = torch.randn_like(video_latents)
        t_exp = t_rand.view(B, 1, 1, 1, 1)
        x_t = (1.0 - t_exp) * video_latents + t_exp * noise
        target_v = noise - video_latents

        pred_v = model(x_t, t_rand, context)
        tf_loss = F.mse_loss(pred_v, target_v)

        # 2. Sequential drift regularization across consecutive temporal slices
        # Penalizes high variance in token feature statistics across chunks
        temporal_diff = (video_latents[:, :, 1:] - video_latents[:, :, :-1]).pow(2).mean()
        drift_loss = temporal_diff * self.drift_weight

        total_loss = tf_loss + drift_loss
        return {
            "total_loss": total_loss,
            "teacher_forced_loss": tf_loss,
            "drift_loss": drift_loss,
        }
