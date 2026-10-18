"""3D Rotary Position Embedding (3D RoPE) for Spacetime Attention.

Splits head_dim into three parts (time, height, width) and rotates each
component along its respective spatial/temporal axis.

Wan 3D RoPE specifications:
- head_dim = 128 -> (dt, dh, dw) = (44, 42, 42)
- head_dim = 64  -> (dt, dh, dw) = (22, 22, 20)
- Precomputes and caches frequency tables for sub-millisecond execution.
- Provably relative-position invariant:
    score(q @ pos1, k @ pos2) == score(q @ (pos1 + delta), k @ (pos2 + delta))
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict
import math
import torch
import torch.nn as nn


def partition_head_dim(head_dim: int) -> Tuple[int, int, int]:
    """Partitions head_dim into (d_t, d_h, d_w) with all even components.

    Wan standard uses 44/42/42 for head_dim=128.
    For head_dim=64, uses 22/22/20.
    """
    if head_dim == 128:
        return (44, 42, 42)
    elif head_dim == 64:
        return (22, 22, 20)
    elif head_dim == 96:
        return (32, 32, 32)

    # General partition: roughly 1/3 each, all even
    base = (head_dim // 3) // 2 * 2
    rem = head_dim - 2 * base
    assert rem % 2 == 0, f"head_dim ({head_dim}) must be divisible by 2"
    return (base + (rem - base), base, base) if rem >= base else (base, base, rem)


def precompute_freqs_1d(dim: int, max_pos: int = 1024, theta: float = 10000.0) -> torch.Tensor:
    """Precompute 1D rotary frequencies: [max_pos, dim // 2]."""
    assert dim % 2 == 0, f"Dimension {dim} must be even for rotary pairs"
    half_dim = dim // 2
    freq_seq = torch.arange(half_dim, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (freq_seq / half_dim))
    pos = torch.arange(max_pos, dtype=torch.float32)
    # Outer product: [max_pos, dim // 2]
    freqs = torch.outer(pos, inv_freq)
    return freqs


def apply_rotary_emb_1d(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Applies 1D rotary embedding to tensor.

    Args:
        x: [B, num_heads, N, dim] or [B, N, dim]
        freqs: [N, dim // 2]
    """
    orig_shape = x.shape
    dim = orig_shape[-1]
    # Reshape x to pairs [..., dim // 2, 2]
    x_pairs = x.view(*orig_shape[:-1], dim // 2, 2)
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    # Broadcast freqs to match x batch and head dims
    while cos.ndim < x1.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    # Givens rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    out = torch.stack([out1, out2], dim=-1).flatten(-2)
    return out


class RotaryEmbedding3D(nn.Module):
    """3D Spacetime Rotary Position Embedding (3D RoPE).

    Partition head_dim into (d_t, d_h, d_w), looks up precomputed freqs
    along each axis for spacetime tokens, and rotates query and key representations.
    """

    def __init__(
        self,
        head_dim: int = 128,
        theta: float = 10000.0,
        max_t: int = 256,
        max_h: int = 256,
        max_w: int = 256,
        custom_partition: Optional[Tuple[int, int, int]] = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.dt, self.dh, self.dw = custom_partition or partition_head_dim(head_dim)
        assert self.dt + self.dh + self.dw == head_dim, (
            f"Partition ({self.dt}, {self.dh}, {self.dw}) does not sum to head_dim ({head_dim})"
        )

        # Precompute frequencies for each axis
        self.register_buffer("freqs_t", precompute_freqs_1d(self.dt, max_pos=max_t, theta=theta), persistent=False)
        self.register_buffer("freqs_h", precompute_freqs_1d(self.dh, max_pos=max_h, theta=theta), persistent=False)
        self.register_buffer("freqs_w", precompute_freqs_1d(self.dw, max_pos=max_w, theta=theta), persistent=False)

    def get_spacetime_freqs(
        self,
        grid_size: Tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Construct concatenated 3D rotary frequency table for grid (grid_t, grid_h, grid_w).

        Returns:
            freqs: [N, head_dim // 2]
        """
        grid_t, grid_h, grid_w = grid_size
        N = grid_t * grid_h * grid_w

        # Coordinate meshgrid
        t_coords = torch.arange(grid_t, device=device)
        h_coords = torch.arange(grid_h, device=device)
        w_coords = torch.arange(grid_w, device=device)

        # Meshgrid: [grid_t, grid_h, grid_w]
        grid_t_mesh, grid_h_mesh, grid_w_mesh = torch.meshgrid(t_coords, h_coords, w_coords, indexing="ij")

        flat_t = grid_t_mesh.reshape(-1)
        flat_h = grid_h_mesh.reshape(-1)
        flat_w = grid_w_mesh.reshape(-1)

        f_t = self.freqs_t.to(device)[flat_t]  # [N, dt // 2]
        f_h = self.freqs_h.to(device)[flat_h]  # [N, dh // 2]
        f_w = self.freqs_w.to(device)[flat_w]  # [N, dw // 2]

        # Concatenate along frequency dimension: [N, (dt + dh + dw) // 2]
        freqs_3d = torch.cat([f_t, f_h, f_w], dim=-1)
        return freqs_3d

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 3D RoPE to query and key tensors.

        Args:
            q: [B, num_heads, N, head_dim] or [B, N, num_heads, head_dim]
            k: [B, num_heads, N, head_dim] or [B, N, num_heads, head_dim]
            grid_size: (grid_t, grid_h, grid_w) where N = grid_t * grid_h * grid_w

        Returns:
            q_rotated, k_rotated with exact 3D spatial & temporal position encoded.
        """
        freqs = self.get_spacetime_freqs(grid_size, device=q.device)

        # Check tensor layout: transpose if necessary so sequence N is 2nd to last
        is_transposed = False
        if q.ndim == 4 and q.shape[1] != freqs.shape[0] and q.shape[2] == freqs.shape[0]:
            # Layout [B, num_heads, N, head_dim] -> N is at dim 2
            pass
        elif q.ndim == 4 and q.shape[1] == freqs.shape[0]:
            # Layout [B, N, num_heads, head_dim] -> transpose to [B, num_heads, N, head_dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            is_transposed = True

        # Split q and k into (dt, dh, dw) parts
        q_t, q_h, q_w = torch.split(q, [self.dt, self.dh, self.dw], dim=-1)
        k_t, k_h, k_w = torch.split(k, [self.dt, self.dh, self.dw], dim=-1)

        f_t, f_h, f_w = torch.split(freqs, [self.dt // 2, self.dh // 2, self.dw // 2], dim=-1)

        q_t_rot = apply_rotary_emb_1d(q_t, f_t)
        k_t_rot = apply_rotary_emb_1d(k_t, f_t)

        q_h_rot = apply_rotary_emb_1d(q_h, f_h)
        k_h_rot = apply_rotary_emb_1d(k_h, f_h)

        q_w_rot = apply_rotary_emb_1d(q_w, f_w)
        k_w_rot = apply_rotary_emb_1d(k_w, f_w)

        q_out = torch.cat([q_t_rot, q_h_rot, q_w_rot], dim=-1)
        k_out = torch.cat([k_t_rot, k_h_rot, k_w_rot], dim=-1)

        if is_transposed:
            q_out = q_out.transpose(1, 2)
            k_out = k_out.transpose(1, 2)

        return q_out, k_out
