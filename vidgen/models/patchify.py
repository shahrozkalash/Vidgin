"""3D Spacetime Patchify and Unpatchify for Wan Video Latents.

Features:
- Configurable patch size (pt, ph, pw), default (1, 2, 2)
- Conv3d patch embedding: [B, C, t, h, w] -> [B, N, D]
- Linear unpatchify: [B, N, D] -> [B, C, t, h, w]
- Identity-weight exact inverse guarantees: patchify(x) -> unpatchify() == x
- Variable-length batching support:
  * Packed varlen sequences with cu_seqlens
  * Padded batches with attention masks
  * Per-sample 3D grid coordinate preservation (t, h, w)
- Extra conditioning channel support (image-to-video mask & reference latents)
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpacetimePatchEmbed(nn.Module):
    """3D Spacetime Patch Embedding.

    Projects video latents [B, C, t, h, w] into token sequence [B, N, D]
    using Conv3d with kernel_size = stride = (pt, ph, pw).
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_dim: int = 384,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        extra_channels: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.total_in_channels = in_channels + extra_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.pt, self.ph, self.pw = patch_size

        self.proj = nn.Conv3d(
            self.total_in_channels,
            hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """Forward pass.

        Args:
            x: Video latents [B, in_channels, t, h, w]
            cond: Optional extra conditioning channels [B, extra_channels, t, h, w]
                  (e.g., mask and reference latent frames for I2V)

        Returns:
            tokens: [B, N, hidden_dim]
            grid_size: (grid_t, grid_h, grid_w) where N = grid_t * grid_h * grid_w
        """
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        B, C, t, h, w = x.shape
        assert t % self.pt == 0, f"Temporal dimension ({t}) must be divisible by pt ({self.pt})"
        assert h % self.ph == 0, f"Height ({h}) must be divisible by ph ({self.ph})"
        assert w % self.pw == 0, f"Width ({w}) must be divisible by pw ({self.pw})"

        # [B, hidden_dim, grid_t, grid_h, grid_w]
        x_proj = self.proj(x)
        grid_t, grid_h, grid_w = x_proj.shape[2], x_proj.shape[3], x_proj.shape[4]

        # Flatten spacetime patches to sequence: [B, hidden_dim, N] -> [B, N, hidden_dim]
        tokens = x_proj.flatten(2).transpose(1, 2)
        return tokens, (grid_t, grid_h, grid_w)


class SpacetimeUnpatchify(nn.Module):
    """Linear Unpatchify layer mapping transformer tokens back to 3D video latents.

    Projects [B, N, hidden_dim] -> [B, N, pt*ph*pw*C] -> [B, C, t, h, w].
    """

    def __init__(
        self,
        out_channels: int = 16,
        hidden_dim: int = 384,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        bias: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.pt, self.ph, self.pw = patch_size
        self.patch_dim = out_channels * self.pt * self.ph * self.pw

        self.proj = nn.Linear(hidden_dim, self.patch_dim, bias=bias)

    def forward(
        self,
        tokens: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: [B, N, hidden_dim]
            grid_size: (grid_t, grid_h, grid_w)

        Returns:
            latents: [B, out_channels, t, h, w]
        """
        B, N, D = tokens.shape
        grid_t, grid_h, grid_w = grid_size
        assert N == grid_t * grid_h * grid_w, f"Token count ({N}) does not match grid {grid_size}"

        # Project tokens to patch flattened elements: [B, N, pt*ph*pw*C]
        x = self.proj(tokens)

        # Reshape to [B, grid_t, grid_h, grid_w, C, pt, ph, pw]
        C = self.out_channels
        pt, ph, pw = self.pt, self.ph, self.pw

        x = x.view(B, grid_t, grid_h, grid_w, C, pt, ph, pw)

        # Rearrange to continuous latents [B, C, grid_t * pt, grid_h * ph, grid_w * pw]
        # Permute: (B, C, grid_t, pt, grid_h, ph, grid_w, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        t = grid_t * pt
        h = grid_h * ph
        w = grid_w * pw

        latents = x.view(B, C, t, h, w)
        return latents


def configure_identity_patchify_unpatchify(
    in_channels: int = 16,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
) -> Tuple[SpacetimePatchEmbed, SpacetimeUnpatchify]:
    """Builds a matched PatchEmbed and Unpatchify pair with exact identity weights.

    Guarantees:
        unpatchify(patchify(x)[0], grid) == x (within machine epsilon)
    """
    pt, ph, pw = patch_size
    patch_dim = in_channels * pt * ph * pw

    # Use hidden_dim == patch_dim for shape-exact identity pass-through
    embed = SpacetimePatchEmbed(
        in_channels=in_channels,
        hidden_dim=patch_dim,
        patch_size=patch_size,
        bias=False,
    )
    unpatch = SpacetimeUnpatchify(
        out_channels=in_channels,
        hidden_dim=patch_dim,
        patch_size=patch_size,
        bias=False,
    )

    with torch.no_grad():
        # Set Conv3d weights such that token[p] = pixel[p]
        # Conv3d weight shape: [out_channels, in_channels, pt, ph, pw]
        embed.proj.weight.zero_()
        unpatch.proj.weight.zero_()

        # Wire diagonal identity
        for c in range(in_channels):
            for t_i in range(pt):
                for h_i in range(ph):
                    for w_i in range(pw):
                        d_idx = c * (pt * ph * pw) + t_i * (ph * pw) + h_i * pw + w_i
                        embed.proj.weight[d_idx, c, t_i, h_i, w_i] = 1.0
                        unpatch.proj.weight[d_idx, d_idx] = 1.0

    return embed, unpatch


# ==============================================================================
# Variable-Length Sequence Batching (Padding & Varlen Packing)
# ==============================================================================

@torch.no_grad()
def pack_varlen_tokens(
    token_list: List[torch.Tensor],
    grid_list: List[Tuple[int, int, int]],
) -> Dict[str, Any]:
    """Flattens a list of variable-size token tensors into a single packed 1D batch.

    Compatible with FlashAttention varlen and cu_seqlens.
    """
    seqlens = [t.shape[1] for t in token_list]
    cu_seqlens = [0]
    for l in seqlens:
        cu_seqlens.append(cu_seqlens[-1] + l)

    packed_tokens = torch.cat(token_list, dim=1)  # [1, sum(N_i), D]

    return {
        "packed_tokens": packed_tokens,
        "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32, device=packed_tokens.device),
        "max_seqlen": max(seqlens),
        "grid_shapes": grid_list,
        "num_samples": len(token_list),
    }


@torch.no_grad()
def pad_varlen_tokens(
    token_list: List[torch.Tensor],
    grid_list: List[Tuple[int, int, int]],
) -> Dict[str, Any]:
    """Pads variable-size token tensors to the max token length with an attention mask."""
    B = len(token_list)
    D = token_list[0].shape[-1]
    seqlens = [t.shape[1] for t in token_list]
    max_len = max(seqlens)
    device = token_list[0].device
    dtype = token_list[0].dtype

    padded_tokens = torch.zeros(B, max_len, D, device=device, dtype=dtype)
    attention_mask = torch.zeros(B, max_len, device=device, dtype=torch.bool)

    for i, t in enumerate(token_list):
        l = t.shape[1]
        padded_tokens[i, :l] = t[0]
        attention_mask[i, :l] = True

    return {
        "padded_tokens": padded_tokens,
        "attention_mask": attention_mask,  # True = valid token, False = padded
        "seqlens": seqlens,
        "max_seqlen": max_len,
        "grid_shapes": grid_list,
    }
