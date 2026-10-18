"""Wan-style Diffusion Transformer (DiT) with Full 3D Spacetime Attention and 3D RoPE.

Architecture Highlights:
- Modular DiT backbone consisting of N identical Wan-style blocks.
- Block structure:
    1. Full 3D spacetime self-attention with 3D RoPE and RMSNorm on Q and K,
       adaLN-modulated input (scale & shift) and gated residual add.
    2. Cross-attention to text (Part 5) with a plain LayerNorm, zero modulation.
    3. Feed-Forward network (GELU, tanh approximation) with adaLN modulation
       and gated residual add.
- Timestep conditioning:
    Sinusoidal embedding (dim 256) -> MLP -> shared projection to 6*D modulation values.
    Each block adds its own learnable modulation table to the shared vector to save parameters.
- Output head:
    Final adaLN layer -> Linear to patch dimension -> Spacetime unpatchify to [B, C, t, h, w].
- Initializations:
    Weights init with std 0.02, zero-init gates and final output layer (adaLN-Zero style).
- FlashAttention / PyTorch SDPA and per-block activation checkpointing support.
- Optional cheaper ablation: factorized spatial & temporal attention comparison.
"""

from __future__ import annotations
import math
from typing import Tuple, List, Dict, Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from vidgen.models.patchify import SpacetimePatchEmbed, SpacetimeUnpatchify
from vidgen.models.rope import RotaryEmbedding3D, partition_head_dim
from vidgen.models.text import RMSNorm, TextCrossAttention, TextProjection


# ==============================================================================
# 1. Sinusoidal Timestep Embeddings
# ==============================================================================

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int = 256,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: 1D Tensor of shape [B] with timestep values (e.g. in [0, 1] or [0, 1000]).
        embedding_dim: Dimension of output embedding (default: 256).
        max_period: Maximum frequency period.

    Returns:
        embeddings: [B, embedding_dim] float tensor.
    """
    assert timesteps.ndim == 1, f"timesteps must be 1D, got {timesteps.shape}"

    # If timesteps are in [0, 1] (flow matching convention), scale to standard diffusion range
    if timesteps.dtype.is_floating_point and timesteps.max() <= 1.0 + 1e-4:
        scaled_t = timesteps * 1000.0
    else:
        scaled_t = timesteps.float()

    half = embedding_dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = scaled_t[:, None] * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if embedding_dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into modulation vectors using sinusoidal encoding and MLP."""

    def __init__(self, time_dim: int = 256, hidden_dim: int = 384):
        super().__init__()
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Shared projection to 6 * D modulation values (adaLN)
        self.shared_proj = nn.Linear(hidden_dim, 6 * hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Standard normal for shared projection
        nn.init.normal_(self.shared_proj.weight, std=0.02)
        nn.init.zeros_(self.shared_proj.bias)

    def forward(self, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute shared modulation vector and hidden time embedding.

        Args:
            timesteps: [B]

        Returns:
            shared_mod: [B, 6 * hidden_dim]
            t_hidden: [B, hidden_dim]
        """
        t_emb = get_timestep_embedding(timesteps, embedding_dim=self.time_dim)
        t_hidden = self.mlp(t_emb)
        shared_mod = self.shared_proj(t_hidden)
        return shared_mod, t_hidden


# ==============================================================================
# 2. Feed-Forward Network with GELU (Tanh Approximation)
# ==============================================================================

class FeedForward(nn.Module):
    """Feed-forward network with GELU tanh approximation."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = self.drop(h)
        return self.fc2(h)


# ==============================================================================
# 3. Full 3D Spacetime Self-Attention with 3D RoPE and QK-RMSNorm
# ==============================================================================

class Full3DSpacetimeAttention(nn.Module):
    """Full 3D spacetime self-attention over all spacetime tokens with 3D RoPE and QK-RMSNorm.

    Features:
    - O(N^2) complete spacetime token interactions
    - QK-RMSNorm on query and key heads for numerical stability
    - 3D RoPE encoding relative spacetime coordinates (t, h, w)
    - PyTorch SDPA (scaled_dot_product_attention) with automatic FlashAttention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        head_dim: Optional[int] = None,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout_p = dropout

        self.q_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)

        # QK-RMSNorm per head
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # 3D Rotary Position Embedding
        self.rope = RotaryEmbedding3D(head_dim=self.head_dim, theta=rope_theta)

        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(proj.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Video tokens [B, N, dim]
            grid_size: (grid_t, grid_h, grid_w) where N = grid_t * grid_h * grid_w

        Returns:
            out: [B, N, dim]
        """
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply QK-RMSNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply 3D RoPE
        q, k = self.rope(q, k, grid_size)

        # SDPA (FlashAttention / math kernel)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            scale=self.scale,
        )

        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_dim)
        return self.out_proj(out)


# ==============================================================================
# 4. Optional Factorized Spatial & Temporal Attention (Ablation Comparison)
# ==============================================================================

class FactorizedSpacetimeAttention(nn.Module):
    """Optional cheaper ablation: separate spatial attention and temporal attention.

    Used for ablation comparisons against full 3D spacetime attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Spatial attention
        self.s_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.s_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.s_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.s_out = nn.Linear(self.inner_dim, dim, bias=False)
        self.s_q_norm = RMSNorm(self.head_dim)
        self.s_k_norm = RMSNorm(self.head_dim)

        # Temporal attention
        self.t_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.t_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.t_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.t_out = nn.Linear(self.inner_dim, dim, bias=False)
        self.t_q_norm = RMSNorm(self.head_dim)
        self.t_k_norm = RMSNorm(self.head_dim)

        self._init_weights()

    def _init_weights(self):
        for p in [self.s_q, self.s_k, self.s_v, self.s_out, self.t_q, self.t_k, self.t_v, self.t_out]:
            nn.init.normal_(p.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        B, N, D = x.shape
        grid_t, grid_h, grid_w = grid_size
        hw = grid_h * grid_w

        # Reshape to [B * grid_t, hw, D] for spatial attention
        x_space = x.view(B, grid_t, hw, D).view(B * grid_t, hw, D)
        sq = self.s_q_norm(self.s_q(x_space).view(B * grid_t, hw, self.num_heads, self.head_dim).transpose(1, 2))
        sk = self.s_k_norm(self.s_k(x_space).view(B * grid_t, hw, self.num_heads, self.head_dim).transpose(1, 2))
        sv = self.s_v(x_space).view(B * grid_t, hw, self.num_heads, self.head_dim).transpose(1, 2)
        s_out = F.scaled_dot_product_attention(sq, sk, sv, scale=self.scale)
        s_out = self.s_out(s_out.transpose(1, 2).contiguous().view(B * grid_t, hw, self.inner_dim))
        x_space = (x_space + s_out).view(B, grid_t, hw, D)

        # Reshape to [B * hw, grid_t, D] for temporal attention
        x_time = x_space.permute(0, 2, 1, 3).contiguous().view(B * hw, grid_t, D)
        tq = self.t_q_norm(self.t_q(x_time).view(B * hw, grid_t, self.num_heads, self.head_dim).transpose(1, 2))
        tk = self.t_k_norm(self.t_k(x_time).view(B * hw, grid_t, self.num_heads, self.head_dim).transpose(1, 2))
        tv = self.t_v(x_time).view(B * hw, grid_t, self.num_heads, self.head_dim).transpose(1, 2)
        t_out = F.scaled_dot_product_attention(tq, tk, tv, scale=self.scale)
        t_out = self.t_out(t_out.transpose(1, 2).contiguous().view(B * hw, grid_t, self.inner_dim))
        x_time = (x_time + t_out).view(B, hw, grid_t, D).permute(0, 2, 1, 3).contiguous()

        return x_time.view(B, N, D)


# ==============================================================================
# 5. Wan DiT Transformer Block
# ==============================================================================

class WanDiTBlock(nn.Module):
    """Single Wan-style Diffusion Transformer block.

    Execution Flow:
    1. adaLN-modulated full 3D self-attention with 3D RoPE & RMSNorm on Q and K,
       followed by gated residual add.
    2. Plain LayerNorm followed by cross-attention to text embeddings (Part 5),
       with zero modulation, followed by residual add.
    3. adaLN-modulated FFN (GELU, tanh approximation), followed by gated residual add.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 6,
        head_dim: Optional[int] = None,
        ffn_dim: Optional[int] = None,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        attention_mode: str = "full_3d",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_dim // num_heads)
        self.ffn_dim = ffn_dim or (4 * hidden_dim)
        self.attention_mode = attention_mode

        # 1. Self-Attention components
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        if attention_mode == "factorized":
            self.self_attn = FactorizedSpacetimeAttention(
                dim=hidden_dim,
                num_heads=num_heads,
                head_dim=self.head_dim,
                dropout=dropout,
            )
        else:
            self.self_attn = Full3DSpacetimeAttention(
                dim=hidden_dim,
                num_heads=num_heads,
                head_dim=self.head_dim,
                rope_theta=rope_theta,
                dropout=dropout,
            )

        # 2. Text Cross-Attention components (Part 5)
        self.norm_cross = nn.LayerNorm(hidden_dim, elementwise_affine=True, eps=1e-6)
        self.cross_attn = TextCrossAttention(
            model_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=self.head_dim,
            dropout=dropout,
        )

        # 3. Feed-Forward components
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.ffn = FeedForward(hidden_dim, self.ffn_dim, dropout=dropout)

        # Learnable block modulation table: [6 * hidden_dim]
        # Adds block-specific offset to the shared timestep vector to save parameters
        self.block_mod_table = nn.Parameter(torch.zeros(6 * hidden_dim))

        # Zero-initialize the gates for adaLN-Zero identity pass-through at init
        self._init_adaLN_zero()

    def _init_adaLN_zero(self):
        # block_mod_table initialized to zeros
        nn.init.zeros_(self.block_mod_table)

    def forward(
        self,
        x: torch.Tensor,
        shared_mod: torch.Tensor,
        context: torch.Tensor,
        grid_size: Tuple[int, int, int],
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through Wan DiT block.

        Args:
            x: Video tokens [B, N, hidden_dim]
            shared_mod: Shared timestep modulation [B, 6 * hidden_dim]
            context: Text tokens [B, L, hidden_dim]
            grid_size: (grid_t, grid_h, grid_w)
            text_mask: Optional text padding mask [B, L]

        Returns:
            x: Updated video tokens [B, N, hidden_dim]
        """
        # Combine shared modulation with block-specific learnable table
        mod = shared_mod + self.block_mod_table.unsqueeze(0)  # [B, 6 * hidden_dim]
        scale1, shift1, gate1, scale2, shift2, gate2 = mod.chunk(6, dim=-1)

        # Reshape for broadcasting over token dimension N: [B, 1, hidden_dim]
        scale1 = scale1.unsqueeze(1)
        shift1 = shift1.unsqueeze(1)
        gate1 = gate1.unsqueeze(1)
        scale2 = scale2.unsqueeze(1)
        shift2 = shift2.unsqueeze(1)
        gate2 = gate2.unsqueeze(1)

        # 1. Self-Attention with adaLN modulation and gated residual
        x_norm1 = self.norm1(x) * (1.0 + scale1) + shift1
        attn_out = self.self_attn(x_norm1, grid_size=grid_size)
        x = x + gate1 * attn_out

        # 2. Text Cross-Attention with plain LayerNorm (NO modulation)
        x_norm_cross = self.norm_cross(x)
        cross_out = self.cross_attn(x_norm_cross, context=context, mask=text_mask)
        x = x + cross_out

        # 3. Feed-Forward with adaLN modulation and gated residual
        x_norm2 = self.norm2(x) * (1.0 + scale2) + shift2
        ffn_out = self.ffn(x_norm2)
        x = x + gate2 * ffn_out

        return x


# ==============================================================================
# 6. Wan DiT Backbone (Full Model)
# ==============================================================================

class WanDiT(nn.Module):
    """Wan-style Spacetime Diffusion Transformer Backbone.

    Pipeline:
    1. Patchify latents: [B, C, t, h, w] -> [B, N, D]
    2. Timestep conditioning: t -> [B, 6*D] shared modulation
    3. Text projection: [B, L, text_dim] -> [B, L, D] via Linear+GELU+Linear
    4. N repeated WanDiTBlock layers with full 3D attention & 3D RoPE
    5. Output head: final adaLN -> Linear(D to pt*ph*pw*C) -> Unpatchify to [B, C, t, h, w]
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_layers: int = 8,
        hidden_dim: int = 384,
        num_heads: int = 6,
        head_dim: Optional[int] = None,
        ffn_dim: Optional[int] = None,
        text_dim: int = 384,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        time_dim: int = 256,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        attention_mode: str = "full_3d",
        enable_checkpointing: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_dim // num_heads)
        self.ffn_dim = ffn_dim or (4 * hidden_dim)
        self.patch_size = patch_size
        self.time_dim = time_dim
        self.text_dim = text_dim
        self.attention_mode = attention_mode
        self.enable_checkpointing = enable_checkpointing

        # 1. 3D Spacetime Patch Embedding
        self.patch_embed = SpacetimePatchEmbed(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
        )

        # 2. Timestep Conditioning MLP with Shared 6*D Projection
        self.time_embedder = TimestepEmbedder(
            time_dim=time_dim,
            hidden_dim=hidden_dim,
        )

        # 3. Text Projector: Linear(text_dim, D) -> GELU -> Linear(D, D)
        self.text_projector = TextProjection(
            text_dim=text_dim,
            model_dim=hidden_dim,
        )

        # 4. Backbone Blocks (repeated N times)
        self.blocks = nn.ModuleList([
            WanDiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=self.head_dim,
                ffn_dim=self.ffn_dim,
                rope_theta=rope_theta,
                dropout=dropout,
                attention_mode=attention_mode,
            )
            for _ in range(num_layers)
        ])

        # 5. Output Head: Final adaLN + Linear to patch dimension
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_mod_proj = nn.Linear(hidden_dim, 2 * hidden_dim)

        pt, ph, pw = patch_size
        self.patch_elements = out_channels * pt * ph * pw
        self.final_linear = nn.Linear(hidden_dim, self.patch_elements)

        # 6. Spacetime Unpatchifier
        self.unpatchifier = SpacetimeUnpatchify(
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
        )

        self._init_adaLN_zero_output()

    def _init_adaLN_zero_output(self):
        """Zero-init final modulation and linear projection (adaLN-Zero standard)."""
        nn.init.zeros_(self.final_mod_proj.weight)
        nn.init.zeros_(self.final_mod_proj.bias)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_breakdown(self) -> Dict[str, int]:
        """Return breakdown of parameters by component."""
        patch_p = sum(p.numel() for p in self.patch_embed.parameters())
        time_p = sum(p.numel() for p in self.time_embedder.parameters())
        text_p = sum(p.numel() for p in self.text_projector.parameters())
        blocks_p = sum(p.numel() for p in self.blocks.parameters())
        head_p = sum(p.numel() for p in self.final_mod_proj.parameters()) + sum(p.numel() for p in self.final_linear.parameters())
        return {
            "patch_embed": patch_p,
            "timestep_embedder": time_p,
            "text_projector": text_p,
            "transformer_blocks": blocks_p,
            "output_head": head_p,
            "total": sum(p.numel() for p in self.parameters()),
        }

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the Wan DiT model.

        Args:
            x: Noisy latents [B, in_channels, t, h, w]
            timesteps: Float or long timesteps [B] (in [0, 1] or [0, 1000])
            context: Text embeddings [B, L, text_dim] or projected [B, L, hidden_dim]
            text_mask: Boolean attention mask [B, L] (True = attend, False = pad)
            cond: Optional conditioning latents / mask for image-to-video [B, C_cond, t, h, w]

        Returns:
            v_pred: Predicted flow field [B, out_channels, t, h, w]
        """
        B, C, t, h, w = x.shape

        # 1. Patchify latents to spacetime tokens: [B, N, hidden_dim]
        tokens, grid_size = self.patch_embed(x, cond=cond)

        # 2. Compute timestep shared modulation and hidden vector
        shared_mod, t_hidden = self.time_embedder(timesteps)

        # 3. Project text embeddings to model hidden dimension if not already projected
        if context.shape[-1] != self.hidden_dim:
            context = self.text_projector(context)

        # 4. Pass through N DiT blocks
        for block in self.blocks:
            if self.enable_checkpointing and self.training:
                tokens = checkpoint(
                    block,
                    tokens,
                    shared_mod,
                    context,
                    grid_size,
                    text_mask,
                    use_reentrant=False,
                )
            else:
                tokens = block(
                    tokens,
                    shared_mod,
                    context,
                    grid_size,
                    text_mask=text_mask,
                )

        # 5. Output Head: Final adaLN modulation
        final_mod = self.final_mod_proj(t_hidden).unsqueeze(1)  # [B, 1, 2 * hidden_dim]
        final_scale, final_shift = final_mod.chunk(2, dim=-1)

        tokens = self.final_norm(tokens) * (1.0 + final_scale) + final_shift
        patch_preds = self.final_linear(tokens)  # [B, N, pt*ph*pw*out_channels]

        # 6. Unpatchify back to video latent tensor [B, out_channels, t, h, w]
        # SpacetimeUnpatchify internal projection logic matches patch_preds rearrangement
        pt, ph, pw = self.patch_size
        grid_t, grid_h, grid_w = grid_size

        v_pred = patch_preds.view(B, grid_t, grid_h, grid_w, self.out_channels, pt, ph, pw)
        v_pred = v_pred.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        v_pred = v_pred.view(B, self.out_channels, grid_t * pt, grid_h * ph, grid_w * pw)

        return v_pred


# ==============================================================================
# 7. Model Factory from YAML Configurations
# ==============================================================================

def create_wan_dit_from_config(config: Dict[str, Any]) -> WanDiT:
    """Instantiate WanDiT model from configuration dictionary."""
    model_cfg = config.get("model", {})
    patch_cfg = config.get("patchify", {})

    patch_size_raw = patch_cfg.get("patch_size", [1, 2, 2])
    patch_size = (int(patch_size_raw[0]), int(patch_size_raw[1]), int(patch_size_raw[2]))

    return WanDiT(
        in_channels=patch_cfg.get("in_channels", 16),
        out_channels=patch_cfg.get("in_channels", 16),
        num_layers=model_cfg.get("num_layers", 8),
        hidden_dim=model_cfg.get("hidden_dim", 384),
        num_heads=model_cfg.get("num_heads", 6),
        ffn_dim=model_cfg.get("ffn_dim", 1536),
        text_dim=model_cfg.get("text_dim", 384),
        patch_size=patch_size,
        rope_theta=model_cfg.get("rope_theta", 10000.0),
        dropout=model_cfg.get("dropout", 0.0),
    )
