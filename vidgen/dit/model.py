"""Part 4 (patchifier + 3D RoPE) and Part 6 (DiT backbone).

- 3D spacetime patchify via exact unfold/fold (patchify <-> unpatchify is a
  bit-exact inverse), then a learned Linear projection to the token sequence.
- Full 3D self-attention with partitioned 3D RoPE: head dim is split across
  (time, height, width) so attention scores are invariant under global
  spatiotemporal translation.
- DiT block: AdaLN-Zero timestep modulation (shared sinusoidal MLP + per-block
  zero-init linear), self-attn, text cross-attn ([B,L,Dt] projected to D), FFN.
- forward(latent [B,C,t,h,w], timesteps [B], text, text_mask) -> velocity field
  of the same shape (rectified-flow target lives in latent space).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- patchify
def patchify_latents(x: torch.Tensor, p: tuple[int, int, int]) -> torch.Tensor:
    """[B,C,t,h,w] -> [B,N,C*pt*ph*pw] exact unfold. Inverse: unpatchify_tokens."""
    pt, ph, pw = p
    b, c, t, h, w = x.shape
    assert t % pt == 0 and h % ph == 0 and w % pw == 0, "latent dims must divide by patch"
    x = x.view(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return x.view(b, (t // pt) * (h // ph) * (w // pw), c * pt * ph * pw)


def unpatchify_tokens(tok: torch.Tensor, p: tuple[int, int, int],
                      grid: tuple[int, int, int], channels: int) -> torch.Tensor:
    """[B,N,C*pt*ph*pw] -> [B,C,t,h,w]; exact inverse of patchify_latents."""
    pt, ph, pw = p
    tt, hh, ww = grid
    b = tok.shape[0]
    x = tok.view(b, tt, hh, ww, channels, pt, ph, pw)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return x.view(b, channels, tt * pt, hh * ph, ww * pw)


def token_grid_positions(num_t: int, num_h: int, num_w: int,
                         device=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token (t, h, w) integer positions, row-major over (t, h, w)."""
    t = torch.arange(num_t, device=device)
    h = torch.arange(num_h, device=device)
    w = torch.arange(num_w, device=device)
    tt, hh, ww = torch.meshgrid(t, h, w, indexing="ij")
    return tt.reshape(-1), hh.reshape(-1), ww.reshape(-1)


# ---------------------------------------------------------------- 3D RoPE
def _rope_1d_freqs(pos: torch.Tensor, dim: int, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=pos.device).float() / dim))
    ang = torch.outer(pos.float(), inv)
    return ang.cos(), ang.sin()


def _apply_rope_1d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [..., N, d] with even d; cos/sin: [N, d/2]
    assert x.shape[-1] % 2 == 0, "RoPE needs even dims (use even rope_split)"
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos.unsqueeze(0) if x.dim() == 3 else cos
    s = sin.unsqueeze(0) if x.dim() == 3 else sin
    # broadcast over leading dims
    while c.dim() < x1.dim():
        c = c.unsqueeze(0)
        s = s.unsqueeze(0)
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def apply_rope_3d(q: torch.Tensor, pos_t: torch.Tensor, pos_h: torch.Tensor,
                  pos_w: torch.Tensor, split: tuple[int, int, int]) -> torch.Tensor:
    """Apply partitioned 3D RoPE to q [B, H, N, head_dim].

    head_dim = dt + dh + dw; each axis rotates its own slice with its own
    positions, giving translation invariance of attention scores.
    """
    dt, dh, dw = split
    assert q.shape[-1] == dt + dh + dw
    qt, qh, qw = q.split([dt, dh, dw], dim=-1)
    ct, st = _rope_1d_freqs(pos_t, dt)
    ch, sh = _rope_1d_freqs(pos_h, dh)
    cw, sw = _rope_1d_freqs(pos_w, dw)
    return torch.cat([
        _apply_rope_1d(qt, ct, st),
        _apply_rope_1d(qh, ch, sh),
        _apply_rope_1d(qw, cw, sw),
    ], dim=-1)


# ---------------------------------------------------------------- blocks
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding + MLP (shared across blocks)."""

    def __init__(self, dim: int, hidden_mult: int = 2):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult), nn.SiLU(),
            nn.Linear(dim * hidden_mult, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]
        return self.mlp(torch.cat([args.cos(), args.sin()], dim=1))


class TextProjector(nn.Module):
    """[B, L, Dt] -> [B, L, D] with length clamping (Part 5 audit)."""

    def __init__(self, d_t: int, dim: int, max_len: int):
        super().__init__()
        self.max_len = max_len
        self.proj = nn.Sequential(nn.Linear(d_t, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, text: torch.Tensor, mask: torch.Tensor):
        if text.shape[1] > self.max_len:
            text, mask = text[:, : self.max_len], mask[:, : self.max_len]
        return self.proj(text), mask


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn: int, rope_split: tuple):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.rope_split = rope_split
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_q = nn.Linear(dim, dim)
        self.cross_kv = nn.Linear(dim, 2 * dim)
        self.cross_out = nn.Linear(dim, dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Linear(ffn, dim))

    def forward(self, x, mod9, text, text_mask, pos):
        # x: [B, N, D]; mod9: shared AdaLN-Zero params; text: [B, L, D]
        (s1, g1a, ga1, s2, g2a, ga2, s3, g3a, ga3) = mod9
        b, n, d = x.shape
        # --- self attention with 3D RoPE ---
        h = modulate(self.norm1(x), s1, g1a)
        qkv = self.qkv(h).view(b, n, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3 x [B, H, N, dh]
        q, k, v = qkv[0], qkv[1], qkv[2]
        pt, ph, pw = pos
        q = apply_rope_3d(q, pt, ph, pw, self.rope_split)
        k = apply_rope_3d(k, pt, ph, pw, self.rope_split)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        x = x + ga1.unsqueeze(1) * self.attn_out(attn)
        # --- text cross attention (masked; masked tokens -> -inf) ---
        h = modulate(self.norm2(x), s2, g2a)
        q = self.cross_q(h).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        kv = self.cross_kv(text).view(b, -1, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn_mask = None
        if text_mask is not None:
            # True = keep; SDPA wants True = attend
            attn_mask = text_mask[:, None, None, :].expand(b, 1, n, -1)
        ca = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        ca = ca.transpose(1, 2).reshape(b, n, d)
        x = x + ga2.unsqueeze(1) * self.cross_out(ca)
        # --- ffn ---
        h = modulate(self.norm3(x), s3, g3a)
        x = x + ga3.unsqueeze(1) * self.ffn(h)
        return x


class DiT(nn.Module):
    """Full model: patchify -> DiT blocks -> head -> unpatchify.

    forward(latent [B,C,t,h,w], timesteps [B], text [B,L,Dt], text_mask [B,L])
    -> velocity field [B,C,t,h,w].
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.dit
        self.patch = (cfg.patch_t, cfg.patch_h, cfg.patch_w)
        self.grid = (cfg.token_t, cfg.token_h, cfg.token_w)
        patch_vol = cfg.patch_t * cfg.patch_h * cfg.patch_w
        self.proj_in = nn.Linear(cfg.latent_channels * patch_vol, d.dim)
        self.t_embedder = TimestepEmbedder(d.dim)
        # Shared AdaLN-Zero modulation (Part 6 audit: shared 6xD projection
        # saves >80% of modulation params vs per-block). Zero-init -> identity.
        self.adaLN_shared = nn.Linear(d.dim, 9 * d.dim)
        nn.init.zeros_(self.adaLN_shared.weight)
        nn.init.zeros_(self.adaLN_shared.bias)
        self.text_proj = TextProjector(cfg.text_dim, d.dim, cfg.text_max_len)
        self.blocks = nn.ModuleList(
            [DiTBlock(d.dim, d.heads, d.ffn, d.rope_split) for _ in range(d.layers)]
        )
        self.norm_final = nn.LayerNorm(d.dim)
        self.adaLN_final = nn.Linear(d.dim, 2 * d.dim)
        nn.init.zeros_(self.adaLN_final.weight)
        nn.init.zeros_(self.adaLN_final.bias)
        self.head = nn.Linear(d.dim, cfg.latent_channels * patch_vol)
        self._pos_cache = {}
        self.grad_checkpoint = False

    def set_grad_checkpoint(self, enabled: bool = True) -> None:
        """Activation checkpointing: recompute block activations in the
        backward pass instead of storing them. Trades ~30% speed for a
        large activation-memory cut — required to train the Small preset
        (and above) in limited RAM/VRAM. No effect at inference."""
        self.grad_checkpoint = bool(enabled)

    def _positions(self, device):
        key = str(device)
        if key not in self._pos_cache:
            self._pos_cache[key] = token_grid_positions(*self.grid, device=device)
        return self._pos_cache[key]

    def forward(self, latent, timesteps, text, text_mask=None):
        b = latent.shape[0]
        tok = patchify_latents(latent, self.patch)      # [B, N, C*p]
        x = self.proj_in(tok)
        c = self.t_embedder(timesteps)
        mod9 = self.adaLN_shared(c).chunk(9, dim=-1)
        text, text_mask = self.text_proj(text, text_mask)
        pos = self._positions(latent.device)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, mod9, text, text_mask, pos, use_reentrant=False)
            else:
                x = blk(x, mod9, text, text_mask, pos)
        shift, scale = self.adaLN_final(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.head(x)
        return unpatchify_tokens(x, self.patch, self.grid, self.cfg.latent_channels)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
