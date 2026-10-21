"""Text-conditioned audio diffusion model (DiT-style) for Vidgin.

Architecture:
- Generates 80-channel log-mel spectrograms from text captions.
- 2D spacetime/frequency-time patchify with exact fold/unfold inverse.
- 2D partitioned RoPE (frequency, time) for spatial/temporal translational invariance.
- DiT backbone with AdaLN-Zero timestep modulation, self-attention, text cross-attention, FFN.
- Rectified flow formulation (v-prediction, x_t = (1-t) x_0 + t x_1).
- Target size: ~71M parameters (in target 30-80M range) for fast training on free Colab T4.
- 100% in-repo code, randomly initialized, zero external weights or downloads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from vidgen.audio.torch_compat import torch, nn, F, HAS_TORCH
from vidgen.audio.mel import denormalize_mel, normalize_mel


# Standard sound vocabulary for sound model text conditioning
SOUND_VOCAB = [
    "<pad>", "<unk>", "ambient", "sound", "noise", "audio", "background",
    "city", "traffic", "street", "car", "cars", "horn", "passing", "siren", "bus",
    "rain", "rainfall", "steady", "drops", "heavy", "thunder", "thunderstorm", "lightning",
    "nature", "forest", "tree", "trees", "wind", "breeze", "leaves", "rustling",
    "birds", "bird", "chirping", "singing", "wildlife", "stream", "water", "river",
    "beach", "ocean", "waves", "sea", "crashing", "coast", "shore", "seagulls",
    "crowd", "people", "chatter", "murmur", "walking", "footsteps", "park", "square",
    "fire", "crackling", "flames", "campfire", "wood", "burning",
    "dog", "barking", "cat", "meow", "animals", "clock", "ticking",
    "office", "indoor", "room", "door", "bell", "music", "distant",
    "night", "crickets", "gentle", "loud", "soft", "muffled", "deep",
]


@dataclass
class AudioDiTConfig:
    dim: int = 512
    depth: int = 12
    heads: int = 8
    ffn_dim: int = 2048
    text_dim: int = 256
    text_max_len: int = 32
    n_mels: int = 80
    patch_freq: int = 16  # 80 / 16 = 5 frequency tokens
    patch_time: int = 2   # e.g. 256 frames -> 128 time tokens
    rope_split: Tuple[int, int] = (32, 32)  # head_dim = 64 = 32 + 32
    cfg_dropout: float = 0.1


# ----------------------------------------------------------- Patchify Helpers
def patchify_mel(x: torch.Tensor, patch_f: int, patch_t: int) -> torch.Tensor:
    """[B, 1, F, T] or [B, F, T] -> [B, N, patch_f * patch_t]."""
    if x.dim() == 3:
        x = x.unsqueeze(1)
    b, c, f, t = x.shape
    assert f % patch_f == 0 and t % patch_t == 0, (
        f"Mel dims (F={f}, T={t}) must divide by patch ({patch_f}, {patch_t})"
    )
    x = x.view(b, c, f // patch_f, patch_f, t // patch_t, patch_t)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    num_tokens = (f // patch_f) * (t // patch_t)
    return x.view(b, num_tokens, c * patch_f * patch_t)


def unpatchify_mel(
    tok: torch.Tensor,
    grid_f: int,
    grid_t: int,
    patch_f: int,
    patch_t: int,
    channels: int = 1,
) -> torch.Tensor:
    """[B, N, patch_f * patch_t] -> [B, channels, grid_f * patch_f, grid_t * patch_t]."""
    b, _, _ = tok.shape
    x = tok.view(b, grid_f, grid_t, channels, patch_f, patch_t)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(b, channels, grid_f * patch_f, grid_t * patch_t)


# ------------------------------------------------------------- 2D RoPE
def _rope_1d_freqs(pos: torch.Tensor, dim: int, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=pos.device).float() / dim))
    ang = torch.outer(pos.float(), inv)
    return ang.cos(), ang.sin()


def _apply_rope_1d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0, "RoPE requires even dimension"
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos.unsqueeze(0) if x.dim() == 3 else cos
    s = sin.unsqueeze(0) if x.dim() == 3 else sin
    while c.dim() < x1.dim():
        c = c.unsqueeze(0)
        s = s.unsqueeze(0)
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def apply_rope_2d(
    q: torch.Tensor, pos_f: torch.Tensor, pos_t: torch.Tensor, split: Tuple[int, int]
) -> torch.Tensor:
    """Apply partitioned 2D RoPE to [B, H, N, head_dim]."""
    df, dt = split
    assert q.shape[-1] == df + dt, f"RoPE head dim {q.shape[-1]} != {df} + {dt}"
    qf, qt = q.split([df, dt], dim=-1)
    cf, sf = _rope_1d_freqs(pos_f, df)
    ct, st = _rope_1d_freqs(pos_t, dt)
    return torch.cat([_apply_rope_1d(qf, cf, sf), _apply_rope_1d(qt, ct, st)], dim=-1)


# ---------------------------------------------------------- Text Encoder for Audio
class AudioTextEncoder(nn.Module):
    """Learned token embeddings over sound vocabulary -> [B, L, text_dim] + mask."""

    def __init__(self, vocab: list[str] | None = None, d_t: int = 256, max_len: int = 32):
        super().__init__()
        self.vocab = vocab or SOUND_VOCAB
        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        self.d_t = d_t
        self.max_len = max_len
        self.embedding = nn.Embedding(len(self.vocab), d_t, padding_idx=0)
        if HAS_TORCH:
            self.register_buffer("_null_ids", torch.zeros(1, 1, dtype=torch.long))

    def tokenize(self, caption: str) -> list[int]:
        tokens = caption.lower().replace(",", " ").replace(".", " ").split()
        return [self.word2id.get(w, self.word2id["<unk>"]) for w in tokens][: self.max_len]

    def forward(self, captions: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not HAS_TORCH:
            return None, None
        device = self.embedding.weight.device
        b = len(captions)
        ids = torch.zeros(b, self.max_len, dtype=torch.long, device=device)
        mask = torch.zeros(b, self.max_len, dtype=torch.bool, device=device)
        for i, cap in enumerate(captions):
            toks = self.tokenize(cap)
            if toks:
                ids[i, : len(toks)] = torch.tensor(toks, device=device)
                mask[i, : len(toks)] = True
        return self.embedding(ids), mask

    def null_embedding(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not HAS_TORCH:
            return None, None
        emb = self.embedding(self._null_ids)
        mask = torch.zeros(1, 1, dtype=torch.bool, device=emb.device)
        return emb, mask


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding + MLP for diffusion timestep."""

    def __init__(self, dim: int, hidden_mult: int = 2):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.SiLU(),
            nn.Linear(dim * hidden_mult, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if not HAS_TORCH:
            return None
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]
        return self.mlp(torch.cat([args.cos(), args.sin()], dim=1))


class AudioDiTBlock(nn.Module):
    """DiT Block with AdaLN-Zero, 2D RoPE Self-Attention, Text Cross-Attention, FFN."""

    def __init__(self, dim: int, heads: int, ffn_dim: int, rope_split: Tuple[int, int]):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        if sum(rope_split) != self.head_dim:
            # Configured split was sized for a different head_dim (e.g. the
            # small test/training configs): re-derive an even split that
            # sums to this block's actual head_dim.
            half = (self.head_dim // 4) * 2
            rope_split = (half, self.head_dim - half)
        self.rope_split = rope_split

        # AdaLN-Zero parameters: 6 scale/shift/gate multipliers
        self.ada_lin = nn.Linear(dim, 6 * dim)
        if HAS_TORCH:
            nn.init.zeros_(self.ada_lin.weight)
            nn.init.zeros_(self.ada_lin.bias)

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_out = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.cross_q = nn.Linear(dim, dim)
        self.cross_kv = nn.Linear(dim, 2 * dim)
        self.cross_out = nn.Linear(dim, dim)

        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        text: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        pos: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        b, n, d = x.shape
        pos_f, pos_t = pos

        # AdaLN-Zero modulation chunking
        mod = self.ada_lin(t_emb)  # [B, 6*dim]
        s1, g1, a1, s2, g2, a2 = mod.chunk(6, dim=-1)

        # 1. Self-Attention with 2D RoPE
        h = self.norm1(x) * (1.0 + s1.unsqueeze(1)) + g1.unsqueeze(1)
        qkv = self.qkv(h).view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope_2d(q, pos_f, pos_t, self.rope_split)
        k = apply_rope_2d(k, pos_f, pos_t, self.rope_split)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        x = x + a1.unsqueeze(1) * self.attn_out(attn)

        # 2. Text Cross-Attention
        h = self.norm2(x) * (1.0 + s2.unsqueeze(1)) + g2.unsqueeze(1)
        cq = self.cross_q(h).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        ckv = self.cross_kv(text).view(b, -1, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        ck, cv = ckv[0], ckv[1]

        attn_mask = None
        if text_mask is not None:
            attn_mask = text_mask[:, None, None, :].expand(b, 1, n, -1)
        c_attn = F.scaled_dot_product_attention(cq, ck, cv, attn_mask=attn_mask)
        c_attn = c_attn.transpose(1, 2).reshape(b, n, d)
        x = x + self.cross_out(c_attn)

        # 3. Feed Forward Network
        h = self.norm3(x)
        x = x + a2.unsqueeze(1) * self.ffn(h)
        return x


class AudioDiT(nn.Module):
    """Vidgin's own Audio Diffusion Transformer (DiT).

    Generates 80-channel log-mel spectrograms conditioned on text captions.
    Rectified flow velocity prediction: forward(mel, t, text, text_mask) -> velocity.
    """

    def __init__(self, cfg: AudioDiTConfig = AudioDiTConfig()):
        super().__init__()
        self.cfg = cfg
        patch_dim = cfg.patch_freq * cfg.patch_time
        self.proj_in = nn.Linear(patch_dim, cfg.dim)
        self.t_embedder = TimestepEmbedder(cfg.dim)
        self.text_proj = nn.Sequential(
            nn.Linear(cfg.text_dim, cfg.dim),
            nn.GELU(),
            nn.Linear(cfg.dim, cfg.dim),
        )
        self.blocks = nn.ModuleList([
            AudioDiTBlock(cfg.dim, cfg.heads, cfg.ffn_dim, cfg.rope_split)
            for _ in range(cfg.depth)
        ])
        self.norm_final = nn.LayerNorm(cfg.dim)
        self.proj_out = nn.Linear(cfg.dim, patch_dim)
        if HAS_TORCH:
            nn.init.zeros_(self.proj_out.weight)
            nn.init.zeros_(self.proj_out.bias)
        self.text_encoder = AudioTextEncoder(d_t=cfg.text_dim, max_len=cfg.text_max_len)

    def _get_grid_positions(
        self, grid_f: int, grid_t: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        f = torch.arange(grid_f, device=device)
        t = torch.arange(grid_t, device=device)
        ff, tt = torch.meshgrid(f, t, indexing="ij")
        return ff.reshape(-1), tt.reshape(-1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        captions: list[str] | torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict velocity field [B, 1, F, T] or [B, F, T] for rectified flow."""
        original_dim = x.dim()
        if x.dim() == 3:
            x = x.unsqueeze(1)
        b, c, f, t_frames = x.shape
        grid_f = f // self.cfg.patch_freq
        grid_t = t_frames // self.cfg.patch_time

        # 1. Patchify mel spectrogram
        tokens = patchify_mel(x, self.cfg.patch_freq, self.cfg.patch_time)
        h = self.proj_in(tokens)

        # 2. Timestep conditioning
        t_emb = self.t_embedder(t)

        # 3. Text conditioning
        if isinstance(captions, list):
            text_emb, text_mask = self.text_encoder(captions)
        else:
            text_emb = captions
        text_feat = self.text_proj(text_emb)

        # 4. RoPE Grid Positions
        pos = self._get_grid_positions(grid_f, grid_t, x.device)

        # 5. Transformer Blocks
        for block in self.blocks:
            h = block(h, t_emb, text_feat, text_mask, pos)

        # 6. Unpatchify back to mel shape
        h = self.norm_final(h)
        out_tokens = self.proj_out(h)
        velocity = unpatchify_mel(
            out_tokens, grid_f, grid_t, self.cfg.patch_freq, self.cfg.patch_time, channels=1
        )
        if original_dim == 3:
            velocity = velocity.squeeze(1)
        return velocity

    def sample(
        self,
        caption: str,
        duration_sec: float = 3.0,
        sr: int = 22050,
        hop_length: int = 256,
        steps: int = 25,
        cfg_scale: float = 3.0,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Sample a mel spectrogram from a text caption using Rectified Flow Euler ODE."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for AudioDiT.sample")
        with torch.no_grad():
            self.eval()
            dev = torch.device(device)
            self.to(dev)

            # Determine frame count divisible by patch_time
            raw_frames = int(round(duration_sec * sr / hop_length))
            frames = math.ceil(raw_frames / self.cfg.patch_time) * self.cfg.patch_time
            frames = max(frames, self.cfg.patch_time * 4)

            # Initial Gaussian noise x_0
            x = torch.randn(1, 1, self.cfg.n_mels, frames, device=dev)
            dt = 1.0 / steps

            # Pre-embed text and null text for CFG
            text_emb, text_mask = self.text_encoder([caption])
            null_emb, null_mask = self.text_encoder.null_embedding()

            for step in range(steps):
                t_val = step / steps
                t_tensor = torch.full((1,), t_val, device=dev)

                if cfg_scale > 1.0:
                    v_cond = self.forward(x, t_tensor, text_emb, text_mask)
                    v_uncond = self.forward(x, t_tensor, null_emb, null_mask)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = self.forward(x, t_tensor, text_emb, text_mask)

                # Euler ODE step: x_{t+dt} = x_t + dt * v
                x = x + dt * v

            mel_norm = x.squeeze(1)
            return denormalize_mel(mel_norm)


def count_sound_params(model_or_cfg: AudioDiT | AudioDiTConfig) -> int:
    """Calculate parameter count for the Audio DiT model."""
    if isinstance(model_or_cfg, AudioDiTConfig):
        cfg = model_or_cfg
        dim = cfg.dim
        depth = cfg.depth
        ffn = cfg.ffn_dim
        p_dim = cfg.patch_freq * cfg.patch_time

        # Tokenizer / Text
        text_emb = len(SOUND_VOCAB) * cfg.text_dim
        text_proj = (cfg.text_dim * dim + dim) + (dim * dim + dim)
        t_emb = (dim * (dim * 2) + (dim * 2)) + ((dim * 2) * dim + dim)
        p_in = p_dim * dim + dim
        p_out = dim * p_dim + p_dim
        norm_final = 2 * dim

        # DiT block
        per_block = (
            (dim * (6 * dim) + 6 * dim)  # AdaLN
            + (dim * (3 * dim) + 3 * dim + dim * dim + dim)  # Self attn
            + (dim * dim + dim + dim * (2 * dim) + 2 * dim + dim * dim + dim)  # Cross attn
            + (dim * ffn + ffn + ffn * dim + dim)  # FFN
            + (6 * dim)  # 3 LayerNorms
        )
        return text_emb + text_proj + t_emb + p_in + p_out + norm_final + depth * per_block

    if HAS_TORCH and isinstance(model_or_cfg, nn.Module):
        return sum(p.numel() for p in model_or_cfg.parameters())
    return count_sound_params(AudioDiTConfig())
