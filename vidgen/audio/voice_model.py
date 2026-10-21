"""Vidgin text-to-mel TTS voice model for ONE speaker.

Architecture:
- Text tokenizer: Character-level English vocabulary (letters, numbers, punctuation).
- Text Encoder: Learned character embeddings + 6 Transformer encoder layers (dim=256, heads=4, ffn=1024).
- Alignment Mechanism & Duration Predictor: 2-layer 1D Conv duration predictor + Length Regulator
  expanding phoneme/char tokens to mel frame tokens.
- Mel Decoder: 6 Transformer layers (dim=256) projecting to 80 log-mel bins + 5-layer Post-Net.
- Outputs 80-channel log-mel spectrograms ready for synthesis by Vidgin's HiFi-GAN vocoder.
- 100% in-repo PyTorch code, randomly initialized, zero external weights or downloads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from vidgen.audio.torch_compat import torch, nn, F, HAS_TORCH
from vidgen.audio.mel import AudioConfig, DEFAULT_AUDIO_CFG


CHAR_VOCAB = (
    ["<pad>", "<unk>", "<sos>", "<eos>", " "]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [str(d) for d in range(10)]
    + list(".,!?'\"-:;")
)


@dataclass
class VoiceModelConfig:
    vocab_size: int = len(CHAR_VOCAB)
    dim: int = 256
    encoder_layers: int = 6
    decoder_layers: int = 6
    heads: int = 4
    ffn_dim: int = 1024
    n_mels: int = 80
    max_text_len: int = 256
    max_mel_len: int = 1024
    postnet_channels: int = 256
    postnet_layers: int = 5


class DurationPredictor(nn.Module):
    """Predicts duration (number of mel frames) for each text token."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> [B, D, L]
        h = x.transpose(1, 2)
        h = F.relu(self.conv1(h)).transpose(1, 2)
        h = self.norm1(h)
        h = h.transpose(1, 2)
        h = F.relu(self.conv2(h)).transpose(1, 2)
        h = self.norm2(h)
        # Predict duration in log scale
        dur_log = self.proj(h).squeeze(-1)  # [B, L]
        return dur_log


def length_regulator(
    x: torch.Tensor, durations: torch.Tensor, max_len: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand text representations according to frame durations."""
    if not HAS_TORCH:
        return x, durations
    b, l, d = x.shape
    expanded_list = []
    mel_lens = []

    for i in range(b):
        cur_x = x[i]  # [L, D]
        cur_dur = torch.clamp(torch.round(durations[i]).long(), min=1)
        repeated = torch.repeat_interleave(cur_x, cur_dur, dim=0)
        expanded_list.append(repeated)
        mel_lens.append(repeated.shape[0])

    target_len = max_len or max(mel_lens)
    padded = torch.zeros(b, target_len, d, device=x.device)
    for i, exp in enumerate(expanded_list):
        dur = min(exp.shape[0], target_len)
        padded[i, :dur] = exp[:dur]

    return padded, torch.tensor(mel_lens, device=x.device)


class PostNet(nn.Module):
    """5-layer 1D convolutional Post-Net for fine-grained mel refinement."""

    def __init__(self, n_mels: int = 80, channels: int = 256, layers: int = 5):
        super().__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(nn.Conv1d(n_mels, channels, 5, padding=2))
        for _ in range(layers - 2):
            self.convolutions.append(nn.Conv1d(channels, channels, 5, padding=2))
        self.convolutions.append(nn.Conv1d(channels, n_mels, 5, padding=2))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, 80, T]
        x = mel
        for i, conv in enumerate(self.convolutions[:-1]):
            x = torch.tanh(conv(x)) if HAS_TORCH else conv(x)
        res = self.convolutions[-1](x)
        return mel + res


class TransformerBlock(nn.Module):
    """Transformer Encoder/Decoder block with Pre-LN."""

    def __init__(self, dim: int, heads: int, ffn_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_attn = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :].expand(b, 1, n, -1)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(b, n, d)
        x = x + self.out_attn(attn)
        x = x + self.ffn(self.norm2(x))
        return x


class VidginVoiceTTS(nn.Module):
    """Vidgin's in-repo Single-Speaker Text-to-Mel TTS Model."""

    def __init__(self, cfg: VoiceModelConfig = VoiceModelConfig()):
        super().__init__()
        self.cfg = cfg
        self.char2id = {c: i for i, c in enumerate(CHAR_VOCAB)}

        # Text Encoder
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx=0)
        self.pos_emb_text = nn.Embedding(cfg.max_text_len, cfg.dim)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(cfg.dim, cfg.heads, cfg.ffn_dim)
            for _ in range(cfg.encoder_layers)
        ])

        # Alignment & Duration Predictor
        self.duration_predictor = DurationPredictor(cfg.dim)

        # Mel Decoder
        self.pos_emb_mel = nn.Embedding(cfg.max_mel_len, cfg.dim)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(cfg.dim, cfg.heads, cfg.ffn_dim)
            for _ in range(cfg.decoder_layers)
        ])
        self.mel_proj = nn.Linear(cfg.dim, cfg.n_mels)
        self.postnet = PostNet(cfg.n_mels, cfg.postnet_channels, cfg.postnet_layers)

    def tokenize(self, text: str) -> list[int]:
        """Convert raw dialogue string into token ids."""
        text_clean = text.lower().strip()
        ids = [self.char2id.get("<sos>")]
        for char in text_clean:
            ids.append(self.char2id.get(char, self.char2id["<unk>"]))
        ids.append(self.char2id.get("<eos>"))
        return ids[: self.cfg.max_text_len]

    def forward(
        self,
        token_ids: torch.Tensor,
        mel_target: Optional[torch.Tensor] = None,
        durations_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training and evaluation.

        Returns:
            (mel_coarse, mel_refined, log_durations)
        """
        b, l = token_ids.shape
        device = token_ids.device

        # 1. Text Encoder
        pos_t = torch.arange(l, device=device).unsqueeze(0).expand(b, -1)
        h = self.embedding(token_ids) + self.pos_emb_text(pos_t)
        for block in self.encoder_blocks:
            h = block(h)

        # 2. Duration prediction
        log_dur = self.duration_predictor(h)

        # 3. Alignment expansion
        if durations_target is not None:
            dur = durations_target
        else:
            dur = torch.clamp(torch.exp(log_dur), min=1.0)

        expanded, _ = length_regulator(h, dur, max_len=self.cfg.max_mel_len)

        # 4. Mel Decoder
        mel_len = expanded.shape[1]
        pos_m = torch.arange(mel_len, device=device).unsqueeze(0).expand(b, -1)
        h_mel = expanded + self.pos_emb_mel(pos_m)
        for block in self.decoder_blocks:
            h_mel = block(h_mel)

        mel_coarse = self.mel_proj(h_mel).transpose(1, 2)  # [B, 80, T]
        mel_refined = self.postnet(mel_coarse)

        return mel_coarse, mel_refined, log_dur

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        duration_scale: float = 1.0,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Synthesize dialogue into an 80-channel log-mel spectrogram [1, 80, frames]."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for VidginVoiceTTS.synthesize")
        self.eval()
        dev = torch.device(device)
        self.to(dev)

        tokens = self.tokenize(text)
        token_ids = torch.tensor([tokens], device=dev)
        b, l = token_ids.shape

        pos_t = torch.arange(l, device=dev).unsqueeze(0)
        h = self.embedding(token_ids) + self.pos_emb_text(pos_t)
        for block in self.encoder_blocks:
            h = block(h)

        log_dur = self.duration_predictor(h)
        dur = torch.clamp(torch.exp(log_dur) * duration_scale, min=1.0)
        expanded, _ = length_regulator(h, dur)

        mel_len = expanded.shape[1]
        pos_m = torch.arange(mel_len, device=dev).unsqueeze(0)
        h_mel = expanded + self.pos_emb_mel(pos_m)
        for block in self.decoder_blocks:
            h_mel = block(h_mel)

        mel_coarse = self.mel_proj(h_mel).transpose(1, 2)
        mel_refined = self.postnet(mel_coarse)
        return mel_refined


def count_voice_params(model_or_cfg: VidginVoiceTTS | VoiceModelConfig) -> int:
    """Calculate parameter count for the Vidgin single-speaker voice TTS model."""
    if isinstance(model_or_cfg, VoiceModelConfig):
        cfg = model_or_cfg
        dim = cfg.dim
        ffn = cfg.ffn_dim

        # Text embedding + pos
        emb = cfg.vocab_size * dim + cfg.max_text_len * dim
        # Single transformer block
        block = (
            2 * dim  # norm1
            + (dim * (3 * dim) + 3 * dim + dim * dim + dim)  # qkv + out
            + 2 * dim  # norm2
            + (dim * ffn + ffn + ffn * dim + dim)  # ffn
        )
        enc = cfg.encoder_layers * block
        # Duration predictor
        dur = (dim * dim * 3 + dim + 2 * dim) * 2 + (dim * 1 + 1)
        # Mel decoder
        pos_mel = cfg.max_mel_len * dim
        dec = cfg.decoder_layers * block
        mel_proj = dim * cfg.n_mels + cfg.n_mels
        # Postnet
        pn = (cfg.n_mels * cfg.postnet_channels * 5 + cfg.postnet_channels)
        pn += (cfg.postnet_layers - 2) * (cfg.postnet_channels * cfg.postnet_channels * 5 + cfg.postnet_channels)
        pn += (cfg.postnet_channels * cfg.n_mels * 5 + cfg.n_mels)
        return emb + enc + dur + pos_mel + dec + mel_proj + pn

    if HAS_TORCH and isinstance(model_or_cfg, nn.Module):
        return sum(p.numel() for p in model_or_cfg.parameters())
    return count_voice_params(VoiceModelConfig())
