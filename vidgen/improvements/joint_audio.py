"""Joint Audio-Video Co-Generation Attention Layer (Part 12.7).

Target:
Overcome Wan's silent video output by enabling joint audio-video synthesis:
1. Dual Stream Architecture:
   - Video latent stream [B, N_video, D_video]
   - Audio spectrogram latent stream [B, N_audio, D_audio]
2. Joint Cross-Modal Attention:
   - Video tokens attend to audio tokens to synchronize motion beats (e.g. footsteps, door slams).
   - Audio tokens attend to video tokens to synthesize ambient soundscapes matching scene optics.
3. Shared Temporal RoPE Alignment:
   - Normalizes audio sample rate (e.g. 16kHz) to video frame rate (16 FPS) so time t=1.0s
     corresponds to the exact same temporal position angle in both streams.
"""

from __future__ import annotations
import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from vidgen.models.text import RMSNorm


class JointAudioVideoAttention(nn.Module):
    """Bidirectional cross-attention between video tokens and audio spectrogram tokens."""

    def __init__(
        self,
        video_dim: int = 384,
        audio_dim: int = 256,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
    ):
        super().__init__()
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (video_dim // num_heads)
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Video queries audio
        self.v_to_a_q = nn.Linear(video_dim, self.inner_dim, bias=False)
        self.v_to_a_k = nn.Linear(audio_dim, self.inner_dim, bias=False)
        self.v_to_a_v = nn.Linear(audio_dim, self.inner_dim, bias=False)
        self.v_out = nn.Linear(self.inner_dim, video_dim, bias=False)

        # Audio queries video
        self.a_to_v_q = nn.Linear(audio_dim, self.inner_dim, bias=False)
        self.a_to_v_k = nn.Linear(video_dim, self.inner_dim, bias=False)
        self.a_to_v_v = nn.Linear(video_dim, self.inner_dim, bias=False)
        self.a_out = nn.Linear(self.inner_dim, audio_dim, bias=False)

        self.norm_v = RMSNorm(video_dim)
        self.norm_a = RMSNorm(audio_dim)

    def forward(
        self,
        video_tokens: torch.Tensor,  # [B, N_v, D_v]
        audio_tokens: torch.Tensor,  # [B, N_a, D_a]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs joint cross-modal exchange.

        Returns:
            video_updated: [B, N_v, D_v]
            audio_updated: [B, N_a, D_a]
        """
        B, N_v, _ = video_tokens.shape
        _, N_a, _ = audio_tokens.shape

        v_norm = self.norm_v(video_tokens)
        a_norm = self.norm_a(audio_tokens)

        # 1. Video queries Audio
        vq = self.v_to_a_q(v_norm).view(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)
        ak = self.v_to_a_k(a_norm).view(B, N_a, self.num_heads, self.head_dim).transpose(1, 2)
        av = self.v_to_a_v(a_norm).view(B, N_a, self.num_heads, self.head_dim).transpose(1, 2)

        v_cross = F.scaled_dot_product_attention(vq, ak, av, scale=self.scale)
        v_cross = self.v_out(v_cross.transpose(1, 2).contiguous().view(B, N_v, self.inner_dim))
        video_updated = video_tokens + v_cross

        # 2. Audio queries Video
        aq = self.a_to_v_q(a_norm).view(B, N_a, self.num_heads, self.head_dim).transpose(1, 2)
        vk = self.a_to_v_k(v_norm).view(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)
        vv = self.a_to_v_v(v_norm).view(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)

        a_cross = F.scaled_dot_product_attention(aq, vk, vv, scale=self.scale)
        a_cross = self.a_out(a_cross.transpose(1, 2).contiguous().view(B, N_a, self.inner_dim))
        audio_updated = audio_tokens + a_cross

        return video_updated, audio_updated
