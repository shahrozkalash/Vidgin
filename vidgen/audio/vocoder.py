"""HiFi-GAN style neural vocoder for Vidgin (mel -> waveform).

Architecture:
- Generator:
  - Initial Conv1d(80, hidden_dim, kernel_size=7)
  - 4 Transposed Convolutions with upsample rates [8, 8, 2, 2] (8*8*2*2 = 256x upsampling)
  - Multi-Receptive Field Fusion (MRF) ResNet blocks with kernel sizes [3, 7, 11]
    and dilations [1, 3, 5]
  - Final Conv1d(..., 1, 7) + Tanh() producing 22050 Hz 1D audio in [-1.0, 1.0]
- Discriminators (for training):
  - Multi-Period Discriminator (MPD): periods [2, 3, 5, 7, 11]
  - Multi-Scale Discriminator (MSD): 1x, 2x, 4x pooling
- 100% in-repo PyTorch implementation, randomly initialized, zero external weights or downloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from vidgen.audio.torch_compat import torch, nn, F, HAS_TORCH
from vidgen.audio.mel import wav_to_mel


@dataclass
class VocoderConfig:
    in_channels: int = 80
    upsample_initial_channel: int = 256
    upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 1], [3, 1], [5, 1]]
    )


class ResBlock1D(nn.Module):
    """Residual block with dilated 1D convolutions for Multi-Receptive Field Fusion."""

    def __init__(self, channels: int, kernel_size: int, dilations: List[int]):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=d,
                padding=(kernel_size * d - d) // 2,
            )
            for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=1,
                padding=(kernel_size - 1) // 2,
            )
            for _ in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class HiFiGANGenerator(nn.Module):
    """Vidgin's in-repo neural vocoder generator. Converts log-mel [B, 80, T] to waveform [B, 1, T*256]."""

    def __init__(self, cfg: VocoderConfig = VocoderConfig()):
        super().__init__()
        self.cfg = cfg
        self.num_kernels = len(cfg.resblock_kernel_sizes)
        self.num_upsamples = len(cfg.upsample_rates)

        # Initial projection
        self.conv_pre = nn.Conv1d(cfg.in_channels, cfg.upsample_initial_channel, 7, 1, padding=3)

        # Transposed convolution upsampling layers
        self.ups = nn.ModuleList()
        curr_channels = cfg.upsample_initial_channel
        for u, k in zip(cfg.upsample_rates, cfg.upsample_kernel_sizes):
            out_channels = curr_channels // 2
            self.ups.append(
                nn.ConvTranspose1d(
                    curr_channels,
                    out_channels,
                    k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )
            curr_channels = out_channels

        # MRF ResNet blocks at each upsampling resolution
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = cfg.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes):
                self.resblocks.append(ResBlock1D(ch, k, d))

        # Final projection to audio waveform
        self.conv_post = nn.Conv1d(curr_channels, 1, 7, 1, padding=3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert [B, 80, T_mel] to [B, 1, T_wav] in [-1.0, 1.0]."""
        x = self.conv_pre(mel)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            # MRF fusion: sum outputs of all resblocks at this resolution
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[idx](x)
                else:
                    xs = xs + self.resblocks[idx](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x, 0.01)
        x = self.conv_post(x)
        if HAS_TORCH:
            x = torch.tanh(x)
        return x


# ------------------------------------------------ Discriminator Architecture
class PeriodDiscriminator(nn.Module):
    """Sub-discriminator for a single period p in Multi-Period Discriminator."""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=((kernel_size - 1) // 2, 0)),
            nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=((kernel_size - 1) // 2, 0)),
            nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=((kernel_size - 1) // 2, 0)),
            nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=((kernel_size - 1) // 2, 0)),
            nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=((kernel_size - 1) // 2, 0)),
        ])
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            if HAS_TORCH:
                x = F.pad(x, (0, n_pad), "reflect")
                t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        if HAS_TORCH:
            x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator (MPD) evaluating 1D waveforms across periods."""

    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        real_scores, fake_scores, real_fmaps, fake_fmaps = [], [], [], []
        for d in self.discriminators:
            r_s, r_f = d(y)
            f_s, f_f = d(y_hat)
            real_scores.append(r_s)
            fake_scores.append(f_s)
            real_fmaps.append(r_f)
            fake_fmaps.append(f_f)
        return real_scores, fake_scores, real_fmaps, fake_fmaps


class ScaleDiscriminator(nn.Module):
    """Sub-discriminator for Multi-Scale Discriminator."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 128, 15, 1, padding=7),
            nn.Conv1d(128, 128, 41, 2, groups=4, padding=20),
            nn.Conv1d(128, 256, 41, 2, groups=16, padding=20),
            nn.Conv1d(256, 512, 41, 4, groups=16, padding=20),
            nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 5, 1, padding=2),
        ])
        self.conv_post = nn.Conv1d(1024, 1, 3, 1, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        if HAS_TORCH:
            x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator (MSD) evaluating audio at original and pooled scales."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        real_scores, fake_scores, real_fmaps, fake_fmaps = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i > 0 and HAS_TORCH:
                y = F.avg_pool1d(y, 4, 2, padding=2)
                y_hat = F.avg_pool1d(y_hat, 4, 2, padding=2)
            r_s, r_f = d(y)
            f_s, f_f = d(y_hat)
            real_scores.append(r_s)
            fake_scores.append(f_s)
            real_fmaps.append(r_f)
            fake_fmaps.append(f_f)
        return real_scores, fake_scores, real_fmaps, fake_fmaps


def feature_loss(fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
    """Feature matching loss between real and generated discriminator representations."""
    if not HAS_TORCH:
        return 0.0
    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss = loss + torch.mean(torch.abs(rl - gl))
    return loss * 2.0


def generator_loss(disc_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Adversarial loss for generator."""
    if not HAS_TORCH:
        return 0.0, []
    loss = 0.0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1.0 - dg) ** 2)
        gen_losses.append(l)
        loss = loss + l
    return loss, gen_losses


def discriminator_loss(
    disc_real: List[torch.Tensor], disc_fake: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """Least-squares GAN loss for discriminators."""
    if not HAS_TORCH:
        return 0.0, [], []
    loss = 0.0
    r_losses, f_losses = [], []
    for dr, df in zip(disc_real, disc_fake):
        r_l = torch.mean((1.0 - dr) ** 2)
        f_l = torch.mean(df ** 2)
        r_losses.append(r_l)
        f_losses.append(f_l)
        loss = loss + (r_l + f_l)
    return loss, r_losses, f_losses


def count_vocoder_params(model_or_cfg: HiFiGANGenerator | VocoderConfig) -> int:
    """Calculate parameter count for the vocoder generator."""
    if isinstance(model_or_cfg, VocoderConfig):
        cfg = model_or_cfg
        # Initial conv
        params = cfg.in_channels * cfg.upsample_initial_channel * 7 + cfg.upsample_initial_channel
        curr = cfg.upsample_initial_channel
        # Transposed convs
        for u, k in zip(cfg.upsample_rates, cfg.upsample_kernel_sizes):
            out_ch = curr // 2
            params += curr * out_ch * k + out_ch
            # Resblocks at this stage
            for k_size, dilations in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes):
                for _ in dilations:
                    # conv1 + conv2
                    params += (out_ch * out_ch * k_size + out_ch) * 2
            curr = out_ch
        # Final conv
        params += curr * 1 * 7 + 1
        return params

    if HAS_TORCH and isinstance(model_or_cfg, nn.Module):
        return sum(p.numel() for p in model_or_cfg.parameters())
    return count_vocoder_params(VocoderConfig())
