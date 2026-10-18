"""Part 3: causal 3D VAE — 4x temporal, 8x spatial compression, 16 channels.

Causality: every 3D conv pads the temporal axis only on the *past* side, so a
latent at t' depends only on input frames t <= f(t'). This is what makes
first-frame-solo encoding (T=4k+1 -> k+1 latents, T=1 -> 1 latent) and causal
chunked decoding possible. Spatial padding is symmetric.

Tiny geometry: [B,3,17,64,64] -> [B,16,5,8,8].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv3d(nn.Module):
    """3D conv with past-only temporal padding (strictly causal in time)."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel: tuple = (3, 3, 3), stride: tuple = (1, 1, 1)):
        super().__init__()
        kt, kh, kw = kernel
        st, sh, sw = stride
        self.stride = stride
        # F.pad order: (w_l, w_r, h_l, h_r, t_l, t_r); past-only in time.
        # Spatial pad is (k-1)//2 (NOT k//2) so k=4,s=2 downsamples 64->32.
        self.pad = ((kw - 1) // 2, (kw - 1) // 2,
                    (kh - 1) // 2, (kh - 1) // 2,
                    kt - 1, 0)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel, stride=stride, padding=0)

    def out_time(self, t_in: int) -> int:
        kt, st = self.conv.kernel_size[0], self.stride[0]
        return (t_in + (kt - 1) - kt) // st + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, self.pad))


class FramewiseGroupNorm(nn.Module):
    """GroupNorm applied independently per time step.

    Standard GroupNorm3d-style normalization pools statistics over (C, T, H, W),
    which mixes future frames into past latents and breaks the causal receptive
    field. This variant normalizes each frame on its own (over C, H, W only),
    preserving strict temporal causality.
    """

    def __init__(self, num_groups: int, num_channels: int):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.gn(x)
        return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


class ResBlock3d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.n1 = FramewiseGroupNorm(8, ch)
        self.c1 = CausalConv3d(ch, ch)
        self.n2 = FramewiseGroupNorm(8, ch)
        self.c2 = CausalConv3d(ch, ch)

    def forward(self, x):
        h = F.silu(self.n1(x))
        h = self.c1(h)
        h = F.silu(self.n2(h))
        return x + self.c2(h)


class Encoder3d(nn.Module):
    """[B,3,T,H,W] -> mu, logvar [B,16,t,h,w] with 4x temporal / 8x spatial."""

    def __init__(self, latent_ch: int = 16, base_ch: int = 32):
        super().__init__()
        self.in_conv = CausalConv3d(3, base_ch)
        self.down1 = CausalConv3d(base_ch, base_ch * 2, kernel=(3, 4, 4), stride=(2, 2, 2))
        self.down2 = CausalConv3d(base_ch * 2, base_ch * 4, kernel=(3, 4, 4), stride=(2, 2, 2))
        self.down3 = CausalConv3d(base_ch * 4, base_ch * 4, kernel=(3, 3, 3), stride=(1, 2, 2))
        self.mid = ResBlock3d(base_ch * 4)
        self.mu = nn.Conv3d(base_ch * 4, latent_ch, 1)
        self.logvar = nn.Conv3d(base_ch * 4, latent_ch, 1)

    def forward(self, x):
        h = self.in_conv(x)
        h = F.silu(self.down1(h))
        h = F.silu(self.down2(h))
        h = F.silu(self.down3(h))
        h = self.mid(h)
        return self.mu(h), self.logvar(h)


class Decoder3d(nn.Module):
    """[B,16,t,h,w] -> [B,3,T,H,W] in [-1, 1]."""

    def __init__(self, latent_ch: int = 16, base_ch: int = 32):
        super().__init__()
        self.in_conv = nn.Conv3d(latent_ch, base_ch * 4, 1)
        self.mid = ResBlock3d(base_ch * 4)
        # mirror of down3 (3,3,3) s=(1,2,2): t -> t, h -> 2h
        self.up3 = nn.ConvTranspose3d(base_ch * 4, base_ch * 4, (3, 3, 3),
                                      stride=(1, 2, 2), padding=(1, 1, 1),
                                      output_padding=(0, 1, 1))
        # mirror of down2 (3,4,4) s=(2,2,2): t -> 2t-1, h -> 2h
        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, (3, 4, 4),
                                      stride=(2, 2, 2), padding=(1, 1, 1))
        # mirror of down1 (3,4,4) s=(2,2,2): t -> 2t-1, h -> 2h
        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, (3, 4, 4),
                                      stride=(2, 2, 2), padding=(1, 1, 1))
        self.out_conv = CausalConv3d(base_ch, 3)

    def forward(self, z):
        h = self.in_conv(z)
        h = self.mid(h)
        h = F.silu(self.up3(h))   # (t, 2h, 2w)
        h = F.silu(self.up2(h))   # (2t-1, 4h, 4w)
        h = F.silu(self.up1(h))   # (4t-3, 8h, 8w)
        return torch.tanh(self.out_conv(h))


class CausalVAE(nn.Module):
    """Full VAE with reparameterization + KL. Encoder is the causal path that
    matters for training latents; decoder is trained jointly."""

    def __init__(self, latent_ch: int = 16, base_ch: int = 32):
        super().__init__()
        self.encoder = Encoder3d(latent_ch, base_ch)
        self.decoder = Decoder3d(latent_ch, base_ch)
        self.latent_ch = latent_ch

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * torch.clamp(logvar, -10, 10))
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        # KL per element, averaged
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return recon, mu, logvar, kl


def vae_loss(recon, x, kl, kl_weight: float = 1e-4):
    recon_loss = F.mse_loss(recon, x)
    return recon_loss + kl_weight * kl, recon_loss, kl
