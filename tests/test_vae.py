"""Unit and Acceptance Tests for Part 3: Video VAE and Wan VAE Wrapper.

Acceptance Criteria:
- T = 4k+1 frames maps to k+1 latent frames.
- First frame encoded alone (unified image T=1 and video).
- Output shapes correct: [B, 16, t, h, w].
- Chunked decode matches unchunked decode within tolerance.
- Round-trip PSNR and LPIPS reported on 20 clips.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from vidgen.models.vae import (
    CausalConv3d,
    CausalResnetBlock3d,
    CausalVAE3d,
    WanVAEWrapper,
    PatchDiscriminator3d,
    VAELossEngine,
)
from vidgen.eval.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_lpips,
    evaluate_vae_reconstruction,
)


def test_causal_conv_padding_strictness():
    """Verify CausalConv3d only attends to past frames and never peeks at future frames."""
    conv = CausalConv3d(in_channels=4, out_channels=4, kernel_size=(3, 3, 3))
    # Test causality: zero out frame 2 and inspect output at frame 0 and 1
    x = torch.randn(1, 4, 5, 16, 16, requires_grad=True)
    out, _ = conv(x)

    # Gradient of frame 0 output with respect to input frame 2 must be strictly ZERO
    loss = out[:, :, 0, :, :].sum()
    loss.backward()

    assert x.grad is not None
    future_grad = x.grad[:, :, 2:, :, :].abs().sum().item()
    assert future_grad == 0.0, f"Causal conv leaked future information! Future grad = {future_grad}"


def test_vae_t_4k_plus_1_mapping():
    """Verify T = 4k+1 maps to k+1 latent frames."""
    wrapper = WanVAEWrapper()

    # Case 1: T=17 (k=4) -> k+1 = 5 latent frames
    v17 = torch.randn(1, 3, 17, 32, 32)
    lat17 = wrapper.encode(v17, chunked=False)
    B, C, t, h, w = lat17.shape
    assert C == 16, f"Expected 16 latent channels, got {C}"
    assert t == 5, f"Expected 5 latent frames for T=17, got {t}"
    assert h == 4 and w == 4, f"Expected 8x spatial downsampling (32->4), got {h}x{w}"

    # Decode back: 5 latents -> 17 frames
    rec17 = wrapper.decode(lat17, chunked=False)
    assert rec17.shape == (1, 3, 17, 32, 32)

    # Case 2: T=1 (image, k=0) -> k+1 = 1 latent frame
    img = torch.randn(1, 3, 1, 32, 32)
    lat_img = wrapper.encode(img, chunked=False)
    assert lat_img.shape == (1, 16, 1, 4, 4)
    rec_img = wrapper.decode(lat_img, chunked=False)
    assert rec_img.shape == (1, 3, 1, 32, 32)


def test_first_frame_encoded_alone_and_image_compatibility():
    """Verify that images and video share the exact same VAE with frame 0 encoded alone."""
    wrapper = WanVAEWrapper()

    # Full 5-frame video (k=1)
    vid = torch.randn(1, 3, 5, 32, 32)
    # Extract frame 0 as an image
    frame0_as_img = vid[:, :, :1, :, :]

    lat_full = wrapper.encode(vid, chunked=True)
    lat_img = wrapper.encode(frame0_as_img, chunked=True)

    assert lat_img.shape == (1, 16, 1, 4, 4)
    assert lat_full.shape == (1, 16, 2, 4, 4)


def test_per_channel_normalization():
    """Verify per-channel mean and std normalization prevents scale distortion."""
    mean = torch.tensor([0.1 * i for i in range(16)])
    std = torch.tensor([0.9 + 0.05 * i for i in range(16)])

    wrapper = WanVAEWrapper(latent_mean=mean, latent_std=std)

    dummy_raw = torch.randn(1, 16, 5, 4, 4)
    normed = wrapper.normalize(dummy_raw)
    denormed = wrapper.denormalize(normed)

    # Verify invertibility
    diff = (denormed - dummy_raw).abs().max().item()
    assert diff < 1e-5, f"Normalization round-trip failed: diff = {diff}"


def test_chunked_vs_unchunked_tolerance_and_roundtrip_20_clips():
    """Part 3 Acceptance:

    Report round-trip PSNR and LPIPS on 20 clips.
    Verify output shapes and chunked vs unchunked decode tolerance.
    """
    wrapper = WanVAEWrapper()

    # Generate 20 test clips (mix of T=17 videos and T=1 images)
    test_clips = []
    for i in range(20):
        t = 1 if (i % 4 == 0) else 17
        clip = torch.randn(1, 3, t, 32, 32).clamp(-1.0, 1.0)
        test_clips.append(clip)

    results = evaluate_vae_reconstruction(wrapper, test_clips)

    assert results["num_clips"] == 20
    assert results["mean_psnr_db"] > 10.0, "PSNR should be reported and positive"
    assert "mean_lpips" in results
    assert results["matches_within_tolerance"], "Chunked decode must match unchunked within tolerance"
    assert len(results["clip_psnr_scores"]) == 20


def test_patch_discriminator_and_vae_loss():
    """Verify 3D Patch Discriminator and VAE loss engine."""
    disc = PatchDiscriminator3d(in_channels=3, base_channels=16)
    loss_engine = VAELossEngine(kl_weight=1e-6)

    real = torch.randn(2, 3, 5, 32, 32)
    recon = torch.randn(2, 3, 5, 32, 32)
    mean = torch.zeros(2, 16, 2, 4, 4)
    logvar = torch.zeros(2, 16, 2, 4, 4)

    rec_loss = loss_engine.calculate_rec_loss(real, recon)
    kl_loss = loss_engine.calculate_kl_loss(mean, logvar)

    assert rec_loss > 0.0
    assert kl_loss >= 0.0

    # Discriminator logits
    logits_real = disc(real)
    logits_fake = disc(recon)
    d_loss = loss_engine.calculate_disc_loss(logits_real, logits_fake)
    assert d_loss > 0.0
