"""Part 3: VAE geometry, causality, round-trip."""

import torch

from vidgen.vae.model import CausalVAE


def test_vae_roundtrip_shape():
    vae = CausalVAE().eval()
    x = torch.rand(1, 3, 17, 64, 64) * 2 - 1
    with torch.no_grad():
        mu, logvar = vae.encode(x)
        assert mu.shape == (1, 16, 5, 8, 8), mu.shape
        assert logvar.shape == (1, 16, 5, 8, 8)
        z = vae.reparameterize(mu, logvar)
        recon = vae.decode(z)
    assert recon.shape == x.shape
    assert recon.min() >= -1.0 - 1e-5 and recon.max() <= 1.0 + 1e-5


def test_vae_first_frame_solo_encoding():
    vae = CausalVAE().eval()
    x = torch.rand(1, 3, 1, 64, 64) * 2 - 1
    with torch.no_grad():
        mu, _ = vae.encode(x)
    assert mu.shape == (1, 16, 1, 8, 8), "T=1 must encode to exactly 1 latent"


def test_vae_causal_receptive_field():
    """Perturbing only the last input frame must not change latents t' < 4."""
    vae = CausalVAE().eval()
    torch.manual_seed(0)
    x = torch.rand(1, 3, 17, 64, 64) * 2 - 1
    x2 = x.clone()
    x2[:, :, 16] = torch.rand(1, 3, 64, 64) * 2 - 1  # change last frame only
    with torch.no_grad():
        mu1, _ = vae.encode(x)
        mu2, _ = vae.encode(x2)
    early = (mu1[:, :, :4] - mu2[:, :, :4]).abs().max()
    assert early < 1e-5, f"future frame leaked into past latents: {early}"
    late = (mu1[:, :, 4:] - mu2[:, :, 4:]).abs().max()
    assert late > 1e-3, "last frame should affect the last latent"
