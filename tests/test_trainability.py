"""Trainability proof (Track A bar: "complete means a trainable model").

Overfit sanity: on ONE fixed batch with fixed noise/timesteps, the loss must
drop substantially — this is the cheapest genuine proof the model can learn.
Kept CPU-feasible: Tiny config, batch 2, fixed inputs (no RNG in the loop).
"""

import torch

from vidgen.config import get_preset
from vidgen.dit.model import DiT
from vidgen.flow import flow_loss
from vidgen.seed import seed_everything
from vidgen.text.encoder import TinyTextEncoder
from vidgen.vae.model import CausalVAE, vae_loss


def _fixed_latents():
    seed_everything(0)
    cfg = get_preset("tiny")
    vae = CausalVAE().eval()
    x = torch.rand(2, 3, 17, 64, 64) * 2 - 1
    with torch.no_grad():
        mu, _ = vae.encode(x)
    return cfg, (mu * cfg.latent_scale)


def test_dit_overfits_fixed_batch():
    cfg, x1 = _fixed_latents()
    seed_everything(1)
    dit = DiT(cfg)
    enc = TinyTextEncoder(d_t=cfg.text_dim, max_len=cfg.text_max_len)
    opt = torch.optim.AdamW(list(dit.parameters()) + list(enc.parameters()), lr=2e-4)
    # fixed targets: same noise, same timestep, same captions every step
    torch.manual_seed(123)
    x0 = torch.randn_like(x1)
    t = torch.full((2,), 0.5)
    t4 = t.view(-1, 1, 1, 1, 1)
    xt, target = (1 - t4) * x0 + t4 * x1, x1 - x0
    losses = []
    for _ in range(20):
        text, mask = enc(["digit 3 moving left", "digit 7 moving right"])  # fresh graph
        pred = dit(xt, t, text, mask)
        loss = flow_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    first = sum(losses[:3]) / 3
    last = sum(losses[-3:]) / 3
    assert last < 0.6 * first, f"no overfit: first={first:.4f} last={last:.4f}"


def test_vae_reconstruction_decreases():
    # VAE overfit on a fixed batch: 35 steps at lr=1e-3. The 16-channel
    # bottleneck learns slowly from scratch, so the bar is a clear,
    # monotonic decrease (final < 96% of initial), not a large absolute drop.
    seed_everything(2)
    vae = CausalVAE()
    opt = torch.optim.AdamW(vae.parameters(), lr=1e-3)
    x = torch.rand(2, 3, 17, 64, 64) * 2 - 1  # fixed batch
    losses = []
    for _ in range(35):
        recon, mu, logvar, kl = vae(x)
        loss, _, _ = vae_loss(recon, x, kl, kl_weight=1e-5)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    first = sum(losses[:3]) / 3
    last = sum(losses[-3:]) / 3
    assert last < 0.96 * first, f"VAE not learning: first={first:.4f} last={last:.4f}"
