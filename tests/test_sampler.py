"""Part 8: ODE sampler oracle recovery, determinism, CFG."""

import torch

from vidgen.sampler import sample_latents


class OracleModel(torch.nn.Module):
    """Fake model returning the exact constant velocity x1 - x0."""

    def __init__(self, x0: torch.Tensor, x1: torch.Tensor):
        super().__init__()
        self.x0 = x0
        self.x1 = x1
        self.calls = 0

    def forward(self, x, t, text, mask):
        self.calls += 1
        return (self.x1 - self.x0).expand_as(x)


def _oracle_setup():
    torch.manual_seed(0)
    shape = (1, 4, 2, 2, 2)  # small stand-in latent
    x1 = torch.randn(shape)
    # the sampler draws its own x0 from seed; oracle must know it:
    gen = torch.Generator().manual_seed(0)
    x0 = torch.randn(shape, generator=gen)
    model = OracleModel(x0, x1)
    text = torch.randn(1, 2, 8)
    mask = torch.ones(1, 2, dtype=torch.bool)
    return model, shape, text, mask, x1


def test_oracle_exact_recovery_euler():
    model, shape, text, mask, x1 = _oracle_setup()
    out = sample_latents(model, shape, text, mask, method="euler",
                         num_steps=10, cfg_scale=1.0, seed=0)
    assert (out - x1).abs().max() < 1e-5


def test_oracle_exact_recovery_midpoint_rk4():
    for method in ("midpoint", "rk4"):
        model, shape, text, mask, x1 = _oracle_setup()
        out = sample_latents(model, shape, text, mask, method=method,
                             num_steps=10, cfg_scale=1.0, seed=0)
        assert (out - x1).abs().max() < 1e-5, method


def test_sampler_seed_determinism():
    model, shape, text, mask, _ = _oracle_setup()
    a = sample_latents(model, shape, text, mask, seed=3, num_steps=5)
    b = sample_latents(model, shape, text, mask, seed=3, num_steps=5)
    c = sample_latents(model, shape, text, mask, seed=4, num_steps=5)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_cfg_blends_cond_uncond():
    calls = []

    class Probe(torch.nn.Module):
        def forward(self, x, t, text, mask):
            calls.append(text.abs().sum().item())
            return torch.zeros_like(x)

    shape = (1, 4, 2, 2, 2)
    text = torch.randn(1, 2, 8) + 5.0
    mask = torch.ones(1, 2, dtype=torch.bool)
    null = torch.zeros(1, 1, 8)
    null_mask = torch.zeros(1, 1, dtype=torch.bool)
    sample_latents(Probe(), shape, text, mask, null, null_mask,
                   num_steps=2, cfg_scale=7.5, seed=0)
    # every step calls the model twice (cond + uncond)
    assert len(calls) == 4
    assert any(c > 1.0 for c in calls) and any(c == 0.0 for c in calls)
