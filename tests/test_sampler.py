"""Unit and Acceptance Tests for Part 8: Sampler and Inference Pipeline.

Acceptance Criteria:
1. Oracle test: a fake model that returns eps - x0 for a known x0 recovers x0 exactly,
   for any number of Euler steps (error < 1e-5).
2. Same seed gives the same video (deterministic bit-exact reproducibility).
3. Invalid frame counts raise a clear error (not 4k+1).
4. Sampling from a model overfit on one clip reproduces that clip.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vidgen.sample.sampler import (
    FlowSampler,
    get_time_schedule,
    OracleFlowModel,
)
from vidgen.sample.pipeline import (
    WanInferencePipeline,
    validate_video_dimensions,
)
from vidgen.models.dit import WanDiT
from vidgen.models.vae import WanVAEWrapper, CausalVAE3d
from vidgen.models.text import WanTextConditioner


# ==============================================================================
# 1. Acceptance Criterion 1: Oracle Test
# ==============================================================================

def test_oracle_euler_exact_recovery():
    """Oracle test: A fake model that returns (eps - x0) for a known x0

    recovers x0 exactly, for any number of Euler steps.
    """
    torch.manual_seed(42)
    shape = (1, 16, 5, 8, 8)  # [B, C, t, h, w]
    x0 = torch.randn(shape)
    eps = torch.randn(shape)

    oracle = OracleFlowModel(x0=x0, eps=eps)

    # Test with varying step counts: 1 step, 5 steps, 10 steps, 30 steps, 50 steps
    for steps in [1, 5, 10, 30, 50]:
        for shift in [1.0, 3.0, 5.0]:
            sampler = FlowSampler(steps=steps, shift=shift, solver="euler", guidance_scale=1.0)
            recovered = sampler.sample(
                model_fn=oracle,
                x_init=eps,
                steps=steps,
                shift=shift,
            )

            max_err = (recovered - x0).abs().max().item()
            assert max_err < 1e-5, (
                f"Oracle recovery failed for steps={steps}, shift={shift}: max error = {max_err:.2e}"
            )


def test_oracle_higher_order_solvers():
    """Verify Heun, UniPC, and DPM-Solver++ also recover x0 on the linear flow ODE."""
    torch.manual_seed(123)
    shape = (1, 8, 3, 4, 4)
    x0 = torch.randn(shape)
    eps = torch.randn(shape)
    oracle = OracleFlowModel(x0=x0, eps=eps)

    for solver in ["heun", "unipc", "dpm_solver"]:
        sampler = FlowSampler(steps=10, shift=3.0, solver=solver, guidance_scale=1.0)
        recovered = sampler.sample(
            model_fn=oracle,
            x_init=eps,
            steps=10,
        )
        max_err = (recovered - x0).abs().max().item()
        assert max_err < 1e-4, f"Oracle failed for solver={solver}: max error = {max_err:.2e}"


# ==============================================================================
# 2. Acceptance Criterion 2: Same Seed Gives Same Video
# ==============================================================================

def test_deterministic_seed_reproducibility():
    """Verify that running sampling with the same seed yields bit-identical results."""
    device = torch.device("cpu")
    dit = WanDiT(
        in_channels=16,
        out_channels=16,
        num_layers=2,
        hidden_dim=64,
        num_heads=2,
        text_dim=64,
        patch_size=(1, 2, 2),
    )
    vae = WanVAEWrapper(device=device)
    tc = WanTextConditioner(text_dim=64, model_dim=64)

    pipeline = WanInferencePipeline(dit=dit, vae=vae, text_conditioner=tc, device=device)

    prompt = "A red sports car on highway"
    seed = 2026

    video1 = pipeline(
        prompt=prompt,
        height=16,
        width=16,
        frames=5,  # 5 is 4*1 + 1
        steps=2,
        seed=seed,
        guidance_scale=1.0,
    )

    video2 = pipeline(
        prompt=prompt,
        height=16,
        width=16,
        frames=5,
        steps=2,
        seed=seed,
        guidance_scale=1.0,
    )

    diff = (video1 - video2).abs().max().item()
    assert diff == 0.0, f"Same seed failed to produce identical video: max diff = {diff}"


# ==============================================================================
# 3. Acceptance Criterion 3: Invalid Frame Counts Raise Clear Error
# ==============================================================================

def test_invalid_frame_counts_raise_error():
    """Verify that non-(4k+1) frame counts immediately raise ValueError with explanation."""
    invalid_counts = [0, -4, 16, 20, 24, 30, 48, 60, 64]
    for frames in invalid_counts:
        with pytest.raises(ValueError) as excinfo:
            validate_video_dimensions(frames=frames, height=32, width=32)
        assert "4k + 1" in str(excinfo.value) or "positive" in str(excinfo.value)

    # Valid counts should pass without raising
    valid_counts = [1, 5, 9, 17, 33, 49, 65, 81]
    for frames in valid_counts:
        validate_video_dimensions(frames=frames, height=32, width=32)


def test_invalid_spatial_dimensions_raise_error():
    """Verify non-multiples of 16 raise descriptive ValueError."""
    with pytest.raises(ValueError) as excinfo:
        validate_video_dimensions(frames=17, height=250, width=256)
    assert "divisible by 16" in str(excinfo.value)


# ==============================================================================
# 4. Acceptance Criterion 4: Sampling from Overfit Model Reproduces Clip
# ==============================================================================

def test_overfit_model_reproduces_clip():
    """Verify that sampling from a model overfit on one clip reproduces that clip."""
    torch.manual_seed(999)
    device = torch.device("cpu")

    # Small mock DiT
    dim = 64
    dit = WanDiT(
        in_channels=16,
        out_channels=16,
        num_layers=3,
        hidden_dim=dim,
        num_heads=2,
        text_dim=dim,
        patch_size=(1, 2, 2),
    )
    tc = WanTextConditioner(text_dim=dim, model_dim=dim)

    # Synthetic single clip latents
    target_x0 = torch.randn(1, 16, 3, 4, 4)
    prompt = "golden sunset over ocean waves"
    ctx, mask = tc.encode([prompt], device=device, apply_dropout=False)

    optimizer = torch.optim.AdamW(dit.parameters(), lr=1e-2)

    # Overfit for 45 steps on target clip flow
    for step in range(45):
        optimizer.zero_grad()
        # Random sigma
        sigma = torch.rand(1, 1, 1, 1, 1)
        eps = torch.randn_like(target_x0)
        # Rectified flow: x_t = (1 - sigma) * x0 + sigma * eps
        x_t = (1.0 - sigma) * target_x0 + sigma * eps
        target_v = eps - target_x0

        t_input = sigma.view(1) * 1000.0
        v_pred = dit(x_t, t_input, ctx, text_mask=mask)

        loss = F.mse_loss(v_pred, target_v)
        loss.backward()
        optimizer.step()

    # Final loss should be low
    assert loss.item() < 0.25, f"Overfit training failed: final loss={loss.item():.4f}"

    # Sample using Euler
    def model_fn(latents: torch.Tensor, timesteps: torch.Tensor, is_uncond: bool) -> torch.Tensor:
        return dit(latents, timesteps, ctx, text_mask=mask)

    sampler = FlowSampler(steps=20, shift=1.0, solver="euler", guidance_scale=1.0)
    eps_sample = torch.randn_like(target_x0)
    recovered_latent = sampler.sample(model_fn, eps_sample, steps=20)

    # Check correlation / similarity
    correlation = (
        (recovered_latent - recovered_latent.mean()) * (target_x0 - target_x0.mean())
    ).sum() / (recovered_latent.std() * target_x0.std() * target_x0.numel())

    assert correlation.item() > 0.60, f"Overfit reproduction correlation too low: {correlation.item():.3f}"


# ==============================================================================
# 5. Schedule & CFG Tests
# ==============================================================================

def test_time_schedule_properties():
    """Verify that sigma starts at 1.0 (pure noise) and ends at 0.0 (clean data)."""
    for shift in [1.0, 3.0, 5.0, 7.0]:
        sigmas = get_time_schedule(steps=30, shift=shift)
        assert len(sigmas) == 31
        assert math.isclose(sigmas[0].item(), 1.0, rel_tol=1e-5)
        assert math.isclose(sigmas[-1].item(), 0.0, rel_tol=1e-5)
        # Monotonically strictly decreasing
        diffs = sigmas[1:] - sigmas[:-1]
        assert (diffs < 0).all(), f"Schedule not monotonically decreasing for shift={shift}"


def test_cfg_formula():
    """Verify CFG vector calculation: v_guided = v_uncond + g * (v_cond - v_uncond)."""
    sampler = FlowSampler(guidance_scale=6.0)
    v_cond = torch.tensor([2.0, 5.0])
    v_uncond = torch.tensor([1.0, 1.0])

    # v = [1, 1] + 6 * ([2, 5] - [1, 1]) = [1, 1] + 6 * [1, 4] = [7, 25]
    guided = sampler.apply_cfg(v_cond, v_uncond, guidance_scale=6.0)
    expected = torch.tensor([7.0, 25.0])
    assert torch.allclose(guided, expected)
