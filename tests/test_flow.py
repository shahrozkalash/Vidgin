"""Part 7: rectified flow objective."""

import torch

from vidgen.flow import flow_loss, rectified_flow_targets, sample_shifted_logit_normal


def test_interpolation_formula_exact():
    torch.manual_seed(0)
    x1 = torch.randn(2, 16, 5, 8, 8)
    t = torch.tensor([0.25, 0.75])
    xt, target = rectified_flow_targets(x1, t)
    # recover the noise that was drawn and verify x_t = (1-t) x0 + t x1
    torch.manual_seed(0)
    x1b = torch.randn(2, 16, 5, 8, 8)  # same draw sequence start
    assert torch.equal(x1, x1b)
    # xt - t*x1 over (1-t) must equal a standard normal draw independent of t
    t4 = t.view(-1, 1, 1, 1, 1)
    x0_rec = (xt - t4 * x1) / (1 - t4)
    assert x0_rec.shape == x1.shape
    assert torch.isfinite(x0_rec).all()
    assert torch.allclose(target, x1 - x0_rec, atol=1e-6)


def test_velocity_target_direction():
    x1 = torch.zeros(1, 2, 1, 1, 1)
    t = torch.tensor([0.5])
    xt, target = rectified_flow_targets(x1, t)
    # target points from noise toward data: target = x1 - x0
    assert torch.allclose(target, x1 - ((xt - 0.5 * x1) / 0.5), atol=1e-6)


def test_shifted_logit_normal_range():
    t = sample_shifted_logit_normal((1000,), s=5.0)
    assert ((t > 0) & (t < 1)).all()
    # shifted schedule (s=5) skews mass toward t -> 1 relative to uniform
    assert 0.6 < t.mean() < 0.95
    # s=1 must reduce to plain logit-normal (symmetric, mean ~0.5)
    t1 = sample_shifted_logit_normal((4000,), s=1.0)
    assert abs(t1.mean().item() - 0.5) < 0.05


def test_flow_loss_finite():
    pred = torch.randn(2, 16, 5, 8, 8)
    tgt = torch.randn(2, 16, 5, 8, 8)
    loss = flow_loss(pred, tgt)
    assert torch.isfinite(loss) and loss.item() >= 0
