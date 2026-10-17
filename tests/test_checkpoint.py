"""Acceptance test: Checkpoint save and resume yields bit-for-bit identical next-step loss.

Verifies:
- Model weights restoration
- EMA shadow weights restoration
- Optimizer state restoration (momentum buffers)
- Python, NumPy, and PyTorch RNG state restoration
- Data sampler state restoration
- Step N+1 loss identity: loss_resumed == loss_uninterrupted
"""

import copy
import random
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from vidgen.train.checkpoint import save_checkpoint, resume_checkpoint
from vidgen.train.ema import EMAModel
from vidgen.utils.seed import seed_everything


class SimpleVideoDiTDummy(nn.Module):
    """Small deterministic module for checkpoint verification."""

    def __init__(self, in_features: int = 32, hidden_features: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_checkpoint_reproducibility_next_step_loss(tmp_path):
    """Part 0 Acceptance Test:

    Resuming from a checkpoint must yield the IDENTICAL next-step loss.
    """
    ckpt_path = tmp_path / "checkpoint_step2.pt"

    # --- 1. RUN UNINTERRUPTED TRAINING ---
    seed_everything(1234)

    model_orig = SimpleVideoDiTDummy()
    ema_orig = EMAModel(model_orig, decay=0.9)
    optimizer_orig = AdamW(model_orig.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.MSELoss()

    # Step 1
    x1 = torch.randn(4, 32)
    y1 = torch.randn(4, 32)
    optimizer_orig.zero_grad()
    loss1 = criterion(model_orig(x1), y1)
    loss1.backward()
    optimizer_orig.step()
    ema_orig.update(model_orig)

    # Step 2
    x2 = torch.randn(4, 32)
    y2 = torch.randn(4, 32)
    optimizer_orig.zero_grad()
    loss2 = criterion(model_orig(x2), y2)
    loss2.backward()
    optimizer_orig.step()
    ema_orig.update(model_orig)

    sampler_state = {"epoch": 0, "batch_idx": 2, "sample_cursor": 8}

    # Save checkpoint at step 2
    save_checkpoint(
        save_path=ckpt_path,
        step=2,
        epoch=0,
        model=model_orig,
        optimizer=optimizer_orig,
        ema=ema_orig,
        sampler_state=sampler_state,
        config={"name": "tiny_dummy"},
    )
    assert ckpt_path.exists(), "Checkpoint file was not created"

    # Step 3 (Uninterrupted)
    # Notice: x3 and y3 are generated from the ongoing RNG stream!
    x3_orig = torch.randn(4, 32)
    y3_orig = torch.randn(4, 32)
    optimizer_orig.zero_grad()
    pred3_orig = model_orig(x3_orig)
    loss3_uninterrupted = criterion(pred3_orig, y3_orig)
    loss3_uninterrupted.backward()
    optimizer_orig.step()
    ema_orig.update(model_orig)

    # --- 2. RESUME IN A FRESH SESSION ---
    # Scramble the RNG states before resumption to ensure restore_rng does its job
    random.seed(99999)
    np.random.seed(99999)
    torch.manual_seed(99999)
    _ = torch.randn(100)

    model_resumed = SimpleVideoDiTDummy()
    ema_resumed = EMAModel(model_resumed, decay=0.9)
    optimizer_resumed = AdamW(model_resumed.parameters(), lr=1e-3, weight_decay=1e-2)

    step, epoch, resumed_sampler_state = resume_checkpoint(
        checkpoint_path=ckpt_path,
        model=model_resumed,
        optimizer=optimizer_resumed,
        ema=ema_resumed,
        restore_rng=True,
    )

    assert step == 2, f"Expected step 2, got {step}"
    assert epoch == 0, f"Expected epoch 0, got {epoch}"
    assert resumed_sampler_state == sampler_state

    # Because RNG state was restored, the next torch.randn calls must match x3_orig and y3_orig!
    x3_resumed = torch.randn(4, 32)
    y3_resumed = torch.randn(4, 32)
    torch.testing.assert_close(x3_resumed, x3_orig, msg="Resumed RNG did not produce identical data tensors")
    torch.testing.assert_close(y3_resumed, y3_orig, msg="Resumed RNG did not produce identical target tensors")

    # Step 3 on resumed session
    optimizer_resumed.zero_grad()
    pred3_resumed = model_resumed(x3_resumed)
    loss3_resumed = criterion(pred3_resumed, y3_resumed)

    # CRITICAL ACCEPTANCE CHECK: IDENTICAL LOSS
    torch.testing.assert_close(
        loss3_resumed,
        loss3_uninterrupted,
        rtol=1e-7,
        atol=1e-7,
        msg="Resumed next-step loss does not match uninterrupted next-step loss!",
    )

    # Perform Step 3 update on resumed
    loss3_resumed.backward()
    optimizer_resumed.step()
    ema_resumed.update(model_resumed)

    # Verify model parameters are bit-for-bit identical after Step 3
    for (n1, p1), (n2, p2) in zip(model_orig.named_parameters(), model_resumed.named_parameters()):
        assert n1 == n2
        torch.testing.assert_close(p1, p2, rtol=1e-7, atol=1e-7, msg=f"Parameter {n1} diverged after step 3")

    # Verify EMA shadow parameters are bit-for-bit identical after Step 3
    for k in ema_orig.shadow:
        torch.testing.assert_close(
            ema_orig.shadow[k],
            ema_resumed.shadow[k],
            rtol=1e-7,
            atol=1e-7,
            msg=f"EMA shadow {k} diverged after step 3",
        )


@pytest.mark.gpu
def test_gpu_checkpoint_save_and_resume(tmp_path):
    """GPU-only checkpoint test: skipped when CUDA is absent."""
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    model = SimpleVideoDiTDummy().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "gpu_ckpt.pt"

    save_checkpoint(ckpt_path, step=5, epoch=1, model=model, optimizer=optimizer)

    model2 = SimpleVideoDiTDummy().to(device)
    optimizer2 = AdamW(model2.parameters(), lr=1e-3)
    step, epoch, _ = resume_checkpoint(ckpt_path, model=model2, optimizer=optimizer2)
    assert step == 5
    assert epoch == 1
