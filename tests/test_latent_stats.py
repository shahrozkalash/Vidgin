"""Tests for Phase 3: measured latent statistics + latent_scale wiring."""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vidgen.data.moving_mnist import MovingMNISTDataset
from vidgen.train import DiTTrainer
from vidgen.vae.model import CausalVAE
from vidgen.vae.stats import measure_latent_stats, save_recon_preview


@pytest.fixture(scope="module")
def dataset():
    return MovingMNISTDataset(8, height=64, width=64, frames=17, base_seed=5)


def test_measure_latent_stats(dataset):
    vae = CausalVAE()
    report = measure_latent_stats(vae, dataset, max_clips=8, batch_size=4)
    assert report["clips_measured"] == 8
    assert report["latent_channels"] == 16
    assert len(report["mean_per_channel"]) == 16
    assert len(report["std_per_channel"]) == 16
    assert report["global_std"] > 0
    assert report["recommended_latent_scale"] == pytest.approx(
        1.0 / report["global_std"], rel=1e-4)
    assert math.isfinite(report["recon_mse"]) and report["recon_mse"] > 0
    assert math.isfinite(report["recon_psnr_db"])


def test_recon_preview_png(dataset, tmp_path):
    vae = CausalVAE()
    clip, _ = dataset[0]
    out = str(tmp_path / "preview.png")
    save_recon_preview(vae, clip, out)
    from PIL import Image
    img = Image.open(out)
    # 17 frames across, input+recon rows, upscaled x4 from 64px tiles.
    assert img.width == 64 * 17 * 4
    assert img.height == 64 * 2 * 4


def test_latent_scale_override_and_resume(tmp_path):
    tr = DiTTrainer(preset="tiny", seed=0, num_clips=8, batch_size=2,
                    ckpt_dir=None, latent_scale=2.5)
    assert tr.cfg.latent_scale == 2.5
    path = str(tmp_path / "dit_scaled.pt")
    tr.save(path)

    from vidgen.checkpoint import load_checkpoint
    assert load_checkpoint(path)["config"]["latent_scale"] == 2.5

    # Resume must restore the checkpoint's scale, not the preset's 1.0.
    tr2 = DiTTrainer.resume(path, seed=0, num_clips=8, batch_size=2,
                            ckpt_dir=None)
    assert tr2.cfg.latent_scale == 2.5

    with pytest.raises(ValueError):
        DiTTrainer(preset="tiny", num_clips=8, batch_size=2,
                   ckpt_dir=None, latent_scale=0)
