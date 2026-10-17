"""Unit tests for configuration system, preset dimensions, and token counting."""

import pytest
import torch
from pathlib import Path

from vidgen.utils.config import (
    VidGenConfig,
    SystemConfig,
    VaeConfig,
    PatchifyConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    SamplerConfig,
    load_config,
    save_config,
)


def test_tiny_config_loading():
    """Verify tiny.yaml loads accurately with Track A settings."""
    cfg_path = Path("vidgen/configs/tiny.yaml")
    assert cfg_path.exists(), "vidgen/configs/tiny.yaml must exist"

    cfg = load_config(cfg_path)
    assert cfg.track == "A"
    assert cfg.name == "tiny"
    assert cfg.seed == 42

    # Presets for Tiny: 64x64, 17 frames, latent 5x8x8, 80 tokens
    assert cfg.data.frames == 17
    assert cfg.data.height == 64
    assert cfg.data.width == 64

    # Latent shape: t = (17 - 1)//4 + 1 = 5; h = 64//8 = 8; w = 64//8 = 8
    t, h, w = cfg.compute_latent_shape()
    assert (t, h, w) == (5, 8, 8), f"Expected latent (5, 8, 8), got {(t, h, w)}"

    # Patch size (1, 2, 2) -> (5/1) * (8/2) * (8/2) = 5 * 4 * 4 = 80 tokens
    tokens = cfg.compute_token_count()
    assert tokens == 80, f"Expected 80 tokens for Tiny, got {tokens}"

    # Model architecture: 8 layers, 384 dim, 6 heads, 1536 ffn
    assert cfg.model.num_layers == 8
    assert cfg.model.hidden_dim == 384
    assert cfg.model.num_heads == 6
    assert cfg.model.head_dim == 64
    assert cfg.model.ffn_dim == 1536


def test_small_preset_dimensions():
    """Verify small.yaml preset: 256x256, 33 frames, latent 9x32x32, 2,304 tokens."""
    cfg = load_config("vidgen/configs/small.yaml")
    assert cfg.data.frames == 33
    assert cfg.data.height == 256
    assert cfg.data.width == 256

    t, h, w = cfg.compute_latent_shape()
    assert (t, h, w) == (9, 32, 32)

    tokens = cfg.compute_token_count()
    assert tokens == 2304  # 9 * 16 * 16 = 2304

    assert cfg.model.num_layers == 16
    assert cfg.model.hidden_dim == 1024
    assert cfg.model.num_heads == 16
    assert cfg.model.head_dim == 64
    assert cfg.model.ffn_dim == 4096


def test_wan_480p_preset_dimensions():
    """Verify Wan 1.3B preset: 480x832, 81 frames, latent 21x60x104, 32,760 tokens."""
    cfg = load_config("vidgen/configs/wan1p3b.yaml")
    assert cfg.data.frames == 81
    assert cfg.data.height == 480
    assert cfg.data.width == 832

    t, h, w = cfg.compute_latent_shape()
    assert (t, h, w) == (21, 60, 104)

    tokens = cfg.compute_token_count()
    assert tokens == 32760  # 21 * 30 * 52 = 32760

    assert cfg.model.num_layers == 30
    assert cfg.model.hidden_dim == 1536
    assert cfg.model.num_heads == 12
    assert cfg.model.head_dim == 128
    assert cfg.model.ffn_dim == 8960


def test_device_agnostic_resolution(monkeypatch):
    """Verify device-agnostic behavior and CPU float32 fallback."""
    # Force CPU
    sys_cfg = SystemConfig(device="cpu", dtype="bfloat16")
    device = sys_cfg.resolve_device()
    dtype = sys_cfg.resolve_dtype()
    assert device == torch.device("cpu")
    # On CPU, resolve_dtype must return float32 to avoid unsupported CPU bfloat16 kernels
    assert dtype == torch.float32

    # Auto resolution without GPU
    sys_auto = SystemConfig(device="auto", dtype="float32")
    assert sys_auto.resolve_device() == torch.device("cpu")
    assert sys_auto.resolve_dtype() == torch.float32


def test_frame_rule_validation():
    """Verify Wan rule T = 4k + 1 assertion."""
    # Valid: 17 = 4*4+1, 33 = 4*8+1, 81 = 4*20+1
    DataConfig(frames=17)
    DataConfig(frames=33)
    DataConfig(frames=81)

    # Invalid: 16, 20, 64
    with pytest.raises(AssertionError):
        DataConfig(frames=16)

    with pytest.raises(AssertionError):
        DataConfig(frames=20)


def test_config_roundtrip(tmp_path):
    """Verify config can be saved and loaded without loss."""
    cfg = load_config("vidgen/configs/tiny.yaml")
    out_file = tmp_path / "roundtrip.yaml"
    save_config(cfg, out_file)

    loaded = load_config(out_file)
    assert loaded.name == cfg.name
    assert loaded.data.frames == cfg.data.frames
    assert loaded.model.hidden_dim == cfg.model.hidden_dim
    assert loaded.patchify.patch_size == cfg.patchify.patch_size


@pytest.mark.gpu
def test_gpu_config_resolution():
    """GPU-only test: automatically skipped if CUDA is absent."""
    assert torch.cuda.is_available(), "CUDA must be available if this runs"
    sys_cfg = SystemConfig(device="cuda", dtype="bfloat16")
    assert sys_cfg.resolve_device().type == "cuda"
    assert sys_cfg.resolve_dtype() == torch.bfloat16
