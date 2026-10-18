"""Part 0: config loading, preset dimensions, frame rule, roundtrip."""

import pytest

from vidgen.config import VidgenConfig, get_preset, resolve_device


def test_tiny_config_loading():
    cfg = get_preset("tiny")
    assert cfg.height == 64 and cfg.width == 64 and cfg.frames == 17
    assert (cfg.latent_channels, cfg.latent_t, cfg.latent_h, cfg.latent_w) == (16, 5, 8, 8)
    assert cfg.num_tokens == 80
    d = cfg.dit
    assert (d.layers, d.dim, d.heads, d.ffn) == (8, 384, 6, 1536)


def test_small_preset_dimensions():
    cfg = get_preset("small")
    assert (cfg.height, cfg.width, cfg.frames) == (256, 256, 33)
    assert (cfg.latent_t, cfg.latent_h, cfg.latent_w) == (9, 32, 32)
    assert cfg.num_tokens == 2304
    assert (cfg.dit.layers, cfg.dit.dim, cfg.dit.heads, cfg.dit.ffn) == (16, 1024, 16, 4096)


def test_vidgen_480p_preset_dimensions():
    cfg = get_preset("vidgen480p")
    assert (cfg.height, cfg.width, cfg.frames) == (480, 832, 81)
    assert (cfg.latent_t, cfg.latent_h, cfg.latent_w) == (21, 60, 104)
    assert cfg.num_tokens == 32760
    assert (cfg.dit.layers, cfg.dit.dim, cfg.dit.heads, cfg.dit.ffn) == (30, 1536, 12, 8960)


def test_vidgen_14b_preset_dimensions():
    cfg = get_preset("vidgen14b")
    assert (cfg.height, cfg.width, cfg.frames) == (720, 1280, 81)
    assert (cfg.latent_t, cfg.latent_h, cfg.latent_w) == (21, 90, 160)
    assert cfg.num_tokens == 75600
    assert (cfg.dit.layers, cfg.dit.dim, cfg.dit.heads, cfg.dit.ffn) == (40, 5120, 40, 13824)


def test_frame_rule_validation():
    bad = get_preset("tiny")
    bad.frames = 16
    with pytest.raises(ValueError, match="4k\\+1"):
        bad.validate()
    bad = get_preset("tiny")
    bad.height = 60
    with pytest.raises(ValueError, match="multiples of 16"):
        bad.validate()


def test_single_frame_allowed():
    cfg = get_preset("tiny")
    cfg.frames = 1
    cfg.latent_t = 1
    cfg.validate()  # T=1 -> 1 latent (unified image/video latent space)


def test_config_roundtrip(tmp_path):
    cfg = get_preset("tiny")
    p = str(tmp_path / "cfg.json")
    cfg.save(p)
    cfg2 = VidgenConfig.load(p)
    assert cfg2.to_dict() == cfg.to_dict()


def test_device_agnostic_resolution():
    # no hardcoded device strings: vidgen must resolve device dynamically
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "cuda")
    import pathlib

    for f in pathlib.Path("vidgen").rglob("*.py"):
        src = f.read_text()
        assert '.cuda(' not in src, f"hardcoded .cuda() in {f}"
        assert 'torch.device("cuda' not in src, f"hardcoded cuda device in {f}"


@pytest.mark.skipif(__import__("torch").cuda.is_available() is False, reason="needs CUDA")
def test_gpu_config_resolution():
    assert resolve_device("auto") == "cuda"
