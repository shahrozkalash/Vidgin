"""VAE growth: transplant the trained 16/32 VAE into the grown 32/64
geometry (vidgen/vae/grow.py), plus the config plumbing that makes
VAE shape a per-preset, checkpoint-owned property.

The smoke test uses the real finished Stage 2 checkpoint and a real
held-out clip when they exist on this machine, and skips otherwise —
the synthetic tests pin the transplant rules everywhere.
"""

import math
import os

import pytest
import torch

from vidgen.checkpoint import load_checkpoint, vae_weights
from vidgen.config import VidgenConfig, get_preset
from vidgen.vae.grow import (
    grow_vae_payload,
    grow_vae_state_dict,
)
from vidgen.vae.model import CausalVAE, build_vae, infer_vae_shape

REAL_CKPT = "/home/hatch/workspace/tmp/vae_r4cont_final/step_020500.pt"
REAL_CLIP = "/home/hatch/workspace/round5_local/city1/clip_000021.npy"

OLD_PARAMS = 3_650_851
NEW_PARAMS = 14_588_483


def _fresh_old_sd(seed=0):
    torch.manual_seed(seed)
    return CausalVAE(16, 32).state_dict()


# ------------------------------------------------------------- (a) size
def test_grown_param_count():
    grown, stats = grow_vae_state_dict(_fresh_old_sd())
    assert stats["vae_total_params"] == NEW_PARAMS
    assert stats["vae_source_params"] == OLD_PARAMS
    assert sum(v.numel() for v in grown.values()) == NEW_PARAMS
    with torch.device("meta"):
        ref = CausalVAE(32, 64)
    assert set(grown) == set(ref.state_dict())
    for k, v in grown.items():
        assert tuple(v.shape) == tuple(ref.state_dict()[k].shape)


# ------------------------------------------- (b) transplant rules exact
def test_overlapping_slices_copied_exactly():
    src = _fresh_old_sd()
    grown, stats = grow_vae_state_dict(src)
    assert stats["vae_copied_params"] == OLD_PARAMS
    for k, s in src.items():
        g = grown[k]
        slices = tuple(slice(0, n) for n in s.shape)
        assert torch.equal(g[slices], s), f"{k}: overlap not copied exactly"


def test_new_capacity_starts_silent():
    src = _fresh_old_sd()
    grown, _ = grow_vae_state_dict(src)
    # New latent channels: mu/logvar rows are zero weight AND zero bias,
    # so the new channels start at mu=0, logvar=0.
    for head in ("encoder.mu", "encoder.logvar"):
        assert torch.all(grown[f"{head}.weight"][16:] == 0)
        assert torch.all(grown[f"{head}.bias"][16:] == 0)
    # Decoder ignores the new latent channels at init: their in_conv
    # columns are zero.
    assert torch.all(grown["decoder.in_conv.weight"][:, 16:] == 0)
    # New output rows of every widened conv are zero (silent channels).
    assert torch.all(grown["encoder.in_conv.conv.weight"][32:] == 0)
    assert torch.all(grown["encoder.down2.conv.weight"][128:] == 0)
    # ConvTranspose weights are [in, out, ...]: new INPUT rows are zero
    # and the new OUTPUT columns on old rows are zero.
    assert torch.all(grown["decoder.up2.weight"][128:] == 0)
    assert torch.all(grown["decoder.up2.weight"][:128, 64:] == 0)
    assert torch.all(grown["decoder.up1.weight"][64:] == 0)
    assert torch.all(grown["decoder.up1.weight"][:64, 32:] == 0)
    # Old outputs do not read new inputs: (old out) x (new in) is zero.
    assert torch.all(grown["encoder.down1.conv.weight"][:64, 32:] == 0)
    assert torch.all(grown["decoder.out_conv.conv.weight"][:, 32:] == 0)
    # GroupNorm affine: old entries copied, new entries weight 1 / bias 0.
    key = next(k for k in grown if k.endswith(".gn.weight"))
    assert grown[key].shape[0] == 256 and src[key].shape[0] == 128
    assert torch.equal(grown[key][:128], src[key][:128])
    assert torch.all(grown[key][128:] == 1)
    assert torch.all(grown[key.replace(".weight", ".bias")][128:] == 0)


def test_grow_rejects_shrink_and_shape_lies():
    src = _fresh_old_sd()
    with pytest.raises(ValueError, match="declared"):
        grow_vae_state_dict(src, old_latent_ch=8, old_base_ch=32)
    with pytest.raises(ValueError, match="smaller"):
        grow_vae_state_dict(src, new_latent_ch=32, new_base_ch=16)


# ------------------------------------------------------- payload rules
def test_payload_growth_step_config_and_dropped_optimizer():
    torch.manual_seed(0)
    donor = {
        "kind": "vae", "preset": "small", "step": 20500,
        "vae": _fresh_old_sd(), "opt": {"state": {}}, "sampler": {},
        "config": get_preset("small").to_dict(),
        "loss_history": [0.01],
    }
    # The donor payload's stored config is the OLD small config (16/32).
    donor["config"]["vae_latent_ch"] = 16
    donor["config"]["vae_base_ch"] = 32
    donor["config"]["latent_channels"] = 16
    out = grow_vae_payload(donor, "small")
    assert out["kind"] == "vae" and out["preset"] == "small"
    assert out["step"] == 20500                    # step preserved
    assert "opt" not in out and "sampler" not in out  # weights-only
    cfg = VidgenConfig.from_dict(out["config"])
    assert (cfg.vae_latent_ch, cfg.vae_base_ch) == (32, 64)
    assert cfg.latent_channels == 32
    assert infer_vae_shape(out["vae"]) == (32, 64)
    assert out["grow_vae"]["vae_total_params"] == NEW_PARAMS


def test_payload_growth_refuses_same_shape():
    donor = {"kind": "vae", "preset": "tiny", "step": 10,
             "vae": _fresh_old_sd()}
    with pytest.raises(ValueError, match="no growth needed"):
        grow_vae_payload(donor, "tiny")


def test_payload_growth_rejects_non_vae():
    with pytest.raises(ValueError, match="VAE checkpoint"):
        grow_vae_payload({"kind": "dit"}, "small")


# ------------------------------------------------------- (e) config
def test_config_vae_shapes():
    small = get_preset("small")
    assert (small.vae_latent_ch, small.vae_base_ch) == (32, 64)
    assert small.latent_channels == 32
    tiny = get_preset("tiny")
    assert (tiny.vae_latent_ch, tiny.vae_base_ch) == (16, 32)
    assert tiny.latent_channels == 16
    for name in ("vidgen480p", "vidgen9b", "vidgen14b"):
        cfg = get_preset(name)
        assert (cfg.vae_latent_ch, cfg.vae_base_ch) == (16, 32), name
        assert cfg.latent_channels == 16, name


def test_config_old_dict_without_vae_fields_defaults_to_16_32():
    # Checkpoints saved before the vae_* fields existed must still
    # parse, with the historical shape as the default.
    d = get_preset("small").to_dict()
    for k in ("vae_latent_ch", "vae_base_ch", "latent_channels"):
        d.pop(k, None)
    cfg = VidgenConfig.from_dict(d)
    assert (cfg.vae_latent_ch, cfg.vae_base_ch) == (16, 32)
    assert cfg.latent_channels == 16


def test_config_rejects_vae_dit_channel_mismatch():
    cfg = get_preset("tiny")
    cfg.vae_latent_ch = 24
    with pytest.raises(ValueError, match="must equal"):
        cfg.validate()


# -------------------------------------- (d) old checkpoints keep shape
def test_build_vae_sizes_from_payload_not_preset():
    payload = {"kind": "vae", "vae": _fresh_old_sd()}
    vae = build_vae(payload=payload)
    assert infer_vae_shape(vae.state_dict()) == (16, 32)
    vae.load_state_dict(payload["vae"])  # strict: must not raise
    assert sum(p.numel() for p in vae.parameters()) == OLD_PARAMS
    # ... while the small preset builds the grown shape.
    grown = build_vae(get_preset("small"))
    assert infer_vae_shape(grown.state_dict()) == (32, 64)
    assert sum(p.numel() for p in grown.parameters()) == NEW_PARAMS


# --------------------------------- (c) real checkpoint + clip smoke
def _psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10.0 * math.log10(4.0 / mse)


@pytest.fixture(scope="module")
def real_pair():
    if not (os.path.isfile(REAL_CKPT) and os.path.isfile(REAL_CLIP)):
        pytest.skip("real Stage 2 checkpoint / held-out clip not on "
                    "this machine")
    payload = load_checkpoint(REAL_CKPT)
    old = build_vae(payload=payload)
    old.load_state_dict(vae_weights(payload))
    old.eval()
    grown_sd, _ = grow_vae_state_dict(vae_weights(payload))
    new = CausalVAE(32, 64)
    new.load_state_dict(grown_sd)
    new.eval()
    import numpy as np
    arr = np.load(REAL_CLIP)
    x = (torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous().float()
         / 127.5 - 1.0).unsqueeze(0)
    return old, new, x


@torch.no_grad()
def test_real_checkpoint_loads_with_own_shape(real_pair):
    old, _, _ = real_pair
    assert infer_vae_shape(old.state_dict()) == (16, 32)
    assert sum(p.numel() for p in old.parameters()) == OLD_PARAMS


@torch.no_grad()
def test_grown_at_init_reconstruction_ballpark(real_pair):
    # Measured at build time on this clip: donor 25.98 dB, grown-at-init
    # 25.50 dB (the ~0.5 dB mid-GroupNorm transplant tax documented in
    # vidgen/vae/grow.py). The bar is "same ballpark", with headroom
    # for CPU float variation — NOT bit-equality.
    old, new, x = real_pair
    mu_o, _ = old.encode(x)
    r_o = old.decode(mu_o)
    mu_n, _ = new.encode(x)
    r_n = new.decode(mu_n)
    old_psnr, new_psnr = _psnr(r_o, x), _psnr(r_n, x)
    assert old_psnr > 20.0, f"donor recon suspiciously bad: {old_psnr}"
    assert new_psnr > old_psnr - 1.5, \
        f"grown-at-init {new_psnr:.2f} dB vs donor {old_psnr:.2f} dB"
    # The grown encoder's trunk is exact up to the mid block; the new
    # latent channels start at exactly mu=0, logvar=0.
    assert torch.all(mu_n[:, 16:] == 0)
    # The grown decoder fed the donor's own latents (zero-padded)
    # reproduces the donor's reconstruction closely.
    z = torch.zeros(1, 32, *mu_o.shape[2:])
    z[:, :16] = mu_o
    r_dec = new.decode(z)
    assert _psnr(r_dec, x) > old_psnr - 1.0
    assert float((r_dec - r_o).abs().max()) < 0.5
