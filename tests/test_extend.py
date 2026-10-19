"""Stage 1 upgrade: segment-chained long video (vidgen.extend).

CPU-feasible: Tiny preset, real DiT/VAE/text-encoder, minimal sampler steps.
Proves frame arithmetic, value sanity, prefix conditioning, determinism, and
that a single chained segment matches the plain sample path bit-for-bit.
"""

import numpy as np
import pytest
import torch

from vidgen.config import get_preset
from vidgen.dit.model import DiT
from vidgen.extend import (
    FRAME_MS,
    context_pixel_frames,
    generate_long,
    new_frames_per_segment,
    segments_for_seconds,
)
from vidgen.sampler import sample_latents
from vidgen.seed import seed_everything
from vidgen.text.encoder import TinyTextEncoder
from vidgen.vae.model import CausalVAE


@pytest.fixture()
def rig():
    seed_everything(7)
    cfg = get_preset("tiny")
    dit = DiT(cfg).eval()
    vae = CausalVAE().eval()
    enc = TinyTextEncoder(d_t=cfg.text_dim, max_len=cfg.text_max_len).eval()
    return cfg, dit, vae, enc


def test_frame_arithmetic():
    assert FRAME_MS == 290
    assert context_pixel_frames(2) == 5   # 4*2-3: causal VAE geometry
    assert context_pixel_frames(3) == 9
    cfg = get_preset("tiny")
    assert new_frames_per_segment(cfg, 2) == 12  # 17 - 5
    # ~30 s at 290 ms/frame -> target ceil(103.45)=104 -> 9 segments -> 113
    n = segments_for_seconds(cfg, 30.0, 2)
    assert n == 9
    total = cfg.frames + (n - 1) * new_frames_per_segment(cfg, 2)
    assert total == 113
    assert total * FRAME_MS / 1000.0 >= 30.0
    with pytest.raises(ValueError):
        segments_for_seconds(cfg, 0.0, 2)
    with pytest.raises(ValueError):
        context_pixel_frames(0)


def test_chained_output_shape_and_range(rig):
    cfg, dit, vae, enc = rig
    frames = generate_long(dit, vae, enc, cfg, "digit 3 moving left",
                           segments=3, steps=2, seed=11)
    assert frames.shape == (41, 64, 64, 3)  # 17 + 2*12
    assert np.isfinite(frames).all()
    assert frames.min() >= -1.0 - 1e-5 and frames.max() <= 1.0 + 1e-5
    # chaining must actually change the picture: later segments differ
    assert not np.allclose(frames[:17], frames[17:34])


def test_chained_deterministic(rig):
    cfg, dit, vae, enc = rig
    a = generate_long(dit, vae, enc, cfg, "digit 3 moving left",
                      segments=3, steps=2, seed=42)
    b = generate_long(dit, vae, enc, cfg, "digit 3 moving left",
                      segments=3, steps=2, seed=42)
    assert np.array_equal(a, b)
    c = generate_long(dit, vae, enc, cfg, "digit 3 moving left",
                      segments=3, steps=2, seed=43)
    assert not np.array_equal(a, c)


def test_single_segment_matches_sample_path(rig):
    cfg, dit, vae, enc = rig
    frames = generate_long(dit, vae, enc, cfg, "digit 3 moving left",
                           segments=1, steps=2, seed=5)
    # reference: the exact sample.py path with the same seed
    text, text_mask = enc(["digit 3 moving left"])
    null_text, null_mask = enc.null_embedding()
    lat_shape = (1, cfg.latent_channels, cfg.latent_t, cfg.latent_h, cfg.latent_w)
    z = sample_latents(dit, lat_shape, text, text_mask, null_text, null_mask,
                       method="euler", num_steps=2, cfg_scale=7.5,
                       seed=5, device="cpu")
    with torch.no_grad():
        ref = vae.decode(z / cfg.latent_scale)[0].permute(1, 2, 3, 0).numpy()
    assert np.array_equal(frames, ref.astype(np.float32))


def test_prefix_hook_clamps_context(rig):
    cfg, dit, vae, enc = rig
    text, text_mask = enc(["digit 3 moving left"])
    null_text, null_mask = enc.null_embedding()
    lat_shape = (1, cfg.latent_channels, cfg.latent_t, cfg.latent_h, cfg.latent_w)
    prefix = torch.randn(1, cfg.latent_channels, 2, cfg.latent_h, cfg.latent_w)
    out = sample_latents(dit, lat_shape, text, text_mask, null_text, null_mask,
                         num_steps=2, seed=0, prefix_latent=prefix, prefix_t=2)
    # final step clamps the context to the straight-line endpoint: exactly
    # the given context at t=1
    assert torch.allclose(out[..., :2, :, :], prefix.to(out.dtype), atol=1e-6)
    # ...and the free region was still generated, not copied from the prefix
    free = out[..., 2:, :, :]
    assert torch.isfinite(free).all() and free.std() > 0
    assert not torch.allclose(free, prefix[..., :1, :, :].expand_as(free))


def test_prefix_hook_rejects_bad_shape(rig):
    cfg, dit, vae, enc = rig
    text, text_mask = enc(["digit 3 moving left"])
    null_text, null_mask = enc.null_embedding()
    lat_shape = (1, cfg.latent_channels, cfg.latent_t, cfg.latent_h, cfg.latent_w)
    bad = torch.randn(1, cfg.latent_channels, 3, cfg.latent_h, cfg.latent_w)
    with pytest.raises(ValueError):
        sample_latents(dit, lat_shape, text, text_mask, null_text, null_mask,
                       num_steps=2, seed=0, prefix_latent=bad, prefix_t=2)
    with pytest.raises(ValueError):
        sample_latents(dit, lat_shape, text, text_mask, null_text, null_mask,
                       num_steps=2, seed=0, prefix_t=2)  # no context given
    with pytest.raises(ValueError):
        generate_long(dit, vae, enc, cfg, "x", segments=0)
    with pytest.raises(ValueError):
        generate_long(dit, vae, enc, cfg, "x", segments=2, ctx_latent=5)


def test_crossfade_stays_in_range(rig):
    cfg, dit, vae, enc = rig
    # blend larger than default still yields convex blends in [-1, 1]
    frames = generate_long(dit, vae, enc, cfg, "digit 3 moving left",
                           segments=3, steps=2, seed=9, blend=5)
    assert frames.shape == (41, 64, 64, 3)
    assert np.isfinite(frames).all()
    assert frames.min() >= -1.0 - 1e-5 and frames.max() <= 1.0 + 1e-5
    with pytest.raises(ValueError):
        generate_long(dit, vae, enc, cfg, "x", segments=2, blend=6)
