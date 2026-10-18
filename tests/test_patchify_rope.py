"""Unit and Acceptance Tests for Part 4: Spacetime Patchifier and 3D RoPE.

Acceptance Criteria:
1. Patchify then unpatchify is a shape-exact inverse (identity weights test).
2. RoPE relative-position test: attention score between two tokens depends only on their offset.
3. The 480p, 81-frame preset gives exactly 32,760 tokens.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from vidgen.models.patchify import (
    SpacetimePatchEmbed,
    SpacetimeUnpatchify,
    configure_identity_patchify_unpatchify,
    pack_varlen_tokens,
    pad_varlen_tokens,
)
from vidgen.models.rope import (
    RotaryEmbedding3D,
    partition_head_dim,
    precompute_freqs_1d,
)


def test_wan480p_81_frames_exact_32760_tokens():
    """Acceptance Test 1:

    Wan 480p (480x832), 81-frame preset gives EXACTLY 32,760 tokens.

    Calculation:
    - Frames T=81 after 4x temporal VAE: t = (81-1)//4 + 1 = 21
    - Spatial 480x832 after 8x spatial VAE: h = 480//8 = 60, w = 832//8 = 104
    - Patch size (1, 2, 2):
        grid_t = 21 // 1 = 21
        grid_h = 60 // 2 = 30
        grid_w = 104 // 2 = 52
        Total Tokens N = 21 * 30 * 52 = 32,760
    """
    patch_embed = SpacetimePatchEmbed(
        in_channels=16,
        hidden_dim=384,
        patch_size=(1, 2, 2),
    )

    t = (81 - 1) // 4 + 1  # 21
    h = 480 // 8           # 60
    w = 832 // 8           # 104

    dummy_latents = torch.zeros(1, 16, t, h, w)
    tokens, grid = patch_embed(dummy_latents)

    B, N, D = tokens.shape
    assert N == 32760, f"Expected exactly 32,760 tokens, got {N}"
    assert grid == (21, 30, 52), f"Expected grid (21, 30, 52), got {grid}"
    assert 21 * 30 * 52 == 32760


def test_patchify_unpatchify_shape_exact_inverse_identity():
    """Acceptance Test 2:

    Patchify then unpatchify is a shape-exact, value-exact inverse under identity weights.
    """
    in_channels = 16
    patch_size = (1, 2, 2)
    B, C, t, h, w = 2, 16, 5, 8, 8

    embed, unpatch = configure_identity_patchify_unpatchify(
        in_channels=in_channels,
        patch_size=patch_size,
    )

    x = torch.randn(B, C, t, h, w)
    tokens, grid = embed(x)
    recon = unpatch(tokens, grid)

    # Verify shapes match exactly
    assert recon.shape == x.shape, f"Recon shape {recon.shape} != original shape {x.shape}"

    # Verify bit-exact identity within floating point precision
    diff = (recon - x).abs().max().item()
    assert diff < 1e-6, f"Identity round-trip reconstruction failed! Max diff = {diff}"


def test_3d_rope_relative_position_test():
    """Acceptance Test 3:

    RoPE relative-position test: The attention score between two tokens depends ONLY on their offset.
    score(q @ pos1, k @ pos2) == score(q @ (pos1 + delta), k @ (pos2 + delta))
    """
    head_dim = 128
    rope = RotaryEmbedding3D(head_dim=head_dim, theta=10000.0)

    # Grid: [grid_t=5, grid_h=8, grid_w=8]
    grid = (5, 8, 8)
    N = 5 * 8 * 8  # 320 tokens

    # Token 1 at (t=1, h=2, w=2) -> index = 1*(8*8) + 2*8 + 2 = 64 + 16 + 2 = 82
    idx1 = 1 * (8 * 8) + 2 * 8 + 2
    # Token 2 at (t=2, h=4, w=5) -> index = 2*(8*8) + 4*8 + 5 = 128 + 32 + 5 = 165
    idx2 = 2 * (8 * 8) + 4 * 8 + 5

    # Shift both tokens by delta: delta_t = 1, delta_h = 2, delta_w = 1
    # Shifted Token 1 at (t=2, h=4, w=3) -> index = 2*(8*8) + 4*8 + 3 = 128 + 32 + 3 = 163
    idx1_shifted = (1 + 1) * (8 * 8) + (2 + 2) * 8 + (2 + 1)
    # Shifted Token 2 at (t=3, h=6, w=6) -> index = 3*(8*8) + 6*8 + 6 = 192 + 48 + 6 = 246
    idx2_shifted = (2 + 1) * (8 * 8) + (4 + 2) * 8 + (5 + 1)

    torch.manual_seed(42)
    q = torch.randn(1, 1, N, head_dim)
    k = torch.randn(1, 1, N, head_dim)

    # Make query at idx1 and idx1_shifted have the same feature representation
    q[0, 0, idx1_shifted] = q[0, 0, idx1].clone()
    # Make key at idx2 and idx2_shifted have the same feature representation
    k[0, 0, idx2_shifted] = k[0, 0, idx2].clone()

    q_rot, k_rot = rope(q, k, grid)

    # Attention score between pos1 and pos2
    score_original = (q_rot[0, 0, idx1] * k_rot[0, 0, idx2]).sum().item()
    # Attention score between pos1+delta and pos2+delta
    score_shifted = (q_rot[0, 0, idx1_shifted] * k_rot[0, 0, idx2_shifted]).sum().item()

    diff = abs(score_original - score_shifted)
    assert diff < 1e-4, f"RoPE relative position invariance failed! Score diff = {diff}"


def test_rope_head_dim_partitions():
    """Verify Wan 44/42/42 partition for head_dim=128 and 22/22/20 for head_dim=64."""
    p128 = partition_head_dim(128)
    assert p128 == (44, 42, 42), f"Expected (44, 42, 42) for head_dim 128, got {p128}"
    assert sum(p128) == 128
    assert all(dim % 2 == 0 for dim in p128)

    p64 = partition_head_dim(64)
    assert p64 == (22, 22, 20), f"Expected (22, 22, 20) for head_dim 64, got {p64}"
    assert sum(p64) == 64
    assert all(dim % 2 == 0 for dim in p64)


def test_varlen_packing_and_padding_mask():
    """Verify variable-length sequence handling: cu_seqlens packing and padded batching."""
    patch_embed = SpacetimePatchEmbed(in_channels=16, hidden_dim=64, patch_size=(1, 2, 2))

    # Sample 1: 5 frames, 8x8 -> 5*4*4 = 80 tokens
    x1 = torch.randn(1, 16, 5, 8, 8)
    t1, g1 = patch_embed(x1)
    assert t1.shape[1] == 80

    # Sample 2: 1 frame (image), 8x8 -> 1*4*4 = 16 tokens
    x2 = torch.randn(1, 16, 1, 8, 8)
    t2, g2 = patch_embed(x2)
    assert t2.shape[1] == 16

    # 1. Packed Varlen Attention
    packed_info = pack_varlen_tokens([t1, t2], [g1, g2])
    assert packed_info["packed_tokens"].shape == (1, 96, 64)
    assert packed_info["cu_seqlens"].tolist() == [0, 80, 96]
    assert packed_info["max_seqlen"] == 80
    assert packed_info["grid_shapes"] == [(5, 4, 4), (1, 4, 4)]

    # 2. Padded with Attention Mask
    padded_info = pad_varlen_tokens([t1, t2], [g1, g2])
    assert padded_info["padded_tokens"].shape == (2, 80, 64)
    assert padded_info["attention_mask"].shape == (2, 80)
    # Sample 2 should have 16 True and 64 False
    assert padded_info["attention_mask"][1, :16].all()
    assert not padded_info["attention_mask"][1, 16:].any()


def test_image_to_video_extra_conditioning_channels():
    """Verify extra conditioning frame support (mask + reference latent frames)."""
    # 16 latent channels + 1 mask channel + 16 reference latent channels = 33 channels
    embed_i2v = SpacetimePatchEmbed(
        in_channels=16,
        extra_channels=17,
        hidden_dim=128,
        patch_size=(1, 2, 2),
    )

    noisy_latents = torch.randn(1, 16, 5, 8, 8)
    cond = torch.randn(1, 17, 5, 8, 8)

    tokens, grid = embed_i2v(noisy_latents, cond=cond)
    assert tokens.shape == (1, 80, 128)
    assert grid == (5, 4, 4)
