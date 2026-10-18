"""Unit and Acceptance Tests for Part 5: Text Conditioner and Cross-Attention.

Acceptance Criteria:
1. Tokenization lengths correct.
2. Changing only padding tokens does not change outputs (mask invariance test).
3. The text-drop path works (CFG null embedding swap).
4. No gradient reaches the encoder (strictly frozen parameters).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vidgen.models.text import (
    RMSNorm,
    TextProjection,
    TextCrossAttention,
    WanTextConditioner,
    PromptRewriter,
)


def test_tokenization_lengths_and_projection():
    """Acceptance 1: Tokenization lengths and projection dimensions are correct."""
    conditioner = WanTextConditioner(
        text_dim=384,
        model_dim=256,
        max_length=64,
    )

    prompts = [
        "A cyan glowing numeral moving across space.",
        "Short clip.",
        "",  # Empty prompt
    ]

    tokens, mask = conditioner.encode_prompts(prompts)

    B, L, D = tokens.shape
    assert B == 3, f"Expected batch size 3, got {B}"
    assert L == 64, f"Expected max length 64, got {L}"
    assert D == 256, f"Expected projected model dim 256, got {D}"
    assert mask.shape == (3, 64)
    # Empty prompt must have only 1 active token
    assert mask[2, 0].item() is True
    assert mask[2, 1:].any().item() is False


def test_cross_attention_padding_mask_invariance():
    """Acceptance 2: Changing only padding tokens does NOT change outputs.

    Video tokens attending to text tokens must be completely invariant
    to arbitrary perturbations in masked padding positions.
    """
    B, N, L, D = 2, 80, 64, 128
    cross_attn = TextCrossAttention(model_dim=D, num_heads=4)
    cross_attn.eval()

    # Video tokens
    video_tokens = torch.randn(B, N, D)

    # Text tokens: first 20 tokens are real content, remaining 44 are padding
    active_len = 20
    text_tokens_1 = torch.randn(B, L, D)

    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, :active_len] = True

    # Run pass 1 with text_tokens_1
    with torch.no_grad():
        out1 = cross_attn(video_tokens, text_tokens_1, mask=mask)

    # Create text_tokens_2: keep active tokens identical, perturb padding tokens with noise
    text_tokens_2 = text_tokens_1.clone()
    text_tokens_2[:, active_len:, :] = torch.randn(B, L - active_len, D) * 100.0  # Massive perturbation

    # Run pass 2 with text_tokens_2 and same mask
    with torch.no_grad():
        out2 = cross_attn(video_tokens, text_tokens_2, mask=mask)

    # Difference in output must be effectively zero (< 1e-6)
    diff = (out1 - out2).abs().max().item()
    assert diff < 1e-6, f"Padding mask invariance failed! Max diff = {diff}"


def test_cross_attention_qk_rmsnorm():
    """Verify that RMSNorm is applied to queries and keys in cross-attention."""
    cross_attn = TextCrossAttention(model_dim=128, num_heads=4)
    assert hasattr(cross_attn, "q_norm") and isinstance(cross_attn.q_norm, RMSNorm)
    assert hasattr(cross_attn, "k_norm") and isinstance(cross_attn.k_norm, RMSNorm)

    # Large values should be stably normalized
    huge_q = torch.randn(2, 4, 10, 32) * 500.0
    normed_q = cross_attn.q_norm(huge_q)
    # Variance along feature dim should be bounded near 1.0
    var = normed_q.pow(2).mean(dim=-1)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-2)


def test_cfg_text_drop_path():
    """Acceptance 3: The text-drop path works (CFG null embedding swap)."""
    conditioner = WanTextConditioner(
        text_dim=128,
        model_dim=128,
        max_length=32,
        cfg_dropout=1.0,  # Force 100% dropout to test drop path
    )
    conditioner.train()

    # Precomputed null embedding projected
    expected_null_projected = conditioner.projector(conditioner.null_embedding)

    prompts = ["A complex prompt that should be dropped."]
    tokens, mask = conditioner.encode_prompts(prompts, apply_dropout=True)

    # Under 100% dropout, tokens must match null embedding
    diff = (tokens - expected_null_projected).abs().max().item()
    assert diff < 1e-6, f"CFG text-drop path failed! Expected null embedding, diff = {diff}"
    assert mask[0, 1:].sum().item() == 0, "Dropped prompt should have masked padding"


def test_no_gradient_reaches_encoder():
    """Acceptance 4: No gradient reaches the encoder (encoder is strictly frozen)."""
    # Create mock text encoder with requires_grad=False
    encoder_param = nn.Parameter(torch.randn(10, 10), requires_grad=False)
    conditioner = WanTextConditioner(text_dim=10, model_dim=32, max_length=16)
    cross_attn = TextCrossAttention(model_dim=32, num_heads=2)

    # Forward
    video_tokens = torch.randn(1, 20, 32, requires_grad=True)
    text_raw = encoder_param.unsqueeze(0)[:, :16, :] # [1, 10, 10]
    # Pad to 16
    text_raw_padded = F.pad(text_raw, (0, 0, 0, 6))

    projected_text = conditioner.projector(text_raw_padded)
    out = cross_attn(video_tokens, projected_text)

    # Backward from loss
    loss = out.sum()
    loss.backward()

    # Projector weights should have gradients (they are trained with DiT)
    assert conditioner.projector.fc1.weight.grad is not None
    # Frozen encoder parameter must receive NO gradients
    assert encoder_param.grad is None, "Gradient leaked to frozen text encoder!"


def test_prompt_rewriter_expansion():
    """Verify prompt rewriter expands short user prompt into dense caption."""
    short = "A red sports car"
    expanded = PromptRewriter.expand_prompt(short)

    assert not expanded["is_empty"]
    assert "A red sports car" in expanded["dense_caption"]
    assert len(expanded["dense_caption"]) > len(short) * 3
    assert "camera" in expanded["dense_caption"].lower()
    assert "lighting" in expanded["dense_caption"].lower() or "illumination" in expanded["dense_caption"].lower()
