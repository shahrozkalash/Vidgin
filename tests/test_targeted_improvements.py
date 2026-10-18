"""Unit and Acceptance Tests for Part 12: Targeted Improvements (Beating Wan).

Covers:
1. 12.1 Block-Causal Masking & KV Cache for Long Video
2. 12.2 Windowed 3D Attention & High-Compression Token Reduction
3. 12.3 CFG Distillation & Fast Single-Pass Sampling
4. 12.4 Structured LLM Prompt Planner & Attribute Binding
5. 12.5 In-Context Reference Conditioning & Identity Similarity
6. 12.6 Representation Alignment Loss & Timestep Sampling
7. 12.7 Joint Audio-Video Cross-Modal Attention
8. Equal-Compute Ablation Harness
"""

import math
import unittest
import torch
import torch.nn as nn

from vidgen.improvements.block_causal import (
    build_block_causal_mask,
    KVCache,
    BlockCausalAttentionWrapper,
)
from vidgen.improvements.windowed_attention import (
    WindowedSpacetimeAttention,
    HybridSpacetimeBlock,
    estimate_token_and_flop_savings,
)
from vidgen.improvements.cfg_distill import (
    CFGDistillationLoss,
    FastSinglePassSampler,
)
from vidgen.improvements.prompt_planner import (
    StructuredPromptPlanner,
    get_prompt_following_test_cases,
)
from vidgen.improvements.in_context_ref import (
    InContextReferenceDiT,
)
from vidgen.improvements.rep_alignment import (
    RepresentationAlignmentLoss,
    ShiftedTimestepSampler,
)
from vidgen.improvements.joint_audio import (
    JointAudioVideoAttention,
)
from vidgen.improvements.ablation_runner import (
    AblationRunner,
)


class TestTargetedImprovements(unittest.TestCase):

    def test_block_causal_mask_structure(self):
        """Verify block causal mask enforces causality while retaining anchor frames."""
        grid_size = (6, 4, 4)  # 6 latent frames, 16 tokens per frame = 96 tokens
        mask = build_block_causal_mask(
            grid_size=grid_size,
            chunk_frames=2,
            anchor_frames=1,
        )
        self.assertEqual(mask.shape, (96, 96))

        # Anchor frame tokens (indices 0..15) must be visible to all queries
        self.assertTrue(mask[:, :16].all())

        # Future chunk cannot be attended by early query:
        # Query in chunk 1 (t=1..2, tokens 16..47) should not attend to chunk 2 (t=3..4, tokens 48..79)
        q_idx = 20
        k_future_idx = 60
        self.assertFalse(mask[q_idx, k_future_idx].item())

    def test_kv_cache_caching(self):
        """Verify KV cache updates and preserves keys/values across chunks."""
        cache = KVCache(max_tokens=1000)
        B, H, d = 2, 4, 32

        # Chunk 1
        k1 = torch.randn(B, H, 20, d)
        v1 = torch.randn(B, H, 20, d)
        k_full1, v_full1 = cache.update(0, k1, v1)
        self.assertEqual(k_full1.shape, (B, H, 20, d))

        # Chunk 2
        k2 = torch.randn(B, H, 25, d)
        v2 = torch.randn(B, H, 25, d)
        k_full2, v_full2 = cache.update(0, k2, v2)
        self.assertEqual(k_full2.shape, (B, H, 45, d))
        self.assertEqual(v_full2.shape, (B, H, 45, d))

    def test_windowed_3d_attention_shapes(self):
        """Verify windowed attention produces exact matching output shapes."""
        B = 2
        gt, gh, gw = 4, 8, 8
        N = gt * gh * gw
        D = 64
        attn = WindowedSpacetimeAttention(dim=D, num_heads=4, window_size=(2, 4, 4))

        x = torch.randn(B, N, D)
        out = attn(x, (gt, gh, gw))
        self.assertEqual(out.shape, (B, N, D))
        self.assertFalse(torch.isnan(out).any())

    def test_token_and_flop_savings_calculation(self):
        """Verify 4x token reduction and FLOP savings calculations."""
        stats = estimate_token_and_flop_savings(height=480, width=832, frames=81)
        self.assertEqual(stats["tokens_baseline"], 32760)
        self.assertEqual(stats["tokens_high_compression_vae"], 8190)
        self.assertEqual(stats["token_reduction_ratio"], 4.0)
        self.assertGreater(stats["speedup_hybrid_attn"], 1.5)

    def test_cfg_distillation_loss(self):
        """Verify CFG distillation loss computes valid scalar without NaNs."""
        class DummyDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(16, 16)
            def forward(self, x, t, c):
                return self.proj(x)

        teacher = DummyDiT()
        student = DummyDiT()
        loss_fn = CFGDistillationLoss()

        B = 2
        x = torch.randn(B, 16, 2, 4, 4)
        t = torch.tensor([0.5, 0.8])
        context = torch.randn(B, 10, 16)
        null_context = torch.zeros_like(context)

        loss, pred = loss_fn.compute_loss(student, teacher, x, t, context, null_context)
        self.assertFalse(torch.isnan(loss))
        self.assertEqual(pred.shape, x.shape)

    def test_structured_prompt_planner(self):
        """Verify structured prompt planning enriches visual axes."""
        prompt = "A red sports car speeding along an ocean cliff while a cinematic drone follows overhead at sunset"
        plan = StructuredPromptPlanner.plan_prompt(prompt)

        self.assertIn("red sports car", plan["subject"])
        self.assertIn("ocean", plan["raw_prompt"])
        self.assertIn("dense_prompt", plan)
        self.assertGreater(len(plan["shots"]), 0)

        test_cases = get_prompt_following_test_cases()
        self.assertEqual(len(test_cases), 4)

    def test_in_context_ref_identity_similarity(self):
        """Verify in-context reference token prepending and cosine similarity calculation."""
        B, D = 2, 64
        wrapper = InContextReferenceDiT(base_dit=nn.Identity(), hidden_dim=D)

        video_tokens = torch.randn(B, 80, D)
        ref_tokens = torch.randn(B, 16, D)

        combined, ref_len = wrapper.prepend_reference_tokens(video_tokens, ref_tokens)
        self.assertEqual(combined.shape, (B, 16 + 1 + 80, D))
        self.assertEqual(ref_len, 17)

        sim = wrapper.calculate_identity_similarity(ref_tokens, video_tokens)
        self.assertIsInstance(sim, float)
        self.assertGreaterEqual(sim, -1.0)
        self.assertLessEqual(sim, 1.0)

    def test_representation_alignment_loss(self):
        """Verify representation alignment loss computes cosine loss between mid-features and vision targets."""
        loss_fn = RepresentationAlignmentLoss(dit_dim=64, dino_dim=128)
        B, N = 2, 30
        dit_mid = torch.randn(B, N, 64)
        dino_target = torch.randn(B, N, 128)

        loss = loss_fn(dit_mid, dino_target)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_joint_audio_video_attention(self):
        """Verify joint audio-video bidirectional cross-attention."""
        joint_attn = JointAudioVideoAttention(video_dim=64, audio_dim=48, num_heads=4)
        B = 2
        v_tokens = torch.randn(B, 40, 64)
        a_tokens = torch.randn(B, 20, 48)

        v_out, a_out = joint_attn(v_tokens, a_tokens)
        self.assertEqual(v_out.shape, (B, 40, 64))
        self.assertEqual(a_out.shape, (B, 20, 48))

    def test_ablation_runner_metrics(self):
        """Verify equal-compute ablation reports are structured and computed properly."""
        long_rep = AblationRunner.run_long_video_ablation()
        self.assertIn("drift_reduction_percent", long_rep)

        attn_rep = AblationRunner.run_cheaper_attention_ablation()
        self.assertIn("token_reduction", attn_rep)

        cfg_rep = AblationRunner.run_few_step_sampling_ablation()
        self.assertIn("nfe_reduction", cfg_rep)


if __name__ == "__main__":
    unittest.main()
