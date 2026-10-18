"""
Unit tests for Seedance Batch 1: Decoupled & Hybrid Attention.
Covers:
1. Default configuration flags must be OFF (attention_mode="full", enable_decoupled_attention=False).
2. Spatial attention isolation: Perturbing frame i does NOT change spatial-layer output of frame j (i != j).
3. Temporal attention isolation: Perturbing position (h, w) does NOT change other positions in the temporal layer.
4. Block-causal chunked generation: Zero leakage from future chunks.
5. KV-cache equivalence: Streaming chunked generation with KV-cache produces exact identical output to full sequence with block-causal mask.
6. Hybrid attention: Decoupled layers with full-attention layer every N layers.
7. RoPE 3D partition: Valid even dimensions and correct Givens rotation.
"""

import math
import random
import unittest
from typing import List, Tuple

from vidgen.config import (
    AttentionConfig,
    PresetConfig,
    LongVideoConfig,
    TINY_PRESET,
)
from vidgen.attention import (
    SpatialAttention,
    TemporalAttention,
    FullSpacetimeAttention,
    DiTAttentionBlock,
    DecoupledDiTModel,
    estimate_attention_flops_and_memory,
    get_rope_partition,
    apply_rope_3d,
    vector_dot,
    vector_sub,
)


class TestDecoupledAttention(unittest.TestCase):

    def test_default_config_flags_are_off(self):
        """Rule check: All new feature flags MUST default to OFF / baseline behavior."""
        cfg = AttentionConfig()
        self.assertEqual(cfg.attention_mode, "full", "attention_mode must default to 'full'")
        self.assertFalse(cfg.enable_decoupled_attention, "enable_decoupled_attention must default to False")
        self.assertIsNone(cfg.temporal_window_size, "temporal_window_size must default to None")
        self.assertEqual(cfg.hybrid_full_attn_interval, 4)

        # PresetConfig and LongVideoConfig defaults
        preset = PresetConfig()
        self.assertEqual(preset.attention.attention_mode, "full")
        self.assertFalse(preset.attention.enable_decoupled_attention)

        long_cfg = LongVideoConfig()
        self.assertEqual(long_cfg.attention.attention_mode, "full")
        self.assertFalse(long_cfg.attention.enable_decoupled_attention)

    def test_spatial_layer_frame_isolation(self):
        """
        Part 1.4: Perturbing frame i does NOT change spatial-layer output of frame j (i != j).
        Because spatial attention operates inside each frame independently.
        """
        dim = 32
        num_heads = 2
        head_dim = 16
        spatial_layer = SpatialAttention(dim=dim, num_heads=num_heads, head_dim=head_dim, seed=42)

        T = 4
        H, W = 2, 2
        S = H * W
        spatial_coords = [(h, w) for h in range(H) for w in range(W)]

        # Generate synthetic visual tokens for T frames
        rng = random.Random(1337)
        frames_visual_orig: List[List[List[float]]] = []
        for t in range(T):
            frame = []
            for s in range(S):
                token = [rng.gauss(0, 1) for _ in range(dim)]
                frame.append(token)
            frames_visual_orig.append(frame)

        # Shared text tokens
        text_tokens = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(3)]

        # Baseline forward pass
        out_orig, _ = spatial_layer.forward(frames_visual_orig, spatial_coords, text_tokens=text_tokens)

        # Perturb frame 0 heavily (add noise to all tokens in frame 0)
        frames_visual_perturbed = [
            [list(token) for token in frame] for frame in frames_visual_orig
        ]
        for s in range(S):
            for d in range(dim):
                frames_visual_perturbed[0][s][d] += 99.0

        # Forward pass on perturbed input
        out_perturbed, _ = spatial_layer.forward(frames_visual_perturbed, spatial_coords, text_tokens=text_tokens)

        # 1. Output of frame 0 MUST change
        diff_frame_0 = 0.0
        for s in range(S):
            for d in range(dim):
                diff_frame_0 += abs(out_orig[0][s][d] - out_perturbed[0][s][d])
        self.assertGreater(diff_frame_0, 1.0, "Frame 0 output should change when frame 0 is perturbed")

        # 2. Outputs of frame 1, 2, 3 MUST BE IDENTICAL to machine precision (zero leakage)
        for t in range(1, T):
            for s in range(S):
                for d in range(dim):
                    diff = abs(out_orig[t][s][d] - out_perturbed[t][s][d])
                    self.assertAlmostEqual(
                        diff, 0.0, places=12,
                        msg=f"Frame {t} leaked from frame 0 perturbation at token {s}, dim {d}"
                    )

    def test_temporal_layer_spatial_position_isolation(self):
        """
        Part 1.4: Perturbing one (h, w) position does NOT change others in the temporal layer.
        Because temporal attention operates across time for each spatial position independently.
        """
        dim = 32
        num_heads = 2
        head_dim = 16
        temporal_layer = TemporalAttention(dim=dim, num_heads=num_heads, head_dim=head_dim, seed=77)

        T = 4
        H, W = 2, 2
        S = H * W
        spatial_coords = [(h, w) for h in range(H) for w in range(W)]

        rng = random.Random(2024)
        frames_visual_orig: List[List[List[float]]] = []
        for t in range(T):
            frame = []
            for s in range(S):
                token = [rng.gauss(0, 1) for _ in range(dim)]
                frame.append(token)
            frames_visual_orig.append(frame)

        # Baseline forward pass
        out_orig, _ = temporal_layer.forward(frames_visual_orig, spatial_coords)

        # Perturb spatial position 0 (which is (0, 0)) across all time frames
        frames_visual_perturbed = [
            [list(token) for token in frame] for frame in frames_visual_orig
        ]
        for t in range(T):
            for d in range(dim):
                frames_visual_perturbed[t][0][d] += 42.0

        out_perturbed, _ = temporal_layer.forward(frames_visual_perturbed, spatial_coords)

        # 1. Output at spatial position 0 must change
        diff_pos_0 = 0.0
        for t in range(T):
            for d in range(dim):
                diff_pos_0 += abs(out_orig[t][0][d] - out_perturbed[t][0][d])
        self.assertGreater(diff_pos_0, 1.0, "Position 0 output should change when position 0 is perturbed")

        # 2. Outputs at positions 1, 2, 3 ((0,1), (1,0), (1,1)) must remain EXACTLY unchanged
        for s in range(1, S):
            for t in range(T):
                for d in range(dim):
                    diff = abs(out_orig[t][s][d] - out_perturbed[t][s][d])
                    self.assertAlmostEqual(
                        diff, 0.0, places=12,
                        msg=f"Spatial position {spatial_coords[s]} leaked from position (0,0) at time {t}, dim {d}"
                    )

    def test_block_causal_no_future_leakage(self):
        """
        Part 1.4: In chunked temporal attention, tokens in chunk c have ZERO leakage from future chunks k > c.
        """
        dim = 16
        num_heads = 2
        head_dim = 8
        temporal_layer = TemporalAttention(dim=dim, num_heads=num_heads, head_dim=head_dim, seed=99)

        # 2 chunks: Chunk 0 (frames 0, 1), Chunk 1 (frames 2, 3)
        T = 4
        H, W = 1, 1
        S = 1
        spatial_coords = [(0, 0)]
        time_indices = [0, 1, 2, 3]

        rng = random.Random(888)
        frames_orig = [[[rng.gauss(0, 1) for _ in range(dim)]] for _ in range(T)]

        out_orig, _ = temporal_layer.forward(frames_orig, spatial_coords, time_indices=time_indices)

        # Perturb future chunk (frames 2 and 3)
        frames_perturbed = [[[val for val in token] for token in frame] for frame in frames_orig]
        for t in [2, 3]:
            for d in range(dim):
                frames_perturbed[t][0][d] += 123.45

        out_perturbed, _ = temporal_layer.forward(frames_perturbed, spatial_coords, time_indices=time_indices)

        # Chunk 0 (frames 0 and 1) must be completely unaffected by future chunk 1!
        for t in [0, 1]:
            for d in range(dim):
                diff = abs(out_orig[t][0][d] - out_perturbed[t][0][d])
                self.assertAlmostEqual(
                    diff, 0.0, places=12,
                    msg=f"Frame {t} in chunk 0 leaked from future chunk at dim {d} (diff={diff})"
                )

    def test_kv_cache_equivalence(self):
        """
        Part 1.4: Streaming chunked generation with KV-cache produces exact identical output
        to the complete un-chunked sequence under the block-causal attention rule.
        """
        dim = 16
        num_heads = 2
        head_dim = 8
        temporal_layer = TemporalAttention(dim=dim, num_heads=num_heads, head_dim=head_dim, seed=314)

        # 3 chunks of 2 frames each: Chunk 0 = [0, 1], Chunk 1 = [2, 3], Chunk 2 = [4, 5]
        chunk_0 = [0, 1]
        chunk_1 = [2, 3]
        chunk_2 = [4, 5]
        all_frames = chunk_0 + chunk_1 + chunk_2
        T = len(all_frames)

        H, W = 2, 2
        S = H * W
        spatial_coords = [(h, w) for h in range(H) for w in range(W)]

        rng = random.Random(777)
        frames_visual: List[List[List[float]]] = []
        for t in range(T):
            frame = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(S)]
            frames_visual.append(frame)

        # Run 1: Full sequence in one go (ground truth)
        out_full_seq, _ = temporal_layer.forward(
            frames_visual, spatial_coords, time_indices=all_frames
        )

        # Run 2: Chunk-by-chunk using KV cache
        kv_cache: dict = {}
        # Chunk 0
        frames_c0 = [frames_visual[t] for t in chunk_0]
        out_c0, kv_cache = temporal_layer.forward(
            frames_c0, spatial_coords, time_indices=chunk_0, kv_cache=kv_cache
        )

        # Chunk 1 with cached Chunk 0
        frames_c1 = [frames_visual[t] for t in chunk_1]
        out_c1, kv_cache = temporal_layer.forward(
            frames_c1, spatial_coords, time_indices=chunk_1, kv_cache=kv_cache
        )

        # Chunk 2 with cached Chunks 0 and 1
        frames_c2 = [frames_visual[t] for t in chunk_2]
        out_c2, kv_cache = temporal_layer.forward(
            frames_c2, spatial_coords, time_indices=chunk_2, kv_cache=kv_cache
        )

        out_streamed = out_c0 + out_c1 + out_c2

        # Assert bit-for-bit equivalence across all frames, spatial positions, and dimensions
        max_diff = 0.0
        for t in range(T):
            for s in range(S):
                for d in range(dim):
                    diff = abs(out_full_seq[t][s][d] - out_streamed[t][s][d])
                    max_diff = max(max_diff, diff)
                    self.assertAlmostEqual(
                        diff, 0.0, places=10,
                        msg=f"KV cache divergence at frame {t}, position {spatial_coords[s]}, dim {d} (diff={diff})"
                    )
        self.assertLess(max_diff, 1e-10, f"Max KV cache divergence {max_diff} exceeds 1e-10")

    def test_hybrid_attention_execution(self):
        """
        Part 1.1: Hybrid mode executes decoupled layers, and a full-attention layer every N layers.
        """
        dim = 16
        depth = 6
        num_heads = 2
        head_dim = 8
        interval = 3  # Layers 2 and 5 (0-indexed) are full attention

        cfg_hybrid = AttentionConfig(
            attention_mode="hybrid",
            hybrid_full_attn_interval=interval,
            enable_decoupled_attention=True,
        )

        model = DecoupledDiTModel(
            dim=dim, depth=depth, num_heads=num_heads, head_dim=head_dim, config=cfg_hybrid
        )

        # Check block effective types
        expected_types = ["decoupled", "decoupled", "full", "decoupled", "decoupled", "full"]
        actual_types = [block.effective_type for block in model.blocks]
        self.assertEqual(actual_types, expected_types)

        # Forward pass runs cleanly
        T, S = 2, 4
        spatial_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
        frames = [[[0.5] * dim for _ in range(S)] for _ in range(T)]
        text = [[0.1] * dim for _ in range(2)]

        out_frames, out_text, _ = model.forward(frames, spatial_coords, text_tokens=text)
        self.assertEqual(len(out_frames), T)
        self.assertEqual(len(out_frames[0]), S)
        self.assertEqual(len(out_frames[0][0]), dim)

    def test_rope_partition_and_3d_rotation(self):
        """
        Part 1.3: 3D RoPE partition satisfies even dimensions and correct relative angle translation invariance.
        """
        # Head dim 64 -> (22, 22, 20)
        p64 = get_rope_partition(64)
        self.assertEqual(sum(p64), 64)
        self.assertTrue(all(d % 2 == 0 for d in p64))

        # Head dim 128 -> (44, 42, 42)
        p128 = get_rope_partition(128)
        self.assertEqual(sum(p128), 128)
        self.assertEqual(p128, (44, 42, 42))

        # Dot product of RoPE(q, t, h, w) and RoPE(k, t', h', w') satisfies relative translation invariance
        q = [math.sin(i * 0.1) for i in range(64)]
        k = [math.cos(i * 0.2) for i in range(64)]

        # Pos A: (t=5, h=3, w=4) and (t=8, h=6, w=7) -> deltas: (3, 3, 3)
        rot_q1 = apply_rope_3d(q, t=5, h=3, w=4, partition=p64)
        rot_k1 = apply_rope_3d(k, t=8, h=6, w=7, partition=p64)
        dot1 = vector_dot(rot_q1, rot_k1)

        # Pos B: shifted by (10, 10, 10): (t=15, h=13, w=14) and (t=18, h=16, w=17) -> deltas: (3, 3, 3)
        rot_q2 = apply_rope_3d(q, t=15, h=13, w=14, partition=p64)
        rot_k2 = apply_rope_3d(k, t=18, h=16, w=17, partition=p64)
        dot2 = vector_dot(rot_q2, rot_k2)

        self.assertAlmostEqual(dot1, dot2, places=6, msg="RoPE dot product must be shift-invariant")

    def test_flops_and_memory_estimator(self):
        """
        Part 1.5: FLOPs and memory estimator comparing full, decoupled, and hybrid at 5s and 30s.
        """
        est_5s = estimate_attention_flops_and_memory(
            duration_seconds=5.0, fps=16, height=480, width=832, dim=1536, depth=30, num_heads=12
        )
        est_30s = estimate_attention_flops_and_memory(
            duration_seconds=30.0, fps=16, height=480, width=832, dim=1536, depth=30, num_heads=12
        )

        # 5s: 21 latents, 32,760 visual tokens
        self.assertEqual(est_5s["latent_frames"], 21)
        self.assertEqual(est_5s["total_visual_tokens"], 32760)

        # 30s: 121 latents, 188,760 visual tokens
        self.assertEqual(est_30s["latent_frames"], 121)
        self.assertEqual(est_30s["total_visual_tokens"], 188760)

        # Decoupled attention reduces FLOPs and peak activation memory substantially at 30s
        flops_red_30s = est_30s["decoupled_attention"]["flops_reduction_ratio"]
        mem_red_30s = est_30s["decoupled_attention"]["memory_reduction_ratio"]

        self.assertGreater(flops_red_30s, 5.0, "Decoupled attention should reduce 30s FLOPs by > 5x")
        self.assertGreater(mem_red_30s, 100.0, "Decoupled attention should reduce peak activation memory by > 100x")


if __name__ == "__main__":
    unittest.main()
