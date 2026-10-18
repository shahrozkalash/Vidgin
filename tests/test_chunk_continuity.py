"""
Unit tests for Batch 1: Smooth chunk-to-chunk continuity.
Covers:
1. Chunk overlap (1-2 latent frames, fixed clean conditioning vs blended)
2. Linear cross-fade in the overlap region with configurable blend width
3. RoPE re-indexing relative to window start with translation invariance
4. Motion carry-over from last N clean frames (velocity/acceleration)
5. Seam metric calculation and oracle ground truth matching within tolerance
6. Verification that all config flags default to OFF
"""

import math
import unittest
from vidgen.config import (
    ChunkContinuityConfig,
    LongVideoConfig,
    TINY_PRESET,
    SMALL_PRESET
)
from vidgen.continuity import (
    LatentFrame,
    ChunkContinuityManager,
    compute_seam_metric,
    compute_rope_frequencies,
    apply_temporal_rope,
    compute_rope_dot_product
)
from vidgen.oracle import SyntheticSmoothVideo, OracleChunkedGenerator


class TestChunkContinuity(unittest.TestCase):

    def test_default_config_flags_are_off(self):
        """Rule check: All new improvement flags MUST default to OFF (False)."""
        cfg = ChunkContinuityConfig()
        self.assertFalse(cfg.enable_chunk_overlap, "enable_chunk_overlap must default to False")
        self.assertFalse(cfg.enable_linear_crossfade, "enable_linear_crossfade must default to False")
        self.assertFalse(cfg.enable_rope_reindexing, "enable_rope_reindexing must default to False")
        self.assertFalse(cfg.enable_motion_carryover, "enable_motion_carryover must default to False")

        # LongVideoConfig default continuity
        long_cfg = LongVideoConfig()
        self.assertFalse(long_cfg.continuity.enable_chunk_overlap)
        self.assertFalse(long_cfg.continuity.enable_linear_crossfade)
        self.assertFalse(long_cfg.continuity.enable_rope_reindexing)
        self.assertFalse(long_cfg.continuity.enable_motion_carryover)

    def test_chunk_overlap_modes_fixed_and_blended(self):
        """1. Chunk overlap: sharing 1 to 2 frames in fixed clean or blended mode."""
        dim = 4
        prev_chunk = [
            LatentFrame([1.0] * dim),
            LatentFrame([2.0] * dim),
            LatentFrame([3.0] * dim),
            LatentFrame([4.0] * dim),  # Overlap frame 0 (tail)
            LatentFrame([5.0] * dim),  # Overlap frame 1 (tail)
        ]
        curr_chunk = [
            LatentFrame([10.0] * dim),  # Overlap frame 0 (head)
            LatentFrame([20.0] * dim),  # Overlap frame 1 (head)
            LatentFrame([30.0] * dim),
            LatentFrame([40.0] * dim),
        ]

        # Mode A: fixed clean conditioning (overlap=2)
        cfg_fixed = ChunkContinuityConfig(
            enable_chunk_overlap=True,
            overlap_frames=2,
            overlap_mode="fixed",
            enable_linear_crossfade=False
        )
        mgr_fixed = ChunkContinuityManager(cfg_fixed)
        stitched_fixed = mgr_fixed.stitch_two_chunks(prev_chunk, curr_chunk)

        # Length: 5 - 2 + 4 = 7 frames
        self.assertEqual(len(stitched_fixed), 7)
        # In fixed mode, the 2 overlap frames must exactly match prev_chunk's tail ([4.0], [5.0])
        self.assertEqual(stitched_fixed[3].data, [4.0] * dim)
        self.assertEqual(stitched_fixed[4].data, [5.0] * dim)
        self.assertEqual(stitched_fixed[5].data, [30.0] * dim)

        # Mode B: blended mode (overlap=2, blend_width=2)
        cfg_blended = ChunkContinuityConfig(
            enable_chunk_overlap=True,
            overlap_frames=2,
            overlap_mode="blended",
            enable_linear_crossfade=True,
            blend_width=2
        )
        mgr_blended = ChunkContinuityManager(cfg_blended)
        stitched_blended = mgr_blended.stitch_two_chunks(prev_chunk, curr_chunk)

        self.assertEqual(len(stitched_blended), 7)
        # For overlap frame 0: w = 1/3 -> (2/3)*4.0 + (1/3)*10.0 = 2.6667 + 3.3333 = 6.0
        expected_0 = (2.0 / 3.0) * 4.0 + (1.0 / 3.0) * 10.0
        # For overlap frame 1: w = 2/3 -> (1/3)*5.0 + (2/3)*20.0 = 1.6667 + 13.3333 = 15.0
        expected_1 = (1.0 / 3.0) * 5.0 + (2.0 / 3.0) * 20.0

        for val in stitched_blended[3].data:
            self.assertAlmostEqual(val, expected_0, places=5)
        for val in stitched_blended[4].data:
            self.assertAlmostEqual(val, expected_1, places=5)

    def test_linear_crossfade_blend_width(self):
        """2. Linear cross-fade with configurable blend width."""
        mgr = ChunkContinuityManager()
        tail = [LatentFrame([0.0, 10.0]), LatentFrame([0.0, 20.0])]
        head = [LatentFrame([10.0, 10.0]), LatentFrame([10.0, 20.0])]

        # Blend width 1
        res1 = mgr.linear_crossfade(tail, head, blend_width=1)
        self.assertEqual(len(res1), 1)
        # w = 1 / 2 = 0.5
        self.assertAlmostEqual(res1[0].data[0], 5.0, places=5)
        self.assertAlmostEqual(res1[0].data[1], 10.0, places=5)

        # Blend width 2
        res2 = mgr.linear_crossfade(tail, head, blend_width=2)
        self.assertEqual(len(res2), 2)
        # Frame 0: w = 1/3 -> 0.0 * 2/3 + 10.0 * 1/3 = 3.3333
        self.assertAlmostEqual(res2[0].data[0], 10.0 / 3.0, places=5)
        # Frame 1: w = 2/3 -> 0.0 * 1/3 + 10.0 * 2/3 = 6.6667
        self.assertAlmostEqual(res2[1].data[0], 20.0 / 3.0, places=5)

    def test_rope_reindexing_relative_invariance(self):
        """3. RoPE re-indexing: Shifting the window gives identical relative attention output."""
        dim = 16
        freqs = compute_rope_frequencies(dim)

        # Query and key feature vectors
        q = [math.sin(i * 0.7) for i in range(dim)]
        k = [math.cos(i * 0.5) for i in range(dim)]

        # Case A: Window starts at t=0, query at local t=3, key at local t=1
        t_q_local = 3
        t_k_local = 1
        window_0 = 0
        dot_0 = compute_rope_dot_product(q, k, window_0 + t_q_local, window_0 + t_k_local, freqs)

        # Case B: Shifted window starting at t=80 (e.g. Chunk 4 in 30s video)
        # With RoPE re-indexing relative to window start, both coordinates are re-indexed to local slots:
        cfg = ChunkContinuityConfig(enable_rope_reindexing=True)
        mgr = ChunkContinuityManager(cfg)

        reindexed_q = mgr.reindex_rope_positions(window_start_t=80, chunk_local_positions=[t_q_local])[0]
        reindexed_k = mgr.reindex_rope_positions(window_start_t=80, chunk_local_positions=[t_k_local])[0]

        # Re-indexed positions must match local slots exactly (no extrapolation beyond trained span)
        self.assertEqual(reindexed_q, t_q_local)
        self.assertEqual(reindexed_k, t_k_local)

        dot_reindexed = compute_rope_dot_product(q, k, reindexed_q, reindexed_k, freqs)

        # Mathematical invariance test: Attention dot product must be bit-for-bit identical
        self.assertAlmostEqual(dot_0, dot_reindexed, places=6,
                               msg="RoPE dot product shifted across window must be invariant")

    def test_motion_carryover_preserves_velocity(self):
        """4. Motion carry-over conditions each chunk on last N clean frames to continue motion direction."""
        dim = 4
        # Create a chunk with linear motion: v = [1.0, -0.5, 2.0, 0.0] per frame
        v = [1.0, -0.5, 2.0, 0.0]
        chunk = []
        for t in range(5):
            chunk.append(LatentFrame([t * vi for vi in v]))

        cfg = ChunkContinuityConfig(
            enable_motion_carryover=True,
            motion_carryover_frames=2
        )
        mgr = ChunkContinuityManager(cfg)
        motion_info = mgr.extract_motion_carryover(chunk)

        self.assertTrue(motion_info["enabled"])
        self.assertEqual(motion_info["n_frames"], 2)

        # Extracted velocity must exactly match v
        for vi_est, vi_true in zip(motion_info["velocity"].data, v):
            self.assertAlmostEqual(vi_est, vi_true, places=5)

        # Extrapolated frame at t=5 must match 5 * v
        expected_t5 = [5 * vi for vi in v]
        for val_est, val_true in zip(motion_info["extrapolated"].data, expected_t5):
            self.assertAlmostEqual(val_est, val_true, places=5)

    def test_seam_metric_and_oracle_ground_truth(self):
        """5. Seam metric & oracle model test on smooth synthetic video matching GT within tolerance."""
        total_latents = 121  # 30s at 16 fps
        dim = 16
        gt_video = SyntheticSmoothVideo(total_frames=total_latents, dim=dim, seed=1337)

        # Generator with all Batch 1 features enabled
        continuity_cfg = ChunkContinuityConfig(
            enable_chunk_overlap=True,
            overlap_frames=2,
            overlap_mode="blended",
            enable_linear_crossfade=True,
            blend_width=2,
            enable_rope_reindexing=True,
            enable_motion_carryover=True,
            motion_carryover_frames=2
        )
        video_cfg = LongVideoConfig(
            duration_seconds=30.0,
            fps=16,
            preset=TINY_PRESET,
            continuity=continuity_cfg
        )
        generator = OracleChunkedGenerator(video_cfg, dim=dim)

        stitched, boundaries, metrics = generator.generate_and_stitch(
            gt_video,
            num_chunks=7,
            nominal_chunk_size=21
        )

        # Verification 1: Stitched output must match ground truth within tolerance (< 1e-4)
        self.assertLess(metrics["mse_vs_gt"], 1e-4, f"MSE {metrics['mse_vs_gt']} exceeds tolerance")
        self.assertLess(metrics["max_abs_err"], 1e-3, f"Max error {metrics['max_abs_err']} exceeds tolerance")

        # Verification 2: Seam metric across borders must be natural and smooth (< 1.5x of inside diff)
        self.assertGreater(metrics["num_boundaries"], 0)
        self.assertLess(metrics["seam_metric"], 1.5,
                        f"Seam metric {metrics['seam_metric']:.2f} indicates boundary discontinuity")


if __name__ == "__main__":
    unittest.main()
