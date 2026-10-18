"""
Unit tests for Batch 1: Shared Conditioning Framework.
Tests:
1. ConditioningSpec mapping: pixel frames to latent indices under causal VAE (4x8x8).
   - First-frame exact mapping to latent index 0.
   - Last-frame single-frame approximation and error measurement.
2. Model input & Zero-initialization equivalence:
   - Concatenate [noisy_latents (16), mask (1), cond_latents (16)] -> 33 channels.
   - Zero-initialization of new channels guarantees bit-for-bit identical forward output to T2V checkpoint.
   - Checkpoint migration script with strict=False equivalence.
3. Training modes & Masked loss:
   - Multi-mode sampling (none, first_frame, first_and_last, first_n_frames).
   - Loss computed strictly on non-conditioned frames (and full loss flag).
4. Small noise augmentation on conditioning latents.
5. All new configuration flags default to OFF (False).
"""

import math
import random
import unittest
from vidgen.config import (
    ConditioningConfig,
    ConditioningTrainingConfig,
    TINY_PRESET
)
from vidgen.conditioning import (
    ConditioningMode,
    ConditioningItem,
    ConditioningSpec,
    ConditioningTensorBuilder,
    TrainingModeSampler,
    compute_masked_loss,
    measure_causal_approximation_error
)
from vidgen.patch_embed import (
    PatchEmbed3D,
    VideoDiTStubModel,
    migrate_t2v_checkpoint_to_conditioning
)


class TestSharedConditioning(unittest.TestCase):

    def test_default_config_flags_are_off(self):
        """Rule check: All new feature flags MUST default to OFF (False)."""
        cfg = ConditioningConfig()
        self.assertFalse(cfg.enable_conditioning, "enable_conditioning must default to False")
        self.assertFalse(cfg.enable_conditioning_noise, "enable_conditioning_noise must default to False")
        self.assertEqual(cfg.in_channels, 16)
        self.assertEqual(cfg.total_model_channels, 33)

        train_cfg = ConditioningTrainingConfig()
        self.assertTrue(train_cfg.loss_non_conditioned_only)
        self.assertEqual(train_cfg.extension_n_frames, 4)

    def test_conditioning_spec_mapping_and_approximation(self):
        """
        1. Conditioning spec mapping:
        - First frame (p=0) -> latent 0 (exact, causal VAE encodes first frame alone).
        - Last frame (p=T-1) -> last latent (T-1)//4.
        - Approximation measurement between full causal clip and isolated single frame.
        """
        total_pixel_frames = 81  # 5s at 16 fps
        spec = ConditioningSpec(total_pixel_frames=total_pixel_frames, vae_temporal_stride=4)

        dummy_first = [1.0] * 16
        dummy_last = [2.0] * 16

        spec.add_first_frame(dummy_first)
        spec.add_last_frame(dummy_last)

        latent_map = spec.map_to_latent_indices()

        # Expected total latents: (81 - 1) // 4 + 1 = 21
        self.assertEqual(spec.total_latent_frames, 21)
        # First frame must map to exact latent index 0
        self.assertIn(0, latent_map)
        self.assertEqual(latent_map[0], dummy_first)
        # Last frame (p=80) must map to latent index 20
        self.assertIn(20, latent_map)
        self.assertEqual(latent_map[20], dummy_last)

        # Extension prefix: first 17 pixel frames -> (17 - 1) // 4 + 1 = 5 latent frames
        spec_ext = ConditioningSpec(total_pixel_frames=81, vae_temporal_stride=4)
        spec_ext.add_first_n_frames([0.5] * 16, n_pixel_frames=17)
        ext_map = spec_ext.map_to_latent_indices()
        self.assertEqual(len(ext_map), 5)
        for l in range(5):
            self.assertIn(l, ext_map)

        # Document and test causal VAE approximation error on isolated last frame
        dim = 16
        # Simulate full continuous clip latents where temporal causal filtering adds small memory
        full_clip = [[math.sin(t * 0.2 + d * 0.1) for d in range(dim)] for t in range(21)]
        # Isolated single frame has no temporal history (slight delta)
        isolated_last = [full_clip[-1][d] + 0.03 * math.cos(d) for d in range(dim)]

        err_metrics = measure_causal_approximation_error(full_clip, isolated_last)
        self.assertLess(err_metrics["l1_error"], 0.1, "Causal approximation L1 error should be small (< 0.1)")
        self.assertGreater(err_metrics["cosine_similarity"], 0.95, "Directional alignment should be high (> 0.95)")
        self.assertIn("approximation_note", err_metrics)

    def test_model_input_concatenation_and_zero_init_equivalence(self):
        """
        2. Model input concatenation [x_t (16), mask (1), cond_latents (16)] = 33 channels.
        Zero-initialization guarantees bit-for-bit mathematical equivalence to T2V model.
        """
        num_patches = 10
        embed_dim = 64
        patch_size = (1, 2, 2)
        spatial_prod = patch_size[0] * patch_size[1] * patch_size[2]  # 4

        # A. Original Text-to-Video Model (16 channels -> patch_dim = 64)
        t2v_embed = PatchEmbed3D(in_channels=16, embed_dim=embed_dim, patch_size=patch_size)

        # Synthetic noisy latent patch tokens: [num_patches, 64]
        noisy_patch_tokens = [
            [math.sin(p * 0.4 + j * 0.08) for j in range(16 * spatial_prod)]
            for p in range(num_patches)
        ]
        t2v_output = t2v_embed.forward_patch_tokens(noisy_patch_tokens)

        # B. Checkpoint Migration to 33 channels (patch_dim = 132)
        old_state = t2v_embed.state_dict()
        migrated_state = migrate_t2v_checkpoint_to_conditioning(
            old_state,
            original_in_channels=16,
            new_in_channels=33,
            patch_size=patch_size
        )

        # Verify zero-initialization of the new channels
        new_weight = migrated_state["weight"]
        self.assertEqual(len(new_weight), embed_dim)
        self.assertEqual(len(new_weight[0]), 33 * spatial_prod)  # 132

        # Check first 64 channels are identical to old weights
        for d in range(embed_dim):
            self.assertEqual(new_weight[d][:64], old_state["weight"][d])
            # Check remaining 68 channels are EXACT ZERO (0.0)
            self.assertTrue(all(val == 0.0 for val in new_weight[d][64:]),
                            "New channels must be initialized to exact zero")

        # C. Forward through 33-channel Model
        cond_embed = PatchEmbed3D(in_channels=33, embed_dim=embed_dim, patch_size=patch_size)
        cond_embed.load_state_dict(migrated_state, strict=True)

        # Build 33-channel input: [noisy (64), mask (4), cond_latents (64)]
        # Even with non-zero arbitrary masks and conditioning latents:
        cond_patch_tokens = []
        for p in range(num_patches):
            noisy_part = noisy_patch_tokens[p]
            mask_part = [1.0] * (1 * spatial_prod)  # active mask
            cond_part = [math.cos(p * 0.3 + j * 0.1) for j in range(16 * spatial_prod)]  # non-zero cond
            combined = noisy_part + mask_part + cond_part
            cond_patch_tokens.append(combined)

        cond_output = cond_embed.forward_patch_tokens(cond_patch_tokens)

        # EQUIVALENCE ASSERTION:
        # Since new channels have weights 0.0, output MUST BE BIT-FOR-BIT IDENTICAL to t2v_output!
        for p in range(num_patches):
            for d in range(embed_dim):
                diff = abs(t2v_output[p][d] - cond_output[p][d])
                self.assertAlmostEqual(
                    t2v_output[p][d], cond_output[p][d], places=10,
                    msg=f"Zero-init equivalence failed: diff {diff} > 1e-10 at patch {p}, dim {d}"
                )

    def test_strict_false_checkpoint_loading(self):
        """2b. Test strict=False loader automatically zero-pads 16-ch checkpoint into 33-ch model."""
        embed_dim = 64
        t2v_embed = PatchEmbed3D(in_channels=16, embed_dim=embed_dim, patch_size=(1, 2, 2))
        cond_embed = PatchEmbed3D(in_channels=33, embed_dim=embed_dim, patch_size=(1, 2, 2))

        # Directly load 16-channel state dict into 33-channel model with strict=False
        t2v_state = t2v_embed.state_dict()
        cond_embed.load_state_dict(t2v_state, strict=False)

        # Verify weights were correctly padded with zeros
        for d in range(embed_dim):
            self.assertEqual(cond_embed.weight[d][:64], t2v_state["weight"][d])
            self.assertTrue(all(val == 0.0 for val in cond_embed.weight[d][64:]))

    def test_training_modes_and_masked_loss(self):
        """3. Training modes with configurable probabilities and masked loss computation."""
        train_cfg = ConditioningTrainingConfig(
            prob_none=0.1,
            prob_first_frame=0.3,
            prob_first_and_last=0.3,
            prob_first_n_frames=0.3,
            extension_n_frames=3,
            loss_non_conditioned_only=True
        )
        sampler = TrainingModeSampler(train_cfg, seed=999)

        # Test mask construction for all 4 modes
        m_none = sampler.build_training_mask(total_latent_frames=10, mode=ConditioningMode.NONE)
        self.assertEqual(sum(m_none), 0.0)

        m_first = sampler.build_training_mask(total_latent_frames=10, mode=ConditioningMode.FIRST_FRAME)
        self.assertEqual(m_first[0], 1.0)
        self.assertEqual(sum(m_first), 1.0)

        m_fl = sampler.build_training_mask(total_latent_frames=10, mode=ConditioningMode.FIRST_AND_LAST)
        self.assertEqual(m_fl[0], 1.0)
        self.assertEqual(m_fl[-1], 1.0)
        self.assertEqual(sum(m_fl), 2.0)

        m_ext = sampler.build_training_mask(total_latent_frames=10, mode=ConditioningMode.FIRST_N_FRAMES)
        self.assertEqual(m_ext[:3], [1.0, 1.0, 1.0])
        self.assertEqual(sum(m_ext), 3.0)

        # Test masked loss: Conditioned frames must contribute 0 to the loss
        dim = 8
        T = 5
        # v_pred has large error on frame 0, but frame 0 is conditioned (mask[0]=1.0)
        mask = [1.0, 0.0, 0.0, 0.0, 0.0]
        v_target = [[0.0] * dim for _ in range(T)]
        v_pred = [[1000.0] * dim] + [[1.0] * dim for _ in range(T - 1)]

        # When loss_non_conditioned_only=True: frame 0 error is excluded!
        res_masked = compute_masked_loss(v_pred, v_target, mask, loss_non_conditioned_only=True)
        # Expected loss is based strictly on frames 1..4 where error is 1.0^2 = 1.0
        self.assertAlmostEqual(res_masked["loss"], 1.0, places=5)
        self.assertEqual(res_masked["active_training_frames"], 4.0)

        # When loss_non_conditioned_only=False: frame 0 error is included
        res_unmasked = compute_masked_loss(v_pred, v_target, mask, loss_non_conditioned_only=False)
        self.assertGreater(res_unmasked["loss"], 100000.0)

    def test_conditioning_noise_augmentation(self):
        """4. Optional small noise perturbation on conditioning latents."""
        builder_clean = ConditioningTensorBuilder(ConditioningConfig(enable_conditioning_noise=False))
        builder_noisy = ConditioningTensorBuilder(ConditioningConfig(
            enable_conditioning_noise=True,
            conditioning_noise_strength=0.05
        ))

        noisy_latents = [[[1.0]] for _ in range(2)]
        clean_lat = [[5.0]]
        cond_map = {0: clean_lat}

        rng = random.Random(42)
        out_clean, _ = builder_clean.build_concatenated_input_cpu(noisy_latents, cond_map)
        rng = random.Random(42)
        out_noisy, _ = builder_noisy.build_concatenated_input_cpu(noisy_latents, cond_map, noise_rng=rng)

        # Frame 0 conditioning channel should have noise added
        clean_cond_val = out_clean[0][2][0]
        noisy_cond_val = out_noisy[0][2][0]
        self.assertEqual(clean_cond_val, 5.0)
        self.assertNotEqual(clean_cond_val, noisy_cond_val)
        self.assertAlmostEqual(clean_cond_val, noisy_cond_val, delta=0.2)


if __name__ == "__main__":
    unittest.main()
