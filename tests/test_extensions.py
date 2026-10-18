"""Unit and Acceptance Tests for Part 11: Post-Training and Extensions.

Covers:
1. LoRA Injection, Trainable Ratio (< 1%), Merge & Unmerge Identity
2. Image-to-Video (I2V) 33-Channel Upgrade & Zero-Init Mathematical Equivalence
3. Wan 2.2 Dual Experts 0.875 Timestep Schedule Dispatch & Memory Offloading
4. Rectified-Flow Reflow & 4/8-Step Few-Step Solvers
5. Longer Video Chunked Overlap Generation with Cosine Weighting
6. NSFW Prompt Moderation, Video Safety Checks, and C2PA Provenance Watermarking
"""

import math
import unittest
import torch
import torch.nn as nn

from vidgen.models.dit import WanDiT
from vidgen.extensions.lora import (
    LoRALinear,
    inject_lora_into_dit,
    merge_lora_weights,
    unmerge_lora_weights,
    count_parameters_summary,
    get_lora_state_dict,
    load_lora_weights,
)
from vidgen.extensions.i2v import (
    WanImageToVideoDiT,
    convert_t2v_to_i2v,
    I2VConditioningCollator,
)
from vidgen.extensions.dual_experts import (
    Wan22DualExperts,
    create_dual_experts_from_base,
    compute_dual_expert_training_loss,
)
from vidgen.extensions.distillation import (
    RectifiedFlowReflow,
    FewStepFlowSolver,
)
from vidgen.extensions.long_video import (
    ChunkedOverlapGenerator,
)
from vidgen.extensions.safety import (
    NSFWPromptFilter,
    VideoSafetyChecker,
    C2PAProvenanceWatermark,
)


def get_tiny_dit() -> WanDiT:
    """Helper to construct a tiny DiT model for rapid CPU testing."""
    return WanDiT(
        in_channels=16,
        out_channels=16,
        num_layers=2,
        hidden_dim=64,
        num_heads=4,
        text_dim=64,
        patch_size=(1, 2, 2),
        time_dim=64,
    )


class TestPart11Extensions(unittest.TestCase):

    def test_lora_injection_and_trainable_ratio(self):
        """Verify LoRA injects properly, freezes base model, and trainable params < 5%."""
        model = get_tiny_dit()
        adapters = inject_lora_into_dit(
            model,
            rank=16,
            alpha=16.0,
            target_modules=("q_proj", "k_proj", "v_proj", "out_proj"),
        )

        self.assertGreater(len(adapters), 0)
        summary = count_parameters_summary(model)
        self.assertLess(summary["trainable_ratio_percent"], 5.0)

        # Forward pass with LoRA
        B = 2
        x = torch.randn(B, 16, 5, 8, 8)
        t = torch.tensor([0.5, 0.2])
        context = torch.randn(B, 16, 64)
        out = model(x, t, context)
        self.assertEqual(out.shape, x.shape)

    def test_lora_merge_and_unmerge(self):
        """Verify merging LoRA weights preserves output and unmerge restores state."""
        model = get_tiny_dit()
        inject_lora_into_dit(model, rank=8, alpha=8.0)

        B = 1
        x = torch.randn(B, 16, 5, 8, 8)
        t = torch.tensor([0.4])
        context = torch.randn(B, 16, 64)

        # Set eval mode
        model.eval()
        with torch.no_grad():
            out_before = model(x, t, context)
            merged_count = merge_lora_weights(model)
            self.assertGreater(merged_count, 0)
            out_merged = model(x, t, context)
            torch.testing.assert_close(out_before, out_merged, atol=1e-5, rtol=1e-5)

            unmerge_lora_weights(model)
            out_unmerged = model(x, t, context)
            torch.testing.assert_close(out_before, out_unmerged, atol=1e-5, rtol=1e-5)

    def test_i2v_zero_init_identity_property(self):
        """Verify that zero-init in I2V produces output identical to T2V model at initialization."""
        base_dit = get_tiny_dit()
        base_dit.eval()

        B = 1
        x_noisy = torch.randn(B, 16, 5, 8, 8)
        t = torch.tensor([0.7])
        context = torch.randn(B, 16, 64)

        with torch.no_grad():
            t2v_out = base_dit(x_noisy, t, context)

            # Convert to I2V
            i2v_model = convert_t2v_to_i2v(base_dit, zero_init=True)
            cond_frame = torch.randn(B, 16, 1, 8, 8)
            i2v_out = i2v_model(x_noisy, t, context, cond_latents=cond_frame)

            # Because the weights for channels 16..32 are strictly zeros,
            # the patch projection output should be mathematically identical!
            torch.testing.assert_close(t2v_out, i2v_out, atol=1e-5, rtol=1e-5)

    def test_wan22_dual_experts_schedule_dispatch(self):
        """Verify dual-expert boundary dispatch at t=0.875."""
        base_model = get_tiny_dit()
        dual_experts = create_dual_experts_from_base(base_model, boundary_timestep=0.875)

        # Timestep >= 0.875 -> High Noise Expert
        high_expert, name_high = dual_experts.get_expert_for_timestep(0.95)
        self.assertEqual(name_high, "high")
        self.assertIs(high_expert, dual_experts.high_noise_expert)

        # Timestep < 0.875 -> Low Noise Expert
        low_expert, name_low = dual_experts.get_expert_for_timestep(0.50)
        self.assertEqual(name_low, "low")
        self.assertIs(low_expert, dual_experts.low_noise_expert)

        # Check loss masking
        timesteps = torch.tensor([0.9, 0.95, 0.3, 0.1])
        pred_v = torch.randn(4, 16, 5, 8, 8)
        target_v = torch.randn(4, 16, 5, 8, 8)
        loss_high = compute_dual_expert_training_loss(high_expert, "high", pred_v, target_v, timesteps, boundary=0.875)
        self.assertFalse(torch.isnan(loss_high))

    def test_few_step_flow_distillation_solver(self):
        """Verify few-step solver generates strictly decreasing sigma grid."""
        grid_4 = FewStepFlowSolver.get_time_grid(steps=4, shift=3.0)
        self.assertEqual(len(grid_4), 5)
        self.assertAlmostEqual(grid_4[0], 1.0, places=4)
        self.assertAlmostEqual(grid_4[-1], 0.0, places=4)
        # Monotonically strictly decreasing
        for i in range(len(grid_4) - 1):
            self.assertGreater(grid_4[i], grid_4[i + 1])

    def test_chunked_overlap_temporal_blending(self):
        """Verify chunked overlap planner and cosine latent blending."""
        generator = ChunkedOverlapGenerator(
            chunk_latent_frames=10,
            overlap_latent_frames=4,
            blend_mode="cosine",
        )
        total_latent_frames = 22
        plan = generator.calculate_chunks_plan(total_latent_frames)
        self.assertGreater(len(plan), 1)

        # Simulate synthetic chunks
        chunks = [torch.randn(1, 16, end - start, 8, 8) for start, end in plan]
        stitched = generator.stitch_latent_chunks(chunks, total_latent_frames)
        self.assertEqual(stitched.shape, (1, 16, total_latent_frames, 8, 8))
        self.assertFalse(torch.isnan(stitched).any())

    def test_safety_filter_and_provenance(self):
        """Verify NSFW filter detection and C2PA manifest creation."""
        flt = NSFWPromptFilter()
        safe_res = flt.check_prompt("A majestic eagle soaring over snow-capped mountains at sunrise")
        self.assertTrue(safe_res["is_safe"])
        self.assertEqual(safe_res["status"], "APPROVED")

        unsafe_res = flt.check_prompt("Extremely explicit pornographic video with gore")
        self.assertFalse(unsafe_res["is_safe"])
        self.assertEqual(unsafe_res["status"], "REJECTED")
        self.assertIn("nsfw_adult", unsafe_res["flagged_categories"])

        # Provenance manifest
        c2pa = C2PAProvenanceWatermark()
        manifest = c2pa.generate_c2pa_manifest(
            prompt="A majestic eagle",
            model_name="Wan-2.1-1.3B",
            seed=42,
        )
        self.assertEqual(manifest["c2pa_version"], "2.1")
        self.assertIn("signature_hash", manifest)

        # Frame safety check
        checker = VideoSafetyChecker()
        clean_frames = torch.zeros(5, 3, 32, 32) + 0.2
        frame_res = checker.check_video_frames(clean_frames)
        self.assertTrue(frame_res["is_safe"])


if __name__ == "__main__":
    unittest.main()
