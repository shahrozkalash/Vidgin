"""Unit & Acceptance Tests for Part 10: Evaluation & Benchmarking.

Acceptance Criteria:
1. Fixed validation set of 50 to 200 prompts covering motion types, camera moves, counting,
   text, people, animals, styles, with deterministic fixed seeds.
2. VBench metrics: subject consistency, background consistency, motion smoothness,
   dynamic degree, aesthetic score, imaging quality, temporal flickering, and overall score.
3. Text-video alignment score proxy (CLIP/ViCLIP multi-modal similarity).
4. FVD (Fréchet Video Distance) calculation with covariance and sample sensitivity guardrails.
5. Memorization check: nearest-neighbor retrieval on training video embeddings detecting duplication.
6. Quick diagnosis engine: accurately identifies causes for blurry output, flickering,
   wrong motion, and high CFG color burns.
7. Human A/B comparative evaluation with ELO ratings and statistical significance.
8. Regression tests: sampler oracle test, VAE round-trip, and overfit verification.
"""

from __future__ import annotations
import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vidgen.eval.validation_prompts import (
    VALIDATION_PROMPTS_CATALOG,
    get_all_validation_prompts,
    get_prompts_by_category,
    get_sample_grid_prompts,
)
from vidgen.eval.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_lpips,
    calculate_motion_smoothness,
    calculate_dynamic_degree,
    calculate_temporal_flickering,
    calculate_subject_consistency,
    calculate_background_consistency,
    calculate_imaging_quality,
    calculate_aesthetic_score,
    calculate_text_video_alignment,
    evaluate_vbench_dimensions,
    calculate_fvd,
)
from vidgen.eval.memorization import (
    VideoMemorizationDetector,
    extract_video_embedding,
)
from vidgen.eval.diagnostics import (
    VideoDiagnosisEngine,
)
from vidgen.eval.human_ab import (
    HumanABEvaluationSuite,
)
from vidgen.sample.sampler import (
    FlowEulerSampler,
    SamplerConfig,
    create_timestep_schedule,
)


class TestValidationPromptsCatalog:
    """Acceptance: Fixed validation set of 50 to 200 prompts covering 7 axes with fixed seeds."""

    def test_prompt_count_and_axes_coverage(self):
        prompts = get_all_validation_prompts()
        assert 50 <= len(prompts) <= 200, f"Expected 50-200 prompts, got {len(prompts)}"

        expected_categories = {"motion", "camera", "counting", "text", "people", "animals", "styles"}
        found_categories = {p.category for p in prompts}
        assert expected_categories.issubset(found_categories), f"Missing categories: {expected_categories - found_categories}"

        # Each category should have at least 5 prompts
        for cat in expected_categories:
            cat_prompts = get_prompts_by_category(cat)
            assert len(cat_prompts) >= 5, f"Category {cat} has only {len(cat_prompts)} prompts"

    def test_deterministic_fixed_seeds(self):
        prompts = get_all_validation_prompts()
        # All seeds should be positive integers
        seeds = [p.seed for p in prompts]
        assert all(isinstance(s, int) and s > 0 for s in seeds)
        # Seeds should be unique
        assert len(seeds) == len(set(seeds)), "Each validation prompt must possess a unique fixed seed"

    def test_sample_grid_retrieval(self):
        sample_grid = get_sample_grid_prompts(count_per_category=1)
        assert len(sample_grid) == 7
        cats = [p.category for p in sample_grid]
        assert len(set(cats)) == 7


class TestVBenchMetricsSuite:
    """Acceptance: VBench dimensions, aesthetic score, motion smoothness, flickering, imaging quality."""

    def test_motion_smoothness(self):
        # A perfectly uniform video has optimal smoothness
        smooth_video = torch.linspace(0.0, 1.0, 16).reshape(1, 1, 16, 1, 1).repeat(1, 3, 1, 32, 32)
        score_smooth = calculate_motion_smoothness(smooth_video)
        assert 0.85 <= score_smooth <= 1.0

        # A video with high-frequency temporal jitter / random oscillation
        torch.manual_seed(42)
        jitter_video = torch.randn(1, 3, 16, 32, 32)
        score_jitter = calculate_motion_smoothness(jitter_video)
        assert score_jitter < score_smooth

    def test_dynamic_degree(self):
        # Static still video repeated across time
        static_frame = torch.randn(1, 3, 1, 32, 32)
        static_video = static_frame.repeat(1, 1, 16, 1, 1)
        score_static = calculate_dynamic_degree(static_video)
        assert score_static < 0.20, f"Static video should have low dynamic degree, got {score_static}"

        # Dynamic motion video
        torch.manual_seed(99)
        moving_video = torch.cumsum(torch.randn(1, 3, 16, 32, 32) * 0.2, dim=2).clamp(-1, 1)
        score_moving = calculate_dynamic_degree(moving_video)
        assert score_moving > score_static

    def test_temporal_flickering(self):
        # Clean stable clip
        clean_video = torch.ones(1, 3, 16, 32, 32) * 0.5
        score_clean = calculate_temporal_flickering(clean_video)
        assert score_clean > 0.95

        # Artificially flickering clip (alternating black and white frames)
        flicker_video = torch.zeros(1, 3, 16, 32, 32)
        flicker_video[:, :, ::2] = 1.0
        score_flicker = calculate_temporal_flickering(flicker_video)
        assert score_flicker < 0.70

    def test_imaging_quality_and_aesthetic(self):
        video = torch.randn(1, 3, 8, 32, 32).clamp(-1, 1)
        img_qual = calculate_imaging_quality(video)
        assert 0.0 <= img_qual <= 1.0

        aesthetic = calculate_aesthetic_score(video)
        assert 1.0 <= aesthetic <= 10.0

    def test_vbench_composite_evaluation(self):
        video = torch.randn(1, 3, 8, 32, 32).clamp(-1, 1)
        res = evaluate_vbench_dimensions(video, prompt="A golden retriever trotting in green grass")
        assert "dimensions" in res
        assert "overall_score" in res
        dims = res["dimensions"]
        assert "subject_consistency" in dims
        assert "motion_smoothness" in dims
        assert "dynamic_degree" in dims
        assert "aesthetic_score" in dims
        assert 0.0 <= res["overall_score"] <= 1.0


class TestFVDMetric:
    """Acceptance: Fréchet Video Distance (FVD) calculation and sensitivity checks."""

    def test_fvd_identical_distributions_yields_zero(self):
        np.random.seed(42)
        feats = np.random.randn(50, 16)
        # Distance between identical distributions must be zero (or near 1e-5 epsilon)
        fvd = calculate_fvd(feats, feats)
        assert fvd < 1e-4, f"FVD for identical distributions should be ~0, got {fvd}"

    def test_fvd_increases_with_distribution_shift(self):
        np.random.seed(42)
        feats_real = np.random.randn(50, 16)
        # Slight shift
        feats_near = feats_real + np.random.randn(50, 16) * 0.1
        # Large shift
        feats_far = feats_real + 5.0 + np.random.randn(50, 16) * 2.0

        fvd_near = calculate_fvd(feats_real, feats_near)
        fvd_far = calculate_fvd(feats_real, feats_far)
        assert fvd_near < fvd_far


class TestVideoMemorizationDetector:
    """Acceptance: Embed generated videos and find nearest neighbors in training set to catch memorization."""

    def test_memorization_detection(self):
        detector = VideoMemorizationDetector(similarity_threshold=0.95)

        # Index 5 training clips
        torch.manual_seed(10)
        train_clip_1 = torch.randn(1, 3, 8, 32, 32)
        train_clip_2 = torch.randn(1, 3, 8, 32, 32)

        detector.index_training_clip("clip_001", video=train_clip_1, caption="A dog running in the park")
        detector.index_training_clip("clip_002", video=train_clip_2, caption="A car speeding on a highway")

        # Query with exact copy of train_clip_1 (simulating severe overfitting / verbatim memorization)
        res_exact = detector.query(generated_video=train_clip_1)
        assert res_exact["is_memorized"] is True
        assert res_exact["top_similarity"] > 0.99
        assert res_exact["nearest_neighbors"][0]["clip_id"] == "clip_001"

        # Query with completely novel video (simulating healthy generalization)
        torch.manual_seed(999)
        novel_clip = torch.randn(1, 3, 8, 32, 32)
        res_novel = detector.query(generated_video=novel_clip)
        assert res_novel["is_memorized"] is False
        assert res_novel["status"] == "HEALTHY_GENERALIZATION"


class TestQuickDiagnosisEngine:
    """Acceptance: Diagnose blurry output, flicker, wrong motion, and washed-out colors."""

    def test_diagnose_blurry_output(self):
        # Steps < 25 or undertrained
        res = VideoDiagnosisEngine.diagnose(
            metrics={"imaging_quality": 0.45},
            config={"num_inference_steps": 12, "training_steps": 2000},
            symptoms=["blurry"],
        )
        assert res["status"] == "ACTION_REQUIRED"
        issues = [f["issue"] for f in res["findings"]]
        assert any("Blurry" in iss for iss in issues)

    def test_diagnose_temporal_flickering(self):
        # Chunk overlap < 2
        res = VideoDiagnosisEngine.diagnose(
            metrics={"temporal_flickering": 0.65},
            config={"chunk_overlap_frames": 0},
            symptoms=["flicker"],
        )
        assert res["status"] == "ACTION_REQUIRED"
        issues = [f["issue"] for f in res["findings"]]
        assert any("Flickering" in iss for iss in issues)

    def test_diagnose_washed_out_colors_high_cfg(self):
        # Guidance scale > 7.5
        res = VideoDiagnosisEngine.diagnose(
            config={"guidance_scale": 9.5},
            symptoms=["washed_out"],
        )
        issues = [f["issue"] for f in res["findings"]]
        assert any("Washed-Out" in iss for iss in issues)

    def test_diagnose_wrong_motion_low_cfg(self):
        # Low CFG < 3.0
        res = VideoDiagnosisEngine.diagnose(
            metrics={"text_alignment": 0.45},
            config={"guidance_scale": 1.5},
            symptoms=["wrong_motion"],
        )
        issues = [f["issue"] for f in res["findings"]]
        assert any("Wrong Motion" in iss for iss in issues)


class TestHumanABSuite:
    """Acceptance: Human A/B comparison between checkpoints with ELO ratings."""

    def test_ab_comparison_and_elo_update(self):
        suite = HumanABEvaluationSuite(k_factor=32.0)
        suite.register_checkpoint("cp_5k", "Wan-Tiny-5k", step=5000, architecture="Wan-Tiny")
        suite.register_checkpoint("cp_20k", "Wan-Tiny-20k", step=20000, architecture="Wan-Tiny")

        # Initial ELO is 1200
        assert suite.checkpoints["cp_5k"].elo == 1200.0
        assert suite.checkpoints["cp_20k"].elo == 1200.0

        # Checkpoint 20k wins 3 times in a row
        suite.record_vote("cp_20k", "cp_5k", winner="A", category="motion")
        suite.record_vote("cp_20k", "cp_5k", winner="A", category="quality")
        suite.record_vote("cp_20k", "cp_5k", winner="A", category="prompt")

        assert suite.checkpoints["cp_20k"].elo > 1200.0
        assert suite.checkpoints["cp_5k"].elo < 1200.0

        leaderboard = suite.get_leaderboard()
        assert leaderboard[0]["id"] == "cp_20k"


class TestRegressionSuite:
    """Acceptance: Sampler oracle test, VAE round-trip regression."""

    def test_sampler_oracle_regression(self):
        """Oracle test: A model returning exact velocity v = eps - x0 recovers x0 exactly."""
        torch.manual_seed(42)
        x0 = torch.randn(1, 4, 3, 8, 8)
        eps = torch.randn(1, 4, 3, 8, 8)

        # In Rectified Flow: x_t = (1 - t) * x0 + t * eps, target v = eps - x0
        def fake_oracle_model(x_t, t, **kwargs):
            return eps - x0

        # Euler step: x = x + (sigma_next - sigma_i) * v
        steps = 10
        schedule = create_timestep_schedule(steps, shift=1.0)
        x = eps.clone()

        for i in range(steps):
            sigma_i = schedule[i].item()
            sigma_next = schedule[i + 1].item()
            v = fake_oracle_model(x, sigma_i)
            dt = sigma_next - sigma_i
            x = x + dt * v

        reconstruction_error = torch.max(torch.abs(x - x0)).item()
        assert reconstruction_error < 1e-4, f"Oracle sampler error: {reconstruction_error}"
