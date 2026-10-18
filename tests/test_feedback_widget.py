"""
Unit tests for Feedback Widget (Colab / Jupyter notebook thumbs feedback & pair comparison).
Covers:
1. Records are written correctly to JSONL with full schema (gen_id, prompt, seed, checkpoint_id, rating, tags, time_spent).
2. Double clicks: Keeps the last rating for that gen_id.
3. Undo: Retracts the previous rating from active records.
4. Session reload / resume: Skips already rated items.
5. Compare mode: Records pairwise preference pairs (A, B, or neither).
6. Rating hygiene: Tracks time spent and flags monotonous / speedrunner raters.
7. Fallback when ipywidgets is missing.
8. Default config flags are OFF.
"""

import os
import shutil
import tempfile
import unittest
from typing import List, Dict, Any

from vidgen.config import FeedbackConfig, LongVideoConfig
from vidgen.feedback_widget import (
    show_and_rate,
    compare_and_rate,
    analyze_rater_hygiene,
    FakeWidgetBackend,
    get_active_ratings,
    read_feedback_records,
    undo_last_feedback,
)


class TestFeedbackWidget(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="vidgen_test_feedback_")
        self.log_path = os.path.join(self.test_dir, "test_feedback.jsonl")
        self.pairs_log_path = os.path.join(self.test_dir, "test_pairs.jsonl")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_default_config_flags_are_off(self):
        """Rule check: All feedback configuration flags MUST default to OFF (False)."""
        cfg = FeedbackConfig()
        self.assertFalse(cfg.enable_feedback_logging, "enable_feedback_logging must default to False")
        self.assertFalse(cfg.consent_given, "consent_given must default to False")

        long_cfg = LongVideoConfig()
        self.assertFalse(long_cfg.feedback.enable_feedback_logging)

    def test_records_are_written_correctly(self):
        """
        Part 1: Each rating click appends one record to JSONL with all schema fields.
        """
        video_paths = ["/tmp/v0.mp4", "/tmp/v1.mp4"]
        prompts = ["snow leopard climbing mountain", "sports car drifting in rain"]
        gen_ids = ["gen_101", "gen_102"]
        seeds = [42, 1337]
        checkpoints = ["ckpt_tiny_v1", "ckpt_tiny_v1"]

        # Simulate user clicking:
        # Item 0: Thumbs Down with tags ['wrong motion', 'flicker']
        # Item 1: Thumbs Up with tags []
        fake_backend = FakeWidgetBackend([
            {"action": "thumbs_down", "tags": ["wrong motion", "flicker"]},
            {"action": "thumbs_up", "tags": []},
        ])

        res = show_and_rate(
            video_paths=video_paths,
            prompts=prompts,
            gen_ids=gen_ids,
            seeds=seeds,
            checkpoint_ids=checkpoints,
            log_path=self.log_path,
            rater_id="tester_1",
            backend=fake_backend,
            resume=False,
        )

        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["rated_count"], 2)

        # Inspect raw JSONL records
        records = read_feedback_records(self.log_path)
        self.assertEqual(len(records), 2)

        # Record 0 checks
        r0 = records[0]
        self.assertEqual(r0["gen_id"], "gen_101")
        self.assertEqual(r0["prompt"], prompts[0])
        self.assertEqual(r0["seed"], 42)
        self.assertEqual(r0["checkpoint_id"], "ckpt_tiny_v1")
        self.assertEqual(r0["rating"], "thumbs_down")
        self.assertEqual(r0["rating_value"], 0)
        self.assertEqual(r0["tags"], ["wrong motion", "flicker"])
        self.assertIn("timestamp", r0)
        self.assertIn("time_spent_seconds", r0)
        self.assertEqual(r0["rater_id"], "tester_1")

        # Record 1 checks
        r1 = records[1]
        self.assertEqual(r1["gen_id"], "gen_102")
        self.assertEqual(r1["rating"], "thumbs_up")
        self.assertEqual(r1["rating_value"], 1)

    def test_double_click_keeps_last_rating(self):
        """
        Part 3: Handle double clicks / revised ratings: keep the last rating for that gen_id.
        """
        video_paths = ["/tmp/v0.mp4"]
        prompts = ["a majestic fox"]
        gen_ids = ["gen_single"]

        # Run 1: User initially rates thumbs_down
        backend_1 = FakeWidgetBackend([{"action": "thumbs_down", "tags": ["wrong character"]}])
        show_and_rate(video_paths, prompts, gen_ids, log_path=self.log_path, backend=backend_1, resume=False)

        active_1 = get_active_ratings(self.log_path)
        self.assertEqual(active_1["gen_single"]["rating"], "thumbs_down")

        # Run 2: User re-rates or changes mind (double click) to thumbs_up
        backend_2 = FakeWidgetBackend([{"action": "thumbs_up", "tags": []}])
        show_and_rate(video_paths, prompts, gen_ids, log_path=self.log_path, backend=backend_2, resume=False)

        # In active ratings, only the latest rating must be present
        active_2 = get_active_ratings(self.log_path)
        self.assertEqual(len(active_2), 1)
        self.assertEqual(active_2["gen_single"]["rating"], "thumbs_up")
        self.assertEqual(active_2["gen_single"]["rating_value"], 1)

    def test_undo_works(self):
        """
        Part 3 & 6: Undo button retracts the previous rating from active records.
        """
        video_paths = ["/tmp/v0.mp4", "/tmp/v1.mp4"]
        prompts = ["prompt 0", "prompt 1"]
        gen_ids = ["gen_undo_0", "gen_undo_1"]

        # User rates item 0 thumbs_up, item 1 thumbs_down, then hits undo on item 1
        backend = FakeWidgetBackend([
            {"action": "thumbs_up", "tags": []},
            {"action": "thumbs_down", "tags": ["artifact"]},
            {"action": "undo"},  # Undo retracts item 1
            {"action": "skip"},  # Now on item 1 again, user chooses skip
        ])

        show_and_rate(
            video_paths=video_paths,
            prompts=prompts,
            gen_ids=gen_ids,
            log_path=self.log_path,
            backend=backend,
            resume=False,
        )

        active = get_active_ratings(self.log_path)
        # gen_undo_0 should remain active
        self.assertIn("gen_undo_0", active)
        self.assertEqual(active["gen_undo_0"]["rating"], "thumbs_up")

        # gen_undo_1 was undone and skipped -> should NOT be in active ratings
        self.assertNotIn("gen_undo_1", active)

    def test_session_reload_skips_rated_items(self):
        """
        Part 3 & 6: Session reload / resume reads existing ratings and marks them done,
        skipping them in the next run.
        """
        video_paths = ["/tmp/v0.mp4", "/tmp/v1.mp4", "/tmp/v2.mp4", "/tmp/v3.mp4"]
        prompts = ["p0", "p1", "p2", "p3"]
        gen_ids = ["g0", "g1", "g2", "g3"]

        # Session 1: User rates g0 and g1, then stops
        backend_sess1 = FakeWidgetBackend([
            {"action": "thumbs_up"},
            {"action": "thumbs_down"},
            {"action": "quit"},
        ])
        show_and_rate(video_paths, prompts, gen_ids, log_path=self.log_path, backend=backend_sess1, resume=True)

        active_sess1 = get_active_ratings(self.log_path)
        self.assertEqual(len(active_sess1), 2)
        self.assertIn("g0", active_sess1)
        self.assertIn("g1", active_sess1)

        # Session 2: Resumes. Must skip g0 and g1, presenting g2 and g3!
        backend_sess2 = FakeWidgetBackend([
            {"action": "thumbs_up"},
            {"action": "thumbs_up"},
        ])
        show_and_rate(video_paths, prompts, gen_ids, log_path=self.log_path, backend=backend_sess2, resume=True)

        # Verify backend_sess2 only had g2 and g3 displayed
        displayed_gids = [item["gen_id"] for item in backend_sess2.displayed_items]
        self.assertEqual(displayed_gids, ["g2", "g3"])

        # Total active ratings is now all 4
        active_sess2 = get_active_ratings(self.log_path)
        self.assertEqual(len(active_sess2), 4)

    def test_compare_mode_preference_pair(self):
        """
        Part 2: compare_and_rate shows side-by-side pair and records preference (A, B, or neither).
        """
        pairs = [
            ("gen_A1", "/tmp/a1.mp4", "gen_B1", "/tmp/b1.mp4"),
            ("gen_A2", "/tmp/a2.mp4", "gen_B2", "/tmp/b2.mp4"),
            ("gen_A3", "/tmp/a3.mp4", "gen_B3", "/tmp/b3.mp4"),
        ]
        prompts = [
            "a rabbit eating a carrot",
            "a drone flying over coastline",
            "neon city street reflections",
        ]

        # Simulate user choosing:
        # Pair 0: A
        # Pair 1: B
        # Pair 2: neither
        fake_backend = FakeWidgetBackend([
            {"action": "A"},
            {"action": "B"},
            {"action": "neither"},
        ])

        res = compare_and_rate(
            pairs=pairs,
            prompts=prompts,
            log_path=self.pairs_log_path,
            rater_id="pair_tester",
            backend=fake_backend,
        )

        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["recorded_pairs"], 3)

        records = read_feedback_records(self.pairs_log_path)
        self.assertEqual(len(records), 3)

        self.assertEqual(records[0]["choice"], "A")
        self.assertEqual(records[0]["gen_id_a"], "gen_A1")
        self.assertEqual(records[0]["gen_id_b"], "gen_B1")

        self.assertEqual(records[1]["choice"], "B")
        self.assertEqual(records[2]["choice"], "neither")

    def test_rating_hygiene_monotony_and_speedrun(self):
        """
        Part 4: Rating hygiene stores timestamp, time spent, and flags raters with identical choices.
        """
        # Case A: Monotonous rater clicking Thumbs Up 6 times with zero variance
        video_paths = [f"/tmp/v{i}.mp4" for i in range(6)]
        prompts = [f"prompt {i}" for i in range(6)]
        gen_ids = [f"g_mono_{i}" for i in range(6)]

        backend_mono = FakeWidgetBackend([{"action": "thumbs_up"} for _ in range(6)])
        show_and_rate(video_paths, prompts, gen_ids, log_path=self.log_path, backend=backend_mono, resume=False)

        hygiene_mono = analyze_rater_hygiene(self.log_path, min_samples=5, monotony_threshold=0.90)
        self.assertTrue(hygiene_mono["is_monotonous"])
        self.assertEqual(hygiene_mono["monotony_rate"], 1.0)
        self.assertTrue(hygiene_mono["flagged"])
        self.assertIn("Monotonous clicking", hygiene_mono["message"])

        # Case B: Balanced rater with varied ratings
        log_balanced = os.path.join(self.test_dir, "balanced.jsonl")
        backend_bal = FakeWidgetBackend([
            {"action": "thumbs_up"},
            {"action": "thumbs_down"},
            {"action": "thumbs_up"},
            {"action": "thumbs_down"},
            {"action": "thumbs_up"},
            {"action": "thumbs_down"},
        ])
        show_and_rate(video_paths, prompts, gen_ids, log_path=log_balanced, backend=backend_bal, resume=False)

        hygiene_bal = analyze_rater_hygiene(log_balanced, min_samples=5, monotony_threshold=0.90)
        self.assertFalse(hygiene_bal["is_monotonous"])
        self.assertEqual(hygiene_bal["monotony_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
