"""
Unit tests for Memory & Feedback Loop Batch 1: Memory Store & Episodic Retrieval.
Covers:
1. Retrieval returns the right synthetic character (semantic search and @name mention).
2. Real delete: Removes entry from index AND physically deletes latent file from filesystem.
3. Budget enforcement: Never exceeds max_retrieved_references or episodic_budget_chunks.
4. Plumbing into reference bank: role tags (character, background, style, motion, first_frame) and conditioning.
5. Near-duplicate merging: aggregates metadata, updates weighted rating, deletes duplicate from disk.
6. Long-video episodic memory: keyframes stored and retrieved beyond sliding window by caption similarity.
7. Default config flags are OFF (enable_memory_store=False, enable_episodic_memory=False).
"""

import os
import shutil
import tempfile
import unittest
from typing import List

from vidgen.config import MemoryConfig, LongVideoConfig
from vidgen.memory import (
    MemoryStore,
    MemoryEntry,
    StubTextVisualEncoder,
    GenerationMemoryHook,
    EpisodicMemoryBuffer,
    map_type_to_role,
    cosine_similarity,
)


class TestMemoryStore(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test storage
        self.test_dir = tempfile.mkdtemp(prefix="vidgen_test_mem_")
        self.encoder = StubTextVisualEncoder(dim=32, seed=42)
        self.cfg = MemoryConfig(
            enable_memory_store=True,
            enable_episodic_memory=True,
            store_dir=self.test_dir,
            max_retrieved_references=2,
            similarity_threshold=0.35,
            merge_cosine_threshold=0.92,
            episodic_budget_chunks=1,
            episodic_budget_tokens=16,
            embedding_dim=32,
        )
        self.store = MemoryStore(store_dir=self.test_dir, encoder=self.encoder, config=self.cfg)

    def tearDown(self):
        # Clean up temporary test files
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_default_config_flags_are_off(self):
        """Rule check: All memory feature flags MUST default to OFF (False)."""
        default_cfg = MemoryConfig()
        self.assertFalse(default_cfg.enable_memory_store, "enable_memory_store must default to False")
        self.assertFalse(default_cfg.enable_episodic_memory, "enable_episodic_memory must default to False")
        self.assertEqual(default_cfg.max_retrieved_references, 3)

        long_cfg = LongVideoConfig()
        self.assertFalse(long_cfg.memory.enable_memory_store)
        self.assertFalse(long_cfg.memory.enable_episodic_memory)

    def test_retrieval_returns_right_synthetic_character(self):
        """
        Part 1.5: Retrieval returns the right synthetic character by prompt or @mention.
        """
        # Character A: Amber Fox
        latents_fox = [[0.8, 0.2] * 8 for _ in range(4)]
        entry_fox = MemoryEntry(
            entry_id="char_fox",
            name="amber_fox",
            entry_type="character",
            description="A cute amber fox with bushy tail and white chest fur standing in snowy forest",
            latents=latents_fox,
            metadata={"tags": ["animal", "fox", "nature"]},
        )
        self.store.add(entry_fox)

        # Character B: Cyber Robot
        latents_robot = [[-0.5, 0.9] * 8 for _ in range(4)]
        entry_robot = MemoryEntry(
            entry_id="char_robot",
            name="cyber_robot",
            entry_type="character",
            description="A futuristic chrome cyborg robot with glowing blue optical sensors",
            latents=latents_robot,
            metadata={"tags": ["sci-fi", "robot", "cyberpunk"]},
        )
        self.store.add(entry_robot)

        # Style C: Watercolor
        entry_style = MemoryEntry(
            entry_id="style_watercolor",
            name="watercolor_wash",
            entry_type="style",
            description="Soft dreamy pastel watercolor painting with fluid pigment bleed",
            latents=[[0.1, 0.1] * 8],
        )
        self.store.add(entry_style)

        # 1. Search by text description
        results_fox = self.store.search(
            query="close-up of an amber fox in winter snow",
            entry_type="character",
            top_k=1,
            threshold=0.3,
        )
        self.assertEqual(len(results_fox), 1)
        self.assertEqual(results_fox[0][0].id, "char_fox")
        self.assertEqual(results_fox[0][0].name, "amber_fox")

        results_robot = self.store.search(
            query="cybernetic robot walking through high-tech hangar",
            entry_type="character",
            top_k=1,
            threshold=0.3,
        )
        self.assertEqual(len(results_robot), 1)
        self.assertEqual(results_robot[0][0].id, "char_robot")

        # 2. Hook retrieval by @mention
        hook = GenerationMemoryHook(self.store, self.cfg)
        attached = hook.retrieve_prompt_references("Cinematic shot of @amber_fox leaping over a fallen log")
        self.assertGreaterEqual(len(attached), 1)
        self.assertEqual(attached[0].name, "amber_fox")
        self.assertEqual(attached[0].role, "character")
        self.assertEqual(attached[0].mention_token, "@amber_fox")

    def test_real_delete_removes_data_and_disk_files(self):
        """
        Part 1.2: Real delete removes entry from index AND physically deletes latent file from filesystem.
        """
        entry = MemoryEntry(
            entry_id="del_target_01",
            name="temporary_prop",
            entry_type="scene",
            description="A rustic wooden chair in an abandoned attic room",
            latents=[[1.23, 4.56] * 8 for _ in range(2)],
        )
        self.store.add(entry)

        # Confirm file exists on disk
        latent_file = os.path.join(self.test_dir, "latents", "del_target_01.json")
        self.assertTrue(os.path.exists(latent_file), "Latent file must exist on disk before deletion")
        self.assertIn("del_target_01", self.store.entries)

        # Execute real delete
        deleted = self.store.delete("del_target_01")
        self.assertTrue(deleted)

        # Verify entry is removed from in-memory index
        self.assertNotIn("del_target_01", self.store.entries)
        self.assertIsNone(self.store.get("del_target_01"))

        # Verify file is physically removed from filesystem
        self.assertFalse(os.path.exists(latent_file), "Latent file must be completely deleted from disk")

        # Reload store from disk to ensure persistence of deletion
        reloaded_store = MemoryStore(store_dir=self.test_dir, encoder=self.encoder, config=self.cfg)
        self.assertNotIn("del_target_01", reloaded_store.entries)
        self.assertFalse(os.path.exists(latent_file))

    def test_budget_is_never_exceeded(self):
        """
        Part 1.3 & 1.5: Generation hook and episodic memory strictly enforce budget caps.
        """
        # Populate store with 5 entries
        for i in range(5):
            entry = MemoryEntry(
                entry_id=f"item_{i}",
                name=f"character_{i}",
                entry_type="character",
                description=f"Persistent hero character number {i} with unique costume",
                latents=[[float(i)] * 16],
            )
            self.store.add(entry)

        # Generation hook with max_retrieved_references = 2
        hook = GenerationMemoryHook(self.store, self.cfg)
        prompt_with_many_mentions = "Epic scene with @character_0 and @character_1 and @character_2 and @character_3"
        attached = hook.retrieve_prompt_references(prompt_with_many_mentions)

        # Must not exceed budget of 2!
        self.assertLessEqual(len(attached), self.cfg.max_retrieved_references)
        self.assertEqual(len(attached), 2)

        # Test episodic memory budget
        episodic = EpisodicMemoryBuffer(config=self.cfg, encoder=self.encoder)
        # Store 4 chunks
        for c in range(1, 5):
            episodic.store_chunk(
                chunk_id=c,
                time_range=(c * 5.0, (c + 1) * 5.0),
                segment_caption=f"Hero exploring ancient temple ruins chamber {c}",
                keyframe_latents=[[0.1] * 16 for _ in range(32)],  # 32 tokens
            )

        # Retrieve at chunk 6 (sliding window = 1 -> candidates are chunks 1, 2, 3, 4)
        retrieved_chunks = episodic.retrieve_beyond_window(
            current_chunk_id=6,
            current_caption="Hero discovers glowing altar in ancient temple ruins",
            sliding_window_chunks=1,
        )

        # Episodic budget cap = 1 chunk
        self.assertEqual(len(retrieved_chunks), 1)
        # Token budget cap = 16 tokens
        self.assertLessEqual(len(retrieved_chunks[0]["keyframe_latents"]), self.cfg.episodic_budget_tokens)
        self.assertEqual(len(retrieved_chunks[0]["keyframe_latents"]), 16)

    def test_plumbing_into_reference_bank_and_roles(self):
        """
        Part 1.3: Reference attachments carry valid role tags (character, background, style, motion, first_frame).
        """
        roles_to_test = [
            ("character", "character"),
            ("scene", "background"),
            ("style", "style"),
            ("motion", "motion"),
            ("clip", "first_frame"),
        ]

        for entry_type, expected_role in roles_to_test:
            role = map_type_to_role(entry_type)
            self.assertEqual(role, expected_role)

            entry = MemoryEntry(
                entry_id=f"test_{entry_type}",
                name=f"ref_{entry_type}",
                entry_type=entry_type,
                description=f"A reference for {entry_type}",
                latents=[[1.0] * 16],
            )
            self.store.add(entry)

        hook = GenerationMemoryHook(self.store, self.cfg)
        attached = hook.retrieve_prompt_references("Shot of @ref_character in @ref_scene with @ref_style")
        # Ensure attached items have expected attributes
        for att in attached:
            self.assertIn(att.role, {"character", "background", "style", "motion", "first_frame"})
            self.assertIsNotNone(att.latents)

    def test_merge_near_duplicates_combines_ratings_and_deletes_disk_file(self):
        """
        Part 1.2: Merge near-duplicates detects redundant entries, merges stats, and deletes file.
        """
        # Duplicate A
        e1 = MemoryEntry(
            entry_id="dup_a",
            name="warrior_knight",
            entry_type="character",
            description="Medieval armored knight holding a broadsword",
            latents=[[0.5] * 16],
            metadata={"use_count": 5, "average_rating": 4.0, "rating_count": 2, "tags": ["knight"]},
        )
        self.store.add(e1)

        # Duplicate B (almost identical description -> very high cosine similarity)
        e2 = MemoryEntry(
            entry_id="dup_b",
            name="armored_knight",
            entry_type="character",
            description="Medieval armored knight holding a heavy broadsword",
            latents=[[0.51] * 16],
            metadata={"use_count": 3, "average_rating": 5.0, "rating_count": 1, "tags": ["medieval"]},
        )
        self.store.add(e2)

        b_file = os.path.join(self.test_dir, "latents", "dup_b.json")
        self.assertTrue(os.path.exists(b_file))

        # Merge duplicates
        merged = self.store.merge_near_duplicates(cosine_threshold=0.85, dry_run=False)
        self.assertGreaterEqual(len(merged), 1)

        # Winner should be dup_a (higher use_count)
        winner = self.store.get("dup_a")
        self.assertIsNotNone(winner)
        self.assertEqual(winner.metadata["use_count"], 8)  # 5 + 3
        # Weighted rating: (4.0*2 + 5.0*1) / 3 = 13/3 = 4.333
        self.assertAlmostEqual(winner.metadata["average_rating"], 13.0 / 3.0, places=2)
        self.assertIn("medieval", winner.metadata["tags"])
        self.assertIn("knight", winner.metadata["tags"])

        # Loser dup_b must be completely deleted
        self.assertIsNone(self.store.get("dup_b"))
        self.assertFalse(os.path.exists(b_file), "Duplicate file must be deleted from disk")

    def test_export_and_import(self):
        """Part 1.2: Export to standalone JSON and re-import preserves entries and latents."""
        e = MemoryEntry(
            entry_id="exp_01",
            name="sunset_coast",
            entry_type="scene",
            description="Golden hour sunset over rolling ocean waves hitting rocky cliffs",
            latents=[[0.7] * 16],
            metadata={"tags": ["sunset", "ocean"]},
        )
        self.store.add(e)

        export_path = os.path.join(self.test_dir, "backup.json")
        self.store.export(export_path)
        self.assertTrue(os.path.exists(export_path))

        # Import into fresh store directory
        new_store_dir = os.path.join(self.test_dir, "fresh_store")
        fresh_store = MemoryStore(store_dir=new_store_dir, encoder=self.encoder, config=self.cfg)
        imported_count = fresh_store.import_entries(export_path)

        self.assertEqual(imported_count, 1)
        imported_entry = fresh_store.get("exp_01", load_latents=True)
        self.assertIsNotNone(imported_entry)
        self.assertEqual(imported_entry.name, "sunset_coast")
        self.assertEqual(imported_entry.latents, [[0.7] * 16])


if __name__ == "__main__":
    unittest.main()
