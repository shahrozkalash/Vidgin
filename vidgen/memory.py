"""
Vidgen Long-Term Memory and Episodic Feedback Store (Batch 1)
Supports:
1. MemoryStore on disk (JSON index + latent JSON files).
   Entry: id, name, type (character, scene, style, motion, clip), text description,
   embedding (stub encoder interface), reference latents, metadata (created_at, source_video_id, use_count, average_rating).
2. Full CRUD and store operations:
   - add, search (text or embedding, type filter, top-k), update, merge near-duplicates (cosine threshold), pin, export, and real delete that removes files.
3. Generation hook:
   - Parses @name tokens or searches top-k by text similarity.
   - Enforces max_references budget.
   - Attaches role tags (character, background, style, motion, first_frame).
4. Long-video episodic memory:
   - Stores keyframes and embeddings per finished chunk.
   - Retrieves earlier chunks beyond sliding window by semantic similarity.
   - Enforces strict chunk and token budgets.
"""

import os
import json
import math
import time
import shutil
import re
from typing import List, Dict, Tuple, Optional, Any, Union
from vidgen.config import MemoryConfig


def vector_dot(u: List[float], v: List[float]) -> float:
    return sum(a * b for a, b in zip(u, v))


def vector_norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(u: List[float], v: List[float]) -> float:
    nu = vector_norm(u)
    nv = vector_norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return vector_dot(u, v) / (nu * nv)


class StubTextVisualEncoder:
    """
    Deterministic stub text and latent encoder for CPU testing and verification.
    Generates semantic embeddings where shared keywords have high cosine similarity.
    """
    def __init__(self, dim: int = 64, seed: int = 42):
        self.dim = dim
        self.seed = seed

    def encode_text(self, text: str) -> List[float]:
        """
        Embeds text into a unit-norm vector.
        Words and subword n-grams contribute additively with stopword downweighting.
        """
        words = re.findall(r"\w+", text.lower())
        if not words:
            return [1.0 / math.sqrt(self.dim)] * self.dim

        vec = [0.0] * self.dim
        stopwords = {'a', 'an', 'the', 'in', 'on', 'at', 'with', 'and', 'of', 'for', 'over', 'through', 'to'}
        for w in words:
            is_stop = w in stopwords
            weight = 0.2 if is_stop else 2.0
            h = self.seed
            for ch in w:
                h = (h * 31 + ord(ch)) & 0xFFFFFFFF
            vec[h % self.dim] += weight
            vec[(h * 37) % self.dim] += weight * 0.7
            if not is_stop:
                for i in range(len(w) - 2):
                    g = w[i : i + 3]
                    hg = self.seed
                    for ch in g:
                        hg = (hg * 31 + ord(ch)) & 0xFFFFFFFF
                    vec[hg % self.dim] += 0.4

        norm = vector_norm(vec)
        if norm < 1e-9:
            return [1.0 / math.sqrt(self.dim)] * self.dim
        return [x / norm for x in vec]

    def encode_latents(self, latents: List[List[float]]) -> List[float]:
        """
        Pools spacetime latents into a single representation vector of dimension self.dim.
        """
        if not latents or not latents[0]:
            return [0.0] * self.dim

        in_dim = len(latents[0])
        # Mean pool across tokens
        mean_token = [0.0] * in_dim
        for token in latents:
            for d in range(in_dim):
                mean_token[d] += token[d]
        count = len(latents)
        mean_token = [x / count for x in mean_token]

        # Project to self.dim
        out = [0.0] * self.dim
        for i in range(self.dim):
            val = sum(mean_token[j] * math.sin((i + 1) * 0.13 + (j + 1) * 0.27) for j in range(in_dim))
            out[i] = val

        norm = vector_norm(out)
        if norm < 1e-9:
            return [1.0 / math.sqrt(self.dim)] * self.dim
        return [x / norm for x in out]


class MemoryEntry:
    """
    Representation of an entry in the long-term memory store.
    """
    VALID_TYPES = {"character", "scene", "style", "motion", "clip"}

    def __init__(
        self,
        entry_id: str,
        name: str,
        entry_type: str,
        description: str,
        embedding: Optional[List[float]] = None,
        latents: Optional[List[List[float]]] = None,
        latents_path: Optional[str] = None,
        pinned: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        assert entry_type in self.VALID_TYPES, f"Invalid entry_type '{entry_type}'. Must be one of {self.VALID_TYPES}"
        self.id = entry_id
        # Normalize name (strip leading @)
        self.name = name.lstrip("@").strip()
        self.type = entry_type
        self.description = description
        self.embedding = embedding or []
        self.latents = latents
        self.latents_path = latents_path
        self.pinned = pinned

        # Standard metadata schema
        default_meta: Dict[str, Any] = {
            "created_at": time.time(),
            "source_video_id": None,
            "use_count": 0,
            "average_rating": 0.0,
            "rating_count": 0,
            "tags": [],
        }
        if metadata:
            default_meta.update(metadata)
        self.metadata = default_meta

    def to_index_dict(self) -> Dict[str, Any]:
        """Dictionary representation saved in index.json (excluding heavy latents)."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "embedding": self.embedding,
            "latents_path": self.latents_path,
            "pinned": self.pinned,
            "metadata": self.metadata,
        }

    def to_full_dict(self) -> Dict[str, Any]:
        """Complete dictionary including latent tensor data."""
        d = self.to_index_dict()
        d["latents"] = self.latents
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            entry_id=data["id"],
            name=data.get("name", data["id"]),
            entry_type=data["type"],
            description=data.get("description", ""),
            embedding=data.get("embedding", []),
            latents=data.get("latents"),
            latents_path=data.get("latents_path"),
            pinned=data.get("pinned", False),
            metadata=data.get("metadata", {}),
        )


class MemoryStore:
    """
    On-disk long-term memory store.
    Filesystem layout:
      <store_dir>/
        index.json
        latents/
          <entry_id>.json
    """
    def __init__(
        self,
        store_dir: str = ".vidgen_memory",
        encoder: Optional[StubTextVisualEncoder] = None,
        config: Optional[MemoryConfig] = None,
    ):
        self.store_dir = os.path.abspath(store_dir)
        self.latents_dir = os.path.join(self.store_dir, "latents")
        self.index_path = os.path.join(self.store_dir, "index.json")
        self.config = config or MemoryConfig()
        self.encoder = encoder or StubTextVisualEncoder(dim=self.config.embedding_dim)

        self.entries: Dict[str, MemoryEntry] = {}
        self._ensure_storage()
        self._load_index()

    def _ensure_storage(self):
        os.makedirs(self.latents_dir, exist_ok=True)
        if not os.path.exists(self.index_path):
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump({"version": "1.0", "entries": {}}, f, indent=2)

    def _load_index(self):
        if not os.path.exists(self.index_path):
            return
        with open(self.index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.entries = {}
        for entry_id, entry_dict in data.get("entries", {}).items():
            self.entries[entry_id] = MemoryEntry.from_dict(entry_dict)

    def _save_index(self):
        data = {
            "version": "1.0",
            "updated_at": time.time(),
            "count": len(self.entries),
            "entries": {eid: entry.to_index_dict() for eid, entry in self.entries.items()},
        }
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add(self, entry: MemoryEntry) -> str:
        """
        Adds a memory entry to the store.
        Saves latent payload to disk at <store_dir>/latents/<id>.json.
        """
        # Ensure embedding exists
        if not entry.embedding:
            if entry.description:
                entry.embedding = self.encoder.encode_text(entry.description)
            elif entry.latents:
                entry.embedding = self.encoder.encode_latents(entry.latents)

        # Save latents to disk
        latent_filename = f"{entry.id}.json"
        latent_path = os.path.join(self.latents_dir, latent_filename)
        rel_latent_path = os.path.join("latents", latent_filename)

        with open(latent_path, "w", encoding="utf-8") as f:
            json.dump({"id": entry.id, "latents": entry.latents or []}, f)

        entry.latents_path = rel_latent_path
        self.entries[entry.id] = entry
        self._save_index()
        return entry.id

    def get(self, entry_id: str, load_latents: bool = True) -> Optional[MemoryEntry]:
        """Retrieves entry by ID. If load_latents=True, reads latents from disk if not in memory."""
        if entry_id not in self.entries:
            return None
        entry = self.entries[entry_id]
        if load_latents and entry.latents is None and entry.latents_path:
            full_path = os.path.join(self.store_dir, entry.latents_path)
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    entry.latents = data.get("latents")
        return entry

    def get_by_name(self, name: str, load_latents: bool = True) -> Optional[MemoryEntry]:
        """Case-insensitive lookup by name (e.g. '@amber_fox' or 'amber_fox')."""
        clean_name = name.lstrip("@").strip().lower()
        for entry in self.entries.values():
            if entry.name.lower() == clean_name or entry.id.lower() == clean_name:
                return self.get(entry.id, load_latents=load_latents)
        return None

    def search(
        self,
        query: Union[str, List[float]],
        entry_type: Optional[str] = None,
        top_k: int = 3,
        threshold: float = 0.0,
        load_latents: bool = True,
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Semantic similarity search across entries.
        query: prompt string or embedding vector.
        Filters by entry_type and cosine threshold.
        Returns up to top_k entries sorted by similarity score descending.
        """
        if isinstance(query, str):
            q_emb = self.encoder.encode_text(query)
        else:
            q_emb = query

        scored: List[Tuple[MemoryEntry, float]] = []
        for entry in self.entries.values():
            if entry_type and entry.type != entry_type:
                continue
            if not entry.embedding:
                continue
            score = cosine_similarity(q_emb, entry.embedding)
            if score >= threshold:
                scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = scored[:top_k]

        if load_latents:
            for entry, _ in results:
                if entry.latents is None and entry.latents_path:
                    full_path = os.path.join(self.store_dir, entry.latents_path)
                    if os.path.exists(full_path):
                        with open(full_path, "r", encoding="utf-8") as f:
                            entry.latents = json.load(f).get("latents")

        return results

    def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Updates metadata, description, latents, or pinned state."""
        entry = self.get(entry_id, load_latents=False)
        if not entry:
            return False

        if "description" in updates:
            entry.description = updates["description"]
            entry.embedding = self.encoder.encode_text(entry.description)
        if "pinned" in updates:
            entry.pinned = bool(updates["pinned"])
        if "name" in updates:
            entry.name = updates["name"].lstrip("@").strip()
        if "metadata" in updates:
            entry.metadata.update(updates["metadata"])
        if "latents" in updates:
            entry.latents = updates["latents"]
            latent_path = os.path.join(self.store_dir, entry.latents_path or f"latents/{entry.id}.json")
            with open(latent_path, "w", encoding="utf-8") as f:
                json.dump({"id": entry.id, "latents": entry.latents}, f)

        self._save_index()
        return True

    def delete(self, entry_id: str) -> bool:
        """
        REAL DELETE:
        1. Removes entry from in-memory dictionary.
        2. Removes entry from index.json.
        3. Physically removes the latent file from disk using os.remove.
        Guarantees that files are deleted and cannot be resurrected.
        """
        if entry_id not in self.entries:
            return False

        entry = self.entries[entry_id]
        if entry.latents_path:
            full_latent_path = os.path.join(self.store_dir, entry.latents_path)
            if os.path.exists(full_latent_path):
                os.remove(full_latent_path)
                assert not os.path.exists(full_latent_path), f"File {full_latent_path} was not deleted!"

        del self.entries[entry_id]
        self._save_index()
        return True

    def merge_near_duplicates(
        self,
        cosine_threshold: float = 0.95,
        dry_run: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Detects and merges near-duplicate entries of the same type.
        Retains the entry with more uses / higher rating (or pinned).
        Aggregates use_count and updates weighted average rating.
        Physically deletes the merged duplicate from disk.
        """
        entries_list = list(self.entries.values())
        n = len(entries_list)
        merged_pairs = []
        deleted_ids = set()

        for i in range(n):
            e1 = entries_list[i]
            if e1.id in deleted_ids:
                continue
            for j in range(i + 1, n):
                e2 = entries_list[j]
                if e2.id in deleted_ids:
                    continue
                if e1.type != e2.type:
                    continue
                if not e1.embedding or not e2.embedding:
                    continue

                sim = cosine_similarity(e1.embedding, e2.embedding)
                if sim >= cosine_threshold:
                    # Decide winner: pinned beats unpinned, then higher use_count, then higher rating
                    if e1.pinned and not e2.pinned:
                        winner, loser = e1, e2
                    elif e2.pinned and not e1.pinned:
                        winner, loser = e2, e1
                    elif e1.metadata.get("use_count", 0) >= e2.metadata.get("use_count", 0):
                        winner, loser = e1, e2
                    else:
                        winner, loser = e2, e1

                    # Merge statistics into winner
                    u1 = winner.metadata.get("use_count", 0)
                    u2 = loser.metadata.get("use_count", 0)
                    winner.metadata["use_count"] = u1 + u2

                    r1 = winner.metadata.get("average_rating", 0.0)
                    c1 = winner.metadata.get("rating_count", 0)
                    r2 = loser.metadata.get("average_rating", 0.0)
                    c2 = loser.metadata.get("rating_count", 0)
                    if c1 + c2 > 0:
                        winner.metadata["average_rating"] = (r1 * c1 + r2 * c2) / (c1 + c2)
                        winner.metadata["rating_count"] = c1 + c2

                    tags = set(winner.metadata.get("tags", [])) | set(loser.metadata.get("tags", []))
                    winner.metadata["tags"] = list(tags)

                    merged_pairs.append({
                        "kept_id": winner.id,
                        "deleted_id": loser.id,
                        "type": winner.type,
                        "similarity": sim,
                    })

                    deleted_ids.add(loser.id)

        if not dry_run:
            for pair in merged_pairs:
                self.delete(pair["deleted_id"])
            self._save_index()

        return merged_pairs

    def pin(self, entry_id: str, pinned: bool = True) -> bool:
        return self.update(entry_id, {"pinned": pinned})

    def export(self, export_path: str) -> Dict[str, Any]:
        """Exports index and all latent payloads into a single standalone JSON."""
        full_entries = {}
        for eid in self.entries:
            entry = self.get(eid, load_latents=True)
            if entry:
                full_entries[eid] = entry.to_full_dict()

        payload = {
            "version": "1.0",
            "exported_at": time.time(),
            "count": len(full_entries),
            "entries": full_entries,
        }
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload

    def import_entries(self, import_data_or_path: Union[str, Dict[str, Any]]) -> int:
        """Imports entries from a standalone JSON bundle."""
        if isinstance(import_data_or_path, str):
            with open(import_data_or_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = import_data_or_path

        imported_count = 0
        for entry_dict in data.get("entries", {}).values():
            entry = MemoryEntry.from_dict(entry_dict)
            self.add(entry)
            imported_count += 1
        return imported_count


class ReferenceAttachment:
    """Attached reference for model conditioning with explicit role tags."""
    def __init__(
        self,
        entry_id: str,
        name: str,
        role: str,
        latents: List[List[float]],
        similarity_score: float = 1.0,
        mention_token: Optional[str] = None,
    ):
        self.entry_id = entry_id
        self.name = name
        self.role = role  # "character", "background", "style", "motion", "first_frame"
        self.latents = latents
        self.similarity_score = similarity_score
        self.mention_token = mention_token

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "name": self.name,
            "role": self.role,
            "latent_tokens": len(self.latents),
            "similarity_score": self.similarity_score,
            "mention_token": self.mention_token,
        }


def map_type_to_role(entry_type: str) -> str:
    """Maps store entry types to Vidgen reference roles."""
    mapping = {
        "character": "character",
        "scene": "background",
        "style": "style",
        "motion": "motion",
        "clip": "first_frame",
    }
    return mapping.get(entry_type, "character")


class GenerationMemoryHook:
    """
    Hook that connects the MemoryStore to video generation.
    1. Scans prompt for @name mentions and retrieves referenced entities.
    2. Runs semantic search for additional context above similarity_threshold.
    3. Enforces max_retrieved_references budget.
    4. Attaches role tags and updates usage telemetry.
    """
    def __init__(self, store: MemoryStore, config: Optional[MemoryConfig] = None):
        self.store = store
        self.config = config or MemoryConfig()

    def retrieve_prompt_references(self, prompt: str) -> List[ReferenceAttachment]:
        """
        Retrieves references for a prompt while strictly enforcing budget limits.
        """
        if not self.config.enable_memory_store:
            return []

        attachments: List[ReferenceAttachment] = []
        attached_ids = set()
        max_refs = self.config.max_retrieved_references

        # 1. Parse @name mentions in prompt (e.g. "@amber_fox", "@neon_cyberpunk")
        mentions = re.findall(r"@([a-zA-Z0-9_-]+)", prompt)
        for mention in mentions:
            if len(attachments) >= max_refs:
                break
            entry = self.store.get_by_name(mention, load_latents=True)
            if entry and entry.id not in attached_ids and entry.latents:
                role = map_type_to_role(entry.type)
                attachments.append(
                    ReferenceAttachment(
                        entry_id=entry.id,
                        name=entry.name,
                        role=role,
                        latents=entry.latents,
                        similarity_score=1.0,
                        mention_token=f"@{mention}",
                    )
                )
                attached_ids.add(entry.id)
                # Increment usage counter
                entry.metadata["use_count"] = entry.metadata.get("use_count", 0) + 1

        # 2. Semantic search for remaining slots up to max_refs
        remaining_slots = max_refs - len(attachments)
        if remaining_slots > 0:
            search_results = self.store.search(
                query=prompt,
                top_k=remaining_slots * 2,  # Oversample for filtering
                threshold=self.config.similarity_threshold,
                load_latents=True,
            )
            for entry, score in search_results:
                if len(attachments) >= max_refs:
                    break
                if entry.id not in attached_ids and entry.latents:
                    role = map_type_to_role(entry.type)
                    attachments.append(
                        ReferenceAttachment(
                            entry_id=entry.id,
                            name=entry.name,
                            role=role,
                            latents=entry.latents,
                            similarity_score=score,
                        )
                    )
                    attached_ids.add(entry.id)
                    entry.metadata["use_count"] = entry.metadata.get("use_count", 0) + 1

        # Persist updated use counts
        self.store._save_index()

        # Strict budget guarantee: len(attachments) <= max_refs
        assert len(attachments) <= max_refs, f"Budget exceeded: {len(attachments)} > {max_refs}"
        return attachments


class EpisodicMemoryBuffer:
    """
    Episodic memory for Part 13 Long-Video generation.
    Stores key frames and embeddings of each completed chunk.
    Retrieves earlier chunks by similarity to current segment caption as budgeted extra context beyond the sliding window.
    """
    def __init__(self, config: Optional[MemoryConfig] = None, encoder: Optional[StubTextVisualEncoder] = None):
        self.config = config or MemoryConfig()
        self.encoder = encoder or StubTextVisualEncoder(dim=self.config.embedding_dim)
        # List of stored episodes: chunk_id, time_range, caption, embedding, keyframe_latents
        self.episodes: List[Dict[str, Any]] = []

    def store_chunk(
        self,
        chunk_id: int,
        time_range: Tuple[float, float],
        segment_caption: str,
        keyframe_latents: List[List[float]],
    ):
        """Stores keyframe latents and semantic embedding of a finished chunk."""
        embedding = self.encoder.encode_text(segment_caption)
        self.episodes.append({
            "chunk_id": chunk_id,
            "time_range": time_range,
            "segment_caption": segment_caption,
            "embedding": embedding,
            "keyframe_latents": keyframe_latents,
        })

    def retrieve_beyond_window(
        self,
        current_chunk_id: int,
        current_caption: str,
        sliding_window_chunks: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves earlier chunks beyond the sliding window matching current caption.
        Strictly respects episodic_budget_chunks and episodic_budget_tokens.
        """
        if not self.config.enable_episodic_memory:
            return []

        # Chunks strictly outside active sliding window (and not anchor chunk 0 which is already cached)
        # i.e. chunk_id < current_chunk_id - sliding_window_chunks and chunk_id > 0
        candidate_episodes = [
            ep for ep in self.episodes
            if ep["chunk_id"] < (current_chunk_id - sliding_window_chunks) and ep["chunk_id"] > 0
        ]
        if not candidate_episodes:
            return []

        curr_emb = self.encoder.encode_text(current_caption)
        scored = []
        for ep in candidate_episodes:
            sim = cosine_similarity(curr_emb, ep["embedding"])
            if sim >= self.config.similarity_threshold:
                scored.append((ep, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Budget enforcement
        budget = self.config.episodic_budget_chunks
        selected = [ep for ep, _ in scored[:budget]]

        # Token budget enforcement on keyframes
        for ep in selected:
            if len(ep["keyframe_latents"]) > self.config.episodic_budget_tokens:
                ep["keyframe_latents"] = ep["keyframe_latents"][:self.config.episodic_budget_tokens]

        assert len(selected) <= budget, f"Episodic chunk budget exceeded: {len(selected)} > {budget}"
        return selected
