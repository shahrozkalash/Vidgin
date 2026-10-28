"""Part 2: bucket geometry validation, nearest-bucket assignment, manifest schema,
and a deterministic BucketSampler with state_dict() resumption (Part 2 audit).
"""

from __future__ import annotations

import json

import torch

MEDIA_TYPES = ("video", "image")


def validate_bucket(frames: int, height: int, width: int) -> None:
    """Strict bucket rules: (T-1) % 4 == 0 and H, W multiples of 16."""
    if (frames - 1) % 4 != 0:
        raise ValueError(f"Bucket frame rule violated: T={frames}, need (T-1) % 4 == 0")
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(f"Bucket spatial rule violated: {(height, width)} not multiples of 16")


def nearest_bucket(frames: int, height: int, width: int, buckets: list[tuple]) -> tuple:
    """Pick the bucket minimizing aspect-ratio + frame distance."""
    aspect = width / height
    best, best_d = None, float("inf")
    for b in buckets:
        bt, bh, bw = b
        d = abs(bw / bh - aspect) + abs(bt - frames) / max(frames, 1)
        if d < best_d:
            best, best_d = b, d
    return best


# ---------------------------------------------------------------- manifest
MANIFEST_SCHEMA = {
    "id": str,
    "media_type": str,
    "duration": (int, float),
    "fps": (int, float),
    "height": int,
    "width": int,
}


def validate_manifest(records: list[dict]) -> list[dict]:
    """Validate typed manifest records; raises ValueError on first violation."""
    for r in records:
        for key, types in MANIFEST_SCHEMA.items():
            if key not in r:
                raise ValueError(f"Manifest record missing key {key!r}: {r}")
            if not isinstance(r[key], types):
                raise ValueError(f"Manifest record {r.get('id')}: {key} has wrong type")
        if r["media_type"] not in MEDIA_TYPES:
            raise ValueError(f"Bad media_type {r['media_type']!r}")
    return records


def save_manifest(records: list[dict], path: str) -> None:
    validate_manifest(records)
    with open(path, "w") as f:
        json.dump(records, f)


def load_manifest(path: str) -> list[dict]:
    with open(path) as f:
        return validate_manifest(json.load(f))


# ---------------------------------------------------------------- sampler
class BucketSampler:
    """Deterministic index sampler with save/resume.

    state_dict()/load_state_dict() reproduce the next batch bit-for-bit
    (Part 2 audit: deterministic sampler resumption).
    """

    def __init__(self, num_items: int, batch_size: int, seed: int = 0, drop_last: bool = True):
        self.num_items = num_items
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.cursor = 0
        self.perm = self._make_perm()

    def _make_perm(self) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed + self.epoch * 7919)
        return torch.randperm(self.num_items, generator=g)

    def next_batch(self) -> list[int] | None:
        if self.cursor + self.batch_size > self.num_items:
            if not self.drop_last and self.cursor < self.num_items:
                batch = self.perm[self.cursor :].tolist()
                self.cursor = self.num_items
                return batch
            # new epoch
            self.epoch += 1
            self.perm = self._make_perm()
            self.cursor = 0
        batch = self.perm[self.cursor : self.cursor + self.batch_size].tolist()
        self.cursor += self.batch_size
        return batch

    def state_dict(self) -> dict:
        return {
            "num_items": self.num_items,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "drop_last": self.drop_last,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "perm": self.perm,
        }

    def load_state_dict(self, state: dict) -> None:
        self.num_items = state["num_items"]
        self.batch_size = state["batch_size"]
        self.seed = state["seed"]
        self.drop_last = state["drop_last"]
        self.epoch = state["epoch"]
        self.cursor = state["cursor"]
        self.perm = state["perm"]

    def reconcile(self, num_items: int, batch_size: int) -> bool:
        """Adopt the CURRENT dataset size / batch size after a resume.

        A checkpoint's saved sampler describes the dataset it was saved
        on. Restoring it wholesale onto a different dataset silently
        truncates coverage: the saved permutation only ever visits the
        old item range (the Stage 2 VAE incident — sampler num_items
        stayed 4203 across resumes onto a 6,339-clip dataset, so 10,000
        steps sampled only the first 4,203 manifest entries).

        When the current run's values differ, adopt them, regenerate the
        permutation for the new size (same seed discipline: seed +
        epoch * 7919, epoch preserved), restart the epoch (cursor = 0),
        and print one loud line. When they match, do nothing at all —
        resume stays bit-for-bit reproducible. Returns True when a
        reconciliation happened.
        """
        if num_items <= 0:
            raise ValueError("num_items must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_items == self.num_items and batch_size == self.batch_size:
            return False
        old_items, old_batch = self.num_items, self.batch_size
        self.num_items = num_items
        self.batch_size = batch_size
        self.perm = self._make_perm()
        self.cursor = 0
        print(
            f"SAMPLER RECONCILED: checkpoint num_items={old_items} "
            f"batch_size={old_batch} -> current num_items={num_items} "
            f"batch_size={batch_size}; permutation regenerated for the "
            f"current dataset (seed={self.seed}, epoch={self.epoch}), "
            f"cursor reset to 0; reason: current dataset and "
            f"--batch-size take precedence on resume",
            flush=True,
        )
        return True
