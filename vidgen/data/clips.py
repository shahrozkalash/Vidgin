"""Real-clip dataset: FolderClipDataset.

A data directory holds pre-processed clips as uint8 ``.npy`` files of shape
``(T, H, W, 3)`` plus a ``manifest.jsonl`` with one JSON object per line::

    {"file": "clip_000001.npy", "caption": "people crossing a city street"}

Build such a directory with ``scripts/prepare_clips.py`` (see
``notebooks/prepare_clips_drive.ipynb`` for the Colab flow).

Samples are yielded in exactly the same format as
``MovingMNISTDataset.__getitem__``: a tuple
``(tensor, caption)`` where tensor is float32 ``[3, T, H, W]`` in ``[-1, 1]``
and caption is a ``str`` — so the trainers can consume either dataset.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

MANIFEST_NAME = "manifest.jsonl"


def _load_manifest(data_dir: str) -> list[dict]:
    manifest_path = os.path.join(data_dir, MANIFEST_NAME)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Clip data directory not found: {data_dir!r}. "
            "Build it with scripts/prepare_clips.py "
            "(see notebooks/prepare_clips_drive.ipynb)."
        )
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"No {MANIFEST_NAME} in {data_dir!r}. "
            "Expected a FolderClipDataset directory built by "
            "scripts/prepare_clips.py (one .npy clip + manifest.jsonl per clip)."
        )
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{manifest_path}:{lineno}: not valid JSON: {e}"
                ) from e
            if not isinstance(rec, dict) or "file" not in rec or "caption" not in rec:
                raise ValueError(
                    f"{manifest_path}:{lineno}: each line must be "
                    '{"file": "<name>.npy", "caption": "..."}'
                )
            entries.append(rec)
    if not entries:
        raise ValueError(f"{manifest_path}: manifest is empty")
    return entries


class FolderClipDataset(Dataset):
    """Deterministic clip dataset backed by a prepared directory.

    Indexing follows manifest order exactly (no shuffling here — the
    trainers' BucketSampler handles batching), so the dataset is
    reproducible across runs and checkpoint resumes.
    """

    def __init__(
        self,
        data_dir: str,
        frames: int = 17,
        height: int = 64,
        width: int = 64,
        max_clips: int | None = None,
    ):
        self.data_dir = data_dir
        self.frames = frames
        self.height = height
        self.width = width
        entries = _load_manifest(data_dir)
        if max_clips is not None:
            entries = entries[: max(0, int(max_clips))]
        if not entries:
            raise ValueError(
                f"{data_dir}: manifest yielded no clips (max_clips={max_clips})"
            )
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        rec = self.entries[int(i)]
        path = os.path.join(self.data_dir, rec["file"])
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Clip file missing: {path!r} "
                f"(referenced by manifest line {int(i) + 1} of {self.data_dir})"
            )
        clip = np.load(path)
        if clip.dtype != np.uint8:
            raise ValueError(
                f"{path}: expected uint8 array, got {clip.dtype}"
            )
        want = (self.frames, self.height, self.width, 3)
        if clip.shape != want:
            raise ValueError(
                f"{path}: expected shape {want} (T,H,W,3), got {clip.shape}"
            )
        # uint8 [0,255] -> float32 [-1,1], channel-first, matching Moving MNIST.
        t = torch.from_numpy(clip).permute(3, 0, 1, 2).contiguous().to(torch.float32)
        t = t / 127.5 - 1.0
        caption = str(rec["caption"]).strip() or f"clip {os.path.basename(path)}"
        return t, caption
