"""Real-clip dataset: FolderClipDataset.

A data directory holds pre-processed clips as uint8 ``.npy`` files of shape
``(T, H, W, 3)`` plus a ``manifest.jsonl`` with one JSON object per line::

    {"file": "clip_000001.npy", "caption": "people crossing a city street"}

A line may also carry a ``"depth"`` key naming a uint8 ``(T, H, W)`` depth
map (larger = nearer) written by ``scripts/prepare_clips.py --depth`` —
Dawood's 3D conditioning (see ``vidgen/depth.py``). :meth:`get_depth`
loads it; clips without one simply have no map.

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
        shard: tuple[int, int] | None = None,
    ):
        self.data_dir = data_dir
        self.frames = frames
        self.height = height
        self.width = width
        entries = _load_manifest(data_dir)
        if max_clips is not None:
            entries = entries[: max(0, int(max_clips))]
        if shard is not None:
            # Weight-averaging pool worker shard (vidgen/pool.py): entries
            # are first put in deterministic FILE-NAME order (independent
            # of manifest line order, so every platform sees the same
            # numbering), then worker K of N keeps positions % N == K-1.
            # Shards are disjoint; their union is the full dataset.
            from vidgen.pool import parse_data_shard, shard_positions

            k, n = parse_data_shard(shard)
            entries = sorted(entries, key=lambda r: r["file"])
            entries = [entries[i] for i in shard_positions(len(entries), (k, n))]
            if not entries:
                raise ValueError(
                    f"{data_dir}: data shard {k}/{n} selects no clips "
                    f"(dataset too small for {n} shards)"
                )
        if not entries:
            raise ValueError(
                f"{data_dir}: manifest yielded no clips (max_clips={max_clips})"
            )
        self.entries = entries
        self.shard = shard

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

    def get_depth(self, i: int):
        """This clip's 3D depth map as float32 ``[1, T, H, W]`` in ``[0, 1]``
        (larger = nearer), or ``None`` when the manifest entry has no
        ``"depth"`` file. Kept separate from :meth:`__getitem__` so the
        ``(tensor, caption)`` sample contract shared with Moving MNIST
        never changes; the DiT trainer asks for maps only when training
        with depth conditioning."""
        rec = self.entries[int(i)]
        name = rec.get("depth")
        if not name:
            return None
        path = os.path.join(self.data_dir, str(name))
        if not os.path.isfile(path):
            return None
        arr = np.load(path)
        if arr.dtype != np.uint8:
            raise ValueError(f"{path}: expected uint8 depth, got {arr.dtype}")
        want = (self.frames, self.height, self.width)
        if arr.shape != want:
            raise ValueError(
                f"{path}: expected depth shape {want} (T,H,W), got {arr.shape}"
            )
        t = torch.from_numpy(arr).to(torch.float32) / 255.0
        return t.unsqueeze(0).contiguous()
