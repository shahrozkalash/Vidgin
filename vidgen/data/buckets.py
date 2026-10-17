"""Multi-aspect ratio and frame-length bucketing for video batches."""

from __future__ import annotations
from typing import Dict, List, Tuple, Iterator
import numpy as np
import torch
from torch.utils.data import Sampler


class AspectRatioBucketSampler(Sampler[List[int]]):
    """Batches dataset items by resolution and frame count buckets to prevent wasteful padding.

    Each bucket is defined by (frames, height, width).
    """

    def __init__(
        self,
        shapes: List[Tuple[int, int, int]],  # List of (T, H, W) for each dataset index
        batch_size: int = 2,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.shapes = shapes
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Group indices by shape bucket
        self.buckets: Dict[Tuple[int, int, int], List[int]] = {}
        for idx, shape in enumerate(shapes):
            if shape not in self.buckets:
                self.buckets[shape] = []
            self.buckets[shape].append(idx)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed + self.epoch)
        batches: List[List[int]] = []

        for shape, indices in self.buckets.items():
            bucket_indices = list(indices)
            if self.shuffle:
                rng.shuffle(bucket_indices)

            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        count = 0
        for indices in self.buckets.values():
            if self.drop_last:
                count += len(indices) // self.batch_size
            else:
                count += (len(indices) + self.batch_size - 1) // self.batch_size
        return count
