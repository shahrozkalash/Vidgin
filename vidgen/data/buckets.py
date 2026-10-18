"""Multi-Aspect Ratio and Spacetime Bucketing for Wan Video Models.

Buckets are (frames, H, W) triples with H and W strictly multiples of 16.
Supports 16:9, 9:16, 1:1, 4:3, 3:4 aspect ratios and T=1 image buckets.
Batch size scales inversely with bucket tokens to maintain constant token budget per step.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterator, Optional, Any
import numpy as np
import torch
from torch.utils.data import Sampler


@dataclass(frozen=True)
class BucketSpec:
    frames: int
    height: int
    width: int

    def __post_init__(self):
        # Validate Wan T = 4k + 1 rule
        assert (self.frames - 1) % 4 == 0, f"Bucket frames ({self.frames}) must satisfy T = 4k + 1"
        assert self.height % 16 == 0, f"Bucket height ({self.height}) must be a multiple of 16"
        assert self.width % 16 == 0, f"Bucket width ({self.width}) must be a multiple of 16"

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def token_count(self, patch_size: Tuple[int, int, int] = (1, 2, 2)) -> int:
        """Token count after 4x temporal (T -> (T-1)//4 + 1) and 8x spatial (H//8, W//8) VAE,

        followed by (pt, ph, pw) patchify.
        """
        t = (self.frames - 1) // 4 + 1
        h = self.height // 8
        w = self.width // 8
        pt, ph, pw = patch_size
        return (t // pt) * (h // ph) * (w // pw)

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.frames, self.height, self.width)


def get_standard_buckets(preset: str = "tiny") -> List[BucketSpec]:
    """Generate standard aspect ratio buckets (16:9, 9:16, 1:1, 4:3, 3:4)

    for both videos (T=4k+1) and images (T=1).
    All heights and widths are multiples of 16.
    """
    preset = preset.lower()

    if preset == "tiny":
        # Base ~4096 px area (64x64), T=17 video and T=1 image
        resolutions = [
            (64, 64),   # 1:1  (aspect 1.00)
            (48, 80),   # 16:9 approx (aspect 1.67, 48*80=3840)
            (80, 48),   # 9:16 approx (aspect 0.60)
            (48, 64),   # 4:3  approx (aspect 1.33)
            (64, 48),   # 3:4  approx (aspect 0.75)
        ]
        video_frames = [17]

    elif preset == "small":
        # Base ~65536 px area (256x256), T=33 video and T=1 image
        resolutions = [
            (256, 256), # 1:1
            (192, 336), # 16:9 approx (aspect 1.75)
            (336, 192), # 9:16 approx (aspect 0.57)
            (224, 288), # 4:3  approx (aspect 1.29)
            (288, 224), # 3:4  approx (aspect 0.78)
        ]
        video_frames = [33]

    else:  # wan1p3b / wan 480p
        # Base ~400k px area (480x832), T=81 video and T=1 image
        resolutions = [
            (480, 832), # 16:9 (aspect 1.73)
            (832, 480), # 9:16 (aspect 0.58)
            (640, 640), # 1:1  (aspect 1.00)
            (560, 736), # 4:3  approx (aspect 1.31)
            (736, 560), # 3:4  approx (aspect 0.76)
        ]
        video_frames = [81]

    buckets = []
    # Video buckets
    for f in video_frames:
        for h, w in resolutions:
            buckets.append(BucketSpec(frames=f, height=h, width=w))

    # Image buckets (T=1, satisfying 1 = 4(0) + 1)
    for h, w in resolutions:
        buckets.append(BucketSpec(frames=1, height=h, width=w))

    return buckets


def find_nearest_bucket(
    frames: int,
    height: int,
    width: int,
    buckets: List[BucketSpec],
    is_image: bool = False,
) -> BucketSpec:
    """Assign input media to the closest bucket by temporal distance and aspect ratio."""
    target_aspect = width / max(1, height)

    if is_image or frames == 1:
        candidates = [b for b in buckets if b.frames == 1]
    else:
        candidates = [b for b in buckets if b.frames > 1]

    if not candidates:
        candidates = buckets

    best_bucket = candidates[0]
    best_dist = float("inf")

    for b in candidates:
        # Distance metric: aspect ratio error + normalized temporal error
        aspect_diff = abs(b.aspect_ratio - target_aspect)
        frame_diff = abs(b.frames - frames) / max(1.0, float(b.frames))
        dist = aspect_diff + 0.5 * frame_diff

        if dist < best_dist:
            best_dist = dist
            best_bucket = b

    return best_bucket


class DistributedBucketSampler(Sampler[List[int]]):
    """Distributed Batch Sampler drawing homogeneous batches from a single bucket.

    Scales batch size per bucket to maintain a roughly constant token count per batch.
    Supports multi-GPU sharding (rank, world_size) and deterministic state resumption.
    """

    def __init__(
        self,
        bucket_assignments: List[BucketSpec],  # Assigned BucketSpec for each dataset index
        target_tokens_per_batch: int = 160,    # Target tokens per step (e.g. 2 * 80 for Tiny)
        min_batch_size: int = 1,
        max_batch_size: int = 16,
        shuffle: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
    ):
        self.bucket_assignments = bucket_assignments
        self.target_tokens_per_batch = target_tokens_per_batch
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last

        self.epoch = 0
        self.batch_idx = 0

        # Group dataset indices by bucket
        self.bucket_to_indices: Dict[BucketSpec, List[int]] = {}
        for idx, bucket in enumerate(bucket_assignments):
            if bucket not in self.bucket_to_indices:
                self.bucket_to_indices[bucket] = []
            self.bucket_to_indices[bucket].append(idx)

        # Compute dynamic batch size for each bucket
        self.bucket_batch_sizes: Dict[BucketSpec, int] = {}
        for bucket in self.bucket_to_indices:
            tokens = max(1, bucket.token_count())
            scaled_bs = max(self.min_batch_size, min(self.max_batch_size, round(self.target_tokens_per_batch / tokens)))
            self.bucket_batch_sizes[bucket] = scaled_bs

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.batch_idx = 0

    def state_dict(self) -> Dict[str, Any]:
        """Save state for exact resume."""
        return {
            "epoch": self.epoch,
            "batch_idx": self.batch_idx,
            "seed": self.seed,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore sampler state."""
        self.epoch = state_dict.get("epoch", 0)
        self.batch_idx = state_dict.get("batch_idx", 0)
        self.seed = state_dict.get("seed", self.seed)

    def _generate_batches(self) -> List[List[int]]:
        """Generate all batches for the current epoch deterministically."""
        rng = np.random.RandomState(self.seed + self.epoch)
        all_batches: List[List[int]] = []

        for bucket, indices in self.bucket_to_indices.items():
            bs = self.bucket_batch_sizes[bucket]
            bucket_idx_list = list(indices)
            if self.shuffle:
                rng.shuffle(bucket_idx_list)

            for i in range(0, len(bucket_idx_list), bs):
                batch = bucket_idx_list[i : i + bs]
                if len(batch) == bs or not self.drop_last:
                    all_batches.append(batch)

        if self.shuffle:
            rng.shuffle(all_batches)

        # Distributed sharding: slice by rank and world_size
        if self.world_size > 1:
            # Ensure divisible by world_size
            total_batches = len(all_batches)
            remainder = total_batches % self.world_size
            if remainder != 0:
                all_batches = all_batches[: total_batches - remainder]
            rank_batches = all_batches[self.rank :: self.world_size]
        else:
            rank_batches = all_batches

        return rank_batches

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._generate_batches()
        # Fast-forward to resume point if resumed mid-epoch
        if self.batch_idx > 0 and self.batch_idx < len(batches):
            batches = batches[self.batch_idx :]

        for batch in batches:
            self.batch_idx += 1
            yield batch

    def __len__(self) -> int:
        batches = self._generate_batches()
        return len(batches)


def worker_init_fn(worker_id: int, base_seed: int = 42) -> None:
    """Set deterministic per-worker seed for PyTorch DataLoader workers.

    Ensures each worker has a unique, reproducible pseudorandom sequence.
    """
    worker_seed = (base_seed + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# Alias for backward compatibility
AspectRatioBucketSampler = DistributedBucketSampler

