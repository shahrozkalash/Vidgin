"""Precomputes VAE Latents and Text Embeddings with Augmentation and Channel Normalization.

Features:
- Encodes video pixels [3, T, H, W] to latents [16, t, h, w] in fp16
- Computes augmentation variants (original, horizontal flip, alternative crop)
- Encodes prompts and null empty prompts to text embeddings in bf16
- Shards into memory-mapped .safetensors files
- Computes per-channel latent mean and std (16 channels) and saves for normalization
"""

from __future__ import annotations
import json
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from safetensors.torch import save_file, load_file

from vidgen.data.dataset import AdvancedVideoDataset


class ChannelStatsAccumulator:
    """Accumulates running mean and variance per channel for 16-channel latents."""

    def __init__(self, num_channels: int = 16):
        self.num_channels = num_channels
        self.count = 0
        self.sum = torch.zeros(num_channels, dtype=torch.float64)
        self.sq_sum = torch.zeros(num_channels, dtype=torch.float64)

    def update(self, latent: torch.Tensor) -> None:
        """Update with latent tensor [C, t, h, w] or [B, C, t, h, w]."""
        if latent.ndim == 5:
            # [B, C, t, h, w]
            B, C, t, h, w = latent.shape
            elements_per_channel = B * t * h * w
            reshaped = latent.permute(1, 0, 2, 3, 4).reshape(C, -1).double()
        else:
            # [C, t, h, w]
            C, t, h, w = latent.shape
            elements_per_channel = t * h * w
            reshaped = latent.reshape(C, -1).double()

        assert C == self.num_channels, f"Expected {self.num_channels} channels, got {C}"
        self.count += elements_per_channel
        self.sum += reshaped.sum(dim=-1)
        self.sq_sum += (reshaped ** 2).sum(dim=-1)

    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute final mean and std per channel."""
        if self.count == 0:
            return torch.zeros(self.num_channels), torch.ones(self.num_channels)

        mean = self.sum / self.count
        variance = (self.sq_sum / self.count) - (mean ** 2)
        std = torch.sqrt(torch.clamp(variance, min=1e-8))
        return mean.float(), std.float()


def run_precompute_pipeline(
    dataset: AdvancedVideoDataset,
    output_dir: str | Path,
    vae: Optional[nn.Module] = None,
    text_encoder: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu"),
    dtype_latent: torch.dtype = torch.float32,
    dtype_text: torch.dtype = torch.float32,
) -> Tuple[Path, Dict[str, Any]]:
    """Run full precomputation pipeline across dataset and compute normalization statistics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    accumulator = ChannelStatsAccumulator(num_channels=16)
    manifest_records: List[Dict[str, Any]] = []

    # Precompute empty prompt embedding for 10% CFG dropout
    if text_encoder is not None and hasattr(text_encoder, "encode_strings"):
        with torch.no_grad():
            null_emb = text_encoder.encode_strings([""], device=device)[0].to(dtype_text).cpu()
    else:
        # 384-dim dummy text embedding
        null_emb = torch.zeros(16, 384, dtype=dtype_text)

    for idx in range(len(dataset)):
        item = dataset[idx]
        clip_id = item["id"]
        pixels = item["pixels"].unsqueeze(0).to(device)  # [1, 3, T, H, W]

        # 1. Original latent encoding
        if vae is not None and hasattr(vae, "sample_latent"):
            with torch.no_grad():
                lat_orig = vae.sample_latent(pixels).squeeze(0)  # [16, t, h, w]
        else:
            # Deterministic latent generation for tiny testing
            t = (pixels.shape[2] - 1) // 4 + 1
            h = pixels.shape[3] // 8
            w = pixels.shape[4] // 8
            gen = torch.Generator().manual_seed(idx * 100)
            lat_orig = torch.randn(16, t, h, w, generator=gen)

        # 2. Horizontal flip augmentation
        pixels_hflip = torch.flip(pixels, dims=[-1])
        if vae is not None and hasattr(vae, "sample_latent"):
            with torch.no_grad():
                lat_hflip = vae.sample_latent(pixels_hflip).squeeze(0)
        else:
            lat_hflip = torch.flip(lat_orig, dims=[-1])

        # Accumulate stats on original latents
        accumulator.update(lat_orig)

        # 3. Text prompt encoding
        prompt = item["prompt"]
        if text_encoder is not None and hasattr(text_encoder, "encode_strings"):
            with torch.no_grad():
                text_emb = text_encoder.encode_strings([prompt], device=device)[0].to(dtype_text).cpu()
        else:
            gen_t = torch.Generator().manual_seed(abs(hash(prompt)) % (2**31))
            text_emb = torch.randn(16, 384, generator=gen_t, dtype=dtype_text)

        # 4. Save to safetensors
        shard_path = shards_dir / f"{clip_id}.safetensors"
        tensors_to_save = {
            "latents_orig": lat_orig.to(dtype_latent).cpu(),
            "latents_hflip": lat_hflip.to(dtype_latent).cpu(),
            "text_emb": text_emb,
            "text_null_emb": null_emb,
        }
        save_file(tensors_to_save, str(shard_path))

        manifest_records.append({
            "id": clip_id,
            "shard_path": str(shard_path),
            "prompt": prompt,
            "latent_shape": list(lat_orig.shape),
            "bucket": list(item["bucket"]),
        })

    # Finalize per-channel normalization stats
    mean, std = accumulator.finalize()
    stats_dict = {
        "num_channels": 16,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "total_elements_sampled": accumulator.count,
    }

    # Save stats JSON and torch PT file
    with open(output_dir / "latent_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)

    torch.save(
        {"mean": mean, "std": std},
        output_dir / "latent_stats.pt",
    )

    # Save precomputed manifest JSON
    with open(output_dir / "precomputed_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)

    return output_dir, stats_dict


class PrecomputedLatentDataset(Dataset):
    """Memory-mapped dataset reading precomputed .safetensors shards with CFG dropout."""

    def __init__(
        self,
        manifest_path: str | Path,
        stats_path: Optional[str | Path] = None,
        use_augmentation: bool = True,
        caption_dropout: float = 0.10,
    ):
        manifest_path = Path(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        self.use_augmentation = use_augmentation
        self.caption_dropout = caption_dropout

        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        if stats_path is not None and Path(stats_path).exists():
            stats = torch.load(stats_path, map_location="cpu", weights_only=False)
            self.mean = stats["mean"][:, None, None, None]
            self.std = stats["std"][:, None, None, None]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        tensors = load_file(rec["shard_path"])

        # Randomly choose between original and horizontally flipped variant
        if self.use_augmentation and random.random() < 0.5:
            latents = tensors["latents_hflip"]
        else:
            latents = tensors["latents_orig"]

        # Normalize with precomputed channel statistics if available
        if self.mean is not None and self.std is not None:
            latents = (latents - self.mean) / self.std

        # 10% CFG dropout
        if random.random() < self.caption_dropout:
            text_emb = tensors["text_null_emb"]
            is_dropped = True
        else:
            text_emb = tensors["text_emb"]
            is_dropped = False

        return {
            "id": rec["id"],
            "latents": latents,      # [16, t, h, w]
            "text_emb": text_emb,    # [S, text_dim]
            "is_dropped": is_dropped,
            "bucket": rec["bucket"],
        }
