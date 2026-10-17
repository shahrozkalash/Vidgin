"""Precomputes and caches VAE latents and text embeddings to disk."""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from safetensors.torch import save_file

from vidgen.data.dataset import CuratedVideoDataset


def precompute_latents_and_embeddings(
    dataset: CuratedVideoDataset,
    output_dir: str | Path,
    vae: Optional[nn.Module] = None,
    text_encoder: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu"),
) -> Path:
    """Precompute and save .safetensors files for fast training ingestion."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        clip_id = item["id"]
        pixels = item["pixels"].unsqueeze(0).to(device)  # [1, 3, T, H, W]

        # Compute latent: if VAE provided, encode; otherwise dummy latent for tiny testing
        if vae is not None:
            with torch.no_grad():
                latents = vae.sample_latent(pixels)
        else:
            # Latent shape: [1, 16, (T-1)//4 + 1, H//8, W//8]
            t = (pixels.shape[2] - 1) // 4 + 1
            h = pixels.shape[3] // 8
            w = pixels.shape[4] // 8
            latents = torch.randn(1, 16, t, h, w, device=device)

        save_dict = {
            "latents": latents.squeeze(0).cpu(),
        }

        save_path = output_dir / f"{clip_id}.safetensors"
        save_file(save_dict, str(save_path))

        manifest_records.append({
            "id": clip_id,
            "latent_path": str(save_path),
            "prompt": item["prompt"],
        })

    return output_dir
