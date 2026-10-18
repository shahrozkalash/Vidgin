"""Rectified Flow / Flow Matching Loss Functions with Logit-Normal Shifted Scheduling.

Formulation:
- Linear Rectified Flow interpolation:
    x_t = (1 - t) * x_0 + t * eps,  where eps ~ N(0, I)
- Target velocity:
    v = eps - x_0
- Loss:
    L = MSE(v_pred, v) = || v_theta(x_t, t * 1000, text) - v ||^2
- Timestep Sampling:
    u ~ Normal(0, 1) -> t_raw = sigmoid(u)
    Shifted timestep: t = s * t_raw / (1 + (s - 1) * t_raw)
    Where s >= 1.0 (s=5.0 for high resolution / Wan video standard).
- Timestep Binning:
    Losses binned into 10 intervals [0.0, 0.1), ..., [0.9, 1.0] for diagnostic tracking.
"""

from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_logit_normal_timesteps(
    batch_size: int,
    shift: float = 5.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample continuous timesteps t in [0, 1] using shifted logit-normal distribution.

    Args:
        batch_size: Number of timesteps to sample.
        shift: Time shift parameter s (s=1.0 is standard logit-normal; s=5.0 allocates more
               noise budget to resolve fine spacetime structure in high-res video).
        device: Target torch device.
        dtype: Floating point type.
        generator: Optional PyTorch random generator for reproducibility.

    Returns:
        Tensor of shape [batch_size] with values in (0, 1).
    """
    # Sample from standard normal
    u = torch.randn(batch_size, device=device, dtype=dtype, generator=generator)
    # Map to (0, 1) via sigmoid
    t_raw = torch.sigmoid(u)
    # Apply time shift: t = s * t_raw / (1 + (s - 1) * t_raw)
    if abs(shift - 1.0) > 1e-5:
        t = shift * t_raw / (1.0 + (shift - 1.0) * t_raw)
    else:
        t = t_raw
    # Clamp slightly away from exact 0 and 1 for numerical stability
    return torch.clamp(t, 1e-5, 1.0 - 1e-5)


class RectifiedFlowLoss(nn.Module):
    """Flow Matching Loss with optional masking, text drop, and timestep binning."""

    def __init__(
        self,
        shift: float = 5.0,
        text_drop_prob: float = 0.1,
        num_bins: int = 10,
    ):
        super().__init__()
        self.shift = float(shift)
        self.text_drop_prob = float(text_drop_prob)
        self.num_bins = num_bins

        # Running timestep-binned statistics
        self.register_buffer("bin_loss_sums", torch.zeros(num_bins, dtype=torch.float64))
        self.register_buffer("bin_counts", torch.zeros(num_bins, dtype=torch.int64))

    def reset_binned_stats(self) -> None:
        """Reset running bin loss statistics."""
        self.bin_loss_sums.zero_()
        self.bin_counts.zero_()

    def get_binned_losses(self) -> Dict[str, float]:
        """Compute average loss per timestep bin."""
        results: Dict[str, float] = {}
        counts = self.bin_counts.cpu().numpy()
        sums = self.bin_loss_sums.cpu().numpy()
        bin_width = 1.0 / self.num_bins

        for i in range(self.num_bins):
            low = i * bin_width
            high = (i + 1) * bin_width
            bin_name = f"loss_t_[{low:.1f}-{high:.1f}]"
            if counts[i] > 0:
                results[bin_name] = float(sums[i] / counts[i])
            else:
                results[bin_name] = 0.0
        return results

    def forward(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        null_embedding: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute Rectified Flow matching loss for a training batch.

        Args:
            model: WanDiT backbone model returning velocity prediction [B, L, D] or [B, C, T, H, W].
            x_0: Clean latent video tensor [B, C, T, H, W] or patchified tokens.
            text_embeddings: Condition text embeddings [B, L_txt, D_txt].
            text_mask: Attention mask for text padding tokens [B, L_txt].
            null_embedding: Precomputed empty string embedding [1, L_txt, D_txt] for CFG drop.
            token_mask: Optional mask for spatial/temporal padding tokens [B, L].
            generator: Optional RNG generator.

        Returns:
            Tuple of (loss_tensor, metrics_dict).
        """
        B = x_0.shape[0]
        device = x_0.device
        dtype = x_0.dtype

        # 1. Sample continuous shifted timesteps t in (0, 1)
        t = sample_logit_normal_timesteps(
            batch_size=B,
            shift=self.shift,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # 2. Sample standard Gaussian noise eps ~ N(0, I)
        eps = torch.randn_like(x_0)

        # 3. Construct noisy latent x_t = (1 - t) * x_0 + t * eps
        # Expand t to broadcast across all latent dims [B, 1, 1, 1, 1]
        shape_expansion = [B] + [1] * (x_0.ndim - 1)
        t_expanded = t.view(*shape_expansion)
        x_t = (1.0 - t_expanded) * x_0 + t_expanded * eps

        # 4. Target velocity v = eps - x_0
        v_target = eps - x_0

        # 5. Classifier-free guidance training drop (10% by default)
        effective_text = text_embeddings
        effective_mask = text_mask
        if self.training and self.text_drop_prob > 0.0 and null_embedding is not None:
            drop_mask = torch.rand(B, device=device, generator=generator) < self.text_drop_prob
            if drop_mask.any():
                # Replace dropped elements with broadcast null_embedding
                null_expanded = null_embedding.expand(B, -1, -1)
                effective_text = torch.where(
                    drop_mask.view(B, 1, 1),
                    null_expanded,
                    text_embeddings,
                )
                if text_mask is not None:
                    # Unconditional null embedding has valid text length 1 or all true
                    effective_mask = torch.where(
                        drop_mask.view(B, 1),
                        torch.ones_like(text_mask),
                        text_mask,
                    )

        # 6. Model timestep input is t * 1000
        t_input = t * 1000.0

        # 7. Model forward pass -> predicts velocity v_pred
        v_pred = model(
            x_t,
            t_input,
            context=effective_text,
            context_mask=effective_mask,
        )

        # 8. Compute Mean Squared Error with token masking
        # Flatten tensors over spatial/temporal dimensions
        err = (v_pred - v_target) ** 2
        if token_mask is not None:
            # Broadcast token_mask if needed
            while token_mask.ndim < err.ndim:
                token_mask = token_mask.unsqueeze(-1)
            err = err * token_mask
            loss = err.sum() / (token_mask.sum() * err.shape[-1] + 1e-8)
        else:
            loss = err.mean()

        # 9. Update timestep-binned statistics (no grad)
        with torch.no_grad():
            per_sample_loss = err.view(B, -1).mean(dim=-1)
            bin_idx = torch.clamp(
                (t * self.num_bins).long(), 0, self.num_bins - 1
            )
            for b in range(B):
                idx = bin_idx[b].item()
                self.bin_loss_sums[idx] += per_sample_loss[b].item()
                self.bin_counts[idx] += 1

        metrics = {
            "loss": float(loss.item()),
            "mean_t": float(t.mean().item()),
            "v_target_norm": float(v_target.norm(dim=-1).mean().item()),
            "v_pred_norm": float(v_pred.norm(dim=-1).mean().item()),
        }

        return loss, metrics
