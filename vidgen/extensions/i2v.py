"""Image-to-Video (I2V) Extension for Wan Diffusion Transformer (Part 11).

Goal:
Convert a trained Text-to-Video (T2V) DiT into an Image-to-Video (I2V) generator:
1. Input channel expansion:
   - 16 channels: Noisy video latents (x_t)
   - 16 channels: Conditioning frame latents (x_cond, replicated across time or frame-isolated)
   - 1 channel: Binary conditioning mask (1 for conditioned frame, 0 for to-be-generated frames)
   Total input channels: 33 (16 + 16 + 1).
2. Zero-initialization:
   - The patch embedding layer is expanded from 16 to 33 channels.
   - The first 16 channel weights are copied directly from the trained T2V checkpoint.
   - The additional 17 channel weights are zero-initialized (zero-init).
   - Property: At step 0, the model output is mathematically identical to the base T2V model!
3. Training collator:
   - Random first-frame conditioning with conditioning dropout (for Classifier-Free Guidance over the conditioning image).
"""

from __future__ import annotations
import copy
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from vidgen.models.dit import WanDiT
from vidgen.models.patchify import SpacetimePatchEmbed


class WanImageToVideoDiT(nn.Module):
    """Wan Diffusion Transformer adapted for Image-to-Video conditioning.
    
    Accepts 33-channel input latents:
      - x[:, 0:16]: noisy latent video
      - x[:, 16:32]: conditioning image latent (broadcast or placed at frame 0)
      - x[:, 32:33]: binary conditioning mask
    """

    def __init__(
        self,
        base_dit: WanDiT,
        zero_init_new_channels: bool = True,
    ):
        super().__init__()
        self.base_dit = base_dit
        self.original_in_channels = base_dit.in_channels
        self.i2v_in_channels = self.original_in_channels + self.original_in_channels + 1  # 16 + 16 + 1 = 33

        # Upgrade patch_embed to 33 channels
        self._upgrade_patch_embed(zero_init_new_channels)

    def _upgrade_patch_embed(self, zero_init: bool = True):
        """Replaces patch_embed with a 33-channel version and zero-inits new channel weights."""
        old_embed = self.base_dit.patch_embed
        hidden_dim = self.base_dit.hidden_dim
        patch_size = self.base_dit.patch_size

        new_embed = SpacetimePatchEmbed(
            in_channels=self.i2v_in_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
        )

        with torch.no_grad():
            # Copy over pretrained weights for original 16 channels
            new_embed.proj.weight[:, :self.original_in_channels, :, :, :].copy_(
                old_embed.proj.weight
            )
            # Copy bias if present
            if old_embed.proj.bias is not None and new_embed.proj.bias is not None:
                new_embed.proj.bias.copy_(old_embed.proj.bias)

            # Zero-initialize the additional 17 channels
            if zero_init:
                new_embed.proj.weight[:, self.original_in_channels:, :, :, :].zero_()

        self.base_dit.patch_embed = new_embed
        self.base_dit.in_channels = self.i2v_in_channels

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        cond_latents: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Either 33-channel concatenated tensor [B, 33, t, h, w],
               OR 16-channel noisy latents [B, 16, t, h, w] when cond_latents is provided.
            timesteps: Flow timesteps [B]
            context: Text embeddings [B, L, text_dim]
            cond_latents: Optional conditioning frame latents [B, 16, t, h, w] or [B, 16, 1, h, w]
            cond_mask: Optional binary mask [B, 1, t, h, w]
            text_mask: Optional text padding mask [B, L]
        """
        if x.shape[1] == self.original_in_channels:
            assert cond_latents is not None, "cond_latents must be provided when x has 16 channels"
            B, C, T, H, W = x.shape

            # Ensure cond_latents has shape [B, 16, T, H, W]
            if cond_latents.shape[2] == 1 and T > 1:
                # Repeat first frame across temporal dimension or keep only frame 0
                cond_expanded = torch.zeros_like(x)
                cond_expanded[:, :, 0:1, :, :] = cond_latents
            else:
                cond_expanded = cond_latents

            if cond_mask is None:
                cond_mask = torch.zeros(B, 1, T, H, W, device=x.device, dtype=x.dtype)
                cond_mask[:, :, 0:1, :, :] = 1.0  # Frame 0 is conditioned

            # Concatenate along channel dimension -> [B, 33, T, H, W]
            x_in = torch.cat([x, cond_expanded, cond_mask], dim=1)
        else:
            x_in = x

        return self.base_dit(x_in, timesteps, context, text_mask=text_mask)


def convert_t2v_to_i2v(
    t2v_model: WanDiT,
    zero_init: bool = True,
) -> WanImageToVideoDiT:
    """Convenience factory to convert any WanDiT model into an Image-to-Video model."""
    return WanImageToVideoDiT(base_dit=t2v_model, zero_init_new_channels=zero_init)


class I2VConditioningCollator:
    """Prepares training batches with random first-frame image conditioning and CFG dropout."""

    def __init__(
        self,
        cond_prob: float = 0.9,
        cond_dropout_prob: float = 0.1,
    ):
        self.cond_prob = cond_prob
        self.cond_dropout_prob = cond_dropout_prob

    def prepare_batch(
        self,
        x_0: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates conditioning latents and binary mask for a clean video x_0 [B, 16, T, H, W].

        Returns:
            cond_latents: [B, 16, T, H, W] with frame 0 clean latents (or zeros if dropped).
            cond_mask: [B, 1, T, H, W] binary mask (1.0 for conditioned frame, 0.0 for unconditioned).
        """
        B, C, T, H, W = x_0.shape
        cond_latents = torch.zeros_like(x_0)
        cond_mask = torch.zeros(B, 1, T, H, W, device=device, dtype=x_0.dtype)

        for b in range(B):
            # Check if conditioning is applied
            if torch.rand(1).item() < self.cond_prob:
                # Check for conditioning dropout (for CFG)
                if torch.rand(1).item() >= self.cond_dropout_prob:
                    # Condition on frame 0
                    cond_latents[b, :, 0:1, :, :] = x_0[b, :, 0:1, :, :]
                    cond_mask[b, :, 0:1, :, :] = 1.0

        return cond_latents, cond_mask
