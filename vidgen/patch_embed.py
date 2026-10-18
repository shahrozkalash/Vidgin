"""
Vidgen 3D Patch Embedding & Checkpoint Migration Module (Batch 1)
Supports:
1. Standard Text-to-Video patch embedding (C channels).
2. Shared Conditioning patch embedding (2C + 1 channels: noisy latents C + mask 1 + cond latents C).
3. Zero-initialization of new channel weights for exact zero-drift initialization.
4. Checkpoint migration (strict=False) guaranteeing identical outputs to text-to-video checkpoints.
"""

import math
from typing import List, Dict, Tuple, Optional, Any


class PatchEmbed3D:
    """
    3D Patch Embedding layer for Video Latent Diffusion.
    Flattens 3D spacetime patches (p_t, p_h, p_w) and projects to embed_dim.
    """

    def __init__(
        self,
        in_channels: int = 16,
        embed_dim: int = 256,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        bias: bool = True
    ):
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]

        # Weights: [embed_dim, patch_dim]
        # Initialized with scaled Xavier uniform for testing
        scale = 1.0 / math.sqrt(self.patch_dim)
        self.weight: List[List[float]] = []
        for i in range(embed_dim):
            row = [math.sin(i * 0.31 + j * 0.17) * scale for j in range(self.patch_dim)]
            self.weight.append(row)

        if bias:
            self.bias: Optional[List[float]] = [0.0] * embed_dim
        else:
            self.bias = None

    def forward_patch_tokens(self, patch_tokens: List[List[float]]) -> List[List[float]]:
        """
        Forward pass on flattened patch tokens.
        patch_tokens: [num_patches, patch_dim]
        Returns: [num_patches, embed_dim]
        """
        out: List[List[float]] = []
        for token in patch_tokens:
            assert len(token) == self.patch_dim, (
                f"Token dimension {len(token)} does not match patch_dim {self.patch_dim}"
            )
            projected = []
            for d in range(self.embed_dim):
                val = sum(token[j] * self.weight[d][j] for j in range(self.patch_dim))
                if self.bias is not None:
                    val += self.bias[d]
                projected.append(val)
            out.append(projected)
        return out

    def state_dict(self) -> Dict[str, Any]:
        """Returns layer state dictionary."""
        d = {"weight": [list(row) for row in self.weight]}
        if self.bias is not None:
            d["bias"] = list(self.bias)
        return d

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True):
        """
        Loads state dictionary. If strict=False and state has fewer input channels,
        automatically zero-pads the new channel weights to guarantee exact equivalence.
        """
        incoming_weight = state["weight"]
        inc_embed_dim = len(incoming_weight)
        assert inc_embed_dim == self.embed_dim, f"Embed dim mismatch: {inc_embed_dim} vs {self.embed_dim}"
        inc_patch_dim = len(incoming_weight[0])

        if inc_patch_dim == self.patch_dim:
            self.weight = [list(row) for row in incoming_weight]
        elif not strict and inc_patch_dim < self.patch_dim:
            # Automatic zero-padding of new channels
            self.weight = []
            for d in range(self.embed_dim):
                row = list(incoming_weight[d]) + [0.0] * (self.patch_dim - inc_patch_dim)
                self.weight.append(row)
        else:
            raise ValueError(
                f"Cannot load weights of patch_dim {inc_patch_dim} into target {self.patch_dim} (strict={strict})"
            )

        if "bias" in state and self.bias is not None:
            self.bias = list(state["bias"])


def migrate_t2v_checkpoint_to_conditioning(
    t2v_checkpoint: Dict[str, Any],
    original_in_channels: int = 16,
    new_in_channels: int = 33,
    patch_size: Tuple[int, int, int] = (1, 2, 2)
) -> Dict[str, Any]:
    """
    Migrates an existing Text-to-Video checkpoint (16 channels) to the 33-channel
    Shared Conditioning model by zero-initializing the new mask and conditioning channels.

    Guarantees bit-for-bit mathematical equivalence at initialization.
    """
    migrated = dict(t2v_checkpoint)
    patch_spatial_prod = patch_size[0] * patch_size[1] * patch_size[2]
    old_patch_dim = original_in_channels * patch_spatial_prod
    new_patch_dim = new_in_channels * patch_spatial_prod

    # Look for patch embedding weight key
    target_key = None
    for k in ["patch_embed.weight", "patch_embed.proj.weight", "weight"]:
        if k in t2v_checkpoint:
            target_key = k
            break

    if target_key is not None:
        old_weight = t2v_checkpoint[target_key]
        embed_dim = len(old_weight)
        new_weight = []
        for d in range(embed_dim):
            # Copy original weights for the first old_patch_dim coordinates
            row = list(old_weight[d][:old_patch_dim])
            # Append exact 0.0 for new channels (mask + cond latents)
            row.extend([0.0] * (new_patch_dim - old_patch_dim))
            new_weight.append(row)
        migrated[target_key] = new_weight

    return migrated


class VideoDiTStubModel:
    """
    Lightweight DiT model stub for CPU verification of the shared conditioning pipeline.
    Combines PatchEmbed3D with residual transformation blocks.
    """

    def __init__(
        self,
        in_channels: int = 33,  # 33 channels (2C + 1)
        embed_dim: int = 256,
        patch_size: Tuple[int, int, int] = (1, 2, 2)
    ):
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

    def forward(self, concatenated_patch_tokens: List[List[float]]) -> List[List[float]]:
        """Forward pass through patch embedding and stub DiT projection."""
        tokens = self.patch_embed.forward_patch_tokens(concatenated_patch_tokens)
        return tokens

    def state_dict(self) -> Dict[str, Any]:
        return {"patch_embed.weight": self.patch_embed.weight, "patch_embed.bias": self.patch_embed.bias}

    def load_state_dict(self, state: Dict[str, Any], strict: bool = False):
        if "patch_embed.weight" in state:
            sub_state = {"weight": state["patch_embed.weight"]}
            if "patch_embed.bias" in state and state["patch_embed.bias"] is not None:
                sub_state["bias"] = state["patch_embed.bias"]
            self.patch_embed.load_state_dict(sub_state, strict=strict)
