"""Low-Rank Adaptation (LoRA) for Wan Video Diffusion Transformer (Part 11).

Goal:
Parameter-efficient fine-tuning (PEFT) of an open Wan 1.3B or 14B DiT checkpoint.
- Rank r in [16, 64] on attention projections (q, k, v, out) and feed-forward networks (fc1, fc2).
- Learning rate ~ 1e-4 on a few hundred to a few thousand clips.
- Fast merge into base weights at inference for zero-overhead inference.
- Save and load compact adapter checkpoints (~10-50MB vs ~5-28GB).
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Low-rank adapter linear layer wrapper.
    
    Computes: y = W_0 x + (alpha / rank) * B(A(x))
    where W_0 is frozen, A is initialized with Kaiming uniform, and B is initialized to zero.
    This guarantees that at initialization, the adapter output is exactly zero (identity behavior).
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Keep original base linear weights frozen
        self.base_linear = base_linear
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        # LoRA matrices: A projects in_features -> rank, B projects rank -> out_features
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank))

        self.merged = False
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform on A, zeros on B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_linear(x)

        # Base forward
        result = self.base_linear(x)

        # LoRA branch: x @ A.T @ B.T * scaling
        lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return result + lora_out

    def merge_weights(self):
        """In-place fold LoRA delta into base linear weight."""
        if self.merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base_linear.weight.data += delta.to(self.base_linear.weight.device, dtype=self.base_linear.weight.dtype)
        self.merged = True

    def unmerge_weights(self):
        """Restore original base linear weight by subtracting LoRA delta."""
        if not self.merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base_linear.weight.data -= delta.to(self.base_linear.weight.device, dtype=self.base_linear.weight.dtype)
        self.merged = False


def inject_lora_into_dit(
    model: nn.Module,
    rank: int = 32,
    alpha: float = 32.0,
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"),
    dropout: float = 0.0,
) -> Dict[str, LoRALinear]:
    """Injects LoRALinear adapters into targeted modules of a Wan DiT model.

    Freezes all base model parameters and enables gradients only for LoRA parameters.

    Args:
        model: WanDiT instance
        rank: Low-rank dimension (typically 16 to 64)
        alpha: Scaling factor (typically equal to rank or 2*rank)
        target_modules: Names of linear layers to adapt
        dropout: LoRA dropout probability

    Returns:
        adapters: Dictionary of path -> LoRALinear instances
    """
    # 1. Freeze entire base model
    for p in model.parameters():
        p.requires_grad = False

    adapters: Dict[str, LoRALinear] = {}

    # Recursive replacement helper
    def _replace_module(parent: nn.Module, parent_name: str = ""):
        for name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{name}" if parent_name else name
            if isinstance(child, nn.Linear) and any(tgt in name for tgt in target_modules):
                # Wrap child in LoRALinear
                lora_layer = LoRALinear(
                    base_linear=child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(parent, name, lora_layer)
                adapters[full_name] = lora_layer
            else:
                _replace_module(child, full_name)

    _replace_module(model)
    return adapters


def merge_lora_weights(model: nn.Module) -> int:
    """Merges all LoRA weights in the model into their base linear weights.
    
    Returns the number of merged layers.
    """
    count = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge_weights()
            count += 1
    return count


def unmerge_lora_weights(model: nn.Module) -> int:
    """Unmerges all LoRA weights in the model from their base linear weights.
    
    Returns the number of unmerged layers.
    """
    count = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge_weights()
            count += 1
    return count


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extracts only LoRA parameters for compact serialization."""
    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state_dict[f"{name}.lora_A"] = module.lora_A.data.cpu()
            state_dict[f"{name}.lora_B"] = module.lora_B.data.cpu()
    return state_dict


def load_lora_weights(model: nn.Module, lora_state_dict: Dict[str, torch.Tensor]):
    """Loads a LoRA adapter state dict into an already-injected model."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            key_a = f"{name}.lora_A"
            key_b = f"{name}.lora_B"
            if key_a in lora_state_dict and key_b in lora_state_dict:
                module.lora_A.data.copy_(lora_state_dict[key_a].to(module.lora_A.device))
                module.lora_B.data.copy_(lora_state_dict[key_b].to(module.lora_B.device))


def count_parameters_summary(model: nn.Module) -> Dict[str, Union[int, float]]:
    """Calculates total, base, trainable, and LoRA parameter fractions."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_ratio = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "trainable_ratio_percent": round(trainable_ratio, 4),
    }
