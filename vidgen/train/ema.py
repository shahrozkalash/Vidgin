"""Exponential Moving Average (EMA) for model parameters."""

from __future__ import annotations
import copy
from typing import Dict, Any
import torch
import torch.nn as nn


class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model parameters updated as:
        shadow_param = decay * shadow_param + (1 - decay) * model_param
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        # Shadow copy of model parameters
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with current model parameters."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary for checkpointing."""
        return {
            "decay": self.decay,
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore shadow parameters from state dictionary."""
        self.decay = state_dict.get("decay", self.decay)
        shadow_dict = state_dict.get("shadow", {})
        for name, tensor in shadow_dict.items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor)

    def copy_to(self, model: nn.Module) -> None:
        """Copy shadow parameters into model (e.g. for evaluation/inference)."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    param.data.copy_(self.shadow[name].data)
