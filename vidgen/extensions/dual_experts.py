"""Wan 2.2-Style Two-Expert Architecture & Inference Pipeline (Part 11).

Goal:
Implement Wan 2.2-style temporal schedule expert partitioning:
1. Two full DiTs with identical architecture:
   - High-Noise Expert: Specializes in global layout, semantics, and high-frequency noise removal
     for timesteps t in [t_boundary, 1.0] (default boundary ~ 0.875 of the schedule).
   - Low-Noise Expert: Specializes in fine details, textures, crisp edges, and motion smoothness
     for timesteps t in [0.0, t_boundary).
2. Initialization & Fine-Tuning:
   - Both experts are initialized from one shared base checkpoint.
   - Each expert is fine-tuned strictly on its designated timestep interval using loss masking.
3. Memory Offloading during Sampling:
   - To keep VRAM footprint equal to a single DiT, exactly ONE expert is resident on the GPU at a time.
   - At t = 1.0 down to t = t_boundary: High-Noise expert is on GPU.
   - At t = t_boundary: High-Noise expert is offloaded to CPU RAM; Low-Noise expert is loaded to GPU.
   - At t = t_boundary down to t = 0.0: Low-Noise expert performs generation.
4. Important Distinction:
   - This is TEMPORAL/SCHEDULE partitioning, NOT token-level sparse Mixture-of-Experts (MoE) routing.
"""

from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from vidgen.models.dit import WanDiT


class Wan22DualExperts(nn.Module):
    """Container for Wan 2.2 High-Noise and Low-Noise Experts with boundary dispatch."""

    def __init__(
        self,
        high_noise_expert: WanDiT,
        low_noise_expert: WanDiT,
        boundary_timestep: float = 0.875,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.high_noise_expert = high_noise_expert
        self.low_noise_expert = low_noise_expert
        self.boundary_timestep = boundary_timestep
        self.target_device = device
        self.current_resident = "none"

    def set_boundary(self, boundary: float):
        """Update boundary threshold (e.g. 0.875)."""
        assert 0.0 <= boundary <= 1.0, f"Boundary must be in [0, 1], got {boundary}"
        self.boundary_timestep = boundary

    def load_expert_to_device(self, expert_name: str):
        """Ensures the requested expert is on target device and the other is offloaded to CPU."""
        if expert_name == self.current_resident:
            return

        if expert_name == "high":
            # Offload low-noise expert if on device
            self.low_noise_expert.to("cpu")
            # Load high-noise expert
            self.high_noise_expert.to(self.target_device)
            self.current_resident = "high"
        elif expert_name == "low":
            # Offload high-noise expert if on device
            self.high_noise_expert.to("cpu")
            # Load low-noise expert
            self.low_noise_expert.to(self.target_device)
            self.current_resident = "low"
        else:
            raise ValueError(f"Unknown expert: {expert_name}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_expert_for_timestep(self, t: float) -> Tuple[WanDiT, str]:
        """Returns the appropriate expert model and its name for timestep t."""
        if t >= self.boundary_timestep:
            return self.high_noise_expert, "high"
        return self.low_noise_expert, "low"

    def predict_velocity(
        self,
        x: torch.Tensor,
        timestep: float,
        context: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        auto_offload: bool = True,
    ) -> torch.Tensor:
        """Inference forward pass with automatic expert dispatch and memory offload."""
        expert, name = self.get_expert_for_timestep(timestep)
        if auto_offload:
            self.load_expert_to_device(name)

        t_tensor = torch.tensor([timestep], dtype=x.dtype, device=x.device).expand(x.shape[0])
        return expert(x, t_tensor, context, text_mask=text_mask)


def create_dual_experts_from_base(
    base_checkpoint_model: WanDiT,
    boundary_timestep: float = 0.875,
    device: torch.device = torch.device("cpu"),
) -> Wan22DualExperts:
    """Clones a single base DiT model into two identical high-noise and low-noise experts."""
    import copy
    high_expert = copy.deepcopy(base_checkpoint_model)
    low_expert = copy.deepcopy(base_checkpoint_model)
    return Wan22DualExperts(
        high_noise_expert=high_expert,
        low_noise_expert=low_expert,
        boundary_timestep=boundary_timestep,
        device=device,
    )


def compute_dual_expert_training_loss(
    model: WanDiT,
    expert_type: str,  # 'high' or 'low'
    pred_v: torch.Tensor,
    target_v: torch.Tensor,
    timesteps: torch.Tensor,
    boundary: float = 0.875,
) -> torch.Tensor:
    """Calculates masked flow matching loss so each expert only learns in its schedule domain."""
    base_loss = (pred_v - target_v).pow(2).mean(dim=[1, 2, 3, 4])  # [B]

    if expert_type == "high":
        mask = (timesteps >= boundary).float()
    elif expert_type == "low":
        mask = (timesteps < boundary).float()
    else:
        raise ValueError(f"Unknown expert type: {expert_type}")

    masked_loss = (base_loss * mask).sum() / (mask.sum() + 1e-6)
    return masked_loss
