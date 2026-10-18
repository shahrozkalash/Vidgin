"""Profiling and Model FLOPs Utilization (MFU) Tracking for Video Diffusion Transformers.

Key Metrics:
- Tokens Per Second:
    T_s = (global_batch_size * sequence_length) / step_duration_sec
- Model FLOPs Utilization (MFU):
    MFU = (T_s * FLOPs_per_token) / (Theoretical_Peak_FLOPs_per_GPU * num_gpus)
- Stall Detector:
    Tracks data fetching vs compute vs communication (AllReduce/AllToAll) bottlenecks.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn as nn


@dataclass
class HardwareSpecs:
    """Peak theoretical dense BF16/FP16 TFLOPS (without structured sparsity)."""
    name: str
    peak_tflops: float  # TeraFLOPs per second (1e12)
    vram_gb: float


KNOWN_HARDWARE: Dict[str, HardwareSpecs] = {
    "h100_sxm": HardwareSpecs(name="NVIDIA H100 SXM5", peak_tflops=989.0, vram_gb=80.0),
    "a100_80gb": HardwareSpecs(name="NVIDIA A100 80GB SXM", peak_tflops=312.0, vram_gb=80.0),
    "a100_40gb": HardwareSpecs(name="NVIDIA A100 40GB PCIe", peak_tflops=312.0, vram_gb=40.0),
    "l40s": HardwareSpecs(name="NVIDIA L40S", peak_tflops=366.0, vram_gb=48.0),
    "rtx4090": HardwareSpecs(name="NVIDIA RTX 4090", peak_tflops=165.0, vram_gb=24.0),
    "v100": HardwareSpecs(name="NVIDIA V100 32GB", peak_tflops=125.0, vram_gb=32.0),
}


def calculate_flops_per_token(
    num_params: int,
    num_layers: int,
    hidden_dim: int,
    seq_len: int,
    text_len: int = 512,
) -> float:
    """Calculate theoretical FLOPs required per token for forward + backward pass.

    Formulation:
        - Dense feed-forward, projections, QKV, output: 6 * num_params
        - Self-attention matrix multiplies (Q*K^T and Attn*V): 12 * num_layers * hidden_dim * seq_len
        - Cross-attention matrix multiplies: 12 * num_layers * hidden_dim * text_len

    Total per token:
        FLOPs/token = 6 * P + 12 * L * D * (seq_len + text_len) / seq_len
    """
    dense_flops = 6.0 * float(num_params)
    attn_flops = (
        12.0 * float(num_layers) * float(hidden_dim) * float(seq_len + text_len) / max(1.0, float(seq_len))
    )
    return dense_flops + attn_flops


class TrainingProfiler:
    """Real-time profiler tracking throughput, MFU, and pipeline stalls."""

    def __init__(
        self,
        model: nn.Module,
        num_gpus: int = 1,
        hardware_key: str = "h100_sxm",
        text_len: int = 512,
    ):
        self.num_gpus = max(1, num_gpus)
        self.hardware = KNOWN_HARDWARE.get(hardware_key, KNOWN_HARDWARE["h100_sxm"])
        self.text_len = text_len

        # Model parameter counting
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Inspect layer count and hidden dim if available
        self.num_layers = getattr(model, "num_layers", 12)
        self.hidden_dim = getattr(model, "dim", 768)

        # Timing states
        self._step_start_time: Optional[float] = None
        self._data_start_time: Optional[float] = None
        self._last_data_duration: float = 0.0

        # Rolling history
        self.step_times: List[float] = []
        self.data_times: List[float] = []
        self.tokens_per_sec_history: List[float] = []
        self.mfu_history: List[float] = []

    def start_data_fetch(self) -> None:
        """Mark start of dataloader batch retrieval."""
        self._data_start_time = time.perf_counter()

    def start_step(self) -> None:
        """Mark start of GPU compute step."""
        if self._data_start_time is not None:
            self._last_data_duration = time.perf_counter() - self._data_start_time
            self._data_start_time = None
        self._step_start_time = time.perf_counter()

    def end_step(
        self,
        global_batch_size: int,
        seq_len: int,
    ) -> Dict[str, float]:
        """Mark completion of optimizer step, compute MFU and throughput."""
        if self._step_start_time is None:
            return {}

        step_duration = time.perf_counter() - self._step_start_time
        self._step_start_time = None

        total_tokens = global_batch_size * seq_len
        tokens_per_sec = total_tokens / max(1e-6, step_duration)

        flops_per_token = calculate_flops_per_token(
            num_params=self.num_params,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            seq_len=seq_len,
            text_len=self.text_len,
        )

        total_flops_per_sec = tokens_per_sec * flops_per_token
        # Theoretical peak for cluster
        cluster_peak_flops = self.hardware.peak_tflops * 1e12 * self.num_gpus
        mfu = (total_flops_per_sec / max(1.0, cluster_peak_flops)) * 100.0  # Percentage

        # Append to history (bounded at 100 entries)
        self.step_times.append(step_duration)
        self.data_times.append(self._last_data_duration)
        self.tokens_per_sec_history.append(tokens_per_sec)
        self.mfu_history.append(mfu)
        if len(self.step_times) > 100:
            self.step_times.pop(0)
            self.data_times.pop(0)
            self.tokens_per_sec_history.pop(0)
            self.mfu_history.pop(0)

        # Detect stalls
        data_ratio = self._last_data_duration / max(1e-6, step_duration + self._last_data_duration)
        data_stall = data_ratio > 0.15

        return {
            "step_time_sec": step_duration,
            "data_fetch_time_sec": self._last_data_duration,
            "tokens_per_sec": tokens_per_sec,
            "mfu_percent": mfu,
            "data_stall_detected": float(data_stall),
            "flops_per_token": flops_per_token,
            "total_tokens_in_step": float(total_tokens),
        }
