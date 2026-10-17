"""Configuration system using dataclasses and YAML."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Tuple
from pathlib import Path
import yaml
import torch


@dataclass
class SystemConfig:
    device: str = "cpu"            # "cpu", "cuda", or "auto"
    dtype: str = "float32"         # "float32", "bfloat16", "float16"
    use_flash_attn: bool = False   # FlashAttention flag; falls back to SDPA
    enable_tf32: bool = False

    def resolve_device(self) -> torch.device:
        """Resolve to available torch.device."""
        if self.device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        elif self.device == "auto":
            return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return torch.device("cpu")

    def resolve_dtype(self) -> torch.dtype:
        """Resolve torch dtype with safe CPU fallback (CPU requires float32 for most ops)."""
        device = self.resolve_device()
        if device.type == "cpu":
            return torch.float32

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.float32)


@dataclass
class VaeConfig:
    in_channels: int = 3
    latent_channels: int = 16
    base_channels: int = 32
    channel_multipliers: List[int] = field(default_factory=lambda: [1, 2, 4])
    temporal_downsamples: int = 2  # 2 downsamplings -> 4x temporal downsampling (17 -> 5)
    spatial_downsamples: int = 3   # 3 downsamplings -> 8x spatial downsampling (64 -> 8)


@dataclass
class PatchifyConfig:
    patch_size: Tuple[int, int, int] = (1, 2, 2)  # [pt, ph, pw]
    in_channels: int = 16


@dataclass
class ModelConfig:
    num_layers: int = 8
    hidden_dim: int = 384
    num_heads: int = 6
    ffn_dim: int = 1536
    text_dim: int = 384
    rope_theta: float = 10000.0
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        assert self.hidden_dim % self.num_heads == 0, f"{self.hidden_dim} not divisible by {self.num_heads}"
        return self.hidden_dim // self.num_heads


@dataclass
class TrainingConfig:
    batch_size: int = 2
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    max_grad_norm: float = 1.0
    total_steps: int = 1000
    warmup_steps: int = 50
    save_every: int = 200
    eval_every: int = 50
    ema_decay: float = 0.999
    output_dir: str = "./checkpoints/tiny"
    tracker: str = "tensorboard"  # "tensorboard" or "wandb"


@dataclass
class DataConfig:
    frames: int = 17       # T = 4k + 1
    height: int = 64
    width: int = 64
    fps: int = 8
    text_seq_len: int = 16

    def __post_init__(self):
        # Validate Wan T = 4k + 1 rule
        assert (self.frames - 1) % 4 == 0, f"Frames ({self.frames}) must satisfy T = 4k + 1"


@dataclass
class SamplerConfig:
    method: str = "euler"  # euler, midpoint, rk4
    num_steps: int = 20
    guidance_scale: float = 5.0


@dataclass
class VidGenConfig:
    track: str = "A"
    name: str = "tiny"
    seed: int = 42
    system: SystemConfig = field(default_factory=SystemConfig)
    vae: VaeConfig = field(default_factory=VaeConfig)
    patchify: PatchifyConfig = field(default_factory=PatchifyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)

    def compute_latent_shape(self) -> Tuple[int, int, int]:
        """Compute (t, h, w) of latents from pixel (T, H, W)."""
        t_compress = 2 ** self.vae.temporal_downsamples  # 4x
        s_compress = 2 ** self.vae.spatial_downsamples   # 8x
        t = (self.data.frames - 1) // t_compress + 1
        h = self.data.height // s_compress
        w = self.data.width // s_compress
        return t, h, w

    def compute_token_count(self) -> int:
        """Compute token count N = t_p * h_p * w_p."""
        t, h, w = self.compute_latent_shape()
        pt, ph, pw = self.patchify.patch_size
        assert t % pt == 0 and h % ph == 0 and w % pw == 0, (
            f"Latent {(t, h, w)} must be divisible by patch size {self.patch_size}"
        )
        return (t // pt) * (h // ph) * (w // pw)


def load_config(path: str | Path) -> VidGenConfig:
    """Load configuration from a YAML file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return VidGenConfig(
        track=data.get("track", "A"),
        name=data.get("name", path.stem),
        seed=data.get("seed", 42),
        system=SystemConfig(**data.get("system", {})),
        vae=VaeConfig(**data.get("vae", {})),
        patchify=PatchifyConfig(
            patch_size=tuple(data.get("patchify", {}).get("patch_size", (1, 2, 2))),
            in_channels=data.get("patchify", {}).get("in_channels", 16)
        ),
        model=ModelConfig(**data.get("model", {})),
        training=TrainingConfig(**data.get("training", {})),
        data=DataConfig(**data.get("data", {})),
        sampler=SamplerConfig(**data.get("sampler", {})),
    )


def save_config(config: VidGenConfig, path: str | Path) -> None:
    """Save configuration to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = asdict(config)
    raw["patchify"]["patch_size"] = list(raw["patchify"]["patch_size"])
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
