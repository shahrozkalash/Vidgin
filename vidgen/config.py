"""Part 0: preset configs, frame rule T = 4k+1, save/load roundtrip.

Dimension tables mirror README.md (Track A). All presets define geometry; only
Tiny is sized to train on CPU. Spatial rule: H, W multiples of 16.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class DiTConfig:
    layers: int      # L transformer blocks
    dim: int         # D model width
    heads: int       # H attention heads
    ffn: int         # feed-forward hidden width
    rope_split: tuple = (22, 22, 20)  # partitioned 3D RoPE dims (t, h, w); sums to head dim


@dataclass
class VidgenConfig:
    name: str
    height: int
    width: int
    frames: int                 # T, must satisfy T = 4k + 1
    latent_channels: int = 16
    latent_t: int = 5
    latent_h: int = 8
    latent_w: int = 8
    patch_t: int = 1
    patch_h: int = 2
    patch_w: int = 2
    dit: DiTConfig = field(default_factory=lambda: DiTConfig(8, 384, 6, 1536))
    text_dim: int = 256         # Dt: text embedding width before projection to D
    text_max_len: int = 32
    text_layers: int = 0        # >0: TransformerTextEncoder depth. 0 = legacy
                                # embedding-only TinyTextEncoder — every preset
                                # and checkpoint from before the flagship text
                                # upgrade keeps the legacy path, weights load as-is.
    text_heads: int = 8         # attention heads inside the transformer encoder
    text_ffn: int = 0           # encoder MLP width (0 = 4 * text_dim)
    text_vocab_size: int = 8192  # word-hash buckets for the transformer encoder
    text_tokenizer: str = "hash"  # "hash" = legacy word-hash / fixed-table
                                # tokenization (every existing checkpoint);
                                # "caption" = CaptionTokenizer trained on the
                                # dataset's own captions (Phase 2, 2026-10-22),
                                # stored inside the checkpoint itself.
    latent_scale: float = 1.0   # fixed latent normalizer (SD-style); applied in train
    target_params_m: float = 20.0  # documented preset size (millions), sanity only
    use_depth: bool = False     # Dawood's 3D idea: depth-map conditioning (vidgen/depth.py).
                                # Default off: every pre-2026-10-20 checkpoint and code path
                                # behaves exactly as before; trainers opt in with --depth.

    # --- derived ---------------------------------------------------------
    @property
    def token_t(self) -> int:
        return self.latent_t // self.patch_t

    @property
    def token_h(self) -> int:
        return self.latent_h // self.patch_h

    @property
    def token_w(self) -> int:
        return self.latent_w // self.patch_w

    @property
    def num_tokens(self) -> int:
        return self.token_t * self.token_h * self.token_w

    @property
    def head_dim(self) -> int:
        return self.dit.dim // self.dit.heads

    def validate(self) -> "VidgenConfig":
        if (self.frames - 1) % 4 != 0 or self.frames < 1:
            raise ValueError(
                f"Frame rule violated: T={self.frames} must satisfy T = 4k+1 "
                f"(got (T-1) % 4 = {(self.frames - 1) % 4})"
            )
        if self.height % 16 != 0 or self.width % 16 != 0:
            raise ValueError(
                f"Spatial rule violated: H={self.height}, W={self.width} "
                "must be multiples of 16"
            )
        if self.latent_t != (self.frames - 1) // 4 + 1:
            raise ValueError(
                f"Latent time mismatch: T={self.frames} needs latent_t="
                f"{(self.frames - 1) // 4 + 1}, got {self.latent_t}"
            )
        if self.latent_h != self.height // 8 or self.latent_w != self.width // 8:
            raise ValueError("Latent spatial dims must be H/8, W/8 (8x compression)")
        if sum(self.dit.rope_split) != self.head_dim:
            raise ValueError(
                f"RoPE split {self.dit.rope_split} must sum to head dim {self.head_dim}"
            )
        if self.dit.dim % self.dit.heads != 0:
            raise ValueError("DiT dim must be divisible by heads")
        if self.text_layers > 0 and self.text_dim % self.text_heads != 0:
            raise ValueError("Text encoder dim must be divisible by text_heads")
        return self

    # --- (de)serialization ------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["dit"]["rope_split"] = list(d["dit"]["rope_split"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VidgenConfig":
        d = dict(d)
        dit = d.pop("dit")
        dit["rope_split"] = tuple(dit["rope_split"])
        return cls(dit=DiTConfig(**dit), **d)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "VidgenConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f)).validate()


def _tiny() -> VidgenConfig:
    return VidgenConfig(
        name="tiny", height=64, width=64, frames=17,
        latent_t=5, latent_h=8, latent_w=8,
        dit=DiTConfig(layers=8, dim=384, heads=6, ffn=1536, rope_split=(22, 22, 20)),
        target_params_m=20.0,
    )


def _small() -> VidgenConfig:
    return VidgenConfig(
        name="small", height=256, width=256, frames=33,
        latent_t=9, latent_h=32, latent_w=32,
        dit=DiTConfig(layers=16, dim=1024, heads=16, ffn=4096, rope_split=(22, 22, 20)),
        target_params_m=270.0,
    )


def _vidgen480p() -> VidgenConfig:
    return VidgenConfig(
        name="vidgen480p", height=480, width=832, frames=81,
        latent_t=21, latent_h=60, latent_w=104,
        dit=DiTConfig(layers=30, dim=1536, heads=12, ffn=8960, rope_split=(44, 42, 42)),
        target_params_m=1300.0,
    )


def _vidgen9b() -> VidgenConfig:
    return VidgenConfig(
        name="vidgen9b", height=720, width=1280, frames=81,
        latent_t=21, latent_h=90, latent_w=160,
        dit=DiTConfig(layers=40, dim=4096, heads=32, ffn=11264, rope_split=(44, 42, 42)),
        text_dim=1024, text_max_len=128,
        text_layers=16, text_heads=16, text_ffn=4096,
        target_params_m=9150.0,
    )


def _vidgen14b() -> VidgenConfig:
    return VidgenConfig(
        name="vidgen14b", height=720, width=1280, frames=81,
        latent_t=21, latent_h=90, latent_w=160,
        dit=DiTConfig(layers=40, dim=5120, heads=40, ffn=13824, rope_split=(44, 42, 42)),
        text_dim=1024, text_max_len=128,
        text_layers=24, text_heads=16, text_ffn=4096,
        target_params_m=14000.0,
    )


PRESETS = {
    "tiny": _tiny,
    "small": _small,
    "vidgen480p": _vidgen480p,
    "vidgen9b": _vidgen9b,
    "vidgen14b": _vidgen14b,
}


def get_preset(name: str) -> VidgenConfig:
    """Return a validated config for a named preset (dimension definitions only
    for non-Tiny presets; only Tiny is sized to train on CPU)."""
    try:
        cfg = PRESETS[name]()
    except KeyError:
        raise ValueError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return cfg.validate()


def resolve_device(prefer: str = "auto") -> str:
    """Device-agnostic resolution; vidgen never hardcodes a CUDA device call."""
    if prefer == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    return prefer
