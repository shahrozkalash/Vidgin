"""Vidgen — latent diffusion text-to-video engine (Track A / Tiny, PyTorch).

Package layout (mirrors README.md / AUDIT.md parts):
  vidgen.config      Part 0: preset configs, frame rule, save/load roundtrip
  vidgen.seed        Part 7: deterministic seeding
  vidgen.logging     Part 7: structured JSONL logging
  vidgen.data        Part 1+2: Moving MNIST synthesis, quality filters, buckets, manifest
  vidgen.vae         Part 3: causal 3D VAE (4x temporal, 8x spatial, 16 channels)
  vidgen.text        Part 5: text conditioning (lightweight Tiny encoder; umT5 swap path)
  vidgen.dit         Part 4+6: 3D patchify, partitioned 3D RoPE, DiT backbone
  vidgen.flow        Part 7: rectified flow training objective
  vidgen.sampler     Part 8: Euler / Midpoint / RK4 ODE samplers with CFG
  vidgen.checkpoint  Part 7: checkpoint save/resume with RNG state
  vidgen.train       Part 7: Tiny training entry point
  vidgen.sample      Part 8: sampling entry point
"""

__version__ = "0.1.0-trackA"

from vidgen.config import VidgenConfig, get_preset, PRESETS  # noqa: F401
from vidgen.seed import seed_everything  # noqa: F401
