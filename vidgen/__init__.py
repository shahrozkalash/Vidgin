"""VidGen: Wan-Style Latent Diffusion Text-to-Video Engine.

Pipeline:
    Causal 3D VAE -> 3D Spacetime Patchify -> DiT with 3D RoPE & Cross-Attention
    -> Rectified Flow Training -> ODE Sampler with CFG -> Causal 3D VAE Decode
"""

__version__ = "0.1.0"
