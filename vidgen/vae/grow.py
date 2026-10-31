"""VAE growth: transplant a trained CausalVAE into a wider geometry.

Dawood's "small brains first" strategy, applied to the VAE (the DiT
counterpart is vidgen/grow.py): the Stage 2 VAE was trained for 20,500
steps at the historical shape — latent 16 channels, base width 32,
3,650,851 params — and its reconstruction softness is a capacity
ceiling, not a data shortage (the +8,000-step Round-4 continuation
measured flat on held-out clips). Rather than retrain from noise, the
trained weights are transplanted into the grown shape — latent 32,
base 64, 14,588,483 params — and training continues on new rounds.

Unlike the DiT there is no depth/head geometry here: the VAE is a
stack of channel-widening 3D convs, and every tensor's channel axes
mean the same thing in both shapes. The transplant rule is therefore
uniform, and it is chosen to make the grown network reproduce the old
network's function as closely as this architecture allows at init
(exact everywhere except the mid ResBlocks' GroupNorm statistics —
measured and documented below; verified numerically in
tests/test_vae_grow.py):

- Every widened conv weight starts as ZEROS with the source tensor
  copied into the leading (old out-channels x old in-channels) block.
  - (old out) x (new in) = 0: old outputs never read the brand-new
    channels at init.
  - (new out) x (anything) = 0: new channels start silent (their conv
    biases are 0 too), so they contribute nothing downstream until
    training wakes them.
  With both blocks zero, the encoder trunk's old channels compute
  exactly the old activations and the new channels are exactly 0 —
  verified numerically: activations are bit-identical up to the mid
  ResBlocks (see the one exception below).
- Encoder ``mu`` / ``logvar`` (1x1 convs): old rows copied exactly.
  New rows are ZERO weight and ZERO bias, so the new latent channels
  start at mu = 0, logvar = 0 (std = 1): a clean standard-normal
  channel the KL term already knows how to price.
- Decoder ``in_conv`` (latent -> trunk): old block copied; the columns
  for the NEW latent channels are ZERO, so at init the decoder ignores
  the new latents entirely and decode([z_old, anything]) equals the
  old decode(z_old) bit-for-bit (up to float associativity).
- GroupNorm affine: old entries copied; new entries get weight 1 /
  bias 0 (the fresh-GroupNorm convention — a zero channel stays zero
  through it either way, and 1 is the scale a fresh model would use).
- The decoder's final ``out_conv`` has 3 output channels in both
  shapes; only its input axis widens: old columns copied, new columns
  zero, so it reads only the old trunk channels at init.

Where exact preservation IS impossible: the two mid ResBlocks.
FramewiseGroupNorm uses 8 groups at BOTH widths, so widening the mid
trunk from 128 to 256 channels doubles each group's size (16 -> 32
channels): an old channel is renormalized against statistics pooled
over its old group PLUS the neighbouring old group, and no per-channel
affine can undo that pooling. Everything upstream of the encoder mid
and downstream of the decoder's latent input is exact (the grown
decoder fed the old model's latents reproduces its reconstruction to
-0.08 dB); the mid renormalization is the sole deviation. Measured on
a held-out street clip (Round 5 city1 clip_000021, decode(mu)):
donor 25.98 dB / SSIM 0.878, grown at init 25.50 dB / 0.872 — the
same ballpark, a ~0.5 dB transplant tax that training (which adapts
the norm affines immediately) is expected to absorb. There is no
better weight-only fix: zeroing the mid residual branch instead
would discard the trained mid computation entirely.

The other honest caveat is the OPTIMIZER: AdamW state is not
transplanted (it is sized to the old tensors and its second-moment
estimates describe the old loss surface), so the grown checkpoint is
WEIGHTS-ONLY — train it via ``train_vae --init-weights``, which
starts a fresh optimizer, exactly like the DiT growth output. The
step counter IS preserved, so run numbering stays cumulative with
the donor's history.

The init is fully deterministic (zeros and ones; no sampling), so no
seed is involved.
"""

from __future__ import annotations

import copy

import torch

from vidgen.config import VidgenConfig, get_preset
from vidgen.vae.model import CausalVAE, infer_vae_shape

_GROW_VAE_FORMAT = "vidgen-grow-vae-v1"


def _grow_tensor(key: str, src: torch.Tensor,
                 tgt_shape: tuple) -> tuple[torch.Tensor, int]:
    """One tensor under the uniform widening rule (see module docs).

    Base is zeros, except GroupNorm weights whose new entries start
    at 1 (fresh-norm convention). The source tensor is copied into the
    leading slice on every axis; returns (tensor, elements copied).
    """
    if any(t < s for s, t in zip(src.shape, tgt_shape)):
        raise ValueError(
            f"cannot grow {key!r}: target shape {tuple(tgt_shape)} is "
            f"smaller than source {tuple(src.shape)} on some axis "
            f"(growth only widens)")
    if key.endswith(".gn.weight"):
        out = torch.ones(tgt_shape, dtype=src.dtype)
    else:
        out = torch.zeros(tgt_shape, dtype=src.dtype)
    slices = tuple(slice(0, s) for s in src.shape)
    out[slices] = src.detach().to(src.dtype)
    return out, int(src.numel())


def grow_vae_state_dict(old_sd: dict, old_latent_ch: int = 16,
                        old_base_ch: int = 32, new_latent_ch: int = 32,
                        new_base_ch: int = 64) -> tuple[dict, dict]:
    """Grow a CausalVAE state dict from (old_latent_ch, old_base_ch)
    to (new_latent_ch, new_base_ch). Returns (grown_sd, stats).

    The declared shapes are cross-checked against the tensors
    themselves (infer_vae_shape); a mismatch raises rather than
    silently producing a wrong-geometry transplant.
    """
    inferred = infer_vae_shape(old_sd)
    if inferred != (int(old_latent_ch), int(old_base_ch)):
        raise ValueError(
            f"source state dict is shape {inferred}, not the declared "
            f"({old_latent_ch}, {old_base_ch})")
    with torch.device("meta"):
        target = CausalVAE(int(new_latent_ch), int(new_base_ch))
    tgt_shapes = {k: tuple(v.shape)
                  for k, v in target.state_dict().items()}
    if set(tgt_shapes) != set(old_sd):
        raise ValueError(
            "source and target VAE state dicts have different keys: "
            f"missing {sorted(set(tgt_shapes) - set(old_sd))}, "
            f"extra {sorted(set(old_sd) - set(tgt_shapes))}")
    grown: dict[str, torch.Tensor] = {}
    copied = 0
    for key, shape in tgt_shapes.items():
        out, n = _grow_tensor(key, old_sd[key], shape)
        grown[key] = out
        copied += n
    total = sum(int(torch.Size(s).numel()) for s in tgt_shapes.values())
    stats = {
        "vae_copied_params": copied,
        "vae_total_params": total,
        "vae_source_params": sum(int(v.numel()) for v in old_sd.values()),
        "source_shape": [int(old_latent_ch), int(old_base_ch)],
        "target_shape": [int(new_latent_ch), int(new_base_ch)],
    }
    return grown, stats


def grow_vae_payload(payload: dict, target: str | VidgenConfig) -> dict:
    """Grow a VAE checkpoint payload into the target preset's VAE shape.

    Returns a NEW weights-only payload: kind/preset/step preserved in
    spirit — ``step`` keeps the donor's number (cumulative run
    numbering), ``config`` is the target preset's config (its
    vae_latent_ch / vae_base_ch describe the grown geometry; the
    donor's ``latent_scale`` is carried over), and a ``grow_vae``
    metadata dict records the transplant. Optimizer / sampler /
    scheduler / EMA state are DROPPED: the grown model starts a fresh
    run via ``train_vae --init-weights``.
    """
    if payload.get("kind") not in ("vae", "vae-weights-fp16") \
            or "vae" not in payload:
        raise ValueError(
            f"grow_vae needs a VAE checkpoint (kind='vae' with 'vae' "
            f"weights), got kind={payload.get('kind')!r}")
    tgt_cfg = get_preset(target) if isinstance(target, str) \
        else copy.deepcopy(target)
    # Source weights: the EMA shadow when the donor carries one — the
    # same weights every consumer would have used (checkpoint.vae_weights
    # rule), so growth transplants the model as it was actually used.
    from vidgen.checkpoint import vae_weights
    src_sd = vae_weights(payload)
    used_ema = payload.get("ema") is not None
    old_latent_ch, old_base_ch = infer_vae_shape(src_sd)
    if (old_latent_ch, old_base_ch) == (tgt_cfg.vae_latent_ch,
                                        tgt_cfg.vae_base_ch):
        raise ValueError(
            f"source VAE is already shape "
            f"({old_latent_ch}, {old_base_ch}) = the {tgt_cfg.name} "
            f"target geometry; no growth needed — use the checkpoint "
            f"directly via --init-weights")
    grown_sd, stats = grow_vae_state_dict(
        src_sd, old_latent_ch, old_base_ch,
        tgt_cfg.vae_latent_ch, tgt_cfg.vae_base_ch)
    cfg_dict = tgt_cfg.to_dict()
    src_cfg = payload.get("config") or {}
    if "latent_scale" in src_cfg:
        # The measured normalizer belongs to the latent space the DiT
        # side already knows; growth preserves the old latent channels
        # exactly, so the donor's scale stays meaningful at init.
        cfg_dict["latent_scale"] = src_cfg["latent_scale"]
    grow_meta = {
        "format": _GROW_VAE_FORMAT,
        "source_preset": payload.get("preset"),
        "source_step": payload.get("step"),
        "target_preset": tgt_cfg.name,
        "source_weights": "ema" if used_ema else "raw",
        "optimizer": "dropped (weights-only output; train via "
                     "train_vae --init-weights with a fresh optimizer)",
        **stats,
    }
    grow_meta["copied_fraction"] = (
        stats["vae_copied_params"] / stats["vae_total_params"]
        if stats["vae_total_params"] else 0.0)
    return {
        "kind": "vae",
        "preset": tgt_cfg.name,
        "step": int(payload.get("step") or 0),
        "config": cfg_dict,
        "vae": grown_sd,
        "grow_vae": grow_meta,
    }


def grow_vae_checkpoint_file(in_path: str, out_path: str,
                             target: str | VidgenConfig) -> dict:
    """File-level wrapper: load a VAE checkpoint, grow it, save the
    weights-only grown checkpoint (plain torch.save, like the DiT
    growth output). Returns the ``grow_vae`` metadata dict."""
    from vidgen.pool import load_payload_file
    payload = load_payload_file(in_path)
    grown = grow_vae_payload(payload, target)
    torch.save(grown, out_path)
    return grown["grow_vae"]
