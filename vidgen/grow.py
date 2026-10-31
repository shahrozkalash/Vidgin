"""Model growth: transplant a trained DiT into a LARGER preset.

Dawood's "small brains first" strategy, for the DiT: train the small
model, then give the big model its learned weights as a head start
instead of starting it from random noise.

A straight copy is impossible — the presets differ in width, depth,
head count and head dim (Small: 16 layers x 1024 wide, 16 heads;
vidgen9b: 40 x 4096, 32 heads; vidgen14b: 40 x 5120, 40 heads), so a
checkpoint only ever loads into its own exact architecture. This module
performs the transplant ("growth surgery"):

- WIDTH: every source weight is placed at its semantically correct
  position in the bigger tensor. Attention projections are expanded
  HEAD-AWARE (a target head inherits the full q/k/v slice of the
  proportionally mapped source head — a naive top-left copy would mix
  one head's queries with another's keys). The "old outputs must not
  read brand-new inputs" blocks of each matrix are zeroed, and the
  remaining new capacity is initialized the way a fresh model would be
  (PyTorch default inits; the architecture's own zero-init parts —
  AdaLN modulation, the depth stem — stay zero, exactly as in a fresh
  model). Per-block modulation offsets (``--per-block-mod``) follow the
  block depth map, their new width zero.
- DEPTH: target block i inherits source block
  ``round(i * (Ls-1) / (Lt-1))`` — the source stack is stretched
  evenly over the deeper target stack (first->first, last->last).
- TEXT: legacy -> legacy and transformer -> transformer encoders grow
  by the same rules (transformer layers map by depth; its attention
  in-projection is head-aware). Legacy -> transformer (Small/480p ->
  9B/14B) can only carry the WORD EMBEDDINGS: each word the legacy
  encoder knows is written into the bucket the transformer encoder's
  word-hash tokenizer assigns it; the transformer stack itself starts
  fresh — there is no legacy counterpart to transplant from.
- The VAE is NOT handled here: within one VAE shape it transfers
  exactly through ``vidgen.train_vae --init-weights``. A VAE shape
  change (the 2026-10-31 growth of Small's VAE from latent 16 / base
  32 to 32 / 64) is a separate transplant — see vidgen/vae/grow.py.

Honest limits: the grown model does NOT reproduce the source model's
function — LayerNorm statistics, the token grid (resolution), and the
RoPE geometry all change with size, so exact preservation is
impossible in this architecture. What growth guarantees is that every
learned weight lands in the slot with the same meaning, and the new
capacity starts at fresh-model scale. The big model still has to
train; growth buys it a head start (empirically measured for the
Tiny -> Small hop in tests/test_grow.py's companion proof run), not a
finished education.

Output is a WEIGHTS-ONLY checkpoint (like a pooled checkpoint): it
loads through ``DiTTrainer.load_dit_weights`` / ``--init-weights`` and
begins a fresh run at step 0 with fresh optimizer / sampler / RNG.
"""

from __future__ import annotations

import copy
import math
import zlib

import torch
import torch.nn as nn

from vidgen.config import VidgenConfig, get_preset
from vidgen.dit.model import DiT
from vidgen.pool import load_payload_file
from vidgen.text.encoder import VOCAB, build_text_encoder

_GROW_FORMAT = "vidgen-grow-v1"


# ------------------------------------------------------------------ helpers
def layer_map(n_src: int, n_tgt: int) -> list[int]:
    """Target layer index -> source layer index (endpoint-preserving).

    The source stack is stretched evenly: target 0 -> source 0 and
    target Lt-1 -> source Ls-1, proportionally in between.
    """
    if n_src < 1 or n_tgt < 1:
        raise ValueError(f"layer counts must be >= 1, got {n_src}->{n_tgt}")
    if n_tgt == 1:
        return [0]
    if n_src == 1:
        return [0] * n_tgt
    return [round(i * (n_src - 1) / (n_tgt - 1)) for i in range(n_tgt)]


def _head_map(n_src: int, n_tgt: int) -> list[int]:
    """Target head index -> source head index (proportional, floor)."""
    return [min(n_src - 1, (h * n_src) // n_tgt) for h in range(n_tgt)]


def _is_norm_weight(key: str) -> bool:
    return key.endswith(".weight") and ("norm" in key)


def _fresh_init(key: str, shape: tuple, gen: torch.Generator,
                dtype: torch.dtype) -> torch.Tensor:
    """A fresh-model-scale starting value for a target tensor.

    Mirrors the constructors: LayerNorm weight 1 / bias 0, the DiT's
    AdaLN modulation (shared + final) and depth stem ZERO (both are
    zero-init by design in vidgen.dit.model.DiT), other biases 0,
    matrices at PyTorch's default Linear init, embeddings N(0, 1)
    (PyTorch's Embedding default). Non-float tensors (buffers) are 0.
    """
    if not torch.empty((), dtype=dtype).is_floating_point():
        return torch.zeros(shape, dtype=dtype)
    if "adaLN" in key or "depth_stem" in key:
        return torch.zeros(shape, dtype=dtype)
    if _is_norm_weight(key) and len(shape) == 1:
        return torch.ones(shape, dtype=dtype)
    if key.endswith(".bias") or (len(shape) == 1 and key.endswith(".weight")):
        return torch.zeros(shape, dtype=dtype)
    t = torch.empty(shape, dtype=torch.float32)
    if "embedding" in key:
        nn.init.normal_(t, mean=0.0, std=1.0, generator=gen)
    elif len(shape) >= 2:
        nn.init.kaiming_uniform_(t, a=math.sqrt(5), generator=gen)
    else:
        nn.init.zeros_(t)
    return t.to(dtype)


# ------------------------------------------------------- tensor expansion
def _expand_generic(src: torch.Tensor, tgt_shape: tuple, key: str,
                    gen: torch.Generator) -> tuple[torch.Tensor, int]:
    """Default expansion: fresh base, source copied into the leading
    slice on every axis; for 2D matrices the (old out) x (new in) block
    is zeroed so transplanted outputs do not read brand-new inputs at
    step 0 (the new inputs reach them only as training learns the
    connection)."""
    out = _fresh_init(key, tgt_shape, gen, src.dtype)
    if src.ndim == 0:
        out.copy_(src)
        return out, 1
    slices = tuple(slice(0, min(s, t)) for s, t in zip(src.shape, tgt_shape))
    out[slices] = src[slices].to(src.dtype)
    copied = 1  # numel of the overlap region
    for s in slices:
        copied *= s.stop
    if src.ndim == 2 and tgt_shape[1] > src.shape[1] and src.shape[0] > 0:
        out[: src.shape[0], src.shape[1]:] = 0
    return out, copied


def _expand_grouped_rows(src: torch.Tensor, tgt_shape: tuple,
                         groups: int, src_heads: int, tgt_heads: int,
                         key: str, gen: torch.Generator
                         ) -> tuple[torch.Tensor, int]:
    """Expand a (groups * heads * head_dim[, in]) projection head-aware.

    Used for qkv / cross_q / cross_kv / text self-attn in-projections:
    each target head receives the mapped source head's full slice
    (head_dim zero-padded when it grows), and the columns beyond the
    source width are zeroed on transplanted rows (old heads must not
    read new token dims at step 0).
    """
    out = _fresh_init(key, tgt_shape, gen, src.dtype)
    hds = src.shape[0] // (groups * src_heads)
    hdt = tgt_shape[0] // (groups * tgt_heads)
    hd = min(hds, hdt)
    hmap = _head_map(src_heads, tgt_heads)
    copied = 0
    for g in range(groups):
        for ht, hs in enumerate(hmap):
            s0 = (g * src_heads + hs) * hds
            t0 = (g * tgt_heads + ht) * hdt
            if src.ndim == 2:
                out[t0:t0 + hd, :src.shape[1]] = src[s0:s0 + hd, :]
                out[t0:t0 + hdt, src.shape[1]:] = 0
                copied += hd * src.shape[1]
            else:
                out[t0:t0 + hd] = src[s0:s0 + hd]
                copied += hd
    return out, copied


def _expand_attn_out(src: torch.Tensor, tgt_shape: tuple,
                     src_heads: int, tgt_heads: int, key: str,
                     gen: torch.Generator) -> tuple[torch.Tensor, int]:
    """Expand an attention output projection [dim_out, heads*head_dim]:
    the input axis is head-structured (each target head's columns come
    from the mapped source head), the output axis grows generically
    (new output rows keep their fresh init, so new token dims are fed
    by every head from step 0)."""
    out = _fresh_init(key, tgt_shape, gen, src.dtype)
    hds = src.shape[1] // src_heads
    hdt = tgt_shape[1] // tgt_heads
    hd = min(hds, hdt)
    hmap = _head_map(src_heads, tgt_heads)
    copied = 0
    rows = min(src.shape[0], tgt_shape[0])
    for ht, hs in enumerate(hmap):
        s0 = hs * hds
        t0 = ht * hdt
        out[:rows, t0:t0 + hd] = src[:rows, s0:s0 + hd]
        copied += rows * hd
    return out, copied


def _expand_modulation(src: torch.Tensor, tgt_shape: tuple, groups: int,
                       key: str, gen: torch.Generator
                       ) -> tuple[torch.Tensor, int]:
    """Expand an AdaLN projection, whose output is ``groups`` stacked
    dim-sized chunks (adaLN_shared: 9 chunks, adaLN_final: 2). Each
    chunk expands generically against the others staying ZERO (the
    architecture's zero-init convention: a fresh chunk modulates
    nothing and gates nothing)."""
    out = torch.zeros(tgt_shape, dtype=src.dtype)
    ds = src.shape[-1] if src.ndim == 2 else src.shape[0] // groups
    dt = tgt_shape[-1] if src.ndim == 2 else tgt_shape[0] // groups
    # For the 2D weight, the chunking is on the OUTPUT (row) axis; the
    # input axis is the plain model dim.
    copied = 0
    if src.ndim == 2:
        src_chunk = src.shape[0] // groups
        tgt_chunk = tgt_shape[0] // groups
        r = min(src_chunk, tgt_chunk)
        c = min(src.shape[1], tgt_shape[1])
        for g in range(groups):
            out[g * tgt_chunk:g * tgt_chunk + r, :c] = \
                src[g * src_chunk:g * src_chunk + r, :c]
            copied += r * c
    else:  # bias: (groups * dim,)
        src_chunk = src.shape[0] // groups
        tgt_chunk = tgt_shape[0] // groups
        r = min(src_chunk, tgt_chunk)
        for g in range(groups):
            out[g * tgt_chunk:g * tgt_chunk + r] = \
                src[g * src_chunk:g * src_chunk + r]
            copied += r
    return out, copied


def _grow_tensor(key: str, src: torch.Tensor, tgt_shape: tuple,
                 src_heads: int, tgt_heads: int,
                 gen: torch.Generator) -> tuple[torch.Tensor, int]:
    """Dispatch one tensor to its expansion rule by key pattern."""
    if tuple(src.shape) == tuple(tgt_shape):
        return src.detach().clone(), int(src.numel())
    base = key.split(".")[-1]
    parent = key.rsplit(".", 1)[0] if "." in key else ""
    if key.startswith("adaLN_shared"):
        return _expand_modulation(src, tgt_shape, 9, key, gen)
    if key.startswith("adaLN_final"):
        return _expand_modulation(src, tgt_shape, 2, key, gen)
    if base in ("weight", "bias") and parent.endswith("qkv"):
        # qkv packs 3 projections (q, k, v), each head-structured.
        return _expand_grouped_rows(src, tgt_shape, 3, src_heads,
                                    tgt_heads, key, gen)
    if base in ("weight", "bias") and parent.endswith("cross_q"):
        return _expand_grouped_rows(src, tgt_shape, 1, src_heads,
                                    tgt_heads, key, gen)
    if base in ("weight", "bias") and parent.endswith("cross_kv"):
        return _expand_grouped_rows(src, tgt_shape, 2, src_heads,
                                    tgt_heads, key, gen)
    if base in ("weight", "bias") and (parent.endswith("attn_out")
                                      or parent.endswith("cross_out")):
        if src.ndim == 2:
            return _expand_attn_out(src, tgt_shape, src_heads, tgt_heads,
                                    key, gen)
        return _expand_generic(src, tgt_shape, key, gen)
    if "self_attn.in_proj" in key:
        return _expand_grouped_rows(src, tgt_shape, 3, src_heads,
                                    tgt_heads, key, gen)
    if "self_attn.out_proj" in key and src.ndim == 2:
        return _expand_attn_out(src, tgt_shape, src_heads, tgt_heads,
                                key, gen)
    return _expand_generic(src, tgt_shape, key, gen)


def _grow_block_mod(src: torch.Tensor, tgt_shape: tuple,
                    blocks: list[int]) -> tuple[torch.Tensor, int]:
    """Grow the per-block modulation offsets ``[layers, 9, dim]``.

    Target block i takes the offsets of its mapped source block (the
    same depth map as the blocks themselves); each of the 9 chunks
    keeps its leading ``dim`` slots and the new width starts at ZERO —
    the parameter's own zero init, so new dims add no offset.
    """
    out = torch.zeros(tgt_shape, dtype=src.dtype)
    d = min(src.shape[-1], tgt_shape[-1])
    for i, j in enumerate(blocks):
        out[i, :, :d] = src[j, :, :d]
    return out, len(blocks) * src.shape[1] * d


# ------------------------------------------------------------- DiT growth
def _meta_state_shapes(module_fn) -> dict:
    """State-dict shapes of a module built on the meta device (no RAM)."""
    with torch.device("meta"):
        module = module_fn()
    return {k: tuple(v.shape) for k, v in module.state_dict().items()}


def grow_dit_state_dict(src_sd: dict, src_cfg: VidgenConfig,
                        tgt_cfg: VidgenConfig, seed: int = 1234
                        ) -> tuple[dict, dict]:
    """Grow a DiT state dict from ``src_cfg`` geometry to ``tgt_cfg``.

    Returns (grown_state_dict, stats). Target shapes come from a meta
    build, so even a 14B target costs no memory here — the grown
    tensors themselves are of course full size.
    """
    tgt_shapes = _meta_state_shapes(lambda: DiT(tgt_cfg))
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    blocks = layer_map(src_cfg.dit.layers, tgt_cfg.dit.layers)
    grown: dict[str, torch.Tensor] = {}
    copied_total = 0
    total = 0
    for key, shape in tgt_shapes.items():
        total += int(torch.Size(shape).numel())
        if key.startswith("blocks."):
            _, idx, rest = key.split(".", 2)
            src_key = f"blocks.{blocks[int(idx)]}.{rest}"
        else:
            src_key = key
        if key == "block_mod" and key in src_sd:
            out, copied = _grow_block_mod(src_sd[key], shape, blocks)
            grown[key] = out
            copied_total += copied
            continue
        if src_key not in src_sd:
            if key.startswith("depth_stem"):
                # Target wants a depth stem the source never had (grow
                # called with force_depth): the constructor's zero init
                # — the stem starts silent, exactly like a fresh model.
                grown[key] = torch.zeros(shape, dtype=torch.float32)
                continue
            raise ValueError(
                f"source checkpoint is missing DiT weight {src_key!r} "
                f"needed to grow target key {key!r}")
        out, copied = _grow_tensor(key, src_sd[src_key], shape,
                                   src_cfg.dit.heads, tgt_cfg.dit.heads,
                                   gen)
        grown[key] = out
        copied_total += copied
    stats = {
        "dit_copied_params": copied_total,
        "dit_total_params": total,
        "block_map": blocks,
    }
    return grown, stats


# ------------------------------------------------------------- text growth
def grow_text_state_dict(src_sd: dict, src_cfg: VidgenConfig,
                         tgt_cfg: VidgenConfig, seed: int = 1234
                         ) -> tuple[dict, dict]:
    """Grow a text-encoder state dict across preset geometries.

    legacy -> legacy and transformer -> transformer use the generic /
    head-aware rules; legacy -> transformer transplants only the word
    embeddings (into their word-hash buckets); transformer -> legacy
    is a downgrade and is rejected.
    """
    if getattr(src_cfg, "text_tokenizer", "hash") == "caption" or \
            getattr(tgt_cfg, "text_tokenizer", "hash") == "caption":
        raise ValueError(
            "growing caption-tokenizer text encoders is not supported: "
            "token ids are vocabulary positions, not hash buckets, so "
            "the transplant rules do not apply. Grow in hash mode, or "
            "train the target with its own tokenizer from scratch.")
    src_transformer = getattr(src_cfg, "text_layers", 0) > 0
    tgt_transformer = getattr(tgt_cfg, "text_layers", 0) > 0
    if src_transformer and not tgt_transformer:
        raise ValueError(
            "cannot grow a transformer text encoder into the legacy "
            "embedding-only encoder (that is a downgrade, not growth)")
    tgt_shapes = _meta_state_shapes(lambda: build_text_encoder(tgt_cfg))
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 777)
    grown: dict[str, torch.Tensor] = {}
    copied_total = 0
    total = 0
    stats: dict = {}

    if tgt_transformer and not src_transformer:
        # Legacy -> transformer: word embeddings into hash buckets;
        # everything else starts fresh (there is nothing to copy from).
        for key, shape in tgt_shapes.items():
            total += int(torch.Size(shape).numel())
            grown[key] = _fresh_init(key, shape, gen, torch.float32)
        src_emb = src_sd["embedding.weight"]           # [V_legacy, Ds]
        tgt_emb = grown["embedding.weight"]            # [V_hash, Dt]
        vocab_size = tgt_cfg.text_vocab_size
        collisions = 0
        placed_rows: dict[int, int] = {}
        for i, word in enumerate(VOCAB):
            if i >= src_emb.shape[0]:
                break
            bucket = 1 + zlib.crc32(word.encode("utf-8")) % (vocab_size - 1)
            if bucket in placed_rows:
                collisions += 1
            placed_rows[bucket] = i
            row = torch.zeros(tgt_emb.shape[1], dtype=tgt_emb.dtype)
            n = min(src_emb.shape[1], tgt_emb.shape[1])
            row[:n] = src_emb[i, :n]
            tgt_emb[bucket] = row
            copied_total += n
        # The transformer encoder's _null_ids buffer matches the legacy
        # one's shape ([1, 1] zeros); carry it over verbatim.
        if "_null_ids" in src_sd and "_null_ids" in grown:
            grown["_null_ids"] = src_sd["_null_ids"].detach().clone()
        stats.update({
            "text_transfer": "embeddings_only",
            "text_words_placed": len(placed_rows),
            "text_bucket_collisions": collisions,
            "text_copied_params": copied_total,
            "text_total_params": total,
        })
        return grown, stats

    layers = None
    if src_transformer and tgt_transformer:
        layers = layer_map(src_cfg.text_layers, tgt_cfg.text_layers)
    for key, shape in tgt_shapes.items():
        total += int(torch.Size(shape).numel())
        if layers is not None and key.startswith("encoder.layers."):
            _, _, idx, rest = key.split(".", 3)
            src_key = f"encoder.layers.{layers[int(idx)]}.{rest}"
        else:
            src_key = key
        if src_key not in src_sd:
            raise ValueError(
                f"source checkpoint is missing text weight {src_key!r} "
                f"needed to grow target key {key!r}")
        out, copied = _grow_tensor(key, src_sd[src_key], shape,
                                   src_cfg.text_heads, tgt_cfg.text_heads,
                                   gen)
        grown[key] = out
        copied_total += copied
    stats.update({
        "text_transfer": "full",
        "text_copied_params": copied_total,
        "text_total_params": total,
    })
    if layers is not None:
        stats["text_layer_map"] = layers
    return grown, stats


# ------------------------------------------------------------ payload API
def grow_payload(payload: dict, target: str | VidgenConfig,
                 seed: int = 1234, force_depth: bool = False) -> dict:
    """Grow a DiT checkpoint payload into a larger preset's geometry.

    ``payload`` is a raw checkpoint dict (see vidgen.pool.load_payload_file);
    ``target`` is a preset name or a VidgenConfig. Returns a NEW
    weights-only payload for the target preset: kind/preset/step=0/
    config + grown ``dit`` and ``text_enc`` weights + a ``grow``
    metadata dict recording the source and the transplant statistics.
    It loads through DiTTrainer.load_dit_weights (``--init-weights``).
    """
    if payload.get("kind") != "dit" or "dit" not in payload \
            or "text_enc" not in payload:
        raise ValueError(
            f"grow needs a DiT checkpoint (kind='dit' with 'dit' and "
            f"'text_enc' weights), got kind={payload.get('kind')!r} — "
            f"for a VAE checkpoint use vidgen.vae.grow (same-shape "
            f"VAEs transfer via train_vae --init-weights)")
    src_cfg_dict = payload.get("config")
    if src_cfg_dict:
        src_cfg = VidgenConfig.from_dict(copy.deepcopy(src_cfg_dict))
    elif payload.get("preset"):
        src_cfg = get_preset(payload["preset"])
    else:
        raise ValueError("source checkpoint carries neither a config "
                         "nor a preset — cannot determine its geometry")
    tgt_cfg = get_preset(target) if isinstance(target, str) \
        else copy.deepcopy(target)
    if tgt_cfg.name == src_cfg.name and tgt_cfg.dit == src_cfg.dit \
            and tgt_cfg.text_dim == src_cfg.text_dim \
            and tgt_cfg.text_layers == src_cfg.text_layers:
        raise ValueError(
            f"source and target are the same geometry ({src_cfg.name}); "
            f"no growth needed — use the checkpoint directly via "
            f"--init-weights")
    # Depth conditioning is inherited from the source (a depth-trained
    # brain keeps its stem, grown like everything else); --depth on the
    # grow CLI can add a fresh zero-init stem to a plain source.
    src_depth = bool(src_cfg_dict.get("use_depth", False)) \
        if src_cfg_dict else False
    tgt_cfg.use_depth = src_depth or bool(force_depth)
    # Per-block modulation is inherited too: its [layers, 9, dim]
    # offsets follow the block depth map (see _grow_block_mod).
    tgt_cfg.per_block_mod = bool(src_cfg_dict.get("per_block_mod", False))         if src_cfg_dict else False
    tgt_cfg.validate()

    dit_grown, dit_stats = grow_dit_state_dict(
        payload["dit"], src_cfg, tgt_cfg, seed=seed)
    text_grown, text_stats = grow_text_state_dict(
        payload["text_enc"], src_cfg, tgt_cfg, seed=seed)
    grow_meta = {
        "format": _GROW_FORMAT,
        "source_preset": src_cfg.name,
        "source_step": payload.get("step"),
        "target_preset": tgt_cfg.name,
        "seed": int(seed),
        "force_depth": bool(force_depth),
        "recommended_lr": (
            "start a grown model at ~1/5 of the from-scratch learning "
            "rate for its first phase (measured on tiny->small: the "
            "default lr 1e-4 disturbed the transplant on step 2, "
            "while 2e-5 preserved the head start)"),
        **dit_stats, **text_stats,
    }
    total = dit_stats["dit_total_params"] + text_stats["text_total_params"]
    copied = dit_stats["dit_copied_params"] + text_stats["text_copied_params"]
    grow_meta["total_params"] = total
    grow_meta["copied_params"] = copied
    grow_meta["copied_fraction"] = copied / total if total else 0.0
    out = {
        "kind": "dit",
        "preset": tgt_cfg.name,
        "step": 0,
        "config": tgt_cfg.to_dict(),
        "dit": dit_grown,
        "text_enc": text_grown,
        "grow": grow_meta,
    }
    if payload.get("vae_ckpt"):
        # The VAE is the same at every size: the grown DiT still lives in
        # (and must be sampled with) the source's latent space.
        out["vae_ckpt"] = payload["vae_ckpt"]
    return out


def grow_checkpoint_file(in_path: str, out_path: str,
                         target: str | VidgenConfig, seed: int = 1234,
                         force_depth: bool = False) -> dict:
    """File-level wrapper: load a DiT checkpoint, grow it, save the
    weights-only grown checkpoint (plain torch.save — no RNG state,
    like pool outputs). Returns the ``grow`` metadata dict."""
    payload = load_payload_file(in_path)
    grown = grow_payload(payload, target, seed=seed,
                         force_depth=force_depth)
    torch.save(grown, out_path)
    return grown["grow"]
