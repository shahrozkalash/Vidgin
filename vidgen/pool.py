"""Weight-averaging pool (DiLoCo-style) primitives.

The pool is Vidgin's multi-*platform* training mode (Colab + Kaggle +
Lightning AI, whose machines cannot talk to each other — see
PYTHON_IMPLEMENTATION.md, "Weight-averaging pool"). It complements, and
never replaces, the in-machine DDP support in ``vidgen.distributed``:

- N workers each start a round from the SAME pooled checkpoint
  (weights-only init: ``--init-weights`` in both trainers), train
  independently for a fixed number of steps on their own disjoint
  ``--data-shard K/N`` slice of the data, and save a normal checkpoint.
- A coordinator (Spark) merges the workers' checkpoints with
  :func:`merge_payloads` (``scripts/average_checkpoints.py``): the merged
  weights are the mean of the workers' weights, optionally pushed through
  one DiLoCo outer step ``base + outer_lr * (mean - base)`` against the
  checkpoint the round started from.
- The merged checkpoint carries NO optimizer / sampler / RNG state: the
  next round starts a fresh inner optimizer on every worker. Optimizer
  state is a worker's private trajectory memory (Adam moments are tied to
  the exact gradients that worker saw on its own data slice); averaging
  those across workers is not meaningful, and DiLoCo's design replaces
  cross-round momentum with the outer step. Sampler and RNG likewise
  belong to a single run, not to the pooled weights.

Workers never communicate with each other; the only sync is one
checkpoint file per worker per round, so the slow, unreliable links and
short sessions of free tiers are fine — a worker that dies mid-round
simply misses that round's merge.
"""

from __future__ import annotations

import copy

import torch

# Which payload keys carry model weights, per checkpoint kind. The DiT
# trainer's checkpoints hold the DiT and the (separately parameterized)
# text encoder; the VAE trainer's hold the VAE. Kinds are never mixed.
WEIGHT_KEYS: dict[str, tuple[str, ...]] = {
    "dit": ("dit", "text_enc"),
    "vae": ("vae",),
}


# ------------------------------------------------------------------ shards
def parse_data_shard(spec) -> tuple[int, int]:
    """Parse a ``--data-shard K/N`` spec into ``(K, N)`` (1-based K).

    Accepts the string form ``"2/4"`` or an already-parsed ``(K, N)``
    tuple. Worker K of N takes dataset entries whose position (in the
    dataset's deterministic order) satisfies ``index % N == K - 1``.
    Raises ValueError with a clear message on anything malformed.
    """
    if isinstance(spec, (tuple, list)):
        if len(spec) != 2:
            raise ValueError(f"data shard must be K/N, got {spec!r}")
        k, n = int(spec[0]), int(spec[1])
    elif isinstance(spec, str):
        parts = spec.strip().split("/")
        if len(parts) != 2:
            raise ValueError(
                f"data shard must look like K/N (e.g. '2/4'), got {spec!r}")
        try:
            k, n = int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(
                f"data shard must look like K/N with integers, got {spec!r}"
            ) from None
    else:
        raise ValueError(f"data shard must look like K/N, got {spec!r}")
    if n < 1:
        raise ValueError(f"data shard N must be >= 1, got {spec!r}")
    if not 1 <= k <= n:
        raise ValueError(
            f"data shard K must satisfy 1 <= K <= N, got {spec!r}")
    return k, n


def shard_positions(num_items: int, shard: tuple[int, int]) -> list[int]:
    """Global positions owned by worker ``shard`` out of ``num_items``.

    Positions are 0-based over the dataset's deterministic order; worker
    (K, N) owns ``[i for i in range(num_items) if i % N == K - 1]``.
    Shards for a fixed N are disjoint and their union is every position.
    """
    k, n = parse_data_shard(shard)
    return [i for i in range(int(num_items)) if i % n == k - 1]


# ------------------------------------------------------------------ merging
def load_payload_file(path: str) -> dict:
    """Load a checkpoint payload WITHOUT any RNG side effects.

    Merging and inspecting are pure file operations and must not rewind
    the caller's random streams the way a training resume does — so the
    RNG restore stays off. Worker checkpoints arrive from other machines,
    so they go through the safe (allowlisted) loader.
    """
    from vidgen.checkpoint import load_checkpoint

    payload = load_checkpoint(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: not a checkpoint payload (got "
                         f"{type(payload).__name__})")
    return payload


def _verify_compatible(payloads: list[dict], names: list[str],
                       require_same_step: bool = True) -> str:
    """All inputs must be mergeable: same kind, preset, config, and
    identical weight keys/shapes/dtypes; same step too, unless
    ``require_same_step`` is off — the round-start BASE checkpoint is
    checked that way, since it legitimately sits one round of steps
    behind the workers it is compared against. Returns the shared kind.
    Raises ValueError naming the first concrete problem found."""
    first = payloads[0]
    kind = first.get("kind")
    if kind not in WEIGHT_KEYS:
        raise ValueError(
            f"{names[0]}: unknown checkpoint kind {kind!r} "
            f"(expected one of {sorted(WEIGHT_KEYS)}) — only DiT "
            f"('dit') and VAE ('vae') trainer checkpoints can be pooled")
    for key in WEIGHT_KEYS[kind]:
        if key not in first:
            raise ValueError(f"{names[0]}: {kind} checkpoint is missing "
                             f"its {key!r} weights")
    if "step" not in first:
        raise ValueError(f"{names[0]}: checkpoint has no 'step'")
    for name, p in zip(names[1:], payloads[1:]):
        if p.get("kind") != kind:
            raise ValueError(
                f"cannot merge different checkpoint kinds: {names[0]} is "
                f"{kind!r} but {name} is {p.get('kind')!r} — DiT and VAE "
                f"checkpoints are pooled separately")
        if p.get("preset") != first.get("preset"):
            raise ValueError(
                f"preset mismatch: {names[0]} is {first.get('preset')!r} "
                f"but {name} is {p.get('preset')!r}")
        if p.get("config") != first.get("config"):
            raise ValueError(
                f"config mismatch between {names[0]} and {name}: pool "
                f"workers must train the identical architecture/config")
        if p.get("tokenizer") != first.get("tokenizer"):
            raise ValueError(
                f"tokenizer mismatch between {names[0]} and {name}: "
                f"averaging text-encoder weights is only meaningful when "
                f"token ids name the same words in every worker")
        if require_same_step and p.get("step") != first.get("step"):
            raise ValueError(
                f"step mismatch: {names[0]} is at step "
                f"{first.get('step')} but {name} is at step "
                f"{p.get('step')} — every worker must finish the round "
                f"(train the same --steps from the same starting "
                f"checkpoint) before merging")
        for key in WEIGHT_KEYS[kind]:
            if key not in p:
                raise ValueError(f"{name}: {kind} checkpoint is missing "
                                 f"its {key!r} weights")
            a, b = first[key], p[key]
            if set(a) != set(b):
                only = sorted(set(a) ^ set(b))[:3]
                raise ValueError(
                    f"weight key mismatch in {key!r} between "
                    f"{names[0]} and {name} (e.g. {only})")
            for wkey in a:
                if a[wkey].shape != b[wkey].shape:
                    raise ValueError(
                        f"weight shape mismatch: {key}/{wkey} is "
                        f"{tuple(a[wkey].shape)} in {names[0]} but "
                        f"{tuple(b[wkey].shape)} in {name}")
                if a[wkey].dtype != b[wkey].dtype:
                    raise ValueError(
                        f"weight dtype mismatch: {key}/{wkey} is "
                        f"{a[wkey].dtype} in {names[0]} but "
                        f"{b[wkey].dtype} in {name}")
    return kind


def _merge_state_dicts(dicts: list[dict], base: dict | None,
                       outer_lr: float) -> dict:
    """Mean of N state dicts (float64 accumulation), optionally followed
    by the DiLoCo outer step against ``base``. Non-float tensors
    (integer/bool buffers) are copied from the first input — they are
    bookkeeping, not learned weights, and must never be averaged."""
    out = {}
    for wkey in dicts[0]:
        t0 = dicts[0][wkey]
        if torch.is_floating_point(t0) or torch.is_complex(t0):
            acc = t0.to(torch.float64).clone()
            for d in dicts[1:]:
                acc += d[wkey].to(torch.float64)
            mean = acc / len(dicts)
            if base is not None:
                mean = base[wkey].to(torch.float64) \
                    + float(outer_lr) * (mean - base[wkey].to(torch.float64))
            out[wkey] = mean.to(t0.dtype)
        else:
            out[wkey] = t0.clone()
    return out


def merge_payloads(payloads: list[dict], base: dict | None = None,
                   outer_lr: float = 1.0,
                   input_names: list[str] | None = None,
                   base_name: str | None = None) -> dict:
    """Merge 2+ worker checkpoint payloads into one pooled payload.

    ``payloads`` are raw payload dicts as loaded by
    :func:`load_payload_file`. With ``base`` (the checkpoint the round
    started from) the result is ``base + outer_lr * (mean - base)``;
    without it the plain mean, and ``outer_lr`` must then be 1.0.

    The output is a WEIGHTS-ONLY checkpoint (see module docstring):
    merged weights + kind/preset/step/config copied from the inputs +
    a ``pool`` metadata dict — no optimizer, sampler, or RNG state. It
    loads through either trainer's ``--init-weights`` path.
    """
    if len(payloads) < 2:
        raise ValueError(
            f"pooling needs at least 2 worker checkpoints, got "
            f"{len(payloads)}")
    if base is None and float(outer_lr) != 1.0:
        raise ValueError(
            f"--outer-lr {outer_lr} without --base has no meaning: "
            f"without a round-start base checkpoint the merge is the "
            f"plain mean (outer-lr 1.0). Pass --base <round-start ckpt> "
            f"to use an outer step size other than 1.0.")
    names = list(input_names) if input_names else \
        [f"input {i}" for i in range(len(payloads))]
    kind = _verify_compatible(payloads, names)
    if base is not None:
        # The base is the checkpoint the round STARTED from: same kind /
        # preset / config / weight shapes, but an earlier step by design.
        _verify_compatible([payloads[0], base],
                           [names[0], base_name or "base"],
                           require_same_step=False)
    merged = {}
    for key in WEIGHT_KEYS[kind]:
        merged[key] = _merge_state_dicts(
            [p[key] for p in payloads],
            base[key] if base is not None else None, outer_lr)
    first = payloads[0]
    out = {
        "kind": kind,
        "preset": first.get("preset"),
        "step": first["step"],
        "config": copy.deepcopy(first.get("config")),
        "pool": {
            "inputs": names,
            "num_workers": len(payloads),
            "outer_lr": float(outer_lr),
            "base": base_name,
            "base_step": base["step"] if base is not None else None,
            "kind": kind,
            "preset": first.get("preset"),
            "step": first["step"],
        },
    }
    # Not weights, but needed to USE the weights: a caption-mode DiT is
    # unusable without its vocabulary (init-weights and sampling refuse
    # it), and the VAE path is what sampling decodes with. Dropping them
    # made every pooled caption-tokenizer round fail to start.
    for key in ("tokenizer", "vae_ckpt"):
        if first.get(key) is not None:
            out[key] = copy.deepcopy(first[key])
    for key in WEIGHT_KEYS[kind]:
        out[key] = merged[key]
    ema = _merge_ema([p.get("ema") for p in payloads])
    out["pool"]["ema"] = ema is not None
    if ema is not None:
        out["ema"] = ema
    return out


def _merge_ema(emas: list) -> dict | None:
    """Merge the workers' EMA shadows (``--ema-decay``), or None.

    Each worker's EMA averages its own trajectory; their plain mean is
    the EMA of the pooled trajectory, so it is carried into the pooled
    checkpoint and the next round continues it (load_pooled_ema in
    vidgen.train_utils) instead of restarting the average from zero.
    It is never pushed through the outer step — it is an average, not a
    trajectory point. ``updates`` takes the maximum, so the decay ramp
    does not restart. If any worker kept no EMA (or a differently
    shaped one), nothing is merged: half an average is not an average.
    """
    if not emas or any(e is None for e in emas):
        return None
    names = set(emas[0]) - {"decay", "updates"}
    if any(set(e) - {"decay", "updates"} != names for e in emas[1:]):
        return None
    out = {"decay": float(emas[0]["decay"]),
           "updates": max(int(e["updates"]) for e in emas)}
    for name in sorted(names):
        dicts = [e[name] for e in emas]
        if any(set(d) != set(dicts[0]) or any(
                tuple(d[k].shape) != tuple(dicts[0][k].shape) for k in d)
               for d in dicts[1:]):
            return None
        out[name] = _merge_state_dicts(dicts, None, 1.0)
    return out


def checkpoint_summary(payload: dict) -> dict:
    """Small inspectable summary of one checkpoint payload (pool_status)."""
    return {
        "kind": payload.get("kind"),
        "preset": payload.get("preset"),
        "step": payload.get("step"),
        "is_pool": "pool" in payload,
        "pool_workers": (payload.get("pool") or {}).get("num_workers"),
    }
