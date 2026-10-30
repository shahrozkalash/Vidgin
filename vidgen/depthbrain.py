"""DepthBrain — Vidgin's own depth estimator (to replace MiDaS in prep).

``scripts/prepare_clips.py --depth`` labels every clip with a depth map
using MiDaS (Intel's pretrained network, downloaded through torch.hub) —
a borrowed model. DepthBrain is a U-Net that lives in this repo, starts
from random weights, trains with ``scripts/train_depthbrain.py`` and loads
from a plain Vidgin checkpoint: no torch.hub, no internet, no third-party
weights at prep time.

Where it learns from: the depth maps MiDaS already wrote for Rounds 1–4
(``<clip>_depth.npy``). MiDaS is the *teacher that makes the labels* —
the same way Gemini makes our captions — while the network and its
weights are ours. DepthBrain can only be as right as its teacher was.

Gate: prep keeps MiDaS as its default. A DepthBrain checkpoint may label
data only once it has passed :func:`agreement_check` — agreement with
MiDaS on held-out clips it never trained on — which the trainer runs at
the end and records inside the checkpoint. (Self-supervised depth from
video motion, with no teacher at all, is a possible v2.)

Same contract as the MiDaS path (``vidgen.depth.estimate_depth``):
uint8 ``[T, H, W]``, min-max normalized over the whole clip, larger =
nearer. Relative depth only, like MiDaS.

Training objective (all on one clip's frames at once, so the scale is
shared across time and flicker is penalized):

- scale-and-shift invariant L1 (MiDaS's own: each side is centered on its
  median and divided by its mean absolute deviation, so only the SHAPE of
  the depth counts; the sign is kept, so an inverted map is wrong);
- gradient matching on the residual at 4 scales (sharp, correctly placed
  edges) plus a temporal term (frame-to-frame change must match the
  teacher's: no flicker).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_KIND = "depthbrain"


@dataclass
class DepthBrainConfig:
    # base 64 -> ~9.3M params (spec: 5-20M). Size is base^2-bound: 48
    # gives ~5.3M, 96 ~21M.
    base: int = 64           # channels at full resolution
    levels: int = 4          # resolutions: full, 1/2, 1/4, 1/8
    groups: int = 8          # GroupNorm groups
    train_size: int = 64     # square size it was trained at (inference resizes)

    def __post_init__(self):
        if self.base < 1 or self.base % self.groups:
            raise ValueError(f"base ({self.base}) must be a positive multiple "
                             f"of groups ({self.groups}) for GroupNorm")
        if self.levels < 2:
            raise ValueError(f"levels must be >= 2, got {self.levels}")

    @property
    def multiple(self) -> int:
        return 2 ** (self.levels - 1)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DepthBrainConfig":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


def _block(cin: int, cout: int, groups: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.SiLU(),
    )


class DepthBrain(nn.Module):
    """RGB frames ``[N, 3, H, W]`` in ``[-1, 1]`` -> relative depth
    ``[N, H, W]`` (larger = nearer; arbitrary scale and offset)."""

    def __init__(self, cfg: DepthBrainConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or DepthBrainConfig()
        ch = [cfg.base * 2 ** i for i in range(cfg.levels)]
        self.stem = _block(3, ch[0], cfg.groups)
        self.down = nn.ModuleList(
            nn.Conv2d(ch[i], ch[i], 3, stride=2, padding=1)
            for i in range(cfg.levels - 1))
        self.enc = nn.ModuleList(
            _block(ch[i], ch[i + 1], cfg.groups) for i in range(cfg.levels - 1))
        self.up = nn.ModuleList(
            nn.Conv2d(ch[i + 1], ch[i], 3, padding=1)
            for i in range(cfg.levels - 1))
        self.dec = nn.ModuleList(
            _block(2 * ch[i], ch[i], cfg.groups) for i in range(cfg.levels - 1))
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"expected [N,3,H,W] frames, got {tuple(x.shape)}")
        h, w = x.shape[-2:]
        m = self.cfg.multiple
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:   # any size works: pad to the U-Net's multiple, crop back
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        skips = [self.stem(x)]
        for down, enc in zip(self.down, self.enc):
            skips.append(enc(down(skips[-1])))
        y = skips.pop()
        for i in reversed(range(self.cfg.levels - 1)):
            y = F.interpolate(y, size=skips[i].shape[-2:], mode="nearest")
            y = self.dec[i](torch.cat([self.up[i](y), skips[i]], dim=1))
        return self.head(y)[:, 0, :h, :w]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------ losses
def _normalize(d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-clip median/MAD normalization of ``[B, T, H, W]`` (sign kept)."""
    flat = d.reshape(d.shape[0], -1)
    med = flat.median(dim=1, keepdim=True).values
    mad = (flat - med).abs().mean(dim=1, keepdim=True)
    return ((flat - med) / (mad + eps)).reshape(d.shape)


def depth_loss(pred: torch.Tensor, target: torch.Tensor,
               grad_weight: float = 0.5, scales: int = 4) -> dict:
    """DepthBrain's training loss for clips ``[B, T, H, W]``.

    ``target`` is the teacher's map in [0, 1]. Clips whose teacher map is
    flat (prep writes all zeros for those) carry no shape and are left
    out. Returns ``{"loss", "ssi", "grad"}`` (the loss is a tensor with
    grad; the parts are floats).
    """
    if pred.shape != target.shape or pred.dim() != 4:
        raise ValueError(f"pred {tuple(pred.shape)} / target "
                         f"{tuple(target.shape)}: need matching [B,T,H,W]")
    keep = target.reshape(target.shape[0], -1).std(dim=1) > 1e-4
    if not bool(keep.any()):
        zero = pred.sum() * 0.0
        return {"loss": zero, "ssi": 0.0, "grad": 0.0}
    p, t = _normalize(pred[keep].float()), _normalize(target[keep].float())
    res = p - t
    ssi = res.abs().mean()
    grad = res.new_zeros(())
    for s in range(scales):
        r = res[..., ::2 ** s, ::2 ** s]
        if min(r.shape[-2:]) < 2:      # frame too small for this scale
            break
        grad = grad + (r[..., :, 1:] - r[..., :, :-1]).abs().mean() \
            + (r[..., 1:, :] - r[..., :-1, :]).abs().mean()
    if res.shape[1] > 1:   # temporal: frame-to-frame change must match
        grad = grad + (res[:, 1:] - res[:, :-1]).abs().mean()
    loss = ssi + grad_weight * grad
    return {"loss": loss, "ssi": float(ssi.detach()),
            "grad": float(grad.detach())}


@torch.no_grad()
def depth_metrics(pred: torch.Tensor, target: torch.Tensor,
                  pairs: int = 4096, seed: int = 0) -> dict:
    """Agreement with the teacher on clips ``[B, T, H, W]``.

    - ``mae``: mean absolute error in the teacher's [0, 1] units after
      the best per-clip NON-NEGATIVE scale + shift (least squares) — 0 is
      a perfect copy; an upside-down map cannot be flipped back to look
      good;
    - ``order``: fraction of random pixel pairs (teacher depth differing
      by > 0.05) that DepthBrain puts in the same near/far order — the
      relative-depth score; 0.5 is chance;
    - ``flicker``: mean frame-to-frame change of the aligned error (0 =
      no flicker beyond what the teacher has);
    - ``pearson``: mean over clips of the Pearson correlation between
      DepthBrain's and the teacher's maps (all pixels of a clip; 1 = same
      shape up to scale+shift, an upside-down map scores negative), with
      ``pearson_per_clip`` aligned to the input clips (``None`` for a
      flat teacher map, which is skipped like in the other metrics).
    """
    keep = target.reshape(target.shape[0], -1).std(dim=1) > 1e-4
    p, t = pred[keep].double(), target[keep].double()
    if p.shape[0] == 0:
        return {"mae": float("nan"), "order": float("nan"),
                "flicker": float("nan"), "pearson": float("nan"),
                "pearson_per_clip": [None] * int(target.shape[0]), "clips": 0}
    b = p.shape[0]
    pf, tf = p.reshape(b, -1), t.reshape(b, -1)
    pc = pf - pf.mean(1, keepdim=True)
    tc = tf - tf.mean(1, keepdim=True)
    scale = ((pc * tc).sum(1, keepdim=True)
             / (pc.pow(2).sum(1, keepdim=True) + 1e-12)).clamp(min=0.0)
    aligned = pc * scale + tf.mean(1, keepdim=True)
    r = ((pc * tc).sum(1) / (pc.pow(2).sum(1).sqrt() * tc.pow(2).sum(1).sqrt()
                             + 1e-12))
    per_clip, kept = [], iter(r.tolist())
    for k in keep.tolist():
        per_clip.append(next(kept) if k else None)
    err = (aligned - tf).reshape(p.shape)
    g = torch.Generator().manual_seed(seed)
    n = pf.shape[1]
    i = torch.randint(0, n, (pairs,), generator=g)
    j = torch.randint(0, n, (pairs,), generator=g)
    dt = tf[:, i] - tf[:, j]
    dp = pf[:, i] - pf[:, j]
    valid = dt.abs() > 0.05
    agree = ((dt > 0) == (dp > 0)) & valid
    order = float(agree.sum()) / max(int(valid.sum()), 1)
    flicker = float((err[:, 1:] - err[:, :-1]).abs().mean()) \
        if p.shape[1] > 1 else 0.0
    return {"mae": float(err.abs().mean()), "order": order,
            "flicker": flicker, "pearson": float(r.mean()),
            "pearson_per_clip": per_clip, "clips": int(b)}


# ------------------------------------------------------------- checkpoints
def depthbrain_weights(payload: dict, use_ema: bool = True) -> dict:
    """The weights to run: the EMA when the checkpoint kept one."""
    ema = payload.get("ema")
    if use_ema and ema and "depthbrain" in ema:
        return ema["depthbrain"]
    return payload["model"]


def load_depthbrain(path: str, device: str = "cpu",
                    use_ema: bool = True) -> DepthBrain:
    """A ready-to-run (eval mode) DepthBrain from a training checkpoint."""
    from vidgen.checkpoint import load_checkpoint

    payload = load_checkpoint(path)
    if payload.get("kind") != CHECKPOINT_KIND:
        raise ValueError(f"{path}: not a DepthBrain checkpoint "
                         f"(kind={payload.get('kind')!r}); train one with "
                         "scripts/train_depthbrain.py")
    model = DepthBrain(DepthBrainConfig.from_dict(payload["config"]))
    model.load_state_dict(depthbrain_weights(payload, use_ema))
    model.step = int(payload.get("step", 0))
    model.agreement = payload.get("agreement")
    return model.to(device).eval()


# ------------------------------------------------------- agreement check
# Provisional thresholds (2026-10-29), set before any real run: change them
# deliberately, never to make a checkpoint pass. Measured on held-out clips
# the student never trained on, with the EMA weights prep would use.
AGREEMENT_GATE = {
    "min_val_clips": 50,   # too few held-out clips proves nothing
    "min_order": 0.85,     # near/far pairs in MiDaS's order (0.5 = chance)
    "max_mae": 0.08,       # [0,1] units after best per-clip scale+shift
    "max_flicker": 0.03,   # frame-to-frame wobble of the aligned error
    "min_pearson": 0.90,   # mean per-clip Pearson correlation with MiDaS
}


def agreement_check(ev: dict, gate: dict | None = None,
                    step: int | None = None) -> dict:
    """Pass/fail of a DepthBrain against its MiDaS teacher.

    ``ev`` is ``DepthBrainTrainer.evaluate`` output (``val_mae``,
    ``val_order``, ``val_flicker``, ``val_pearson``, ``val_clips``).
    Returns ``{"passed", "reasons", "gate", "measured", "step"}``; every
    failed threshold is listed (ALL must hold; a missing measurement
    fails). ``step`` names the weights that were measured.
    """
    gate = dict(AGREEMENT_GATE if gate is None else gate)
    n = int(ev.get("val_clips") or 0)
    measured = {k: ev.get(k) for k in
                ("val_clips", "val_order", "val_mae", "val_flicker",
                 "val_pearson")}
    reasons = []
    if n < gate["min_val_clips"]:
        reasons.append(f"only {n} held-out clips "
                       f"(need >= {gate['min_val_clips']})")
    if n:
        if not ev["val_order"] >= gate["min_order"]:
            reasons.append(f"order {ev['val_order']:.3f} < {gate['min_order']}")
        if not ev["val_mae"] <= gate["max_mae"]:
            reasons.append(f"mae {ev['val_mae']:.4f} > {gate['max_mae']}")
        if not ev["val_flicker"] <= gate["max_flicker"]:
            reasons.append(f"flicker {ev['val_flicker']:.4f} > "
                           f"{gate['max_flicker']}")
        if ev.get("val_pearson") is None:
            reasons.append("no Pearson correlation recorded (checkpoint "
                           "checked before Pearson joined the gate)")
        elif not ev["val_pearson"] >= gate["min_pearson"]:
            reasons.append(f"pearson {ev['val_pearson']:.3f} < "
                           f"{gate['min_pearson']}")
    return {"passed": not reasons, "reasons": reasons, "gate": gate,
            "measured": measured, "step": step}


def verify_agreement(agreement: dict | None, step: int) -> tuple[bool, list]:
    """May a checkpoint at ``step`` label data? Its stored check must be
    for exactly these weights, and its MEASURED numbers are re-scored
    against today's :data:`AGREEMENT_GATE` — the stored ``passed`` flag
    is not trusted (it may come from a looser gate or older weights)."""
    if not agreement or not agreement.get("measured"):
        return False, ["no agreement check recorded"]
    if agreement.get("step") != step:
        return False, [f"the check measured step {agreement.get('step')}, "
                       f"but these weights are step {step}"]
    check = agreement_check(agreement["measured"])
    return check["passed"], check["reasons"]


# --------------------------------------------------------------- inference
def _work_size(h: int, w: int, cfg: DepthBrainConfig) -> tuple[int, int]:
    """Run at the scale DepthBrain trained at (longer side = train_size),
    keeping the aspect ratio, rounded to the U-Net's multiple."""
    k = cfg.train_size / max(h, w)
    m = cfg.multiple
    return (max(m, round(h * k / m) * m), max(m, round(w * k / m) * m))


@torch.no_grad()
def predict_clip(model: DepthBrain, clip: torch.Tensor,
                 batch: int = 16) -> torch.Tensor:
    """Raw DepthBrain output for one clip ``[T, 3, H, W]`` in [-1, 1] ->
    ``[T, H, W]`` float (relative; not normalized)."""
    t, _, h, w = clip.shape
    size = _work_size(h, w, model.cfg)
    outs = []
    for s in range(0, t, batch):
        x = clip[s:s + batch]
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="area")
        y = model(x).unsqueeze(1)
        if y.shape[-2:] != (h, w):
            y = F.interpolate(y, size=(h, w), mode="bicubic", align_corners=False)
        outs.append(y[:, 0].float())
    return torch.cat(outs)


def frames_to_input(clip: np.ndarray, device="cpu") -> torch.Tensor:
    """uint8 ``[T, H, W, 3]`` -> float ``[T, 3, H, W]`` in [-1, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(clip)).to(device)
    return x.permute(0, 3, 1, 2).float() / 127.5 - 1.0


def to_uint8_depth(d: np.ndarray) -> np.ndarray:
    """Per-clip min-max to uint8 (the prep convention; flat -> zeros)."""
    lo, hi = float(d.min()), float(d.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(d.shape, dtype=np.uint8)
    return (((d - lo) / (hi - lo)) * 255.0).round().astype(np.uint8)


def estimate_depth_brain(model: DepthBrain, clip: np.ndarray) -> np.ndarray:
    """uint8 ``[T, H, W, 3]`` -> uint8 ``[T, H, W]`` depth (larger = nearer)."""
    device = next(model.parameters()).device
    d = predict_clip(model, frames_to_input(clip, device))
    return to_uint8_depth(d.cpu().numpy())
