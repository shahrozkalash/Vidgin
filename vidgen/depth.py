"""3D depth conditioning — Dawood's idea (2026-10-20).

Dawood's proposal: "what if the model learns 3D mapping — then generating
video would be easier." He is right about the mechanism: a video is a camera
moving through a 3D world, so a model that is *shown* the world's 3D layout
(a depth map for every frame: how far each pixel is) does not have to
re-invent geometry at every frame. Shapes hold still, motion stays
consistent, and the layout becomes controllable.

How it is wired into Vidgin (all opt-in; nothing changes unless enabled):

1. **Prep** — ``scripts/prepare_clips.py --depth`` estimates a depth map for
   every kept clip and saves ``<clip>_depth.npy`` (uint8 ``[T, H, W]``,
   larger = nearer) next to the clip, recording it in the manifest as
   ``"depth": "<file>"`` and the estimator as ``"depth_source"``.
   ``--depth-source`` picks the estimator: ``midas`` (default — MiDaS, a
   borrowed pretrained network from torch.hub) or ``depthbrain``
   (Vidgin's own, ``vidgen/depthbrain.py``; 2026-10-29). DepthBrain is
   used only from a checkpoint that PASSED its agreement check against
   MiDaS on held-out clips (``vidgen.depthbrain.agreement_check``); it
   becomes the default only once a trained one has.
   ``scripts/label_depth.py`` (re)labels an already prepared dataset the
   same way.
2. **Data** — ``FolderClipDataset.get_depth(i)`` loads a clip's map as
   float ``[1, T, H, W]`` in ``[0, 1]`` (``None`` when the clip has none).
3. **Model** — with ``VidgenConfig.use_depth`` the DiT grows a zero-init
   ``depth_stem`` (see ``vidgen/dit/model.py``): the map, reduced to the
   latent grid by :func:`depth_to_latent_grid`, is embedded and *added* to
   the latent tokens, so every transformer block sees the 3D layout.
   Zero-init means a depth model starts life behaving exactly like the
   plain model and learns to use the maps gradually.
4. **Training** — ``vidgen.train --depth`` feeds each clip's map. With
   probability ``depth_dropout`` the maps are hidden for a whole batch, so
   the model also stays good at imagining geometry from text alone (at
   text-to-video time there is no map unless the user supplies one).
5. **Sampling** — ``sample_latents(..., depth=...)`` /
   ``vidgen.sample --depth-npy`` accept an optional map for controlled
   generation (e.g. re-shoot a known scene layout); omitted, the model
   runs map-free exactly as before.

Honest limits: depth is *relative* (per-clip normalized), single-view, and
estimated — a strong hint about the world's shape, not a true 3D model.
It does not replace training or data; it spends them better. DepthBrain
learns from the maps MiDaS wrote for the earlier rounds (MiDaS is the
teacher that makes the labels, the way Gemini makes captions; DepthBrain's
weights are random-init and trained by our trainer), so it inherits
MiDaS's mistakes.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F


def depth_frame_indices(frames: int, latent_t: int) -> list[int]:
    """Pixel-frame indices aligned to the causal VAE's latent frames.

    The VAE compresses time 4:1 causally: latent frame 0 is pixel frame 0,
    latent frame k (k >= 1) summarizes pixel frames 4k-3 .. 4k. Taking the
    first pixel frame of each group keeps the depth map on the same clock
    as the latents: [0, 1, 5, 9, ...] (length latent_t for T = 4k+1).
    """
    if latent_t < 1 or frames < 1 or (frames - 1) // 4 + 1 != latent_t:
        raise ValueError(
            f"frames={frames} is incompatible with latent_t={latent_t} "
            "(need frames = 4*(latent_t-1)+1)"
        )
    return [0] + [1 + 4 * k for k in range(latent_t - 1)]


def depth_to_latent_grid(depth: torch.Tensor, latent_t: int,
                         latent_h: int, latent_w: int) -> torch.Tensor:
    """Pixel-space depth [B, 1, T, H, W] (float, [0,1]) -> latent grid
    [B, 1, latent_t, latent_h, latent_w], aligned per
    :func:`depth_frame_indices` and area-averaged spatially."""
    if depth.dim() != 5 or depth.shape[1] != 1:
        raise ValueError(f"depth must be [B,1,T,H,W], got {tuple(depth.shape)}")
    idx = depth_frame_indices(depth.shape[2], latent_t)
    sel = depth[:, :, idx]                                  # [B,1,lt,H,W]
    b = sel.shape[0]
    flat = sel.reshape(b * latent_t, 1, sel.shape[3], sel.shape[4])
    small = F.adaptive_avg_pool2d(flat, (latent_h, latent_w))
    return small.reshape(b, 1, latent_t, latent_h, latent_w).contiguous()


def zeros_latent_depth(cfg, batch: int, device, dtype=torch.float32) -> torch.Tensor:
    """The 'no map' input: an all-zeros latent-grid depth tensor."""
    return torch.zeros(batch, 1, cfg.latent_t, cfg.latent_h, cfg.latent_w,
                       device=device, dtype=dtype)


# ---------------------------------------------------------------- estimator
DEPTH_SOURCES = ("midas", "depthbrain")
DEFAULT_DEPTH_SOURCE = "midas"   # until a trained DepthBrain passes its check
DEPTHBRAIN_ENV = "VIDGIN_DEPTHBRAIN"
DEFAULT_DEPTHBRAIN = os.path.join("checkpoints", "depthbrain", "depthbrain.pt")

_MIDAS = None  # (model, transform) lazily loaded once per process
_BRAINS: dict = {}  # (checkpoint path, device) -> loaded DepthBrain


def depthbrain_path(ckpt: str | None = None) -> str:
    """Which DepthBrain checkpoint to use: ``ckpt``, else
    ``$VIDGIN_DEPTHBRAIN``, else ``checkpoints/depthbrain/depthbrain.pt``."""
    return ckpt or os.environ.get(DEPTHBRAIN_ENV) or DEFAULT_DEPTHBRAIN


def load_depthbrain_cached(ckpt: str | None = None, device: str = "cpu",
                           allow_unverified: bool = False):
    """The DepthBrain for ``ckpt`` (see :func:`depthbrain_path`), loaded
    once per process. A missing checkpoint is a clear error up front, and
    so is one that has not passed its agreement check against MiDaS
    (``allow_unverified`` overrides that, for experiments only)."""
    path = depthbrain_path(ckpt)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No DepthBrain checkpoint at {path!r}. Train one with "
            "scripts/train_depthbrain.py (it learns from datasets that "
            "already have MiDaS depth maps), then pass --depthbrain-ckpt "
            f"or set {DEPTHBRAIN_ENV}. Or use --depth-source midas.")
    key = (os.path.abspath(path), str(device))
    if key not in _BRAINS:
        from vidgen.depthbrain import load_depthbrain

        _BRAINS[key] = load_depthbrain(path, device)
    brain = _BRAINS[key]
    from vidgen.depthbrain import verify_agreement

    ok, reasons = verify_agreement(brain.agreement, brain.step)
    if not allow_unverified and not ok:
        raise ValueError(
            f"{path}: this DepthBrain has not passed its agreement check "
            f"against MiDaS on held-out clips ({'; '.join(reasons)}), so "
            "it may not label data. Train it further with "
            "scripts/train_depthbrain.py, or use --depth-source midas "
            "(--allow-unverified-depthbrain overrides, experiments only).")
    return brain


def depth_source_id(source: str = DEFAULT_DEPTH_SOURCE,
                    ckpt: str | None = None, device: str = "cpu",
                    allow_unverified: bool = False) -> str:
    """What the manifest records as ``"depth_source"`` (loads — and so
    checks — a DepthBrain checkpoint)."""
    if source not in DEPTH_SOURCES:
        raise ValueError(
            f"unknown depth source {source!r}; choose {DEPTH_SOURCES}")
    if source == "midas":
        return "midas_small"
    path = depthbrain_path(ckpt)
    brain = load_depthbrain_cached(path, device, allow_unverified)
    from vidgen.depthbrain import verify_agreement

    tag = "" if verify_agreement(brain.agreement, brain.step)[0] else ",unverified"
    return f"depthbrain:{os.path.basename(path)}@step{brain.step}{tag}"


def _trust_hub_repo(owner_repo: str) -> None:
    """Whitelist a torch.hub repo without an interactive prompt.

    MiDaS builds its EfficientNet backbone through an *inner*
    ``torch.hub.load("rwightman/gen-efficientnet-pytorch", ...)`` that
    cannot pass ``trust_repo=True``. On a fresh machine torch.hub stops
    and asks "(y/N)?" on stdin — in a script or notebook that prompt
    never gets an answer and the load hangs forever (found the hard way,
    2026-10-21). Adding the repo to ``<hub dir>/trusted_list`` — the same
    file ``trust_repo=True`` writes — is the documented way out. A
    read-only hub cache just keeps the old behavior, so failures are
    swallowed.
    """
    try:
        hub_dir = torch.hub.get_dir()
        # Fresh sessions (Kaggle/Colab, seen 2026-10-22) have no hub dir at
        # all -- without makedirs the append below raises FileNotFoundError
        # and the trust silently never lands, so the next torch.hub.load
        # hangs on its interactive prompt instead.
        os.makedirs(hub_dir, exist_ok=True)
        path = os.path.join(hub_dir, "trusted_list")
        existing = ""
        if os.path.exists(path):
            with open(path) as f:
                existing = f.read()
        have = existing.split()
        # torch builds differ: some compare the slash form
        # ("owner/repo"), others the underscore-joined cache-dir form
        # ("owner_repo") — write both (observed on Colab, 2026-10-21).
        for name in (owner_repo, owner_repo.replace("/", "_")):
            if name not in have:
                with open(path, "a") as f:
                    f.write(name + "\n")
                have.append(name)
    except OSError:
        pass


def _load_midas(device: str):
    global _MIDAS
    if _MIDAS is None:
        _trust_hub_repo("rwightman/gen-efficientnet-pytorch")
        try:
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small",
                                   trust_repo=True)
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms",
                                        trust_repo=True)
        except Exception as e:  # noqa: BLE001 - re-raised with context
            raise RuntimeError(
                "Could not download the MiDaS depth estimator (needs "
                f"internet at prep time): {e}"
            ) from e
        model.eval()
        _MIDAS = (model, transforms.small_transform)
    model, transform = _MIDAS
    return model.to(device), transform


@torch.no_grad()
def estimate_depth(clip: np.ndarray, device: str = "cpu",
                   source: str = DEFAULT_DEPTH_SOURCE,
                   ckpt: str | None = None,
                   allow_unverified: bool = False) -> np.ndarray:
    """Estimate a depth map for one clip.

    clip: uint8 ``[T, H, W, 3]`` -> uint8 ``[T, H, W]`` depth, per-clip
    min-max normalized (larger = nearer the camera). Relative depth only:
    neither estimator knows real-world meters.

    ``source``: ``"midas"`` (default — torch.hub download, needs internet
    once) or ``"depthbrain"`` (Vidgin's own, from ``ckpt``; see
    :func:`depthbrain_path` — must have passed its agreement check unless
    ``allow_unverified``).
    """
    if clip.dtype != np.uint8 or clip.ndim != 4 or clip.shape[3] != 3:
        raise ValueError(
            f"clip must be uint8 [T,H,W,3], got {clip.dtype} {clip.shape}")
    if source not in DEPTH_SOURCES:
        raise ValueError(
            f"unknown depth source {source!r}; choose {DEPTH_SOURCES}")
    if source == "depthbrain":
        from vidgen.depthbrain import estimate_depth_brain

        return estimate_depth_brain(
            load_depthbrain_cached(ckpt, device, allow_unverified), clip)
    return _estimate_midas(clip, device)


def _estimate_midas(clip: np.ndarray, device: str) -> np.ndarray:
    model, transform = _load_midas(device)
    t, h, w, _ = clip.shape
    frames = []
    for i in range(t):
        inp = transform(clip[i]).to(device)
        pred = model(inp)
        pred = F.interpolate(pred.unsqueeze(1), size=(h, w),
                             mode="bicubic", align_corners=False)
        frames.append(pred.squeeze().float().cpu().numpy())
    d = np.stack(frames)                                    # [T,H,W]
    lo, hi = float(d.min()), float(d.max())
    if hi <= lo:                                            # flat scene
        return np.zeros((t, h, w), dtype=np.uint8)
    return (((d - lo) / (hi - lo)) * 255.0).round().astype(np.uint8)
