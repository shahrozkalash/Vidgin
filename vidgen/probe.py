"""In-training quality probe: is the run actually getting better?

The training loss is noisy (fresh timesteps and noise every step, a
different batch every step), so it cannot answer "did the last 2,000
steps help?". Every ``--probe-every`` steps the trainers now measure the
model on a FIXED probe set with FIXED noise, so successive probes are
directly comparable:

* DiT: the rectified-flow loss on K fixed clips x 5 fixed timesteps
  (``val_loss``, plus per-timestep values) for the raw weights and, when
  the run keeps one, the EMA; then a few short samples from fixed
  prompts and seeds, scored with the pixel metrics of vidgen/evaluate.py
  (motion, flicker, sharpness, diversity, frozen frames) and saved as
  clips under ``<ckpt-dir>/probe/step_XXXXXX/`` for a human to watch.
* VAE: reconstruction MSE / PSNR of K fixed clips (posterior mean, no
  sampling noise), raw and EMA.

The probe set is the ``val`` split when the dataset directory has one
(manifest_val.jsonl written by vidgin_helper.py split), else the first K
training clips. Results go to the training log as ``{"probe": ...}``
records and are printed with the change since the run's first probe.

The probe never changes training: it runs under no_grad, restores the
train/eval mode, and saves/restores every RNG stream around itself, so a
run with probes follows the same trajectory as one without (and resume
stays bit-for-bit).
"""

from __future__ import annotations

import contextlib
import math
import os
import time

import numpy as np
import torch

from vidgen.seed import get_rng_states, set_rng_states

PROBE_TIMESTEPS = (0.1, 0.3, 0.5, 0.7, 0.9)
DEFAULT_PROBE_SEED = 1234


@contextlib.contextmanager
def preserved_rng():
    """Run a block without consuming any python/numpy/torch RNG state."""
    state = get_rng_states()
    try:
        yield
    finally:
        set_rng_states(state)


@contextlib.contextmanager
def eval_mode(*modules):
    """Put modules in eval mode for the block, then restore each one's mode."""
    was = [m.training for m in modules]
    try:
        for m in modules:
            m.eval()
        yield
    finally:
        for m, w in zip(modules, was):
            m.train(w)


@contextlib.contextmanager
def ema_weights(ema, modules: dict):
    """Temporarily load an EMA's shadow weights into ``modules``.

    The raw weights are parked on the CPU and restored afterwards, so
    training continues from exactly the weights it had. ``ema=None``
    is a no-op (the block sees the raw weights).
    """
    if ema is None:
        yield False
        return
    from vidgen.distributed import is_dtensor

    # Plain weights park on the CPU; FSDP shards stay where they are
    # (each rank only holds its own slice, so the copy is small).
    backup = {name: {k: (v.detach().clone() if is_dtensor(v)
                         else v.detach().cpu().clone())
                     for k, v in m.state_dict().items()}
              for name, m in modules.items()}
    try:
        for name, m in modules.items():
            m.load_state_dict({k: v.to(dtype=backup[name][k].dtype)
                               for k, v in ema.shadow[name].items()})
        yield True
    finally:
        for name, m in modules.items():
            m.load_state_dict(backup[name])


def probe_dataset(trainer, data_dir: str | None):
    """(dataset, split_name) the probe draws its fixed clips from."""
    from vidgen.data.clips import FolderClipDataset

    if data_dir and os.path.isfile(os.path.join(data_dir, "manifest_val.jsonl")) \
            and getattr(trainer, "data_split", None) != "val":
        cfg = trainer.cfg
        ds = FolderClipDataset(data_dir, height=cfg.height, width=cfg.width,
                               frames=cfg.frames, manifest_name="manifest_val.jsonl")
        if len(ds):
            return ds, "val"
    return trainer.dataset, "train"


def _psnr(mse: float) -> float:
    # Pixels live in [-1, 1] (peak-to-peak 2), so PSNR = 10 log10(4 / mse).
    return float("inf") if mse <= 0 else 10.0 * math.log10(4.0 / mse)


def _changes(history: list, rec: dict, keys: tuple) -> dict:
    """Change of each key vs. the run's first probe (negative = lower)."""
    if not history:
        return {}
    first = history[0]
    return {f"{k}_change": round(rec[k] - first[k], 6)
            for k in keys if k in rec and k in first
            and rec[k] is not None and first[k] is not None}


class DiTProbe:
    """Fixed-set probe for DiTTrainer (see module docstring)."""

    def __init__(self, trainer, data_dir: str | None = None,
                 num_clips: int = 4, prompts: list[str] | None = None,
                 num_samples: int = 2, sample_steps: int = 8,
                 out_dir: str | None = None, seed: int = DEFAULT_PROBE_SEED):
        self.tr = trainer
        self.sample_steps = int(sample_steps)
        self.out_dir = out_dir
        self.history: list[dict] = []
        dataset, self.split = probe_dataset(trainer, data_dir)
        n = min(int(num_clips), len(dataset))
        if n < 1:
            raise ValueError("the probe needs at least one clip")
        cfg = trainer.cfg
        with preserved_rng(), torch.no_grad():
            mus, caps = [], []
            for i in range(n):
                clip, cap = dataset[i]
                mu, _ = trainer.vae.encode(clip.unsqueeze(0).to(trainer.device))
                mus.append(mu[0].cpu())
                caps.append(cap)
            self.mu = torch.stack(mus)                   # [K,C,t,h,w] CPU
            self.captions = caps
            gen = torch.Generator().manual_seed(int(seed))
            k = len(PROBE_TIMESTEPS)
            self.noise = torch.randn((n * k,) + tuple(self.mu.shape[1:]),
                                     generator=gen)
        self.t = torch.tensor(PROBE_TIMESTEPS).repeat(n)  # [K*5], clip-major
        if prompts:
            self.prompts = list(prompts)
        else:
            self.prompts = (caps * num_samples)[:max(0, int(num_samples))]
        self.seed = int(seed)
        self.depth = None
        if cfg.use_depth:
            # The "no map" condition: probes measure text conditioning.
            self.depth = torch.zeros(1, 1, cfg.latent_t, cfg.latent_h,
                                     cfg.latent_w)

    @torch.no_grad()
    def _val_loss(self, dit, text_enc) -> tuple[float, list[float]]:
        from vidgen.flow import flow_loss, rectified_flow_targets

        tr, dev = self.tr, self.tr.device
        k = len(PROBE_TIMESTEPS)
        per = torch.zeros(len(self.t))
        for c in range(self.mu.shape[0]):
            x1 = (self.mu[c:c + 1] * tr.cfg.latent_scale).expand(
                k, *self.mu.shape[1:]).to(dev)
            t = self.t[c * k:(c + 1) * k].to(dev)
            x0 = self.noise[c * k:(c + 1) * k].to(dev)
            xt = (1 - t.view(-1, 1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1, 1) * x1
            target = x1 - x0
            text, mask = text_enc([self.captions[c]] * k)
            depth = (None if self.depth is None
                     else self.depth.expand(k, -1, -1, -1, -1).to(dev))
            pred = dit(xt, t, text, mask, depth=depth)
            per[c * k:(c + 1) * k] = ((pred.float() - target) ** 2).mean(
                dim=(1, 2, 3, 4)).cpu()
        by_t = per.view(-1, k).mean(dim=0).tolist()
        return float(per.mean()), [round(v, 6) for v in by_t]

    @torch.no_grad()
    def _samples(self, dit, text_enc, step: int) -> dict:
        from vidgen.evaluate import evaluate_clip_set
        from vidgen.sampler import sample_latents
        from vidgen.video_io import frames_to_uint8

        tr, cfg, dev = self.tr, self.tr.cfg, self.tr.device
        clips = []
        shape = (1, cfg.latent_channels, cfg.latent_t, cfg.latent_h,
                 cfg.latent_w)
        null, null_mask = text_enc.null_embedding()
        out_dir = (os.path.join(self.out_dir, f"step_{step:06d}")
                   if self.out_dir else None)
        for j, prompt in enumerate(self.prompts):
            text, mask = text_enc([prompt])
            z = sample_latents(dit, shape, text, mask, null, null_mask,
                               num_steps=self.sample_steps,
                               seed=self.seed + j, device=str(dev),
                               depth=(None if self.depth is None
                                      else self.depth.to(dev)))
            frames = tr.vae.decode(z / cfg.latent_scale)[0] \
                .permute(1, 2, 3, 0).float().cpu().numpy()
            clips.append(frames_to_uint8(frames))
            if out_dir:
                _save_probe_clip(clips[-1], out_dir, j, prompt)
        metrics = evaluate_clip_set(clips)
        metrics.pop("memorization", None)
        metrics.pop("prompt_adherence", None)
        if out_dir:
            metrics["saved_to"] = out_dir
        return metrics

    def run(self, step: int, sample_modules=None) -> dict:
        """Probe the trainer's current weights; returns the log record.

        ``sample_modules``: (dit, text_enc) to evaluate — the trainer's
        unwrapped modules by default (DDP); FSDP passes its sharded
        modules, which every rank must then run together.
        """
        from vidgen.distributed import unwrap_model

        t0 = time.perf_counter()
        tr = self.tr
        dit, text_enc = sample_modules or (unwrap_model(tr.dit),
                                           unwrap_model(tr.text_enc))
        rec = {"probe": int(step), "split": self.split,
               "clips": int(self.mu.shape[0])}
        with preserved_rng(), eval_mode(dit, text_enc):
            rec["val_loss"], rec["val_loss_by_t"] = self._val_loss(dit, text_enc)
            ema = getattr(tr, "ema", None)
            with ema_weights(ema, {"dit": dit, "text_enc": text_enc}) as used:
                if used:
                    rec["val_loss_ema"], _ = self._val_loss(dit, text_enc)
                if self.prompts:
                    rec["samples"] = self._samples(dit, text_enc, step)
                    rec["samples"]["weights"] = "ema" if used else "raw"
        rec.update(_changes(self.history, rec, ("val_loss", "val_loss_ema")))
        rec["secs"] = round(time.perf_counter() - t0, 2)
        self.history.append(rec)
        return rec


class VAEProbe:
    """Fixed-set reconstruction probe for VAETrainer."""

    def __init__(self, trainer, data_dir: str | None = None,
                 num_clips: int = 4):
        self.tr = trainer
        self.history: list[dict] = []
        dataset, self.split = probe_dataset(trainer, data_dir)
        n = min(int(num_clips), len(dataset))
        if n < 1:
            raise ValueError("the probe needs at least one clip")
        with preserved_rng():
            self.clips = torch.stack([dataset[i][0] for i in range(n)])

    @torch.no_grad()
    def _mse(self, vae) -> float:
        total = 0.0
        for c in range(self.clips.shape[0]):
            x = self.clips[c:c + 1].to(self.tr.device)
            mu, _ = vae.encode(x)
            total += float(((vae.decode(mu).float() - x) ** 2).mean())
        return total / self.clips.shape[0]

    def run(self, step: int, vae=None) -> dict:
        from vidgen.distributed import unwrap_model

        t0 = time.perf_counter()
        vae = vae or unwrap_model(self.tr.vae)
        rec = {"probe": int(step), "split": self.split,
               "clips": int(self.clips.shape[0])}
        with preserved_rng(), eval_mode(vae):
            rec["recon_mse"] = round(self._mse(vae), 6)
            rec["psnr"] = round(_psnr(rec["recon_mse"]), 3)
            ema = getattr(self.tr, "ema", None)
            with ema_weights(ema, {"vae": vae}) as used:
                if used:
                    rec["recon_mse_ema"] = round(self._mse(vae), 6)
                    rec["psnr_ema"] = round(_psnr(rec["recon_mse_ema"]), 3)
        rec.update(_changes(self.history, rec,
                            ("recon_mse", "psnr", "recon_mse_ema", "psnr_ema")))
        rec["secs"] = round(time.perf_counter() - t0, 2)
        self.history.append(rec)
        return rec


def _save_probe_clip(uint8: np.ndarray, out_dir: str, j: int,
                     prompt: str) -> None:
    """Save one probe sample: MP4 when a writer exists, else GIF."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "prompts.txt"), "a",
              encoding="utf-8") as f:
        f.write(f"sample_{j}: {prompt}\n")
    try:
        from vidgen.video_io import write_mp4

        write_mp4(uint8.astype(np.float32) / 127.5 - 1.0,
                  os.path.join(out_dir, f"sample_{j}.mp4"), fps=16.0)
        return
    except Exception:  # an export nicety, never fatal to training
        pass
    try:
        from PIL import Image

        imgs = [Image.fromarray(f) for f in uint8]
        imgs[0].save(os.path.join(out_dir, f"sample_{j}.gif"),
                     save_all=True, append_images=imgs[1:], duration=62)
    except Exception:
        pass


def format_probe(rec: dict) -> str:
    """One human-readable line for the console."""
    parts = [f"probe @ step {rec['probe']} ({rec['split']}, {rec['clips']} clips):"]
    for key in ("val_loss", "val_loss_ema", "recon_mse", "psnr",
                "recon_mse_ema", "psnr_ema"):
        if key in rec:
            ch = rec.get(f"{key}_change")
            parts.append(f"{key}={rec[key]:.4f}"
                         + (f" ({ch:+.4f} since first probe)" if ch is not None else ""))
    s = rec.get("samples")
    if s:
        parts.append(f"motion={s['motion_mean_absdiff']:.2f} "
                     f"sharpness={s['sharpness']:.2f} "
                     f"frozen={s['frozen_pair_fraction']:.2f}")
    return " ".join(parts)
