"""Part 7: train the DiT with rectified flow in VAE latent space (Track A / Tiny).

Data: Moving MNIST clips -> frozen VAE encoder (mu * latent_scale) -> latents.
Objective: rectified flow, x_t = (1-t)*x0 + t*x1, target v = x1 - x0,
timesteps from the shifted logit-normal schedule (s = 5.0).
Optim: AdamW + grad-norm clipping (1.0). 10% caption dropout -> null embedding
for CFG. Checkpoints carry model + optimizer + sampler + RNG state, so resume
reproduces the next step bit-for-bit.

Usage (CPU smoke):
  python -m vidgen.train --preset tiny --steps 12 --batch-size 2 --num-clips 16 \\
      --ckpt-dir checkpoints/tiny --log checkpoints/tiny/train.jsonl
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

from vidgen.checkpoint import load_checkpoint, save_checkpoint
from vidgen.config import get_preset, resolve_device
from vidgen.data.buckets import BucketSampler
from vidgen.data.clips import FolderClipDataset
from vidgen.data.moving_mnist import MovingMNISTDataset
from vidgen.flow import flow_loss, rectified_flow_targets, sample_shifted_logit_normal, timestep_bins
from vidgen.logging import StructuredLogger
from vidgen.seed import seed_everything
from vidgen.text.encoder import TinyTextEncoder
from vidgen.dit.model import DiT
from vidgen.vae.model import CausalVAE


class DiTTrainer:
    def __init__(self, preset: str = "tiny", seed: int = 0, lr: float = 1e-4,
                 weight_decay: float = 0.01, batch_size: int = 2, num_clips: int = 64,
                 device: str = "cpu", ckpt_dir: str | None = None,
                 log_path: str | None = None, cfg_dropout: float = 0.1,
                 vae_ckpt: str | None = None, data_dir: str | None = None):
        seed_everything(seed)
        self.seed = seed
        self.cfg = get_preset(preset)
        self.device = torch.device(resolve_device(device))
        self.batch_size = batch_size
        self.cfg_dropout = cfg_dropout
        self.ckpt_dir = ckpt_dir
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

        self.vae = CausalVAE().to(self.device)
        if vae_ckpt:
            payload = load_checkpoint(vae_ckpt, map_location=str(self.device))
            assert payload["kind"] == "vae", f"not a VAE checkpoint: {vae_ckpt}"
            self.vae.load_state_dict(payload["vae"])
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.text_enc = TinyTextEncoder(d_t=self.cfg.text_dim,
                                        max_len=self.cfg.text_max_len).to(self.device)
        self.dit = DiT(self.cfg).to(self.device)
        params = list(self.dit.parameters()) + list(self.text_enc.parameters())
        self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        if data_dir:
            self.dataset = FolderClipDataset(
                data_dir, height=self.cfg.height, width=self.cfg.width,
                frames=self.cfg.frames)
            num_items = len(self.dataset)
        else:
            self.dataset = MovingMNISTDataset(
                num_clips, height=self.cfg.height, width=self.cfg.width,
                frames=self.cfg.frames, base_seed=seed + 1)
            num_items = num_clips
        self.sampler = BucketSampler(num_items, batch_size, seed=seed + 2)

        self.step = 0
        self.logger = StructuredLogger(log_path) if log_path else None
        self.loss_history: list[float] = []

    # ------------------------------------------------------------ one step
    def train_step(self) -> dict:
        batch_idx = self.sampler.next_batch()
        clips, captions = [], []
        for i in batch_idx:
            c, cap = self.dataset[i]
            clips.append(c)
            captions.append(cap)
        x = torch.stack(clips).to(self.device)  # [B,3,T,H,W]

        with torch.no_grad():
            mu, _ = self.vae.encode(x)
            x1 = mu * self.cfg.latent_scale

        b = x1.shape[0]
        t = sample_shifted_logit_normal((b,), s=5.0, device=str(self.device))
        xt, target = rectified_flow_targets(x1, t)

        text, text_mask = self.text_enc(captions)
        # 10% caption dropout -> CFG null embedding (uses torch RNG: checkpointed)
        drop = (torch.rand(b, device=self.device) < self.cfg_dropout)
        null_emb, null_mask = self.text_enc.null_embedding()
        null_emb = null_emb.expand(b, -1, -1)
        null_mask = null_mask.expand(b, -1)
        text = torch.where(drop.view(b, 1, 1), null_emb, text)
        text_mask = torch.where(drop.view(b, 1), null_mask, text_mask)

        pred = self.dit(xt, t, text, text_mask)
        loss = flow_loss(pred, target)

        self.opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.dit.parameters()) + list(self.text_enc.parameters()), 1.0)
        self.opt.step()

        bins = timestep_bins(t)
        bin_loss = torch.zeros(10, device=self.device)
        bin_count = torch.zeros(10, device=self.device)
        per = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3, 4))
        bin_loss.index_add_(0, bins, per.detach())
        bin_count.index_add_(0, bins, torch.ones_like(per))
        bin_mean = (bin_loss / bin_count.clamp_min(1)).tolist()

        rec = {"step": self.step, "loss": float(loss.detach()),
               "lr": self.opt.param_groups[0]["lr"],
               "grad_norm": float(grad_norm),
               "bin_loss": [round(v, 5) for v in bin_mean]}
        self.loss_history.append(rec["loss"])
        if self.logger:
            self.logger.log(rec)
        self.step += 1
        return rec

    # ------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict:
        return {
            "kind": "dit",
            "preset": self.cfg.name,
            "step": self.step,
            "dit": self.dit.state_dict(),
            "text_enc": self.text_enc.state_dict(),
            "opt": self.opt.state_dict(),
            "sampler": self.sampler.state_dict(),
            "loss_history": self.loss_history,
            "config": self.cfg.to_dict(),
        }

    def load_state_dict(self, payload: dict) -> None:
        assert payload["kind"] == "dit"
        self.dit.load_state_dict(payload["dit"])
        self.text_enc.load_state_dict(payload["text_enc"])
        self.opt.load_state_dict(payload["opt"])
        self.sampler.load_state_dict(payload["sampler"])
        self.step = payload["step"]
        self.loss_history = payload["loss_history"]

    def save(self, path: str) -> None:
        save_checkpoint(path, self.state_dict())

    @classmethod
    def resume(cls, path: str, **kwargs) -> "DiTTrainer":
        from vidgen.seed import set_rng_states

        payload = load_checkpoint(path)
        kwargs.pop("preset", None)  # preset always comes from the checkpoint
        tr = cls(preset=payload["preset"], **kwargs)  # fresh init reseeds RNG...
        tr.load_state_dict(payload)
        set_rng_states(payload["rng"])  # ...so restore the saved RNG state last
        return tr

    def close(self):
        if self.logger:
            self.logger.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-clips", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt-dir", default="checkpoints/tiny-dit")
    ap.add_argument("--ckpt-every", type=int, default=6)
    ap.add_argument("--log", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--vae-ckpt", default=None,
                    help="trained VAE checkpoint (from train_vae.py) for latents")
    ap.add_argument("--data-dir", default=None,
                    help="prepared real-clip dataset dir (manifest.jsonl + .npy clips); "
                         "when set, it replaces Moving MNIST")
    args = ap.parse_args()

    if args.resume:
        tr = DiTTrainer.resume(
            args.resume, seed=args.seed, lr=args.lr, batch_size=args.batch_size,
            num_clips=args.num_clips, device=args.device,
            ckpt_dir=args.ckpt_dir, log_path=args.log, vae_ckpt=args.vae_ckpt,
            data_dir=args.data_dir)
        print(f"resumed from {args.resume} at step {tr.step}")
    else:
        tr = DiTTrainer(preset=args.preset, seed=args.seed, lr=args.lr,
                        batch_size=args.batch_size, num_clips=args.num_clips,
                        device=args.device, ckpt_dir=args.ckpt_dir, log_path=args.log,
                        vae_ckpt=args.vae_ckpt, data_dir=args.data_dir)
    n_params = sum(p.numel() for p in tr.dit.parameters()) / 1e6
    print(f"preset={args.preset} dit_params={n_params:.1f}M "
          f"tokens={tr.cfg.num_tokens} device={tr.device}")

    for _ in range(args.steps):
        rec = tr.train_step()
        if tr.ckpt_dir and tr.step % args.ckpt_every == 0:
            path = os.path.join(tr.ckpt_dir, f"step_{tr.step:06d}.pt")
            tr.save(path)
            print(f"checkpoint -> {path}")
    if tr.ckpt_dir:
        tr.save(os.path.join(tr.ckpt_dir, "latest.pt"))
    tr.close()
    hist = tr.loss_history
    print(f"done: steps={tr.step} loss_first={hist[0]:.4f} loss_last={hist[-1]:.4f}")


if __name__ == "__main__":
    main()
