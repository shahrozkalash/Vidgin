"""Part 3: train the causal 3D VAE on Moving MNIST (Track A / Tiny).

Loss: reconstruction MSE + kl_weight * KL. Checkpointing mirrors train.py
(model + optimizer + sampler + RNG state) so resume is bit-reproducible.

Usage (CPU smoke):
  python -m vidgen.train_vae --steps 12 --batch-size 2 --num-clips 16 \\
      --ckpt-dir checkpoints/tiny-vae --log checkpoints/tiny-vae/train.jsonl

The trained VAE's encoder then supplies latents for DiT training
(python -m vidgen.train --vae-ckpt checkpoints/tiny-vae/latest.pt ...).
"""

from __future__ import annotations

import argparse
import os

import torch

from vidgen.checkpoint import load_checkpoint, save_checkpoint
from vidgen.config import get_preset, resolve_device
from vidgen.data.buckets import BucketSampler
from vidgen.data.clips import FolderClipDataset
from vidgen.data.moving_mnist import MovingMNISTDataset
from vidgen.logging import StructuredLogger
from vidgen.seed import seed_everything
from vidgen.vae.model import CausalVAE, vae_loss


class VAETrainer:
    def __init__(self, preset: str = "tiny", seed: int = 0, lr: float = 1e-4,
                 kl_weight: float = 1e-4, batch_size: int = 2, num_clips: int = 64,
                 device: str = "cpu", ckpt_dir: str | None = None,
                 log_path: str | None = None, data_dir: str | None = None):
        seed_everything(seed)
        self.seed = seed
        self.cfg = get_preset(preset)
        self.device = torch.device(resolve_device(device))
        self.batch_size = batch_size
        self.kl_weight = kl_weight
        self.ckpt_dir = ckpt_dir
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

        self.vae = CausalVAE().to(self.device)
        self.opt = torch.optim.AdamW(self.vae.parameters(), lr=lr)
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

    def train_step(self) -> dict:
        batch_idx = self.sampler.next_batch()
        clips = torch.stack([self.dataset[i][0] for i in batch_idx]).to(self.device)
        recon, mu, logvar, kl = self.vae(clips)
        loss, recon_loss, kl_term = vae_loss(recon, clips, kl, self.kl_weight)
        self.opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
        self.opt.step()
        rec = {"step": self.step, "loss": float(loss.detach()),
               "recon": float(recon_loss.detach()), "kl": float(kl_term.detach()),
               "lr": self.opt.param_groups[0]["lr"], "grad_norm": float(grad_norm)}
        self.loss_history.append(rec["loss"])
        if self.logger:
            self.logger.log(rec)
        self.step += 1
        return rec

    def state_dict(self) -> dict:
        return {
            "kind": "vae",
            "preset": self.cfg.name,
            "step": self.step,
            "vae": self.vae.state_dict(),
            "opt": self.opt.state_dict(),
            "sampler": self.sampler.state_dict(),
            "loss_history": self.loss_history,
            "config": self.cfg.to_dict(),
        }

    def load_state_dict(self, payload: dict) -> None:
        assert payload["kind"] == "vae"
        self.vae.load_state_dict(payload["vae"])
        self.opt.load_state_dict(payload["opt"])
        self.sampler.load_state_dict(payload["sampler"])
        self.step = payload["step"]
        self.loss_history = payload["loss_history"]

    def save(self, path: str) -> None:
        save_checkpoint(path, self.state_dict())

    @classmethod
    def resume(cls, path: str, **kwargs) -> "VAETrainer":
        from vidgen.seed import set_rng_states

        payload = load_checkpoint(path)
        kwargs.pop("preset", None)  # preset always comes from the checkpoint
        tr = cls(preset=payload["preset"], **kwargs)
        tr.load_state_dict(payload)
        set_rng_states(payload["rng"])  # restore saved RNG state after fresh init
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
    ap.add_argument("--kl-weight", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt-dir", default="checkpoints/tiny-vae")
    ap.add_argument("--ckpt-every", type=int, default=6)
    ap.add_argument("--log", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--data-dir", default=None,
                    help="prepared real-clip dataset dir (manifest.jsonl + .npy clips); "
                         "when set, it replaces Moving MNIST")
    args = ap.parse_args()

    if args.resume:
        tr = VAETrainer.resume(
            args.resume, seed=args.seed, lr=args.lr, kl_weight=args.kl_weight,
            batch_size=args.batch_size, num_clips=args.num_clips,
            device=args.device, ckpt_dir=args.ckpt_dir, log_path=args.log,
            data_dir=args.data_dir)
        print(f"resumed from {args.resume} at step {tr.step}")
    else:
        tr = VAETrainer(preset=args.preset, seed=args.seed, lr=args.lr,
                        kl_weight=args.kl_weight, batch_size=args.batch_size,
                        num_clips=args.num_clips, device=args.device,
                        ckpt_dir=args.ckpt_dir, log_path=args.log,
                        data_dir=args.data_dir)
    n_params = sum(p.numel() for p in tr.vae.parameters()) / 1e6
    print(f"preset={args.preset} vae_params={n_params:.2f}M device={tr.device}")

    for _ in range(args.steps):
        tr.train_step()
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
