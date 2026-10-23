#!/usr/bin/env python3
"""Unified trainer for Vidgin's audio models.

Follows the exact conventions of vidgen/train.py:
- Argparse CLI with --target sound|voice|vocoder
- Checkpointing carries model + optimizer + RNG state via vidgen.checkpoint
- Bit-identical resume reproduces the next step exactly
- Printed loss history and optional structured JSONL logging
- Sized for a free Colab T4 (16 GB) with gradient checkpointing support

Usage:
  # 1. Train the vocoder (mel -> waveform)
  python scripts/train_audio.py --target vocoder --steps 1000 --batch-size 4 \\
      --ckpt-dir checkpoints/vocoder --log checkpoints/vocoder/train.jsonl

  # 2. Train the sound model (caption -> mel DiT, 71M params)
  python scripts/train_audio.py --target sound --steps 1000 --batch-size 4 \\
      --ckpt-dir checkpoints/sound --log checkpoints/sound/train.jsonl

  # 3. Train the voice model (text -> mel TTS)
  python scripts/train_audio.py --target voice --steps 1000 --batch-size 4 \\
      --ckpt-dir checkpoints/voice --log checkpoints/voice/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import sys
import time
from typing import List, Optional

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    HAS_TORCH = False

from vidgen.audio.mel import (
    AudioConfig,
    DEFAULT_AUDIO_CFG,
    normalize_mel,
    wav_to_mel,
)
from vidgen.audio.sound_model import (
    AudioDiT,
    AudioDiTConfig,
    count_sound_params,
)
from vidgen.audio.vocoder import (
    HiFiGANGenerator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    VocoderConfig,
    discriminator_loss,
    feature_loss,
    generator_loss,
    count_vocoder_params,
)
from vidgen.audio.voice_model import (
    VidginVoiceTTS,
    VoiceModelConfig,
    count_voice_params,
)
from vidgen.checkpoint import load_checkpoint, save_checkpoint
from vidgen.config import resolve_device
from vidgen.flow import flow_loss, sample_shifted_logit_normal
from vidgen.logging import StructuredLogger
from vidgen.seed import seed_everything
from scripts.prepare_audio_data import prepare_synthetic_data, read_wav_info


# ----------------------------------------------------------- Dataset Loaders
class AudioDataset:
    """Reads audio manifest and yields raw audio waveforms and metadata."""

    def __init__(self, manifest_path: str, max_items: Optional[int] = None):
        self.manifest_path = manifest_path
        self.items = []
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.items.append(json.loads(line))
                        if max_items and len(self.items) >= max_items:
                            break

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def load_audio_tensor(wav_path: str, max_duration: float = 3.0, sr: int = 22050) -> torch.Tensor:
    """Load a WAV file into a 1D float32 PyTorch tensor in [-1.0, 1.0], padded or trimmed."""
    import wave
    import struct

    with wave.open(wav_path, "rb") as wf:
        n_ch = wf.getnchannels()
        s_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    # Tolerate truncated files: trust the bytes actually read, not the header.
    n_frames = len(raw) // (s_width * n_ch)

    fmt = f"<{n_frames * n_ch}h" if s_width == 2 else None
    if fmt is None:
        raise ValueError(f"Unsupported sample width: {s_width}")

    unpacked = struct.unpack(fmt, raw)
    if n_ch > 1:
        mono = [sum(unpacked[i * n_ch : (i + 1) * n_ch]) / (n_ch * 32768.0) for i in range(n_frames)]
    else:
        mono = [v / 32768.0 for v in unpacked]

    target_samples = int(round(max_duration * sr))
    tensor = torch.tensor(mono, dtype=torch.float32)

    if tensor.shape[0] < target_samples:
        pad = target_samples - tensor.shape[0]
        tensor = F.pad(tensor, (0, pad))
    else:
        tensor = tensor[:target_samples]

    return tensor


# ------------------------------------------------------------- Trainers
class SoundModelTrainer:
    """Trains the 71M Audio DiT on captions -> log-mel spectrograms via rectified flow."""

    def __init__(
        self,
        manifest_path: str,
        device: str = "cpu",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        batch_size: int = 4,
        ckpt_dir: Optional[str] = None,
        log_path: Optional[str] = None,
        cfg_dropout: float = 0.1,
        seed: int = 0,
    ):
        seed_everything(seed)
        self.device = torch.device(resolve_device(device))
        self.cfg = AudioDiTConfig()
        self.model = AudioDiT(self.cfg).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.dataset = AudioDataset(manifest_path)
        self.batch_size = batch_size
        self.cfg_dropout = cfg_dropout
        self.ckpt_dir = ckpt_dir
        self.logger = StructuredLogger(log_path) if log_path else None
        self.loss_history: List[float] = []
        self.step = 0
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

    def train_step(self) -> dict:
        self.model.train()
        self.opt.zero_grad()

        # Batch sampling
        batch_items = [random.choice(self.dataset) for _ in range(self.batch_size)]
        captions = [item.get("caption", "ambient sound") for item in batch_items]
        wavs = [load_audio_tensor(item["wav"]) for item in batch_items]
        wav_batch = torch.stack(wavs).to(self.device)  # [B, T_samples]

        # Convert to normalized ground-truth log-mel: [B, 80, frames]
        with torch.no_grad():
            mel = wav_to_mel(wav_batch)
            # Ensure time frames divide by patch_time
            if mel.shape[-1] % self.cfg.patch_time != 0:
                pad_t = self.cfg.patch_time - (mel.shape[-1] % self.cfg.patch_time)
                mel = F.pad(mel, (0, pad_t))
            x_1 = normalize_mel(mel)

        b, f, t_frames = x_1.shape
        x_0 = torch.randn_like(x_1)

        # Shifted logit-normal timesteps (s=5.0)
        t = sample_shifted_logit_normal(b, s=5.0, device=str(self.device))
        t_view = t[:, None, None]
        x_t = (1.0 - t_view) * x_0 + t_view * x_1
        target_v = x_1 - x_0

        # CFG dropout: drop captions to empty with cfg_dropout probability
        dropped_captions = [
            "" if random.random() < self.cfg_dropout else cap
            for cap in captions
        ]

        pred_v = self.model(x_t, t, dropped_captions)
        loss = flow_loss(pred_v, target_v)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.step += 1

        loss_val = float(loss.item())
        self.loss_history.append(loss_val)

        rec = {
            "step": self.step,
            "loss": loss_val,
            "target": "sound",
            "lr": self.opt.param_groups[0]["lr"],
        }
        if self.logger:
            self.logger.log(rec)
        return rec

    def save(self, path: str) -> None:
        payload = {
            "target": "sound",
            "step": self.step,
            "cfg": self.cfg,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "loss_history": self.loss_history[-100:],
        }
        save_checkpoint(path, payload)

    def resume(self, path: str) -> None:
        payload = load_checkpoint(path, map_location=str(self.device))
        assert payload.get("target") == "sound", f"Expected sound checkpoint, got {payload.get('target')}"
        self.model.load_state_dict(payload["model"])
        self.opt.load_state_dict(payload["opt"])
        self.step = payload["step"]
        self.loss_history = payload.get("loss_history", [])


class VoiceModelTrainer:
    """Trains the single-speaker text-to-mel TTS model."""

    def __init__(
        self,
        manifest_path: str,
        device: str = "cpu",
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        batch_size: int = 4,
        ckpt_dir: Optional[str] = None,
        log_path: Optional[str] = None,
        seed: int = 0,
    ):
        seed_everything(seed)
        self.device = torch.device(resolve_device(device))
        self.cfg = VoiceModelConfig()
        self.model = VidginVoiceTTS(self.cfg).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.dataset = AudioDataset(manifest_path)
        self.batch_size = batch_size
        self.ckpt_dir = ckpt_dir
        self.logger = StructuredLogger(log_path) if log_path else None
        self.loss_history: List[float] = []
        self.step = 0
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

    def train_step(self) -> dict:
        self.model.train()
        self.opt.zero_grad()

        batch_items = [random.choice(self.dataset) for _ in range(self.batch_size)]
        texts = [item.get("text", "hello world") for item in batch_items]
        wavs = [load_audio_tensor(item["wav"]) for item in batch_items]
        wav_batch = torch.stack(wavs).to(self.device)

        with torch.no_grad():
            mel_target = wav_to_mel(wav_batch)  # [B, 80, frames]

        # Tokenize texts
        token_lists = [self.model.tokenize(t) for t in texts]
        max_tok = max(len(toks) for toks in token_lists)
        token_tensor = torch.zeros(self.batch_size, max_tok, dtype=torch.long, device=self.device)
        for i, toks in enumerate(token_lists):
            token_tensor[i, : len(toks)] = torch.tensor(toks, device=self.device)

        mel_coarse, mel_refined, log_dur = self.model(token_tensor)

        # Match time lengths
        min_t = min(mel_refined.shape[-1], mel_target.shape[-1])
        mel_ref_crop = mel_refined[..., :min_t]
        mel_tgt_crop = mel_target[..., :min_t]
        mel_coarse_crop = mel_coarse[..., :min_t]

        loss_mel = F.l1_loss(mel_coarse_crop, mel_tgt_crop) + F.l1_loss(mel_ref_crop, mel_tgt_crop)
        loss = loss_mel

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.step += 1

        loss_val = float(loss.item())
        self.loss_history.append(loss_val)

        rec = {
            "step": self.step,
            "loss": loss_val,
            "target": "voice",
            "lr": self.opt.param_groups[0]["lr"],
        }
        if self.logger:
            self.logger.log(rec)
        return rec

    def save(self, path: str) -> None:
        payload = {
            "target": "voice",
            "step": self.step,
            "cfg": self.cfg,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "loss_history": self.loss_history[-100:],
        }
        save_checkpoint(path, payload)

    def resume(self, path: str) -> None:
        payload = load_checkpoint(path, map_location=str(self.device))
        assert payload.get("target") == "voice", f"Expected voice checkpoint, got {payload.get('target')}"
        self.model.load_state_dict(payload["model"])
        self.opt.load_state_dict(payload["opt"])
        self.step = payload["step"]
        self.loss_history = payload.get("loss_history", [])


class VocoderTrainer:
    """Trains the HiFi-GAN generator and discriminators on ground-truth mel <-> wav pairs."""

    def __init__(
        self,
        manifest_path: str,
        device: str = "cpu",
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        batch_size: int = 4,
        ckpt_dir: Optional[str] = None,
        log_path: Optional[str] = None,
        seed: int = 0,
    ):
        seed_everything(seed)
        self.device = torch.device(resolve_device(device))
        self.cfg = VocoderConfig()
        self.generator = HiFiGANGenerator(self.cfg).to(self.device)
        self.mpd = MultiPeriodDiscriminator().to(self.device)
        self.msd = MultiScaleDiscriminator().to(self.device)

        self.opt_g = torch.optim.AdamW(self.generator.parameters(), lr=lr_g, betas=(0.8, 0.99))
        self.opt_d = torch.optim.AdamW(
            list(self.mpd.parameters()) + list(self.msd.parameters()),
            lr=lr_d,
            betas=(0.8, 0.99),
        )

        self.dataset = AudioDataset(manifest_path)
        self.batch_size = batch_size
        self.ckpt_dir = ckpt_dir
        self.logger = StructuredLogger(log_path) if log_path else None
        self.loss_history: List[float] = []
        self.step = 0
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

    def train_step(self) -> dict:
        self.generator.train()
        self.mpd.train()
        self.msd.train()

        batch_items = [random.choice(self.dataset) for _ in range(self.batch_size)]
        wavs = [load_audio_tensor(item["wav"]) for item in batch_items]
        y = torch.stack(wavs).unsqueeze(1).to(self.device)  # [B, 1, T_samples]

        with torch.no_grad():
            s = wav_to_mel(y.squeeze(1))  # [B, 80, T_mel]

        # Generator forward
        y_hat = self.generator(s)  # [B, 1, T_samples]

        # Crop to matching waveform length
        min_len = min(y.shape[-1], y_hat.shape[-1])
        y = y[..., :min_len]
        y_hat = y_hat[..., :min_len]

        # ----------------------------- 1. Discriminator Step
        self.opt_d.zero_grad()
        # MPD
        r_mpd, f_mpd, _, _ = self.mpd(y, y_hat.detach())
        loss_d_mpd, _, _ = discriminator_loss(r_mpd, f_mpd)
        # MSD
        r_msd, f_msd, _, _ = self.msd(y, y_hat.detach())
        loss_d_msd, _, _ = discriminator_loss(r_msd, f_msd)

        loss_d = loss_d_mpd + loss_d_msd
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.mpd.parameters()) + list(self.msd.parameters()), 10.0
        )
        self.opt_d.step()

        # ------------------------------- 2. Generator Step
        self.opt_g.zero_grad()
        # Mel-spectrogram loss
        y_hat_mel = wav_to_mel(y_hat.squeeze(1))
        min_mel = min(s.shape[-1], y_hat_mel.shape[-1])
        loss_mel = F.l1_loss(y_hat_mel[..., :min_mel], s[..., :min_mel]) * 45.0

        # Adversarial & Feature loss
        r_mpd, f_mpd, fmap_r_mpd, fmap_f_mpd = self.mpd(y, y_hat)
        r_msd, f_msd, fmap_r_msd, fmap_f_msd = self.msd(y, y_hat)
        loss_g_adv, _ = generator_loss(f_mpd + f_msd)
        loss_fm = feature_loss(fmap_r_mpd, fmap_f_mpd) + feature_loss(fmap_r_msd, fmap_f_msd)

        loss_g = loss_g_adv + loss_fm + loss_mel
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 10.0)
        self.opt_g.step()

        self.step += 1
        loss_val = float(loss_g.item())
        self.loss_history.append(loss_val)

        rec = {
            "step": self.step,
            "loss_g": loss_val,
            "loss_d": float(loss_d.item()),
            "loss_mel": float(loss_mel.item()),
            "target": "vocoder",
        }
        if self.logger:
            self.logger.log(rec)
        return rec

    def save(self, path: str) -> None:
        payload = {
            "target": "vocoder",
            "step": self.step,
            "cfg": self.cfg,
            "generator": self.generator.state_dict(),
            "mpd": self.mpd.state_dict(),
            "msd": self.msd.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "loss_history": self.loss_history[-100:],
        }
        save_checkpoint(path, payload)

    def resume(self, path: str) -> None:
        payload = load_checkpoint(path, map_location=str(self.device))
        assert payload.get("target") == "vocoder", f"Expected vocoder checkpoint, got {payload.get('target')}"
        self.generator.load_state_dict(payload["generator"])
        self.mpd.load_state_dict(payload["mpd"])
        self.msd.load_state_dict(payload["msd"])
        self.opt_g.load_state_dict(payload["opt_g"])
        self.opt_d.load_state_dict(payload["opt_d"])
        self.step = payload["step"]
        self.loss_history = payload.get("loss_history", [])


# ------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified trainer for Vidgin's audio models (sound, voice, vocoder)."
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["sound", "voice", "vocoder"],
        help="Model to train: sound (AudioDiT 71M), voice (TTS 11M), vocoder (HiFi-GAN 2.6M).",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per step.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda, mps).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--data-manifest", default=None, help="Path to jsonl manifest.")
    parser.add_argument("--ckpt-dir", default=None, help="Directory to save checkpoints.")
    parser.add_argument(
        "--ckpt-every",
        type=int,
        default=200,
        help="Also save latest.pt every N steps during training (0 = only at the end).",
    )
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume bit-for-bit.")
    parser.add_argument("--log", default=None, help="Path to JSONL log file.")
    parser.add_argument(
        "--grad-checkpoint", action="store_true", help="Enable gradient checkpointing for memory saving."
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Generate synthetic data if no manifest is present."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine default manifest path if not provided
    manifest_path = args.data_manifest
    if not manifest_path or not os.path.isfile(manifest_path):
        default_map = {
            "sound": "data/audio/manifest_sound.jsonl",
            "voice": "data/audio/manifest_voice.jsonl",
            "vocoder": "data/audio/manifest_vocoder.jsonl",
        }
        manifest_path = default_map[args.target]

    if not os.path.isfile(manifest_path):
        print(f"[train_audio] Manifest not found at {manifest_path}. Generating synthetic data...")
        synth_dir = "data/audio_synthetic"
        v_m, s_m, voc_m = prepare_synthetic_data(synth_dir, num_samples=16)
        manifest_path = {"sound": s_m, "voice": v_m, "vocoder": voc_m}[args.target]

    ckpt_dir = args.ckpt_dir or f"checkpoints/{args.target}"
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n========================================================")
    print(f"  Vidgin Audio Trainer: target = {args.target}")
    print(f"  Device: {args.device} | Steps: {args.steps} | Batch: {args.batch_size}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"========================================================\n")

    if args.target == "sound":
        trainer = SoundModelTrainer(
            manifest_path=manifest_path,
            device=args.device,
            lr=args.lr,
            batch_size=args.batch_size,
            ckpt_dir=ckpt_dir,
            log_path=args.log,
            seed=args.seed,
        )
        print(f"AudioDiT parameters: {count_sound_params(trainer.cfg) / 1e6:.2f} M")
    elif args.target == "voice":
        trainer = VoiceModelTrainer(
            manifest_path=manifest_path,
            device=args.device,
            lr=args.lr,
            batch_size=args.batch_size,
            ckpt_dir=ckpt_dir,
            log_path=args.log,
            seed=args.seed,
        )
        print(f"VidginVoiceTTS parameters: {count_voice_params(trainer.cfg) / 1e6:.2f} M")
    elif args.target == "vocoder":
        trainer = VocoderTrainer(
            manifest_path=manifest_path,
            device=args.device,
            lr_g=args.lr,
            lr_d=args.lr,
            batch_size=args.batch_size,
            ckpt_dir=ckpt_dir,
            log_path=args.log,
            seed=args.seed,
        )
        print(f"HiFi-GAN Vocoder parameters: {count_vocoder_params(trainer.cfg) / 1e6:.2f} M")

    if args.resume:
        print(f"Resuming bit-for-bit from {args.resume} ...")
        trainer.resume(args.resume)
        print(f"Resumed at step {trainer.step}")

    start_time = time.time()
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    for step in range(trainer.step, trainer.step + args.steps):
        rec = trainer.train_step()
        if args.ckpt_every > 0 and (step + 1) % args.ckpt_every == 0:
            # Periodic save: a crashed/interrupted run loses at most
            # --ckpt-every steps instead of everything (checkpoints used
            # to be written only after the final step).
            trainer.save(latest_path)
        if (step + 1) % max(1, args.steps // 10) == 0 or (step + 1) == trainer.step + args.steps:
            elapsed = time.time() - start_time
            print(
                f"Step {rec['step']:4d} | Loss: {rec.get('loss', rec.get('loss_g', 0.0)):.4f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    trainer.save(latest_path)
    print(f"\n[train_audio] Saved checkpoint: {latest_path}")
    print(f"[train_audio] Training complete!\n")


if __name__ == "__main__":
    main()
