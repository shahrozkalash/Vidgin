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

Multi-GPU (DDP, opt-in via torchrun; --batch-size is PER GPU there):
  torchrun --standalone --nproc_per_node=2 -m vidgen.train --preset small \\
      --depth --grad-checkpoint --batch-size 2 --device cuda ...

Weight-averaging pool worker (one round; see PYTHON_IMPLEMENTATION.md):
  python -m vidgen.train --preset small --init-weights pooled.pt \\
      --data-shard 2/4 --steps 500 --ckpt-dir ckpts/worker2 ...
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

from vidgen.checkpoint import load_checkpoint, save_checkpoint
from vidgen.config import get_preset, resolve_device
from vidgen.depth import depth_to_latent_grid
from vidgen.distributed import (
    DistInfo,
    barrier,
    destroy_distributed,
    init_distributed,
    plain_state_dict,
    rank_seed,
    resolve_rank_device,
    shard_batch,
    sync_gradients,
    unwrap_model,
    wrap_ddp,
)
from vidgen.data.buckets import BucketSampler
from vidgen.data.clips import FolderClipDataset
from vidgen.data.moving_mnist import MovingMNISTDataset
from vidgen.flow import flow_loss, rectified_flow_targets, sample_shifted_logit_normal, timestep_bins
from vidgen.logging import StructuredLogger
from vidgen.pool import parse_data_shard
from vidgen.seed import seed_everything
from vidgen.text.encoder import build_text_encoder
from vidgen.text.tokenizer import CaptionTokenizer, load_tokenizer, tokenizer_from_dict
from vidgen.train_utils import (
    amp_context, build_lr_scheduler, make_grad_scaler, validate_train_options,
)
from vidgen.dit.model import DiT
from vidgen.vae.model import CausalVAE


class DiTTrainer:
    def __init__(self, preset: str = "tiny", seed: int = 0, lr: float = 1e-4,
                 weight_decay: float = 0.01, batch_size: int = 2, num_clips: int = 64,
                 device: str = "cpu", ckpt_dir: str | None = None,
                 log_path: str | None = None, cfg_dropout: float = 0.1,
                 vae_ckpt: str | None = None, data_dir: str | None = None,
                 grad_checkpoint: bool = False,
                 use_depth: bool = False, depth_dropout: float = 0.1,
                 data_shard: tuple[int, int] | None = None,
                 data_split: str | None = None,
                 tokenizer: CaptionTokenizer | None = None,
                 latent_scale: float | None = None,
                 warmup_steps: int | None = None,
                 lr_schedule: str | None = None,
                 total_steps: int | None = None,
                 grad_accum: int | None = None,
                 amp: str | None = None,
                 dist_info: DistInfo | None = None):
        # Distributed (torchrun) is opt-in: with no torchrun environment
        # this is the single-process info and every path below is exactly
        # the original single-GPU one.
        if dist_info is None:
            dist_info = init_distributed(device)
        self.dist_info = dist_info
        self.rank = dist_info.rank
        self.world_size = dist_info.world_size
        self.is_main_rank = dist_info.is_main
        seed_everything(seed)
        self.seed = seed
        self.cfg = get_preset(preset)
        if use_depth:
            # Dawood's 3D idea (vidgen/depth.py): the DiT grows a depth
            # stem and learns from each clip's 3D map. Recorded in every
            # checkpoint's config; resume re-derives it from there.
            self.cfg.use_depth = True
        self.depth_dropout = depth_dropout
        self.device = resolve_rank_device(resolve_device(device), dist_info)
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
        # Phase 2: a supplied CaptionTokenizer switches the run into
        # caption-tokenizer mode; the tokenizer is stored in every
        # checkpoint so resume/inference rebuild it exactly.
        self.tokenizer = tokenizer
        if tokenizer is not None:
            self.cfg.text_tokenizer = "caption"
        # Phase 3: latent_scale comes from MEASURING the trained VAE on
        # the target dataset (scripts/measure_latents.py), never a
        # hand-wave. Stored in the checkpoint config, so sampling and
        # continuation inherit it automatically.
        if latent_scale is not None:
            if latent_scale <= 0:
                raise ValueError(f"latent_scale must be positive, got {latent_scale}")
            self.cfg.latent_scale = float(latent_scale)
        self.text_enc = build_text_encoder(self.cfg, tokenizer=tokenizer).to(self.device)
        self.dit = DiT(self.cfg).to(self.device)
        if grad_checkpoint:
            self.dit.set_grad_checkpoint(True)
        params = list(self.dit.parameters()) + list(self.text_enc.parameters())
        self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        # Phase 4/5: LR schedule, AMP, gradient accumulation. Every
        # option defaults to the historical behaviour (constant LR,
        # fp32, accum 1); the resolved set is recorded in each
        # checkpoint's train_config and restored on resume.
        self.grad_accum = 1 if grad_accum is None else int(grad_accum)
        self.amp = "none" if amp is None else amp
        validate_train_options(self.grad_accum, self.amp)
        self.warmup_steps = 0 if warmup_steps is None else int(warmup_steps)
        self.lr_schedule = "none" if lr_schedule is None else lr_schedule
        self.total_steps = total_steps
        self.sched = build_lr_scheduler(
            self.opt, warmup_steps=self.warmup_steps,
            total_steps=self.total_steps, schedule=self.lr_schedule)
        self.scaler = make_grad_scaler(self.device, self.amp)
        self.train_config = {
            "warmup_steps": self.warmup_steps,
            "lr_schedule": self.lr_schedule,
            "total_steps": self.total_steps,
            "grad_accum": self.grad_accum,
            "amp": self.amp,
        }

        # Optional pool worker shard (vidgen/pool.py): this worker trains
        # only on its disjoint slice of the dataset. None = full dataset,
        # the original behavior exactly.
        self.data_shard = data_shard
        self.data_split = data_split
        if data_dir:
            manifest_name = (f"manifest_{data_split}.jsonl" if data_split
                             else "manifest.jsonl")
            self.dataset = FolderClipDataset(
                data_dir, height=self.cfg.height, width=self.cfg.width,
                frames=self.cfg.frames, shard=data_shard,
                manifest_name=manifest_name)
            num_items = len(self.dataset)
        else:
            self.dataset = MovingMNISTDataset(
                num_clips, height=self.cfg.height, width=self.cfg.width,
                frames=self.cfg.frames, base_seed=seed + 1, shard=data_shard)
            num_items = len(self.dataset)
        # Distributed data sharding: ONE sampler, identically seeded on
        # every rank, draws global batches of batch_size * world_size;
        # each rank trains on its own disjoint slice (see _next_batch_idx).
        # The sampler state therefore stays identical across ranks and
        # rank 0's checkpoint carries the complete, authoritative state.
        # Single-process: world_size == 1, so this is the original sampler.
        self.global_batch_size = batch_size * dist_info.world_size
        self.sampler = BucketSampler(num_items, self.global_batch_size,
                                     seed=seed + 2)

        self.step = 0
        self.logger = (StructuredLogger(log_path)
                       if log_path and self.is_main_rank else None)
        self.loss_history: list[float] = []
        self.init_from: str | None = None

        # DDP wrap LAST: the optimizer, checkpoint loading and warm paths
        # all reference the raw modules' parameters (identical objects
        # after wrapping). Activation checkpointing was set on the raw
        # DiT above and uses the non-reentrant implementation, which is
        # the DDP-compatible variant. The frozen VAE is never wrapped.
        # The TEXT ENCODER is deliberately NOT wrapped: DDP only allows
        # forward() through the wrapper, but train_step also calls
        # text_enc.null_embedding(). It stays replicated and its
        # gradients are all-reduced explicitly in train_step
        # (vidgen.distributed.sync_gradients) — the same averaged
        # gradient DDP would have produced.
        self.dit = wrap_ddp(self.dit, dist_info, self.device)
        if dist_info.is_distributed:
            # Model weights were built identically on all ranks from the
            # base seed above; NOW diverge the stochastic streams (flow
            # noise, timesteps, caption/depth dropout) per rank, derived
            # deterministically from (seed, rank) so a resume with the
            # same world size reproduces the same streams.
            seed_everything(rank_seed(seed, dist_info.rank))

    # ------------------------------------------------------------ data
    def _next_batch_idx(self) -> list[int]:
        """This rank's batch indices for the next step.

        Single-process: the sampler's batch, unchanged. Distributed: this
        rank's disjoint slice of the shared global batch.
        """
        global_batch = self.sampler.next_batch()
        if not self.dist_info.is_distributed:
            return global_batch
        return shard_batch(global_batch, self.batch_size, self.rank)

    # ------------------------------------------------------------ one step
    def train_step(self) -> dict:
        import time

        t0 = time.perf_counter()
        self.opt.zero_grad()
        total_loss = 0.0
        bin_mean = [0.0] * 10
        last_batch = 0
        for _micro in range(self.grad_accum):
            batch_idx = self._next_batch_idx()
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
            last_batch = b
            t = sample_shifted_logit_normal((b,), s=5.0, device=str(self.device))
            xt, target = rectified_flow_targets(x1, t)

            with amp_context(self.device, self.amp):
                text, text_mask = self.text_enc(captions)
                # 10% caption dropout -> CFG null embedding (torch RNG: checkpointed)
                drop = (torch.rand(b, device=self.device) < self.cfg_dropout)
                null_emb, null_mask = self.text_enc.null_embedding()
                null_emb = null_emb.expand(b, -1, -1)
                null_mask = null_mask.expand(b, -1)
                text = torch.where(drop.view(b, 1, 1), null_emb, text)
                text_mask = torch.where(drop.view(b, 1), null_mask, text_mask)

                depth = self._depth_batch(batch_idx) if self.cfg.use_depth else None
                pred = self.dit(xt, t, text, text_mask, depth=depth)
                loss = flow_loss(pred, target)

            scaled = loss / self.grad_accum
            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()
            total_loss += float(loss.detach())

            if _micro == self.grad_accum - 1:
                bins = timestep_bins(t)
                bin_loss = torch.zeros(10, device=self.device)
                bin_count = torch.zeros(10, device=self.device)
                per = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3, 4))
                bin_loss.index_add_(0, bins, per.detach())
                bin_count.index_add_(0, bins, torch.ones_like(per))
                bin_mean = (bin_loss / bin_count.clamp_min(1)).tolist()

        # The DiT's gradients were all-reduced by DDP during backward;
        # the (unwrapped) text encoder's are synchronized explicitly so
        # every rank clips and steps from identical gradients. With
        # accumulation the sync happens once, on the summed gradients —
        # all-reduce is linear, so this equals per-micro syncing.
        sync_gradients(self.text_enc, self.dist_info)
        if self.scaler is not None:
            self.scaler.unscale_(self.opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.dit.parameters()) + list(self.text_enc.parameters()), 1.0)
        if self.scaler is not None:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()
        if self.sched is not None:
            self.sched.step()

        secs = time.perf_counter() - t0
        samples = last_batch * self.world_size * self.grad_accum
        rec = {"step": self.step, "loss": total_loss / self.grad_accum,
               "lr": self.opt.param_groups[0]["lr"],
               "grad_norm": float(grad_norm),
               "bin_loss": [round(v, 5) for v in bin_mean],
               "secs": round(secs, 3),
               "clips_per_sec": round(samples / max(secs, 1e-9), 2)}
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            rec["gpu_mem_mb"] = round(
                torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        self.loss_history.append(rec["loss"])
        if self.logger:
            self.logger.log(rec)
        self.step += 1
        return rec

    # ------------------------------------------------------------ 3D depth
    def _depth_batch(self, batch_idx) -> torch.Tensor:
        """Assemble this batch's depth maps on the latent grid.

        Clips without a map (and datasets that have none, like Moving
        MNIST) contribute an all-zeros map — the same "no map" condition
        the sampler uses. Whole-batch map dropout (``depth_dropout``,
        drawn from the checkpointed RNG like caption dropout) keeps the
        model honest at imagining geometry from text alone, because at
        text-to-video time no map exists unless the user supplies one.
        """
        get = getattr(self.dataset, "get_depth", None)
        items = []
        for i in batch_idx:
            d = get(i) if get is not None else None
            if d is None:
                d = torch.zeros(1, self.cfg.frames, self.cfg.height,
                                self.cfg.width)
            items.append(d)
        depth = torch.stack(items).to(self.device)      # [B,1,T,H,W]
        if self.depth_dropout > 0:
            hide = float(torch.rand((), device=self.device)) < self.depth_dropout
            if hide:
                depth = torch.zeros_like(depth)
        return depth_to_latent_grid(depth, self.cfg.latent_t,
                                    self.cfg.latent_h, self.cfg.latent_w)

    # ------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict:
        payload = {
            "kind": "dit",
            "preset": self.cfg.name,
            "step": self.step,
            "dit": plain_state_dict(self.dit),
            "text_enc": plain_state_dict(self.text_enc),
            "opt": self.opt.state_dict(),
            "sampler": self.sampler.state_dict(),
            "loss_history": self.loss_history,
            "config": self.cfg.to_dict(),
        }
        if self.tokenizer is not None:
            payload["tokenizer"] = self.tokenizer.to_dict()
        payload["train_config"] = dict(self.train_config)
        if self.sched is not None:
            payload["sched"] = self.sched.state_dict()
        if self.scaler is not None:
            payload["scaler"] = self.scaler.state_dict()
        return payload

    def load_state_dict(self, payload: dict) -> None:
        assert payload["kind"] == "dit"
        # Loads go into the UNWRAPPED modules: checkpoint keys are plain
        # (no "module." prefix) whether or not this rank runs under DDP.
        unwrap_model(self.dit).load_state_dict(payload["dit"])
        unwrap_model(self.text_enc).load_state_dict(payload["text_enc"])
        self.opt.load_state_dict(payload["opt"])
        if "sched" in payload:
            if self.sched is None:
                raise ValueError(
                    "checkpoint carries LR-scheduler state but this "
                    "trainer was built without the run's schedule; "
                    "resume via DiTTrainer.resume so the schedule is "
                    "restored from train_config")
            self.sched.load_state_dict(payload["sched"])
        if "scaler" in payload and self.scaler is not None:
            self.scaler.load_state_dict(payload["scaler"])
        self.sampler.load_state_dict(payload["sampler"])
        self.step = payload["step"]
        self.loss_history = payload["loss_history"]

    def load_dit_weights(self, path: str,
                         start_step: int | None = None) -> None:
        """Weights-only init from a donor DiT checkpoint.

        Mirrors VAETrainer.load_vae_weights: copies just the trained
        DiT + text-encoder weights into this fresh trainer. Optimizer,
        sampler, loss history and RNG stay fresh — this begins a new run,
        it does not resume one.

        Step offset: an explicit ``start_step`` wins. Otherwise, if the
        donor is a pooled checkpoint (it carries ``pool`` metadata from
        scripts/average_checkpoints.py), training continues numbering
        from the donor's stored step — pool rounds are numbered
        cumulatively across rounds. A plain donor checkpoint starts at
        step 0, exactly like a warm start.
        """
        payload = load_checkpoint(path)  # restores the donor's RNG state...
        # ...which a fresh run must not keep (mirrors the VAE trainer:
        # rank_seed(seed, 0) == seed, so single-process is unchanged).
        seed_everything(rank_seed(self.seed, self.rank))
        if payload.get("kind") != "dit" or "dit" not in payload \
                or "text_enc" not in payload:
            raise ValueError(
                f"{path} is not a DiT checkpoint (kind={payload.get('kind')!r})")
        # Tokenizer compatibility: a caption-tokenizer donor and a
        # hash-mode (or differently-vocabularied) recipient must fail
        # loudly here, not as a cryptic strict-load shape error.
        donor_tok = payload.get("tokenizer")
        if donor_tok is not None and self.tokenizer is None:
            raise ValueError(
                f"{path} was trained with a caption tokenizer "
                f"({len(donor_tok['words'])} words); rebuild this trainer "
                "with the same tokenizer (--tokenizer-vocab) before "
                "loading its weights")
        if donor_tok is None and self.tokenizer is not None:
            raise ValueError(
                f"{path} was trained in legacy hash mode but this "
                "trainer uses a caption tokenizer; the two tokenizations "
                "are incompatible")
        if donor_tok is not None and self.tokenizer is not None \
                and tokenizer_from_dict(donor_tok) != self.tokenizer:
            raise ValueError(
                f"{path} was trained with a DIFFERENT caption tokenizer "
                "vocabulary; token ids would not mean the same words")
        # Strict loads into the UNWRAPPED modules (plain keys either way).
        unwrap_model(self.dit).load_state_dict(payload["dit"])
        self.text_enc.load_state_dict(payload["text_enc"])
        self.init_from = path
        if start_step is not None:
            self.step = int(start_step)
        elif "pool" in payload:
            self.step = int(payload["step"])
        else:
            self.step = 0

    def save(self, path: str) -> None:
        # Only rank 0 writes checkpoints (all ranks hold identical
        # weights thanks to DDP gradient all-reduce; the checkpoint is
        # built from the unwrapped models, so it stays plain).
        if not self.is_main_rank:
            return
        save_checkpoint(path, self.state_dict())

    @classmethod
    def resume(cls, path: str, **kwargs) -> "DiTTrainer":
        from vidgen.seed import set_rng_states

        payload = load_checkpoint(path)
        kwargs.pop("preset", None)  # preset always comes from the checkpoint
        # Depth mode too: a depth checkpoint needs the depth stem built
        # BEFORE its weights load, and a plain checkpoint must not grow
        # one. The checkpoint's stored config is the source of truth.
        stored_cfg = payload.get("config") or {}
        kwargs["use_depth"] = bool(stored_cfg.get("use_depth", False))
        # Tokenizer too: a caption-mode checkpoint carries its exact
        # vocabulary; the text encoder must be built with it BEFORE the
        # weights load. A conflicting --tokenizer-vocab is an error.
        if payload.get("tokenizer") is not None:
            stored_tok = tokenizer_from_dict(payload["tokenizer"])
            given_tok = kwargs.get("tokenizer")
            if given_tok is not None and given_tok != stored_tok:
                raise ValueError(
                    "the --tokenizer-vocab vocabulary differs from the "
                    "tokenizer stored in the checkpoint; resume uses the "
                    "checkpoint's own tokenizer")
            kwargs["tokenizer"] = stored_tok
        elif stored_cfg.get("text_tokenizer") == "caption":
            raise ValueError(
                "checkpoint config is caption-tokenizer mode but the "
                "checkpoint carries no 'tokenizer' payload; it is "
                "incomplete and cannot be resumed safely")
        cli_scale = kwargs.pop("latent_scale", None)
        # Training options (schedule/AMP/accumulation) are part of the
        # run's identity: restore them from the checkpoint's
        # train_config unless the caller explicitly overrode them.
        stored_tc = payload.get("train_config") or {}
        for key in ("warmup_steps", "lr_schedule", "total_steps",
                    "grad_accum", "amp"):
            if kwargs.get(key) is None and key in stored_tc:
                kwargs[key] = stored_tc[key]
        tr = cls(preset=payload["preset"], **kwargs)  # fresh init reseeds RNG...
        # The checkpoint's latent_scale is authoritative on resume (the
        # preset default must not silently change the latent geometry);
        # an explicit CLI --latent-scale wins, for deliberate re-scaling.
        if cli_scale is not None:
            tr.cfg.latent_scale = float(cli_scale)
        elif stored_cfg.get("latent_scale") is not None:
            tr.cfg.latent_scale = float(stored_cfg["latent_scale"])
        tr.load_state_dict(payload)
        set_rng_states(payload["rng"])  # ...so restore the saved RNG state last
        if tr.dist_info.is_distributed:
            # The saved RNG is rank 0's stream; restoring it verbatim on
            # every rank would make all ranks draw identical noise. In
            # distributed mode each rank instead re-derives its own
            # stream from (seed, rank) — deterministic for a given
            # checkpoint + world size + --seed. (Single-GPU resume of
            # the same checkpoint still uses the saved RNG bit-for-bit.)
            seed_everything(rank_seed(tr.seed, tr.rank))
        return tr

    def close(self):
        if self.logger:
            self.logger.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2,
                    help="clips per step PER GPU: under torchrun the global "
                         "batch is batch-size x world_size (e.g. 2 x 2 T4s "
                         "= global batch 4). Single-GPU runs are unaffected.")
    ap.add_argument("--num-clips", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt-dir", default="checkpoints/tiny-dit")
    ap.add_argument("--ckpt-every", type=int, default=6)
    ap.add_argument("--log", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-weights", default=None, metavar="PATH",
                    help="weights-only init: initialize the DiT + text "
                         "encoder from a donor DiT checkpoint and train as "
                         "a fresh run (fresh optimizer/RNG). This is the "
                         "pool worker's round start: if the donor is a "
                         "pooled checkpoint, step numbering continues from "
                         "its stored step. Mutually exclusive with --resume.")
    ap.add_argument("--start-step", type=int, default=None,
                    help="with --init-weights, override the starting step "
                         "number (default: the pooled checkpoint's stored "
                         "step, or 0 for a plain donor checkpoint)")
    ap.add_argument("--data-shard", default=None, metavar="K/N",
                    type=parse_data_shard,
                    help="pool worker data shard, e.g. 2/4: train only on "
                         "this worker's disjoint slice of the dataset "
                         "(entries ordered deterministically, worker K "
                         "takes positions %% N == K-1). Default: the "
                         "full dataset, as before.")
    ap.add_argument("--vae-ckpt", default=None,
                    help="trained VAE checkpoint (from train_vae.py) for latents")
    ap.add_argument("--data-dir", default=None,
                    help="prepared real-clip dataset dir (manifest.jsonl + .npy clips); "
                         "when set, it replaces Moving MNIST")
    ap.add_argument("--data-split", default=None,
                    choices=["train", "val", "test"],
                    help="train on one deterministic split written by "
                         "vidgin_helper.py split (manifest_<split>.jsonl "
                         "inside --data-dir); default: the full manifest")
    ap.add_argument("--require-real", action="store_true",
                    help="fail loudly unless --data-dir names a real-clip "
                         "dataset: never silently train on synthetic "
                         "Moving MNIST when real training was requested")
    ap.add_argument("--tokenizer-vocab", default=None, metavar="PATH",
                    help="CaptionTokenizer JSON built by "
                         "scripts/build_tokenizer.py from the dataset's "
                         "own captions; switches text conditioning to the "
                         "caption-trained vocabulary (stored inside every "
                         "checkpoint). Default: legacy hash/table mode.")
    ap.add_argument("--latent-scale", type=float, default=None,
                    help="latent normalizer applied to VAE mus before DiT "
                         "training; use the value measured by "
                         "scripts/measure_latents.py (1 / latent std). "
                         "Default: the preset's value; stored in the "
                         "checkpoint config either way.")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="activation checkpointing for the DiT (needed to "
                         "train Small and larger presets in limited memory)")
    ap.add_argument("--depth", action="store_true",
                    help="train with 3D depth-map conditioning (Dawood's "
                         "idea, vidgen/depth.py): each clip's depth map "
                         "(prepared with prepare_clips.py --depth) is fed "
                         "to the DiT. On --resume the mode comes from the "
                         "checkpoint, not this flag.")
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="linear LR warmup over this many steps (default 0)")
    ap.add_argument("--lr-schedule", default=None,
                    choices=["none", "cosine", "linear"],
                    help="LR decay schedule after warmup; cosine/linear "
                         "decay over the run's --steps (default: none = "
                         "constant LR). On --resume the schedule stored "
                         "in the checkpoint wins unless overridden here.")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="accumulate this many micro-batches per optimizer "
                         "step (default 1); raises the effective batch "
                         "without raising peak memory")
    ap.add_argument("--amp", default=None, choices=["none", "bf16", "fp16"],
                    help="mixed precision via autocast (default none): "
                         "bf16 suits modern GPUs, fp16 adds a GradScaler "
                         "on CUDA")
    ap.add_argument("--depth-dropout", type=float, default=0.1,
                    help="probability of hiding a whole batch's depth maps "
                         "so text-only generation stays strong")
    args = ap.parse_args()

    if args.resume and args.init_weights:
        ap.error("--resume and --init-weights are mutually exclusive: "
                 "resume continues a run, weights-only init begins a "
                 "fresh one (e.g. a pool round)")
    if args.start_step is not None and not args.init_weights:
        ap.error("--start-step only applies with --init-weights")
    if args.require_real and not args.data_dir:
        ap.error("--require-real needs --data-dir: refusing to fall back "
                 "to synthetic Moving MNIST for a real training run")
    if args.data_split and not args.data_dir:
        ap.error("--data-split needs --data-dir (splits live in the "
                 "real-clip dataset directory)")
    cli_tokenizer = (load_tokenizer(args.tokenizer_vocab)
                     if args.tokenizer_vocab else None)
    # Under torchrun this initializes the process group (nccl on CUDA,
    # gloo on CPU); without torchrun it is a no-op single-process info.
    dist_info = init_distributed(args.device)
    main_rank = dist_info.is_main
    if main_rank and not args.data_dir:
        print("WARNING: no --data-dir given - training on SYNTHETIC "
              "Moving MNIST. This is a smoke test only, NOT real-video "
              "training. Pass --data-dir <dataset> (and --require-real "
              "to make this fallback an error).")
    try:
        if args.resume:
            tr = DiTTrainer.resume(
                args.resume, seed=args.seed, lr=args.lr,
                batch_size=args.batch_size, num_clips=args.num_clips,
                device=args.device, ckpt_dir=args.ckpt_dir, log_path=args.log,
                vae_ckpt=args.vae_ckpt, data_dir=args.data_dir,
                data_shard=args.data_shard, data_split=args.data_split,
                tokenizer=cli_tokenizer, latent_scale=args.latent_scale,
                warmup_steps=args.warmup_steps,
                lr_schedule=args.lr_schedule,
                grad_accum=args.grad_accum, amp=args.amp,
                dist_info=dist_info)
            if args.grad_checkpoint:
                unwrap_model(tr.dit).set_grad_checkpoint(True)
            if main_rank:
                print(f"resumed from {args.resume} at step {tr.step}")
        else:
            use_depth = args.depth
            if args.init_weights and not use_depth:
                # Like resume, a depth-trained donor needs its depth stem
                # built BEFORE its weights load; the donor's stored
                # config is the source of truth.
                donor = torch.load(args.init_weights, map_location="cpu",
                                   weights_only=False)
                if bool((donor.get("config") or {}).get("use_depth", False)):
                    use_depth = True
                    if main_rank:
                        print("note: donor checkpoint was trained with "
                              "depth conditioning; enabling --depth")
            tr = DiTTrainer(preset=args.preset, seed=args.seed, lr=args.lr,
                            batch_size=args.batch_size, num_clips=args.num_clips,
                            device=args.device, ckpt_dir=args.ckpt_dir,
                            log_path=args.log, vae_ckpt=args.vae_ckpt,
                            data_dir=args.data_dir,
                            data_split=args.data_split,
                            tokenizer=cli_tokenizer,
                            latent_scale=args.latent_scale,
                            warmup_steps=args.warmup_steps,
                            lr_schedule=args.lr_schedule,
                            total_steps=args.steps,
                            grad_accum=args.grad_accum, amp=args.amp,
                            grad_checkpoint=args.grad_checkpoint,
                            use_depth=use_depth,
                            depth_dropout=args.depth_dropout,
                            data_shard=args.data_shard,
                            dist_info=dist_info)
            if args.init_weights:
                tr.load_dit_weights(args.init_weights,
                                    start_step=args.start_step)
                if main_rank:
                    print(f"initialized DiT weights from "
                          f"{args.init_weights} (starting step {tr.step})")
        n_params = sum(p.numel() for p in tr.dit.parameters()) / 1e6
        if main_rank:
            print(f"preset={args.preset} dit_params={n_params:.1f}M "
                  f"tokens={tr.cfg.num_tokens} device={tr.device} "
                  f"world_size={tr.world_size} "
                  f"global_batch={tr.global_batch_size}")

        for _ in range(args.steps):
            rec = tr.train_step()
            if tr.ckpt_dir and tr.step % args.ckpt_every == 0:
                # All ranks arrive together; only rank 0 writes (its
                # weights are identical to every other rank's).
                barrier(dist_info)
                if main_rank:
                    path = os.path.join(tr.ckpt_dir, f"step_{tr.step:06d}.pt")
                    tr.save(path)
                    print(f"checkpoint -> {path}")
                barrier(dist_info)
        if tr.ckpt_dir:
            barrier(dist_info)
            if main_rank:
                tr.save(os.path.join(tr.ckpt_dir, "latest.pt"))
            barrier(dist_info)
        tr.close()
        if main_rank:
            hist = tr.loss_history
            if hist:
                print(f"done: steps={tr.step} loss_first={hist[0]:.4f} "
                      f"loss_last={hist[-1]:.4f}")
            else:
                # --steps 0 (e.g. minting a pool round-start checkpoint):
                # the final latest.pt above is the whole point; no losses.
                print(f"done: steps={tr.step} (no steps run)")
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
