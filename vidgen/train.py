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
import sys
import os

import torch
import torch.nn.functional as F

from vidgen.checkpoint import load_checkpoint, save_checkpoint, vae_weights
from vidgen.config import get_preset, resolve_device
from vidgen.depth import depth_to_latent_grid
from vidgen.distributed import (
    DistInfo,
    barrier,
    clip_grad_norm,
    destroy_distributed,
    full_model_state,
    full_optim_state,
    init_distributed,
    load_full_model_state,
    load_full_optim_state,
    plain_state_dict,
    rank_seed,
    resolve_rank_device,
    shard_batch,
    sync_gradients,
    unwrap_model,
    wrap_ddp,
    wrap_fsdp,
)
from vidgen.data.buckets import BucketSampler
from vidgen.data.clips import FolderClipDataset
from vidgen.data.moving_mnist import MovingMNISTDataset
from vidgen.flow import flow_loss, rectified_flow_targets, sample_shifted_logit_normal, timestep_bins
from vidgen.logging import StructuredLogger
from vidgen.pool import parse_data_shard
from vidgen.probe import DiTProbe, format_probe
from vidgen.seed import seed_everything
from vidgen.text.encoder import build_text_encoder
from vidgen.text.tokenizer import CaptionTokenizer, load_tokenizer, tokenizer_from_dict
from vidgen.train_utils import (
    ModelEMA, NonFiniteGuard, amp_context, build_lr_scheduler,
    guarded_optimizer_step, load_pooled_ema, make_grad_scaler,
    oom_guidance, validate_train_options,
)
from vidgen.dit.model import DiT
from vidgen.vae.model import CausalVAE, build_vae

# Storage dtypes for the opt-in latent cache (--cache-dtype).
CACHE_DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


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
                 ema_decay: float | None = None,
                 cache_latents: bool = False,
                 cache_dtype: str = "fp32",
                 per_block_mod: bool = False,
                 fsdp: bool = False,
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
        if per_block_mod:
            # Per-block AdaLN offsets (config.per_block_mod); like depth,
            # recorded in the checkpoint config and re-derived on resume.
            self.cfg.per_block_mod = True
        self.depth_dropout = depth_dropout
        self.device = resolve_rank_device(resolve_device(device), dist_info)
        self.batch_size = batch_size
        self.cfg_dropout = cfg_dropout
        self.ckpt_dir = ckpt_dir
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

        # Recorded in every checkpoint: sampling, evaluation and resume
        # must decode/encode with the SAME VAE, and before this was
        # stored a forgotten --vae-ckpt silently fell back to a random
        # VAE (the payload.get("vae_ckpt") readers never found a value).
        self.vae_ckpt = vae_ckpt
        # Sized from this run's config; when a VAE checkpoint is given
        # it is REBUILT from the checkpoint's own shape below — the
        # checkpoint wins (a 16/32 VAE paired with a grown-config run
        # must encode in its own geometry; a mismatch with the DiT's
        # latent_channels then fails loudly at the first encode).
        self.vae = build_vae(self.cfg).to(self.device)
        if vae_ckpt:
            payload = load_checkpoint(vae_ckpt, map_location=str(self.device))
            assert payload["kind"] == "vae", f"not a VAE checkpoint: {vae_ckpt}"
            self.vae = build_vae(payload=payload).to(self.device)
            # The VAE's EMA weights when it has them — the same rule
            # sampling/evaluation use, so the latent space always matches.
            self.vae.load_state_dict(vae_weights(payload))
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
        # --fsdp (big presets) shards the DiT across ranks (see the wrap
        # below). Decided up front: an FSDP DiT is BUILT on the CPU and
        # each block is moved to the GPU only as it is sharded, so a model
        # that does not fit one GPU whole never has to.
        self.fsdp = bool(fsdp) and dist_info.is_distributed \
            and torch.distributed.is_initialized()
        if fsdp and not self.fsdp and self.is_main_rank:
            print("WARNING: --fsdp needs a torchrun launch with 2+ "
                  "processes; training unsharded", file=sys.stderr)
        self.dit = DiT(self.cfg)
        if not self.fsdp:
            self.dit = self.dit.to(self.device)
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
        if ema_decay is not None:
            # Only when enabled: a default run's train_config stays
            # exactly the historical one.
            self.train_config["ema_decay"] = float(ema_decay)
        # EMA of the trained weights (DiT + text encoder), used by
        # sampling/evaluation. Built from the RAW modules before the DDP
        # wrap below; it never feeds back into training.
        # (Under --fsdp the shadows are built after sharding instead.)
        self.ema = (ModelEMA(self._ema_modules(), ema_decay)
                    if ema_decay is not None and not self.fsdp else None)
        # Opt-in latent cache: the VAE is frozen and every dataset item is
        # deterministic, so each clip's latent is computed once (one clip
        # at a time, so the value never depends on batch composition) and
        # reused every epoch — skipping both the .npy load and the VAE
        # encoder. Kept on CPU; not checkpointed (it is rebuilt on demand).
        # RAM: see latent_cache_bytes() — ~1,152 KB/clip at the Small preset
        # in fp32; cache_dtype "fp16" halves it (latents are rounded to
        # fp16 once, so a cached run is no longer bit-identical to an
        # uncached one).
        self.cache_latents = bool(cache_latents)
        if cache_dtype not in CACHE_DTYPES:
            raise ValueError(f"unknown cache dtype {cache_dtype!r}; "
                             f"choose {sorted(CACHE_DTYPES)}")
        self.cache_dtype = cache_dtype
        self._latent_cache: dict[int, tuple[torch.Tensor, str]] = {}

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
        self._nf_guard = NonFiniteGuard()
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
        # --fsdp (big presets): SHARD the DiT across the ranks instead of
        # replicating it (vidgen.distributed.wrap_fsdp). Sharding swaps
        # the parameters, so the optimizer and LR schedule are rebuilt on
        # the sharded ones and the EMA shadows are created from them
        # (all still fresh here — resume loads state into them
        # afterwards, through the full-state helpers).
        if self.fsdp:
            if self.amp == "fp16":
                raise ValueError(
                    "--fsdp supports --amp bf16 or none, not fp16 (the "
                    "fp16 GradScaler does not handle sharded gradients); "
                    "the GPUs that need FSDP all run bf16")
            self.dit = wrap_fsdp(
                self.dit, dist_info, self.device,
                param_dtype=torch.bfloat16 if self.amp == "bf16" else None)
            params = list(self.dit.parameters()) + list(self.text_enc.parameters())
            self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
            self.sched = build_lr_scheduler(
                self.opt, warmup_steps=self.warmup_steps,
                total_steps=self.total_steps, schedule=self.lr_schedule)
            if ema_decay is not None:
                self.ema = ModelEMA(self._ema_modules(), ema_decay)
        else:
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

    @torch.no_grad()
    def _encode_batch(self, batch_idx) -> tuple[torch.Tensor, list[str]]:
        """VAE latents (mu, before latent_scale) + captions for a batch."""
        if not self.cache_latents:
            clips, captions = [], []
            for i in batch_idx:
                c, cap = self.dataset[i]
                clips.append(c)
                captions.append(cap)
            x = torch.stack(clips).to(self.device)  # [B,3,T,H,W]
            mu, _ = self.vae.encode(x)
            return mu, captions
        mus, captions = [], []
        for i in batch_idx:
            hit = self._latent_cache.get(int(i))
            if hit is None:
                c, cap = self.dataset[i]
                mu, _ = self.vae.encode(c.unsqueeze(0).to(self.device))
                hit = (mu[0].to(CACHE_DTYPES[self.cache_dtype]).cpu(), cap)
                self._latent_cache[int(i)] = hit
            mus.append(hit[0])
            captions.append(hit[1])
        return torch.stack(mus).to(self.device, torch.float32), captions

    def latent_cache_bytes(self) -> int:
        """RAM the latent cache will hold once every clip is cached."""
        per = (self.cfg.latent_channels * self.cfg.latent_t
               * self.cfg.latent_h * self.cfg.latent_w
               * CACHE_DTYPES[self.cache_dtype].itemsize)
        return per * len(self.dataset)

    def _ema_modules(self) -> dict:
        return {"dit": unwrap_model(self.dit),
                "text_enc": unwrap_model(self.text_enc)}

    # ------------------------------------------------------------ one step
    def train_step(self) -> dict:
        import time

        t0 = time.perf_counter()
        self.opt.zero_grad()
        total_loss = 0.0
        # Per-timestep-bin loss over EVERY micro-batch of the step (it
        # used to cover only the last one under --grad-accum).
        bin_loss = torch.zeros(10, device=self.device)
        bin_count = torch.zeros(10, device=self.device)
        last_batch = 0
        for _micro in range(self.grad_accum):
            batch_idx = self._next_batch_idx()
            mu, captions = self._encode_batch(batch_idx)
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

            bins = timestep_bins(t)
            per = F.mse_loss(pred.detach().float(), target,
                             reduction="none").mean(dim=(1, 2, 3, 4))
            bin_loss.index_add_(0, bins, per)
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
        grad_norm = clip_grad_norm(
            list(self.dit.parameters()) + list(self.text_enc.parameters()), 1.0)
        # Non-finite step: the poisoned gradients are discarded WITHOUT
        # stepping, so the weights stay exactly as they were.
        outcome = guarded_optimizer_step(
            self.opt, self.scaler, self._nf_guard,
            total_loss / self.grad_accum, float(grad_norm))
        # The scheduler tracks OPTIMIZER steps: on a skipped step the
        # weights did not move, so the schedule must not advance either
        # — otherwise a burst of bad steps burns the LR horizon without
        # any training happening (found in the 2026-10-23 repair audit).
        if self.sched is not None and outcome == "stepped":
            self.sched.step()
        if self.ema is not None and outcome == "stepped":
            self.ema.update(self._ema_modules())

        secs = time.perf_counter() - t0
        samples = last_batch * self.world_size * self.grad_accum
        rec = {"step": self.step, "loss": total_loss / self.grad_accum,
               "lr": self.opt.param_groups[0]["lr"],
               "grad_norm": float(grad_norm),
               "bin_loss": [round(v, 5) for v in bin_mean],
               "secs": round(secs, 3),
               "clips_per_sec": round(samples / max(secs, 1e-9), 2)}
        if outcome != "stepped":
            rec[outcome] = True
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
    def _opt_modules(self) -> dict:
        return {"dit": self.dit, "text_enc": self.text_enc}

    def state_dict(self) -> dict:
        # Under --fsdp the DiT/optimizer/EMA gathers are collectives:
        # every rank builds the payload (see save); rank 0 gets tensors.
        payload = {
            "kind": "dit",
            "preset": self.cfg.name,
            "step": self.step,
            "dit": full_model_state(self.dit),
            "text_enc": plain_state_dict(self.text_enc),
            "opt": full_optim_state(self._opt_modules(), self.opt),
            "sampler": self.sampler.state_dict(),
            "loss_history": self.loss_history,
            "config": self.cfg.to_dict(),
            "vae_ckpt": self.vae_ckpt,
        }
        if self.tokenizer is not None:
            payload["tokenizer"] = self.tokenizer.to_dict()
        payload["train_config"] = dict(self.train_config)
        if self.sched is not None:
            payload["sched"] = self.sched.state_dict()
        if self.scaler is not None:
            payload["scaler"] = self.scaler.state_dict()
        if self.ema is not None:
            payload["ema"] = self.ema.state_dict()
        return payload

    def load_state_dict(self, payload: dict) -> None:
        assert payload["kind"] == "dit"
        # Loads go into the UNWRAPPED modules: checkpoint keys are plain
        # (no "module." prefix) whether or not this rank runs under DDP.
        load_full_model_state(self.dit, payload["dit"])
        unwrap_model(self.text_enc).load_state_dict(payload["text_enc"])
        load_full_optim_state(self._opt_modules(), self.opt, payload["opt"])
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
        if "ema" in payload:
            if self.ema is None:
                raise ValueError(
                    "checkpoint carries EMA weights but this trainer was "
                    "built without --ema-decay; resume via "
                    "DiTTrainer.resume so the EMA is restored from "
                    "train_config (dropping it would lose the average)")
            self.ema.load_state_dict(payload["ema"])
        elif self.ema is not None:
            # EMA newly enabled on resume: start averaging from the
            # resumed weights, not from this trainer's fresh init.
            self.ema.reset(self._ema_modules())
        # Sampler reconciliation: the checkpoint's sampler describes the
        # dataset IT was saved on; this trainer's fresh sampler carries
        # the CURRENT dataset size and batch size. Capture them before
        # the load overwrites them, then reconcile — a stale sampler
        # silently truncates training coverage (Stage 2 VAE incident:
        # num_items stayed 4203 onto a 6,339-clip dataset).
        want_items = self.sampler.num_items
        want_batch = self.sampler.batch_size
        self.sampler.load_state_dict(payload["sampler"])
        self.sampler.reconcile(want_items, want_batch)
        self.step = payload["step"]
        self.loss_history = payload["loss_history"]

    def load_dit_weights(self, path: str,
                         start_step: int | None = None,
                         payload: dict | None = None) -> None:
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

        ``payload``: the donor's already-loaded payload (``main`` reads it
        once to detect depth/per-block config), to avoid a second load.
        """
        if payload is None:
            payload = load_checkpoint(path)
        # A fresh run starts from its own seeded streams (mirrors the VAE
        # trainer: rank_seed(seed, 0) == seed, so single-process is
        # unchanged).
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
        load_full_model_state(self.dit, payload["dit"])
        self.text_enc.load_state_dict(payload["text_enc"])
        if self.ema is not None:
            # A pooled donor carries the workers' merged EMA: continue
            # that average across rounds. Any other donor: average from
            # the donor's weights.
            load_pooled_ema(self.ema, payload, self._ema_modules())
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
        # built from the unwrapped models, so it stays plain). Under
        # --fsdp EVERY rank must call save: gathering the shards into
        # the plain payload is a collective.
        if self.fsdp:
            payload = self.state_dict()
            if self.is_main_rank:
                save_checkpoint(path, payload)
            return
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
        kwargs["per_block_mod"] = bool(stored_cfg.get("per_block_mod", False))
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
        # The VAE defines the latent space the DiT was trained in. A
        # resume that forgets --vae-ckpt must not quietly continue on a
        # randomly initialized VAE's latents: reuse the stored one.
        stored_vae = payload.get("vae_ckpt")
        if stored_vae and not kwargs.get("vae_ckpt"):
            if not os.path.isfile(stored_vae):
                raise FileNotFoundError(
                    f"checkpoint was trained with VAE {stored_vae!r}, which "
                    "no longer exists at that path; pass --vae-ckpt with "
                    "its new location")
            kwargs["vae_ckpt"] = stored_vae
        elif stored_vae and os.path.abspath(kwargs["vae_ckpt"]) \
                != os.path.abspath(stored_vae):
            print(f"WARNING: resuming with --vae-ckpt {kwargs['vae_ckpt']} "
                  f"but the checkpoint was trained with {stored_vae}; the "
                  "latent space changes unless they hold the same weights",
                  file=sys.stderr)
        cli_scale = kwargs.pop("latent_scale", None)
        # Training options (schedule/AMP/accumulation) are part of the
        # run's identity: restore them from the checkpoint's
        # train_config unless the caller explicitly overrode them.
        stored_tc = payload.get("train_config") or {}
        for key in ("warmup_steps", "lr_schedule", "total_steps",
                    "grad_accum", "amp", "ema_decay"):
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
    ap.add_argument("--ema-decay", type=float, default=None,
                    help="keep an exponential moving average of the "
                         "weights (e.g. 0.999) in every checkpoint; "
                         "sampling and evaluation use it automatically. "
                         "Default: off. On --resume the stored value wins.")
    ap.add_argument("--cache-latents", action="store_true",
                    help="encode each clip with the frozen VAE once and "
                         "reuse the latent every epoch (kept in CPU RAM): "
                         "after the first epoch, steps skip clip loading "
                         "and the VAE encoder entirely. RAM: ~1,152 KB per "
                         "clip at the Small preset (the estimate is "
                         "printed at start)")
    ap.add_argument("--cache-dtype", default="fp32", choices=["fp32", "fp16"],
                    help="storage dtype of --cache-latents: fp16 halves "
                         "the RAM; latents are rounded once, so the run is "
                         "no longer bit-identical to an uncached one")
    ap.add_argument("--per-block-mod", action="store_true",
                    help="give every DiT block its own zero-init offset on "
                         "the shared AdaLN modulation (PixArt-alpha "
                         "adaLN-single); a new architecture option, so "
                         "only for fresh runs. On --resume the "
                         "checkpoint's config decides.")
    ap.add_argument("--fsdp", action="store_true",
                    help="under torchrun: SHARD the DiT's weights, gradients "
                         "and optimizer state across the GPUs (FSDP2) "
                         "instead of copying the whole model to each one; "
                         "the way to train the 9B/14B presets. Checkpoints "
                         "stay plain (resume on any GPU count). Use with "
                         "--amp bf16 (fp16 is not supported).")
    ap.add_argument("--probe-every", type=int, default=0, metavar="N",
                    help="every N steps, measure the model on a FIXED probe "
                         "set (vidgen/probe.py) so you can see whether "
                         "training is still helping; logged as {\"probe\": "
                         "...} records and printed with the change since "
                         "the first probe. Uses the dataset's val split "
                         "when it has one. Default 0 = off; never changes "
                         "the training trajectory.")
    ap.add_argument("--probe-clips", type=int, default=4,
                    help="clips in the fixed probe set (default 4)")
    ap.add_argument("--probe-samples", type=int, default=2,
                    help="short sample clips generated per probe, scored "
                         "with vidgen/evaluate.py's pixel metrics and saved "
                         "under <ckpt-dir>/probe/ (default 2; 0 = val loss "
                         "only)")
    ap.add_argument("--probe-steps", type=int, default=8,
                    help="sampler steps for probe samples (default 8)")
    ap.add_argument("--probe-prompts", default=None,
                    help="comma-separated fixed prompts for probe samples "
                         "(default: the probe clips' own captions)")
    args = ap.parse_args()

    # Resuming without an explicit --ckpt-dir continues writing next
    # to the checkpoint being resumed — never into the fresh-run
    # default folder, where the continued run's checkpoints would be
    # scattered away from the run they belong to (found live in the
    # 2026-10-23 Stage B validation: a resumed run wrote to
    # checkpoints/tiny-dit instead of the original directory).
    if args.resume and args.ckpt_dir == "checkpoints/tiny-dit":
        args.ckpt_dir = os.path.dirname(os.path.abspath(args.resume))
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
                ema_decay=args.ema_decay,
                cache_latents=args.cache_latents,
                cache_dtype=args.cache_dtype,
                fsdp=args.fsdp,
                dist_info=dist_info)
            if args.grad_checkpoint:
                unwrap_model(tr.dit).set_grad_checkpoint(True)
            if main_rank:
                print(f"resumed from {args.resume} at step {tr.step}")
        else:
            use_depth = args.depth
            per_block_mod = args.per_block_mod
            donor = None
            if args.init_weights:
                # Like resume, a donor's architecture options (depth stem,
                # per-block modulation) must be built BEFORE its weights
                # load; the donor's stored config is the source of truth.
                # Loaded once here and handed to load_dit_weights.
                donor = load_checkpoint(args.init_weights)
                donor_cfg = donor.get("config") or {}
                if not use_depth and donor_cfg.get("use_depth", False):
                    use_depth = True
                    if main_rank:
                        print("note: donor checkpoint was trained with "
                              "depth conditioning; enabling --depth")
                if not per_block_mod and donor_cfg.get("per_block_mod", False):
                    per_block_mod = True
                    if main_rank:
                        print("note: donor checkpoint uses per-block "
                              "modulation; enabling --per-block-mod")
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
                            ema_decay=args.ema_decay,
                            cache_latents=args.cache_latents,
                            cache_dtype=args.cache_dtype,
                            per_block_mod=per_block_mod,
                            fsdp=args.fsdp,
                            dist_info=dist_info)
            if args.init_weights:
                tr.load_dit_weights(args.init_weights,
                                    start_step=args.start_step,
                                    payload=donor)
                if main_rank:
                    print(f"initialized DiT weights from "
                          f"{args.init_weights} (starting step {tr.step})")
        n_params = sum(p.numel() for p in tr.dit.parameters()) / 1e6
        if main_rank:
            print(f"preset={args.preset} dit_params={n_params:.1f}M "
                  f"tokens={tr.cfg.num_tokens} device={tr.device} "
                  f"world_size={tr.world_size} "
                  f"global_batch={tr.global_batch_size}"
                  + (" fsdp=sharded" if tr.fsdp else ""))
            if tr.cache_latents:
                print(f"latent cache: {len(tr.dataset)} clips x "
                      f"{tr.latent_cache_bytes() // max(1, len(tr.dataset)) // 1024} KB "
                      f"= {tr.latent_cache_bytes() / 1024**3:.2f} GB of CPU RAM "
                      f"once every clip is cached ({tr.cache_dtype})")

        probe = None
        # Under --fsdp every rank runs the probe (the sharded forward is
        # collective); only rank 0 saves, logs and prints.
        if args.probe_every > 0 and (main_rank or tr.fsdp):
            prompts = ([p.strip() for p in args.probe_prompts.split(",")
                        if p.strip()] if args.probe_prompts else None)
            probe = DiTProbe(
                tr, data_dir=args.data_dir, num_clips=args.probe_clips,
                prompts=prompts, num_samples=args.probe_samples,
                sample_steps=args.probe_steps,
                out_dir=(os.path.join(tr.ckpt_dir, "probe")
                         if tr.ckpt_dir and main_rank else None))
        for _ in range(args.steps):
            try:
                rec = tr.train_step()
            except torch.cuda.OutOfMemoryError:
                print(oom_guidance("DiT training"), file=sys.stderr)
                raise SystemExit(3)
            if args.probe_every > 0 and tr.step % args.probe_every == 0:
                # Rank 0 probes its (identical) weights; the others wait.
                if probe is not None:
                    prec = probe.run(tr.step)
                    if tr.logger:
                        tr.logger.log(prec, echo=False)
                    if main_rank:
                        print(format_probe(prec))
                barrier(dist_info)
            if tr.ckpt_dir and tr.step % args.ckpt_every == 0:
                # All ranks arrive together; only rank 0 writes (its
                # weights are identical to every other rank's).
                barrier(dist_info)
                # Every rank calls save (an FSDP gather is collective);
                # only rank 0 writes.
                path = os.path.join(tr.ckpt_dir, f"step_{tr.step:06d}.pt")
                tr.save(path)
                if main_rank:
                    print(f"checkpoint -> {path}")
                barrier(dist_info)
        if tr.ckpt_dir:
            barrier(dist_info)
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
