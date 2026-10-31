"""Part 3: train the causal 3D VAE on Moving MNIST (Track A / Tiny).

Loss: reconstruction MSE + kl_weight * KL. Checkpointing mirrors train.py
(model + optimizer + sampler + RNG state) so resume is bit-reproducible.

Usage (CPU smoke):
  python -m vidgen.train_vae --steps 12 --batch-size 2 --num-clips 16 \\
      --ckpt-dir checkpoints/tiny-vae --log checkpoints/tiny-vae/train.jsonl

The trained VAE's encoder then supplies latents for DiT training
(python -m vidgen.train --vae-ckpt checkpoints/tiny-vae/latest.pt ...).

Multi-GPU (DDP, opt-in via torchrun; --batch-size is PER GPU there):
  torchrun --standalone --nproc_per_node=2 -m vidgen.train_vae \\
      --preset small --batch-size 2 --device cuda ...

Weight-averaging pool worker (one round; see PYTHON_IMPLEMENTATION.md):
  python -m vidgen.train_vae --preset small --init-weights pooled.pt \\
      --data-shard 2/4 --steps 500 --ckpt-dir ckpts/worker2 ...
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from vidgen.checkpoint import load_checkpoint, save_checkpoint
from vidgen.config import VidgenConfig, get_preset, resolve_device
from vidgen.data.buckets import BucketSampler
from vidgen.distributed import (
    DistInfo,
    barrier,
    destroy_distributed,
    init_distributed,
    plain_state_dict,
    rank_seed,
    resolve_rank_device,
    shard_batch,
    unwrap_model,
    wrap_ddp,
)
from vidgen.data.clips import FolderClipDataset
from vidgen.data.moving_mnist import MovingMNISTDataset
from vidgen.logging import StructuredLogger
from vidgen.pool import parse_data_shard
from vidgen.probe import VAEProbe, format_probe
from vidgen.seed import seed_everything
from vidgen.train_utils import (
    ModelEMA, NonFiniteGuard, amp_context, build_lr_scheduler,
    guarded_optimizer_step, load_pooled_ema, make_grad_scaler,
    oom_guidance, validate_train_options,
)
from vidgen.vae.model import CausalVAE, build_vae, vae_loss


class VAETrainer:
    def __init__(self, preset: str = "tiny", seed: int = 0, lr: float = 1e-4,
                 kl_weight: float = 1e-4, recon_weight: float = 1.0,
                 batch_size: int = 2, num_clips: int = 64,
                 device: str = "cpu", ckpt_dir: str | None = None,
                 log_path: str | None = None, data_dir: str | None = None,
                 data_shard: tuple[int, int] | None = None,
                 data_split: str | None = None,
                 warmup_steps: int | None = None,
                 lr_schedule: str | None = None,
                 total_steps: int | None = None,
                 grad_accum: int | None = None,
                 amp: str | None = None,
                 ema_decay: float | None = None,
                 dist_info: DistInfo | None = None,
                 cfg: VidgenConfig | None = None):
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
        # An explicit cfg wins over the named preset: on resume this is
        # the checkpoint's STORED config, so a pre-growth Small run
        # (VAE 16/32) resumes into its own geometry even though the
        # small preset now describes the grown 32/64 VAE.
        self.cfg = cfg if cfg is not None else get_preset(preset)
        self.device = resolve_rank_device(resolve_device(device), dist_info)
        self.batch_size = batch_size
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.ckpt_dir = ckpt_dir
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

        self.vae = build_vae(self.cfg).to(self.device)
        self.opt = torch.optim.AdamW(self.vae.parameters(), lr=lr)
        # Phase 4/5 (shared with the DiT trainer via vidgen.train_utils):
        # LR schedule, AMP, gradient accumulation — defaults reproduce
        # the historical behaviour; resolved options are recorded in
        # every checkpoint's train_config and restored on resume.
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
        # EMA of the VAE weights (as in the DiT trainer). Everything that
        # USES a trained VAE — DiT training, sampling, evaluation,
        # measure_latents — prefers these weights when present
        # (vidgen.checkpoint.vae_weights), so they all share one latent
        # space. Built from the RAW module before the DDP wrap below.
        self.ema = (ModelEMA({"vae": self.vae}, ema_decay)
                    if ema_decay is not None else None)
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
        # each rank trains on its own disjoint slice (_next_batch_idx).
        # Single-process: world_size == 1, the original sampler.
        self.global_batch_size = batch_size * dist_info.world_size
        self.sampler = BucketSampler(num_items, self.global_batch_size,
                                     seed=seed + 2)
        self.step = 0
        self.logger = (StructuredLogger(log_path)
                       if log_path and self.is_main_rank else None)
        self.loss_history: list[float] = []
        self._nf_guard = NonFiniteGuard()
        self.init_from: str | None = None

        # DDP wrap LAST (optimizer/checkpoint paths use the raw module's
        # parameters — identical objects after wrapping), then diverge
        # the stochastic streams per rank: weights were built identically
        # from the base seed, but the VAE reparameterization noise must
        # differ per rank. Derived from (seed, rank), so a resume with
        # the same world size reproduces the same streams.
        self.vae = wrap_ddp(self.vae, dist_info, self.device)
        if dist_info.is_distributed:
            seed_everything(rank_seed(seed, dist_info.rank))

    # ------------------------------------------------------------ data
    def _next_batch_idx(self) -> list[int]:
        """This rank's batch indices (see DiTTrainer._next_batch_idx)."""
        global_batch = self.sampler.next_batch()
        if not self.dist_info.is_distributed:
            return global_batch
        return shard_batch(global_batch, self.batch_size, self.rank)

    def train_step(self) -> dict:
        import time

        t0 = time.perf_counter()
        self.opt.zero_grad()
        total_loss = 0.0
        last_recon = last_kl = 0.0
        last_batch = 0
        for _micro in range(self.grad_accum):
            batch_idx = self._next_batch_idx()
            last_batch = len(batch_idx)
            clips = torch.stack([self.dataset[i][0] for i in batch_idx]).to(self.device)
            with amp_context(self.device, self.amp):
                recon, mu, logvar, kl = self.vae(clips)
                loss, recon_loss, kl_term = vae_loss(
                    recon, clips, kl, self.kl_weight,
                    getattr(self, "recon_weight", 1.0))
            scaled = loss / self.grad_accum
            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()
            total_loss += float(loss.detach())
            last_recon, last_kl = float(recon_loss.detach()), float(kl_term.detach())
        if self.scaler is not None:
            self.scaler.unscale_(self.opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
        outcome = guarded_optimizer_step(
            self.opt, self.scaler, self._nf_guard,
            total_loss / self.grad_accum, float(grad_norm))
        # Scheduler tracks optimizer steps only — it must not
        # advance on a skipped step (see train.py).
        if self.sched is not None and outcome == "stepped":
            self.sched.step()
        if self.ema is not None and outcome == "stepped":
            self.ema.update({"vae": unwrap_model(self.vae)})
        secs = time.perf_counter() - t0
        samples = last_batch * self.world_size * self.grad_accum
        rec = {"step": self.step, "loss": total_loss / self.grad_accum,
               "recon": last_recon, "kl": last_kl,
               "lr": self.opt.param_groups[0]["lr"], "grad_norm": float(grad_norm),
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

    def state_dict(self) -> dict:
        payload = {
            "kind": "vae",
            "preset": self.cfg.name,
            "step": self.step,
            "vae": plain_state_dict(self.vae),
            "opt": self.opt.state_dict(),
            "sampler": self.sampler.state_dict(),
            "loss_history": self.loss_history,
            "config": self.cfg.to_dict(),
            "train_config": dict(self.train_config),
        }
        if self.sched is not None:
            payload["sched"] = self.sched.state_dict()
        if self.scaler is not None:
            payload["scaler"] = self.scaler.state_dict()
        if self.ema is not None:
            payload["ema"] = self.ema.state_dict()
        return payload

    def load_state_dict(self, payload: dict) -> None:
        assert payload["kind"] == "vae"
        # Loads go into the UNWRAPPED module: keys are plain either way.
        unwrap_model(self.vae).load_state_dict(payload["vae"])
        self.opt.load_state_dict(payload["opt"])
        if "sched" in payload:
            if self.sched is None:
                raise ValueError(
                    "checkpoint carries LR-scheduler state but this "
                    "trainer was built without the run's schedule; "
                    "resume via VAETrainer.resume so the schedule is "
                    "restored from train_config")
            self.sched.load_state_dict(payload["sched"])
        if "scaler" in payload and self.scaler is not None:
            self.scaler.load_state_dict(payload["scaler"])
        if "ema" in payload:
            if self.ema is None:
                raise ValueError(
                    "checkpoint carries EMA weights but this trainer was "
                    "built without --ema-decay; resume via "
                    "VAETrainer.resume so the EMA is restored from "
                    "train_config (dropping it would lose the average)")
            self.ema.load_state_dict(payload["ema"])
        elif self.ema is not None:
            # EMA newly enabled on resume: average from the resumed weights.
            self.ema.reset({"vae": unwrap_model(self.vae)})
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

    def load_vae_weights(self, path: str,
                         start_step: int | None = None) -> None:
        """Weights-only warm start from a donor VAE checkpoint.

        Copies just the trained VAE weights (e.g. Round-1 tiny VAE) into this
        fresh trainer. Optimizer, step counter, loss history and RNG stay
        fresh — this is a new run, not a resume. The load is strict:
        donor and trainer must share the same VAE shape (a tiny 16/32
        donor initializes any 16/32 target, e.g. the 480p VAE). To
        start from a donor of a DIFFERENT shape (the 16/32 Stage 2
        VAE -> the grown 32/64 Small VAE), transplant it first with
        vidgen.vae.grow / scripts/grow_vae.py; a shape mismatch here
        raises immediately (strict load).

        Step offset (pool rounds): an explicit ``start_step`` wins;
        otherwise a pooled checkpoint (carries ``pool`` metadata) donates
        its stored step so round numbering is cumulative; a plain donor
        starts at step 0, as a warm start always has.
        """
        payload = load_checkpoint(path)
        # A fresh run starts from its own seeded streams. Single-process
        # this is the original seed_everything(self.seed);
        # rank_seed(seed, 0) == seed.
        seed_everything(rank_seed(self.seed, self.rank))
        if payload.get("kind") not in ("vae", "vae-weights-fp16") \
                or "vae" not in payload:
            raise ValueError(
                f"{path} is not a VAE checkpoint (kind={payload.get('kind')!r})")
        unwrap_model(self.vae).load_state_dict(payload["vae"])  # strict load
        if self.ema is not None:
            # A pooled donor carries the workers' merged EMA: continue
            # that average across rounds instead of restarting it.
            load_pooled_ema(self.ema, payload, {"vae": unwrap_model(self.vae)})
        self.init_from = path
        if start_step is not None:
            self.step = int(start_step)
        elif "pool" in payload:
            self.step = int(payload["step"])

    def save(self, path: str) -> None:
        # Only rank 0 writes checkpoints (weights are identical across
        # ranks under DDP; the checkpoint itself is plain/unwrapped).
        if not self.is_main_rank:
            return
        save_checkpoint(path, self.state_dict())

    @classmethod
    def resume(cls, path: str, **kwargs) -> "VAETrainer":
        from vidgen.seed import set_rng_states

        payload = load_checkpoint(path)
        kwargs.pop("preset", None)  # preset always comes from the checkpoint
        if payload.get("config"):
            # The stored config is authoritative for the VAE geometry:
            # it is what the checkpoint's weights were built for (e.g.
            # a 16/32 Small checkpoint in a world where the small
            # preset is 32/64).
            kwargs["cfg"] = VidgenConfig.from_dict(payload["config"])
        stored_tc = payload.get("train_config") or {}
        for key in ("warmup_steps", "lr_schedule", "total_steps",
                    "grad_accum", "amp", "ema_decay"):
            if kwargs.get(key) is None and key in stored_tc:
                kwargs[key] = stored_tc[key]
        tr = cls(preset=payload["preset"], **kwargs)
        tr.load_state_dict(payload)
        set_rng_states(payload["rng"])  # restore saved RNG state after fresh init
        if tr.dist_info.is_distributed:
            # As in DiTTrainer.resume: the saved RNG is rank 0's stream;
            # each rank re-derives its own from (seed, rank) instead of
            # every rank replaying rank 0's noise.
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
                         "batch is batch-size x world_size. Single-GPU "
                         "runs are unaffected.")
    ap.add_argument("--num-clips", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl-weight", type=float, default=1e-4)
    ap.add_argument("--recon-weight", type=float, default=1.0,
                    help="weight of the reconstruction term in the VAE "
                         "loss (default 1.0 = historical loss)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt-dir", default="checkpoints/tiny-vae")
    ap.add_argument("--ckpt-every", type=int, default=6)
    ap.add_argument("--log", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-weights", default=None, metavar="PATH",
                    help="weights-only warm start: initialize the VAE from a donor "
                         "VAE checkpoint (e.g. the Round-1 tiny VAE) and train as a "
                         "fresh run. Donor and target must share the same VAE "
                         "shape; transplant a different-shaped donor first with "
                         "scripts/grow_vae.py. For a pool round, pass the "
                         "pooled checkpoint: step numbering then continues from "
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
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="linear LR warmup over this many steps (default 0)")
    ap.add_argument("--lr-schedule", default=None,
                    choices=["none", "cosine", "linear"],
                    help="LR decay after warmup, over --steps (default "
                         "none = constant). On --resume the checkpoint's "
                         "stored schedule wins unless overridden.")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="micro-batches accumulated per optimizer step "
                         "(default 1)")
    ap.add_argument("--amp", default=None, choices=["none", "bf16", "fp16"],
                    help="mixed precision via autocast (default none)")
    ap.add_argument("--ema-decay", type=float, default=None,
                    help="keep an exponential moving average of the VAE "
                         "weights (e.g. 0.999) in every checkpoint; DiT "
                         "training, sampling and evaluation then use it "
                         "automatically. Default: off. On --resume the "
                         "stored value wins.")
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
    args = ap.parse_args()

    # Same rule as the DiT CLI: resuming without an explicit
    # --ckpt-dir writes next to the resumed checkpoint, not into the
    # fresh-run default folder.
    if args.resume and args.ckpt_dir == "checkpoints/tiny-vae":
        args.ckpt_dir = os.path.dirname(os.path.abspath(args.resume))
    if args.resume and args.init_weights:
        ap.error("--resume and --init-weights are mutually exclusive: "
                 "resume continues a run, warm start begins a fresh one")
    if args.start_step is not None and not args.init_weights:
        ap.error("--start-step only applies with --init-weights")
    if args.require_real and not args.data_dir:
        ap.error("--require-real needs --data-dir: refusing to fall back "
                 "to synthetic Moving MNIST for a real training run")
    if args.data_split and not args.data_dir:
        ap.error("--data-split needs --data-dir (splits live in the "
                 "real-clip dataset directory)")
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
            tr = VAETrainer.resume(
                args.resume, seed=args.seed, lr=args.lr,
                kl_weight=args.kl_weight, recon_weight=args.recon_weight,
                batch_size=args.batch_size,
                num_clips=args.num_clips, device=args.device,
                ckpt_dir=args.ckpt_dir, log_path=args.log,
                data_dir=args.data_dir, data_shard=args.data_shard,
                data_split=args.data_split,
                warmup_steps=args.warmup_steps,
                lr_schedule=args.lr_schedule,
                grad_accum=args.grad_accum, amp=args.amp,
                ema_decay=args.ema_decay,
                dist_info=dist_info)
            if main_rank:
                print(f"resumed from {args.resume} at step {tr.step}")
        else:
            tr = VAETrainer(preset=args.preset, seed=args.seed, lr=args.lr,
                            kl_weight=args.kl_weight,
                            recon_weight=args.recon_weight,
                            batch_size=args.batch_size,
                            num_clips=args.num_clips, device=args.device,
                            ckpt_dir=args.ckpt_dir, log_path=args.log,
                            data_dir=args.data_dir, data_shard=args.data_shard,
                            data_split=args.data_split,
                            warmup_steps=args.warmup_steps,
                            lr_schedule=args.lr_schedule,
                            total_steps=args.steps,
                            grad_accum=args.grad_accum, amp=args.amp,
                            ema_decay=args.ema_decay,
                            dist_info=dist_info)
            if args.init_weights:
                tr.load_vae_weights(args.init_weights,
                                    start_step=args.start_step)
                if main_rank:
                    print(f"warm-started VAE weights from "
                          f"{args.init_weights} (starting step {tr.step})")
        n_params = sum(p.numel() for p in tr.vae.parameters()) / 1e6
        if main_rank:
            print(f"preset={args.preset} vae_params={n_params:.2f}M "
                  f"device={tr.device} world_size={tr.world_size} "
                  f"global_batch={tr.global_batch_size}")

        probe = None
        if args.probe_every > 0 and main_rank:
            probe = VAEProbe(tr, data_dir=args.data_dir,
                             num_clips=args.probe_clips)
        for _ in range(args.steps):
            try:
                tr.train_step()
            except torch.cuda.OutOfMemoryError:
                print(oom_guidance("VAE training"), file=sys.stderr)
                raise SystemExit(3)
            if args.probe_every > 0 and tr.step % args.probe_every == 0:
                if probe is not None:
                    prec = probe.run(tr.step)
                    if tr.logger:
                        tr.logger.log(prec, echo=False)
                    print(format_probe(prec))
                barrier(dist_info)
            if tr.ckpt_dir and tr.step % args.ckpt_every == 0:
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
                # --steps 0 (e.g. minting a pool round-start checkpoint).
                print(f"done: steps={tr.step} (no steps run)")
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
