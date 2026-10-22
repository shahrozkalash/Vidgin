# Vidgin PyTorch Implementation (Track A / Tiny)

This document describes the Python implementation of the Vidgen latent-diffusion
text-to-video engine specified in `README.md`, `AUDIT.md`, and `GPU_CHECKLIST.md`.
It was built on 2026-10-18 and verified by a passing CPU test suite plus real
training runs (loss demonstrably decreases; checkpoints save/resume
bit-identically; a trained checkpoint samples a clip).

The existing React/TypeScript studio app is untouched — the Python code is a new
`vidgen/` package, `tests/`, and `requirements.txt` alongside it.

## What was implemented

| File | Spec part | Contents |
|---|---|---|
| `vidgen/config.py` | Part 0 | Preset configs (Tiny/Small/480p/14B) matching the README dimension table; `T = 4k+1` frame rule + multiples-of-16 spatial validation; JSON save/load roundtrip; device-agnostic resolution (no hardcoded CUDA anywhere) |
| `vidgen/seed.py` | Part 7 | `seed_everything` (python/numpy/torch + fixed thread count); RNG snapshot/restore for checkpoints |
| `vidgen/logging.py` | Part 7 | Structured JSONL logger (step, loss, recon/kl, lr, grad_norm, timestamp) |
| `vidgen/data/moving_mnist.py` | Part 1 | Deterministic Moving MNIST clip generator: `[3, T, H, W]` in `[-1, 1]`, controllable digits/speeds/directions, letterbox borders, static clips, near-duplicates; `MovingMNISTDataset` (per-index deterministic) with captions |
| `vidgen/data/filters.py` | Part 1 | CPU quality filters: Laplacian sharpness ≥ 20, motion 0.5–80, entropy aesthetic ≥ 4.0 |
| `vidgen/data/buckets.py` | Part 2 | Bucket geometry validation, nearest-bucket assignment, manifest schema + JSON io, `BucketSampler` with `state_dict()` resumption |
| `vidgen/vae/model.py` | Part 3 | Causal 3D VAE: past-only temporal padding, 4x temporal / 8x spatial compression, 16 latent channels; Tiny geometry `[B,3,17,64,64]` → `[B,16,5,8,8]`; T=1 → 1 latent; reparameterization + KL |
| `vidgen/text/encoder.py` | Part 5 | `TinyTextEncoder`: learned word embeddings over a fixed caption vocab → `[B, L, Dt]` + padding mask; cached CFG null embedding; `TransformerTextEncoder` (2026-10-21 flagship upgrade): word-hash tokenization + pre-norm transformer stack, selected per preset by `build_text_encoder` (`text_layers > 0`) — vidgen9b ~210M / vidgen14b ~311M params at Dt=1024, 128-token captions; `FrozenEncoderStub` documents the umT5-XXL swap path (interface: `encode(captions) -> ([B,L,Dt], mask)`, frozen, null cached) |
| `vidgen/dit/model.py` | Part 4+6 | Exact-unfold 3D patchify (bit-exact inverse), partitioned 3D RoPE (Tiny: 22/22/20 over head-dim 64), DiT blocks (full 3D self-attn + text cross-attn + FFN) with shared AdaLN-Zero modulation; `forward(latent, t, text, mask) -> velocity` |
| `vidgen/flow.py` | Part 7 | Rectified flow `x_t = (1-t)x_0 + t x_1`, velocity target `x_1 - x_0`; shifted logit-normal timesteps (s=5.0); 10-bin diagnostic losses |
| `vidgen/sampler.py` | Part 8 | Euler / Midpoint / RK4 ODE samplers with classifier-free guidance; oracle recovery < 1e-5 |
| `vidgen/checkpoint.py` | Part 7 | Save/load with python+numpy+torch RNG states → resume reproduces next-step loss bit-for-bit |
| `vidgen/train.py` | Part 7 | `DiTTrainer` + CLI: rectified-flow DiT training in VAE latent space on Moving MNIST; AdamW, grad-clip 1.0, 10% caption dropout, `--vae-ckpt` for a trained VAE; `--data-dir` trains on a prepared real-clip directory instead; DDP multi-GPU under `torchrun` (see Multi-GPU section) |
| `vidgen/train_vae.py` | Part 3 | `VAETrainer` + CLI: VAE training (recon MSE + KL), same checkpointing; `--data-dir` for real clips; `--init-weights` for a weights-only warm start from a donor VAE checkpoint (e.g. Round-1 tiny VAE → Small VAE; VAE arch is preset-independent); DDP multi-GPU under `torchrun` |
| `vidgen/distributed.py` | Part 7 | DDP helpers: torchrun env init (nccl on CUDA / gloo on CPU, no-op single-process fallback), per-rank device selection, barrier/destroy, `unwrap_model` / `plain_state_dict` (checkpoints never carry `module.` prefixes), per-rank seed derivation, global-batch sharding, explicit gradient sync for the unwrapped text encoder |
| `tests/test_distributed.py` | Part 7 | Helper unit tests (fallback, env parsing, seed/shard math, plain state dicts), proof single-process trainers are unwrapped/unsharded, and simulated 2-rank tests: disjoint clip slices, divergent-but-deterministic RNG streams, rank-1 writes no checkpoint |
| `vidgen/grow.py` | Part 7 | Model growth ("small brains first"): transplants a trained DiT's weights into a LARGER preset — head-aware width expansion, endpoint-preserving depth layer mapping, AdaLN chunk alignment, legacy→transformer text-embedding transplant; output is a weights-only checkpoint loadable via `train.py --init-weights`. See the Model growth section |
| `scripts/grow_checkpoint.py` | Part 7 | Growth CLI: donor DiT checkpoint + `--to <preset>` → grown weights-only checkpoint, with a transplant report (params carried vs fresh) |
| `tests/test_grow.py` | Part 7 | Growth coverage: layer/head mapping exactness, strict load into the target architecture, modulation chunk alignment, text transfer paths incl. word-bucket placement, payload validation, CLI + trainer `--init-weights` acceptance on the real tiny→small hop |
| `vidgen/pool.py` | Part 7 | Weight-averaging pool (DiLoCo-style) primitives: `--data-shard K/N` parsing + position math, checkpoint merge (float-only averaging in float64, optional outer step against the round-start base), strict compatibility verification (kind/preset/config/step/keys/shapes), RNG-side-effect-free payload loading |
| `scripts/average_checkpoints.py` | Part 7 | Coordinator merge tool: 2+ same-round worker checkpoints → one pooled weights-only checkpoint (no optimizer/sampler/RNG state; `pool` metadata records inputs/base/outer-lr) |
| `scripts/pool_status.py` | Part 7 | Coordinator inbox report: per-file kind/preset/step + merge-ready groups (same kind, preset, step) with the exact merge command |
| `tests/test_pool.py` | Part 7 | Pool coverage: shard parsing/positions, folder + Moving MNIST shard partitioning (disjoint, union = full, sorted-by-file order), exact merge math incl. outer step and buffer handling, every mismatch rejection, real tiny-preset pool rounds for both trainers (divergence, merged `--init-weights` load, cumulative step numbering), pool_status grouping |
| `vidgen/data/clips.py` | Part 1 | `FolderClipDataset`: reads a prepared directory (`manifest.jsonl` + uint8 `(T,H,W,3)` `.npy` clips); yields the exact Moving MNIST sample format; deterministic manifest-order indexing; `max_clips` cap |
| `vidgen/sample.py` | Part 8 | Sample a clip from a DiT checkpoint → VAE decode → `.npy` (+ PNG/GIF if Pillow installed) |
| `scripts/prepare_clips.py` | Part 1 | Colab-oriented CLI: downloads stock videos (Pixabay API via `PIXABAY_API_KEY` env var — recommended, free key issued instantly; Pexels API via `PEXELS_API_KEY` — only if you already have a key, since Pexels has paused issuing new ones; or `--source urls` with direct URLs) → decode (cv2, else imageio) → center 17-frame window → square crop → 64×64 → quality/dedupe filters → `.npy` + captioned manifest; per-video temp-dir cleanup; prints kept/skipped/MB stats |
| `notebooks/prepare_clips_drive.ipynb` | — | Colab notebook: Drive mount → repo → config ("City and people" queries, 500/query) → `getpass` API-key collector (Pixabay/Pexels) → verify cell (loads `FolderClipDataset`, prints stats, renders a filmstrip) → exact real-data `train_vae.py`/`train.py --data-dir` commands |
| `scripts/harvest_urls.py` | Part 1 | No-API-key URL harvester (stdlib only): fetches stock search pages with a browser User-Agent and extracts direct `.mp4` file URLs from the embedded HTML (Pexels: `videos.pexels.com/video-files/<id>/<id>-<variant>.mp4`, pagination via `?page=N`, one sd variant per video; Pixabay: optional secondary source, `_tiny`→`_medium` suffix swap kept only when HEAD returns 200). Pure-function parsing (`extract_pexels_urls`, `choose_variant`) unit-tested without network. Writes `urls.txt` for `prepare_clips.py --source urls` |
| `scripts/caption_clips.py` | Part 1/5 | 100% Free Tier Gemini vision captioner for prepared clips: replaces weak file/tag captions with dense 15-30 word scene descriptions (subject, action, setting, camera move, lighting); 3 frames resized to ≤512px; paced at 6s with `--max-calls` cap (default 240); graceful HTTP 429 quota handling; resume-safe atomic updates |
| `vidgen/data/dataset_tools.py` | Part 1 | Local dataset health + packing logic: memory-mapped clip/depth validation (missing, corrupt, dtype/shape, dark-sample, duplicate-reference, orphan checks) and verified multi-folder packing with collision-proof prefixed names and staged publish |
| `scripts/vidgin_helper.py` | Part 1/5 | Helper bundle CLI with three commands: `captions` (delegates to the free-tier captioner), `check` (trainer-readiness report, optional JSON), and `pack` (merge prepared groups into one verified training dataset) |
| `tests/test_helper_bundle.py` | Part 1/5 | Helper coverage: clean and damaged dataset reports, depth mismatch, orphans/duplicates, same-name chunk packing, bad-source/mixed-shape/non-empty-output refusal, CLI exit codes, and caption delegation |
| `vidgen/audio/` | Audio | Vidgin's own audio brain: `mel.py` (22050 Hz / 80 mels), `sound_model.py` (~71M Audio DiT), `vocoder.py` (~2.6M HiFi-GAN), `voice_model.py` (~11.4M Voice TTS). 100% in-repo code, zero external models, zero downloaded weights |
| `scripts/prepare_audio_data.py` | Audio | Prepares training data from FREE, no-key public sources: LJSpeech-1.1 (Public Domain / CC0) for voice; ESC-50 and UrbanSound8K subsets (CC-BY-NC 3.0) for sound; synthetic CPU generator |
| `scripts/train_audio.py` | Audio | Unified trainer adhering to `vidgen/train.py` conventions (--target sound, voice, vocoder); bit-for-bit resume, AdamW, RNG state saving |
| `scripts/add_audio.py` | Post-proc | End-to-end audio tool: plans scene sound, synthesizes voice dialogue + background ambience via in-repo checkpoints, ducks background (35% under speech), normalizes to -1 dB, muxes with ffmpeg, verifies with ffprobe |
| `requirements.txt` | — | `torch`, `numpy`, `pytest`, `pillow`, `scipy`, `soundfile` |
| `tests/test_audio.py` | Audio | Purity test (scans for forbidden external models/URLs) + parameter count contracts + mel math + forward pass shape tests |

## Notable design decisions / spec deviations

- **Text encoder**: Track A uses the learned `TinyTextEncoder` (small caption
  vocabulary) instead of frozen umT5-XXL. The DiT consumes `[B, L, Dt]` +
  padding mask either way, so a frozen umT5 can replace the module later
  (`FrozenEncoderStub` documents the exact interface contract). This keeps the
  whole pipeline trainable on CPU with zero downloads. The flagship presets
  (vidgen9b / vidgen14b) instead use the trainable `TransformerTextEncoder`
  (config `text_layers` 16 / 24, Dt=1024, 128-token captions, open word-hash
  vocabulary — any word is representable, not just a 27-word table); all other
  presets keep the legacy encoder, so existing checkpoints load unchanged.
- **VAE latents for DiT training**: `DiTTrainer` encodes with the VAE's `mu`
  (deterministic). A VAE trained via `train_vae.py` can be plugged in with
  `--vae-ckpt`; the CPU smoke path uses the randomly-initialized VAE, which is
  still a consistent latent space for the flow objective.
- **Manifest format**: JSON instead of Parquet (keeps `requirements.txt` to
  torch/numpy/pytest; no pyarrow). Schema validation semantics are identical.
- **End-to-end data acceptance**: runs on 120 clips instead of 1000 (same
  generator path; 1000 is just slower). The synthetic acceptance gates on
  sharpness + motion: the README's aesthetic ≥ 4.0 threshold targets natural
  video — black-background Moving MNIST scores ~0.6 entropy by construction.
- **RoPE split**: Tiny/Small use (22, 22, 20) over head-dim 64 (the all-even
  split nearest the 44/42/42 pattern); RoPE pair rotation requires even dims.
- **AdaLN-Zero warm-up in tests**: modulation gates are exactly 0 at init by
  design, so two tests take a couple of SGD steps first (then assert nonzero
  gradients / text sensitivity).
- **VAE normalization**: GroupNorm is applied per-frame (no temporal pooling),
  otherwise future frames leak into past latents through the norm statistics
  and break the causal receptive field (caught by the causality test).

## How to run (CPU smoke)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

# 1. tests (CPU; GPU tests auto-skip)
python -m pytest tests/ -q

# 2. train the VAE a few steps
python -m vidgen.train_vae --steps 12 --batch-size 2 --num-clips 16 \
    --ckpt-dir checkpoints/tiny-vae --log checkpoints/tiny-vae/train.jsonl

# 3. train the DiT in VAE latent space (uses the trained VAE above)
python -m vidgen.train --steps 12 --batch-size 2 --num-clips 16 \
    --vae-ckpt checkpoints/tiny-vae/latest.pt \
    --ckpt-dir checkpoints/tiny-dit --log checkpoints/tiny-dit/train.jsonl

# 4. resume training from a checkpoint (reproduces the next step bit-for-bit)
python -m vidgen.train --resume checkpoints/tiny-dit/latest.pt \
    --steps 6 --batch-size 2 --num-clips 16 --ckpt-dir checkpoints/tiny-dit

# 5. sample a clip from the trained checkpoint
python -m vidgen.sample --ckpt checkpoints/tiny-dit/latest.pt \
    --vae-ckpt checkpoints/tiny-vae/latest.pt \
    --prompt "digit 3 moving left" --steps 20 --outdir outputs/clip1
```

## Real training on a free GPU (Kaggle / Colab T4, 16 GB)

The Tiny preset (~21.5M DiT params + 3.7M VAE) fits comfortably on a T4.
Sensible starting point for Moving MNIST to visible quality:

```bash
# GPU install (no --index-url needed on Kaggle/Colab; torch is preinstalled)
pip install -r requirements.txt

# 1. train VAE: ~3-5k steps, batch 16, lr 1e-4  (~1-2 h on T4)
python -m vidgen.train_vae --device cuda --steps 4000 --batch-size 16 \
    --num-clips 20000 --lr 1e-4 --ckpt-dir checkpoints/tiny-vae \
    --ckpt-every 500 --log checkpoints/tiny-vae/train.jsonl

# 2. train DiT on VAE latents: ~10-20k steps, batch 16, lr 1e-4  (~4-8 h on T4)
python -m vidgen.train --device cuda --steps 15000 --batch-size 16 \
    --num-clips 20000 --lr 1e-4 --vae-ckpt checkpoints/tiny-vae/latest.pt \
    --ckpt-dir checkpoints/tiny-dit --ckpt-every 1000 \
    --log checkpoints/tiny-dit/train.jsonl

# 3. sample
python -m vidgen.sample --device cuda --ckpt checkpoints/tiny-dit/latest.pt \
    --vae-ckpt checkpoints/tiny-vae/latest.pt \
    --prompt "digit 3 moving left and digit 7 moving right" \
    --steps 50 --method midpoint --outdir outputs/clip_final
```

Tips: watch `checkpoints/*/train.jsonl` — DiT flow loss should fall steadily;
the `bin_loss` array shows per-timestep loss (high-t bins usually lag). If loss
plateaus early, raise `--num-clips` (more digit/speed variety) before touching
the model. For longer runs, Kaggle gives ~30 h/week of GPU; split into
sessions with `--resume`.

## Real-data training path (real stock clips, "City and people" theme)

Moving MNIST is synthetic; real clips teach the model real scenes. The real-data
path is three steps: **prepare → verify → train**. It is a *fresh* run —
the VAE retrains from scratch on the real clips.

**Dataset format.** A prepared directory holds `manifest.jsonl` (one JSON
object per line: `{"file": "clip_000001.npy", "caption": "..."}`) plus one
uint8 `.npy` file per clip of shape `(T, H, W, 3)` (e.g. `(17, 64, 64, 3)`).
`vidgen/data/clips.py::FolderClipDataset` reads such a directory and yields
samples in exactly the Moving MNIST format — `(float32 [3, T, H, W] in [-1, 1]
tensor, str caption)` — so both trainers consume either dataset.

**Prepare (Colab).** Open
[`notebooks/prepare_clips_drive.ipynb`](notebooks/prepare_clips_drive.ipynb):
it mounts Drive, clones the repo, and runs `scripts/prepare_clips.py`, which
downloads stock videos (Pixabay API — key via the notebook's `getpass` prompt
(Pixabay is the default: its free key is issued instantly after signup; Pexels
has paused issuing new API keys, so it only works if you already have one —
never hardcoded — or `--source urls` with a list of direct video URLs, no key),
then per video: download to a temp dir (cleaned after each video) → decode
(cv2, else imageio) → center 17-frame window at even stride → center-crop to
square → resize to 64×64 → skip too-short / corrupt / too-dark / near-static
clips (reusing `vidgen/data/filters.py` quality gates) → dedupe by content
hash → write `<id>.npy` + manifest line with a caption (Pixabay: the hit's
tags; Pexels: the search query; URL lists: a caption derived from the file name). It prints
downloaded/kept/skipped counts and total MB at the end. The notebook's
defaults collect the **"City and people"** theme: six queries × 500 videos
(≈3,000 clips, ~600 MB) into `MyDrive/vidgin-data/tiny-clips/`.

**Train on the clips** (after the Moving MNIST run finishes; checkpoints go
to separate Drive folders):

```bash
# 1. VAE from scratch on real clips (~1-2 h on T4)
python -m vidgen.train_vae --device cuda --steps 4000 --batch-size 16 \
    --data-dir /content/drive/MyDrive/vidgin-data/tiny-clips \
    --ckpt-dir /content/drive/MyDrive/vidgin-checkpoints/real-vae --ckpt-every 500 \
    --log /content/drive/MyDrive/vidgin-checkpoints/real-vae/train.jsonl

# 2. DiT on the real-trained VAE's latents (~4-8 h on T4)
python -m vidgen.train --device cuda --steps 15000 --batch-size 16 \
    --data-dir /content/drive/MyDrive/vidgin-data/tiny-clips \
    --vae-ckpt /content/drive/MyDrive/vidgin-checkpoints/real-vae/latest.pt \
    --ckpt-dir /content/drive/MyDrive/vidgin-checkpoints/real-dit --ckpt-every 500 \
    --log /content/drive/MyDrive/vidgin-checkpoints/real-dit/train.jsonl
```

Both trainers accept `--data-dir` (also as a `data_dir=` constructor arg);
when set it replaces Moving MNIST and the sampler sizes itself to the
manifest. Omit it and behavior is byte-identical to the synthetic path.

**Dataset storage, in one line:** clips must live on the training machine's
disk *while training* (the trainer re-reads them every epoch); once training
is done the model file carries everything it learned and the clips are only
needed again if you train more — so keep the dataset on cheap storage
(Drive / an external disk) as a reusable asset, not as a forever requirement.

## Multi-GPU training (DDP) — 2026-10-21

Both trainers (`vidgen/train.py`, `vidgen/train_vae.py`) support
**DistributedDataParallel** via `vidgen/distributed.py`. Motivation: Kaggle
gives 2× Tesla T4 free, and a single-GPU trainer wastes the second card.

**Opt-in, or nothing changes.** Multi-GPU activates only when the process
is launched with `torchrun` (which sets `RANK`/`WORLD_SIZE`/`LOCAL_RANK`).
Without torchrun every helper is a no-op: no process group, no wrapping,
no reseeding, no sampler change — training is byte-identical to before,
and all existing checkpoints load exactly as before (and vice versa:
DDP-trained checkpoints load into plain single-GPU models, verified).

**Batch-size semantics — read this.** `--batch-size` is **PER GPU** in
distributed mode: global batch = `--batch-size` × world size. Raising the
GPU count without touching `--batch-size` therefore *does* raise the
global batch (and changes the training dynamics, like raising
`--batch-size` would). To keep the same global batch on more GPUs, lower
`--batch-size` proportionally.

Kaggle 2× T4, Small preset (global batch 4 = 2 per GPU):

```bash
# VAE first (warm-started from the Round-1 VAE), then the DiT
torchrun --standalone --nproc_per_node=2 -m vidgen.train_vae \
    --preset small --device cuda --steps 4000 --batch-size 2 \
    --init-weights checkpoints/round1-vae.pt \
    --data-dir /kaggle/working/s2_dataset \
    --ckpt-dir /kaggle/working/ckpts/vae --ckpt-every 500

torchrun --standalone --nproc_per_node=2 -m vidgen.train \
    --preset small --depth --grad-checkpoint --device cuda \
    --steps 15000 --batch-size 2 \
    --data-dir /kaggle/working/s2_dataset \
    --vae-ckpt /kaggle/working/ckpts/vae/latest.pt \
    --ckpt-dir /kaggle/working/ckpts/dit --ckpt-every 500
```

**How ranks stay consistent, and where they diverge:**

- *Weights*: every rank builds the model from the same base seed, the
  optimizer is built on the raw modules, the DiT (and the VAE, in
  `train_vae`) is then wrapped in DDP; DDP all-reduces gradients each
  step, so all copies stay identical. Activation checkpointing
  (`--grad-checkpoint`) is applied to the raw DiT before wrapping and
  uses the non-reentrant implementation — verified working under DDP.
- *Text encoder*: deliberately NOT DDP-wrapped — DDP only permits
  `forward()` through the wrapper, but the trainer also calls
  `null_embedding()`. It stays replicated and its gradients are
  all-reduced + averaged explicitly after `backward()`
  (`distributed.sync_gradients`), which is the same averaged gradient
  DDP would have produced.
- *Data*: ONE `BucketSampler`, identically seeded on all ranks, draws
  global batches of `batch_size × world_size`; each rank keeps its own
  disjoint contiguous slice. The sampler state is therefore identical
  across ranks and rank 0's checkpoint carries the complete state.
- *Noise*: after identical weight init, each rank re-seeds its RNG
  stream to `seed + rank × 1_000_003`, so flow noise, timestep draws,
  caption/depth dropout and VAE reparameterization noise differ per
  rank. On a distributed `--resume`, the checkpoint's saved RNG (rank
  0's) is loaded first and then each rank re-derives its own stream
  from `(seed, rank)`: resuming with the same world size and `--seed`
  is deterministic. A *single-GPU* resume of a DDP checkpoint uses the
  saved RNG bit-for-bit, exactly as before. Note the one asymmetry:
  distributed runs are reproducible for a fixed world size; changing
  the world size mid-run changes slices/streams (true of DDP generally).
- *Checkpoints & logs*: only rank 0 prints, logs, and writes; all ranks
  barrier around each save. Checkpoints are written from the unwrapped
  modules, so keys never have a `module.` prefix.

**Honest scope limit:** DDP is data-parallel only — every rank holds the
full model, so DDP speeds up Small/480p training but does **not** make
the 9B/14B presets fit on small GPUs; that needs FSDP/ZeRO model
sharding, which is future work and is not implemented here.

**Verification (this VM, 2 CPU cores, no GPU, 2026-10-21):** real
2-process `torchrun` smokes on the Tiny preset, gloo backend, batch 1
per rank — DiT: 3 steps, loss 1.3106 → 1.1999, exactly one checkpoint
set written (by rank 0), strict-loaded into a plain DiT + text encoder
afterwards; DiT with `--grad-checkpoint`: 2 steps, exit clean; DiT
`--resume` under torchrun: resumed at step 3, ran to step 5; VAE:
3 steps, loss 0.9989 → 0.9960, checkpoint strict-loads into a plain
`CausalVAE`. Unit coverage in `tests/test_distributed.py`.

## Weight-averaging pool (multi-platform) — 2026-10-21

DDP (above) pools GPUs *inside one machine*. The pool pools **machines
that cannot talk to each other** — a Colab account, a Kaggle account and
a Lightning AI Studio — in the style of Google's DiLoCo: workers train
independently and sync rarely, by exchanging one checkpoint file per
round, so slow links and short free-tier sessions stop mattering.
Implemented in `vidgen/pool.py` + `scripts/average_checkpoints.py` +
`scripts/pool_status.py`; both trainers are pool-ready.

**The round loop:**

```
                 pooled checkpoint (round start, step S)
                    |            |            |
              worker 1       worker 2     worker N        <- Colab / Kaggle /
              --init-weights (same file for everyone)        Lightning / ...
              --data-shard 1/N  2/N          N/N   (disjoint data slices)
              --steps R         R            R     (same R for everyone)
                    |            |            |
              worker ckpts, all at step S + R
                    \            |           /
        coordinator: scripts/average_checkpoints.py
          out = base + outer_lr * (mean(workers) - base),  base = round start
                    |
              pooled checkpoint (step S + R)  ->  next round
```

**Coordinator (Spark's machine):**

```bash
# what has arrived, and which rounds are complete:
python scripts/pool_status.py ~/pool_inbox/
# merge a finished round (drop --base/--outer-lr for a plain mean):
python scripts/average_checkpoints.py \
    inbox/worker1.pt inbox/worker2.pt inbox/worker3.pt \
    --base pooled_round3.pt --outer-lr 1.0 --out pooled_round4.pt
# then publish pooled_round4.pt where every worker can fetch it
# (GitHub Release / Drive) — it is the next round's --init-weights.
```

**Worker round — DiT** (run the VAE round first, exactly like the
single-machine flow; `--vae-ckpt` as usual). Colab, Kaggle and Lightning
run the *identical* commands — only the shard number and the file
transport differ:

```bash
python -m vidgen.train --preset small --depth --grad-checkpoint \
    --device cuda --batch-size 2 \
    --init-weights pooled_round4.pt \
    --data-shard 2/3 --steps 500 \
    --data-dir <this platform's copy of the dataset> \
    --vae-ckpt <trained Small VAE checkpoint> \
    --ckpt-dir ckpts/worker2 --ckpt-every 500
# -> send ckpts/worker2/latest.pt (step = round start + 500) to the coordinator
```

**Worker round — VAE:**

```bash
python -m vidgen.train_vae --preset small --device cuda --batch-size 2 \
    --init-weights pooled_vae_round4.pt \
    --data-shard 2/3 --steps 500 \
    --data-dir <dataset> --ckpt-dir ckpts/vae_worker2 --ckpt-every 500
```

Inside one platform session a worker may itself span that machine's
GPUs with DDP — e.g. on Kaggle's 2× T4, prefix the worker command with
`torchrun --standalone --nproc_per_node=2` (per-GPU `--batch-size`
rules from the Multi-GPU section apply unchanged). Pool and DDP
compose: DDP syncs every step inside the machine; the pool syncs once
per round across machines.

**Rules that make it work:**

- *Same start, same steps.* Every worker in a round starts from the
  identical pooled checkpoint and trains the identical `--steps`. The
  merge tool hard-errors on any kind/preset/config/step mismatch, so a
  bad round fails loudly instead of averaging garbage.
- *Disjoint data.* `--data-shard K/N`: dataset entries are ordered
  deterministically (folder datasets: sorted by file name; Moving MNIST:
  clip index), worker K takes positions `% N == K-1`. Shards are
  disjoint and their union is the full dataset.
- *Fresh inner optimizer every round.* A pooled checkpoint is
  weights-only — no optimizer, sampler, or RNG state — because Adam
  moments are private to one worker's trajectory on its own data slice
  and cannot be meaningfully averaged. `--init-weights` therefore
  starts each worker with a fresh optimizer; cross-round momentum
  lives in the merge's outer step instead. Step numbering is
  cumulative: `--init-weights` on a pooled checkpoint continues from
  its stored step (override with `--start-step`).
- *One round of drift, max.* Workers only stay averageable because
  they re-sync every round; letting a worker run many rounds alone
  before merging would let its weights drift into a different basin
  and the average would degrade. Keep R modest (hundreds of steps) —
  that is the whole design point.
- *A missing worker just misses the round.* Nothing is in lockstep:
  if a session dies, merge whoever arrived (2+ checkpoints), or wait.

**Honest limits.** Sync happens per round, not per step, so pooled
training is not bit-equivalent to one big DDP run and nobody should
expect it to match DDP's exact trajectory; DiLoCo's published results
say final quality is close at these sync intervals, but Vidgin's own
evidence so far is the proof run below, not a quality benchmark. The
merged model must still fit on ONE worker's GPU (the pool divides
*data and time*, not model memory — that remains FSDP territory).

**Verification (this VM, 2 CPU cores, no GPU, 2026-10-21):** scripted
proof round on Moving MNIST, tiny preset — one init checkpoint (step
0); worker A `--data-shard 1/2` and worker B `--data-shard 2/2`, 24
steps each from the same init: losses 1.3046 → 0.7277 (A) and
1.3101 → 0.7239 (B); the workers' final weights verifiably differ;
the merged checkpoint equals the float64 mean cast back to float32
exactly (max deviation 7.5e-09, pure float32 cast rounding), and a
`--base` merge with `--outer-lr 1.0` reproduces the plain mean
bit-for-bit; a continuation run from the merged checkpoint resumed
numbering at step 24 and trained 12 more steps, loss 0.7803 → 0.4962,
all finite. Merge math, every mismatch rejection, shard partitioning
and both trainers' round plumbing are unit-covered in
`tests/test_pool.py` (13 tests).

## Model growth (small brains first) — 2026-10-22

The ladder's transfer mechanism: when a smaller model finishes
training, the next size up does not start from random weights.
`vidgen/grow.py` transplants the trained DiT's weights into the
larger preset's geometry and `scripts/grow_checkpoint.py` wraps it as
a CLI; the output is a weights-only checkpoint (with a `grow`
metadata block) that trains as a fresh run via the existing flag:

```
python scripts/grow_checkpoint.py checkpoints/small/latest.pt \
    --to vidgen480p --out checkpoints/grown_480p.pt
python -m vidgen.train --preset vidgen480p \
    --init-weights checkpoints/grown_480p.pt --vae-ckpt <vae ckpt> ...
```

How the transplant works:

- **Width** — every weight lands at its semantically correct
  position. Attention projections expand *head-aware*: target head h
  inherits the full q/k/v slice of source head `h * Hs // Ht` (a naive
  top-left copy would mix heads). In each matrix the (old outputs) ×
  (new inputs) block is zeroed, so transplanted outputs do not read
  brand-new inputs at step 0; remaining new capacity is initialized
  like a fresh model (PyTorch defaults; the architecture's zero-init
  parts — AdaLN modulation, depth stem — stay zero).
- **Depth** — target block i inherits source block
  `round(i * (Ls-1) / (Lt-1))`: the source stack stretches evenly,
  first→first and last→last. AdaLN projections expand per modulation
  chunk (9 shared / 2 final), never as a flat matrix.
- **Text** — legacy→legacy and transformer→transformer grow by the
  same rules. Legacy→transformer (Small/480p → 9B/14B) transplants
  only the word embeddings: each known word is written into the
  bucket the transformer's word-hash tokenizer assigns it (27 words,
  0 collisions for the current vocab); the transformer stack starts
  fresh. Transformer→legacy is rejected as a downgrade.
- **Depth conditioning** is inherited from the source config; the
  CLI's `--depth` adds a fresh zero-init stem to a plain source.
- The **VAE** needs no growth: one identical architecture at every
  size, transferred exactly by `train_vae.py --init-weights`.

Honest limits: the grown model does not reproduce the source's
function (LayerNorm statistics, token grid and RoPE geometry all
change with size); every learned weight lands in the slot with the
same meaning and the new capacity starts at fresh-model scale. The
big model still trains — growth is a head start, not an education.
Same-geometry "growth" is refused (use the checkpoint directly).
Unit coverage in `tests/test_grow.py` (14 tests): mapping exactness,
strict loads, forward finiteness, all text paths, determinism, and
the real tiny→small CLI + trainer-acceptance path.

CPU proof (2026-10-22, this VM): a Tiny donor trained 30 steps
(Moving MNIST, loss 1.3067 → 0.4183) was grown to Small —
71,905,217 of Small's 285,943,617 DiT+text params (25.1%) are
transplanted values, the rest fresh capacity. Fair race (fresh Small
vs grown Small, identical batches/noise via a common re-seed before
stepping): **first-step loss 1.3970 fresh vs 0.7677 grown — the grown
model starts 45% lower**, i.e. the donor's learning demonstrably
carried into the bigger brain before it trained at all. Caveat, also
measured: at the default lr 1e-4 the grown model's second step
jumped to 2.1555 (Adam's first full-size step disturbs the
transplant); re-run at lr 2e-5 the grown model held its lead
(step 2: 0.9989, still ahead of fresh's 1.2050). Prescription —
recorded in every grown checkpoint's `grow` metadata: start a grown
model at ~1/5 of the from-scratch learning rate for its first
phase. Proof artifacts: `~/workspace/tmp/grow_proof/` (donor, grown
checkpoint, JSON).

## What remains GPU/cluster-only (from GPU_CHECKLIST.md)

These are documentation-only — they need hardware nobody here has:

- **100k web-video ingest + cluster VLM captioning** — 8× A100 (80 GB).
- **3D VAE pretraining on 10M clips** — 8× H100 (80 GB), bf16.
- **Frozen umT5-XXL (4.7B) embedding extraction** — ≥ 24 GB VRAM.
- **Vidgen 1.3B DiT training** — 4 nodes × 8 H100, FSDP2 + Ulysses sequence
  parallel (the config dimensions exist in `get_preset("vidgen480p")`; DDP
  data-parallel now exists (see Multi-GPU section), the FSDP2 + Ulysses
  machinery does not).
- **Vidgen 14B / 500k-step run** — 8 nodes × 8 H100 (`get_preset("vidgen14b")`
  is dimension definitions only).
- **64× H100 MFU/throughput benchmark** — 64 physical H100s.
- **VBench 500-sample cluster eval** — 4× A100/H100.
- **30 s Moving MNIST rollout + VRAM profiling** — needs a real NVIDIA GPU
  (the chunked-decode and O(1)-memory machinery is not implemented in
  `vidgen/sample.py`, which decodes whole clips).

## Stage 1 upgrade: long video by segment chaining (2026-10-19)

`vidgen/extend.py` generates videos longer than the model's native window
(Tiny: 17 frames) without retraining. Segment 1 is sampled normally; each
following segment is sampled with its first `ctx_latent` latent frames clamped
to the rectified-flow path toward the VAE encoding (posterior mean) of the
accumulated video's tail — flow-matching inpainting in latent space — then
stitched with a short crossfade over the reconstructed context to hide the
VAE seam. Rationale: the DiT has no cross-segment attention and was trained on
fixed windows, so latent prefix clamping is the closest principled
conditioning the architecture supports; purely pixel-level chaining would
restart motion at every join. Segment k uses seed + k, so a fixed base seed
reproduces the whole video.

CLI (`vidgen/sample.py`): `--seconds 30` (segments computed from the project
playback rate of ~290 ms/frame, so ~103 frames ≈ 30 s), or explicit
`--segments N`; `--ctx-latent` controls context length. Chained GIFs default
to 290 ms/frame.

Tests: `tests/test_extend.py` (7 tests) — segment arithmetic, output frame
counts/shapes, finite in-range values, seed reproducibility. Full suite after
this change: **72 passed, 2 skipped** (was 64 passed, 2 skipped; +1 test for
gradient checkpointing, below). Still unproven until a GPU run: visual seam
quality and motion coherence over many segments (drift is expected to grow
with segment count at Tiny scale).

Small preset (256×256, 33 frames) readiness, verified on this VM
(2 CPU cores, ~8 GB RAM) on 2026-10-19:

- VAE: one full training step via `train_vae.py --preset small`
  (3.65M params), checkpoint saved, loss 0.9982.
- DiT: 285.9M params, 2,304 tokens. Inference forward at full latent shape
  (1, 16, 9, 32, 32) passes (finite, correct shape, ~1.6 GB RSS). A plain
  training step OOM-kills on this VM during the training forward —
  activation memory, not a code bug (attention already uses SDPA).
- Fix: opt-in activation checkpointing — `DiT.set_grad_checkpoint()` and
  `train.py --grad-checkpoint`. With it, the training forward+backward
  peaks at ~3.3 GB RSS and a full CLI training step completes:
  `train.py --preset small --steps 1 --grad-checkpoint` saved
  step_000001.pt, loss 1.4266.
- Sampling: `sample.py` from the 1-step checkpoints produced
  frames (33, 256, 256, 3) + GIF end-to-end (content is near-mean noise, as
  expected after a single step — pipeline proof, not quality proof).

Small training settings for the GPU stage: batch 1–4 with
`--grad-checkpoint` on a 16 GB card; real training is Stage 2 work.

## 3D depth conditioning — Dawood's idea (2026-10-20)

Dawood's proposal: if the model learns 3D mapping, video generation gets
easier — a video is a camera moving through a 3D world, so a model shown the
world's layout does not have to re-invent geometry every frame. Implemented
as depth-map conditioning, end to end (`vidgen/depth.py` documents the
design and its honest limits):

- **Prep** — `scripts/prepare_clips.py --depth` estimates a depth map per
  kept clip (MiDaS-small, per-clip normalized, larger = nearer), saves
  `<clip>_depth.npy`, and records `"depth"` in the manifest.
- **Data** — `FolderClipDataset.get_depth(i)` returns float `[1,T,H,W]`
  in `[0,1]` or `None`; the `(tensor, caption)` sample contract is
  unchanged.
- **Model** — `VidgenConfig.use_depth` (default **off**, so all existing
  checkpoints and code paths behave exactly as before). When on, the DiT
  grows a zero-init `depth_stem` Conv3d: the map is reduced to the latent
  grid by `depth_to_latent_grid` (temporal indices `[0,1,5,9,...]` aligned
  to the causal VAE, spatial area-average) and the stem's tokens are added
  to the latent tokens. Zero-init ⇒ a fresh depth model is exactly the
  plain model until training teaches the stem.
- **Training** — `vidgen.train --depth`; per-clip maps are fed, clips
  without one contribute zeros, and whole-batch map dropout
  (`--depth-dropout`, default 0.1, drawn from the checkpointed RNG) keeps
  text-only generation strong — at text-to-video time there is no map
  unless one is supplied. On resume, the mode is read from the
  checkpoint's stored config, never from the CLI flag; `sample.py`
  likewise rebuilds its config from the checkpoint.
- **Sampling** — `sample_latents(..., depth=...)` and
  `vidgen.sample --depth-npy <uint8 [T,H,W]>` steer a depth-trained model
  with a fixed layout (depth conditions both CFG branches). The sampler
  passes the depth argument only when a map is supplied, so pre-depth
  models (and test doubles) keep working untouched.

Tests: `tests/test_depth.py` (15 tests) — config gate and old-checkpoint
roundtrip, grid alignment, zero-init invariance, signal after the stem
wakes, gradient flow, misuse errors, dataset loading, a depth train step
+ resume keeping the mode, plain resume staying plain, sampler path, and
estimator input validation (the estimator's weights download is not
exercised on this VM). Full suite after this change: **87 passed,
2 skipped** (was 72 passed, 2 skipped).

Not yet proven (needs the GPU stage): that depth conditioning measurably
improves consistency at Small/480p — that comparison is Stage 2/3
measurement work, and the first depth-prepped dataset arrives with the
256×256 re-prep. Depth here is relative and single-view (the visible
surface only); it is a strong geometric hint, not a true 3D model.

## Captioning: Free-tier Gemini Vision upgrade (`scripts/caption_clips.py`)

Stock video captions from file names or tag search queries are weak labels
that hinder prompt adherence. `scripts/caption_clips.py` replaces weak captions
with rich, dense 15–30 word descriptions written by Gemini vision.

### Key Rules & $0 Cost (100% Free Tier)
- **Zero Budget ($0 Cost)**: The tool runs 100% free exclusively using the Gemini API free tier. No paid billing is required or assumed.
- **Key Security**: The API key is read **ONLY** from the `GEMINI_API_KEY` environment variable, or via an interactive `getpass` prompt if unset. It is **never** hardcoded, never printed to console, and never saved to any file.
- **Pacing & Limits**:
  - The script waits at least 6 seconds between requests (`--pace 6.0`) to comfortably respect free-tier RPM limits.
  - `--max-calls` (default `240`): After reaching the specified number of captions, it flushes progress and stops cleanly for the day.
  - **Graceful Quota Handling**: If an HTTP 429 / resource exhausted quota error is returned, the script avoids looping or failing; it immediately saves all progress to `manifest.jsonl`, prints:
    `Free quota reached — run again tomorrow; it will continue where it stopped`
    and exits cleanly (exit code 0).
- **Model Choice**:
  - Default: `gemini-2.5-flash` (`--model gemini-2.5-flash`).
  - Alternative: `gemini-2.5-flash-lite` (`--model gemini-2.5-flash-lite`). Its free tier allows more requests per day, allowing the dataset to complete in fewer days with slightly simpler caption phrasing.
- **Frame Processing & Prompting**:
  - Extracts 3 frames (first, middle, last) from each `.npy` clip (`uint8 (T,H,W,3)`).
  - Resizes each frame so its widest side is at most 512 px (preserving aspect ratio) and encodes as JPEG.
  - Sends all 3 frames with the prompt enforcing: ONE present-tense sentence of 15 to 30 words covering subject, action/motion, setting, camera movement, and lighting/time, omitting forbidden words ("video", "clip", "footage", "audio", "appears", "seems").
- **Resume Safety & Data Preservation**:
  - Moves the original weak caption to `"caption_weak"`, sets `"caption"` to the new caption, and tags `"caption_source": "gemini"`.
  - Skips already-processed clips on subsequent runs unless `--overwrite` is specified.
  - Preserves all other manifest keys (such as `"depth"`) completely untouched.
  - Flushes to a temporary file and atomically replaces `manifest.jsonl` every 25 clips and on exit.

### How to Run

```bash
# Set your free API key
export GEMINI_API_KEY="your-gemini-api-key"

# Run with defaults (gemini-2.5-flash, max 240 calls/day, 6s pacing)
python scripts/caption_clips.py --data-dir /content/drive/MyDrive/vidgin-data/tiny-clips

# Or use gemini-2.5-flash-lite for higher daily free request allowance
python scripts/caption_clips.py --data-dir /content/drive/MyDrive/vidgin-data/tiny-clips \
    --model gemini-2.5-flash-lite --max-calls 500

# Re-run after quota reset: automatically resumes and skips already-captioned clips
python scripts/caption_clips.py --data-dir /content/drive/MyDrive/vidgin-data/tiny-clips
```

Tests: `tests/test_caption_clips.py` (12 tests) verifies frame extraction, resizing, manifest updates, depth key preservation, resume-skip, overwrite, pacing stop, 429 quota handling, and API key prompts using a mocked client.

## Helper bundle (`scripts/vidgin_helper.py`) — 2026-10-22

The helper bundle puts the everyday dataset jobs behind one small tool.
It does not train a model and does not add any external brain to Vidgin:
the caption command reuses the existing free-tier Gemini captioner only
to write training labels, while `check` and `pack` are fully local.

```bash
# 1. Write proper captions (same free-tier rules as caption_clips.py)
python scripts/vidgin_helper.py captions --data-dir /path/to/clips

# 2. Check a folder before training: missing/corrupt files, wrong
#    dtype or shape, too-dark samples, missing/mismatched depth maps,
#    duplicate manifest references, and orphan files. Arrays are
#    memory-mapped, so checking a 256x256 dataset does not load whole
#    clips into RAM. Add --json report.json for a machine-readable copy.
python scripts/vidgin_helper.py check /path/to/clips \
    --frames 33 --height 256 --width 256 --require-depth

# 3. Merge finished groups into one dataset. Every source is checked
#    first; file names get group prefixes so two folders that both
#    contain clip_000001.npy cannot overwrite each other. The pack is
#    built in a staging folder and verified before it is published.
python scripts/vidgin_helper.py pack /path/to/group1 /path/to/group2 \
    --labels group1,group2 --out /path/to/training-dataset
```

`check` exits 0 only when there are no error-level problems (warnings
such as a dark clip are printed but do not fail the command unless
`--strict` is used). `pack` refuses a damaged source, sources with
different clip shapes, and a non-empty output directory unless
`--overwrite` is explicitly given.

Tests: `tests/test_helper_bundle.py` (8 tests) covers the clean/damaged
reports, same-name chunk packing, refusal cases, CLI exit codes, and
caption delegation.

### Dataset splits + no silent synthetic fallback — 2026-10-22

Phase 1 of the targeted improvement programme (Dawood's decision:
keep Vidgin 100% its own model, improve its weakest parts):

```bash
# 4. Deterministic train/val/test split manifests inside a dataset.
#    Clips sharing a manifest "group"/"source"/"source_url" field stay
#    in ONE split (no source leakage); same seed = same split.
python scripts/vidgin_helper.py split /path/to/training-dataset \
    --ratios 0.8,0.1,0.1 --seed 1234

# Train on one split; --require-real makes a missing --data-dir a loud
# error instead of a silent synthetic Moving MNIST run.
python -m vidgen.train_vae --preset small --data-dir /path/to/training-dataset \
    --data-split train --require-real ...
python -m vidgen.train --preset small --data-dir /path/to/training-dataset \
    --data-split train --require-real ...
```

`make_splits` (in `vidgen/data/dataset_tools.py`) writes
`manifest_train.jsonl` / `manifest_val.jsonl` / `manifest_test.jsonl`
plus `splits.json` into the dataset directory; `FolderClipDataset`
gained a `manifest_name` parameter and both trainers a `--data-split`
option to consume them. Without `--data-dir`, both trainers now print
a loud warning that the run is synthetic smoke-only, and
`--require-real` turns that fallback into an error. Tests:
`tests/test_helper_bundle.py` split tests (grouped, deterministic,
disjoint, consumed by the dataset, CLI wiring, `--require-real` guard
in both trainers).

### Caption-trained tokenizer (Phase 2) — 2026-10-22

The text encoders used to tokenize lossily: the legacy encoder's fixed
27-word table mapped every other word to `<unk>`, and the flagship
transformer hashed words with crc32 into 8,192 buckets (distinct words
could collide into one bucket). `vidgen/text/tokenizer.py` adds
`CaptionTokenizer`: a word-level vocabulary **trained on the dataset's
own captions** — deterministic (frequency order, alphabetical ties),
with `<pad>/<unk>/<bos>/<eos>` specials, JSON save/load, and an
`audit_hash_collisions` helper that counts how many distinctions the
old hash scheme collapses for a given vocabulary.

```bash
# Build the vocabulary from a prepared dataset's captions.
python scripts/build_tokenizer.py --data-dir /path/to/dataset \
    --out /path/to/tokenizer.json

# Train with it (both encoders accept it; the mode is recorded in the
# checkpoint config as text_tokenizer="caption").
python -m vidgen.train --preset small --data-dir /path/to/dataset \
    --tokenizer-vocab /path/to/tokenizer.json ...
```

The tokenizer is **stored inside every checkpoint** (`"tokenizer"`
payload key): resume rebuilds it from the checkpoint, `sample.py`
rebuilds it for inference, and loading weights across tokenizers (or
across hash/caption modes) fails with a clear `ValueError` instead of
a shape error or silent corruption. Hash mode remains the default, so
every existing checkpoint behaves exactly as before. Honest limits: a
tokenizer only gives the encoder distinct symbols — word *meaning* is
still learned during training, unseen words stay `<unk>`, the DiT
grow tool refuses caption-mode encoders (ids are vocabulary positions,
not hash buckets), and pool merging of caption-mode checkpoints is not
yet supported (the merged payload would lack the tokenizer, and
resume then refuses it loudly). Tests: `tests/test_tokenizer.py`
(18 tests).

### Measured latent scale + VAE quality gate (Phase 3) — 2026-10-22

`latent_scale` used to be a hand-set 1.0 that was never measured.
`vidgen/vae/stats.py` now measures the trained VAE's actual latent
distribution (per-channel and global mean/std, streaming over the
dataset) plus reconstruction quality (MSE / PSNR of decode(encode(x))),
and `scripts/measure_latents.py` turns that into a report, optional
input-vs-reconstruction preview PNGs, and a pass/fail PSNR gate:

```bash
# After VAE training, BEFORE DiT training:
python scripts/measure_latents.py --vae-ckpt /path/to/vae.pt \
    --preset small --data-dir /path/to/dataset --max-clips 200 \
    --out latent_report.json --preview-dir previews --min-psnr 15

# Train the DiT with the measured value (stored in the checkpoint
# config; resume restores it, sampling inherits it):
python -m vidgen.train --preset small --data-dir /path/to/dataset \
    --vae-ckpt /path/to/vae.pt --latent-scale <recommended> ...
```

First real measurement (tool proof, 2026-10-22): the Round 1 VAE
measured over Moving MNIST clips gives global latent std 1.686, i.e.
a recommended scale of **0.593, not 1.0** — the hand-set value was
indeed wrong; the number that matters must be measured on each run's
own dataset with that run's VAE (the Stage 2 runbook does this on the
assembled dataset before DiT training). Tests:
`tests/test_latent_stats.py` (3 tests).

### Training efficiency: schedules, AMP, accumulation (Phases 4/5) — 2026-10-22

Both trainers share `vidgen/train_utils.py`:

- **LR warmup + schedule** — `--warmup-steps N`, `--lr-schedule
  {none,cosine,linear}`. Cosine/linear decay over the run's `--steps`
  (a schedule without a horizon is an error, not a silent change);
  cosine floors at 5% of peak. Default stays constant LR.
- **Mixed precision** — `--amp {none,bf16,fp16}` wraps the forward in
  autocast; fp16 on CUDA adds a GradScaler (saved/restored in
  checkpoints). Default off.
- **Gradient accumulation** — `--grad-accum N` runs N micro-batches
  per optimizer step (loss scaled 1/N, sync/clip/step once), raising
  the effective batch without raising peak memory.
- **Throughput logging** — every step record now carries `secs` and
  `clips_per_sec` alongside loss/lr/grad_norm.

The resolved options are stored in each checkpoint's `train_config`
(and scheduler/scaler state alongside); resume restores them from the
checkpoint unless explicitly overridden on the command line. Defaults
reproduce the historical behaviour exactly — old checkpoints carry no
`sched`/`scaler` keys and load as before. Tests:
`tests/test_train_utils.py` (5 tests).

### Evaluation suite + MP4 export (Phases 6/7) — 2026-10-22

Falling loss alone never proved quality. `vidgen/evaluate.py` adds
non-learned pixel metrics (no pretrained evaluator — ownership rule):
motion (mean abs frame diff; 0 = frozen), flicker (brightness
second-difference), sharpness (gradient magnitude), diversity across
prompts/seeds (near 0 = mode collapse), and a memorization check
(nearest-training-clip distance; near 0 = copying). Prompt adherence
is honestly **not scored** — it needs a learned evaluator or human
review of the saved videos.

```bash
python scripts/evaluate_checkpoint.py \
    --dit-ckpt /path/to/dit.pt --vae-ckpt /path/to/vae.pt \
    --baseline /path/to/old_dit.pt --train-dir /path/to/dataset \
    --out-dir /path/to/eval_out
```

Fixed prompts/seeds/settings are used for every checkpoint; each
generation is saved as a verified MP4 (`vidgen/video_io.py`, OpenCV
with an ffmpeg fallback, frame count re-checked after writing) next
to a `report.json`/`report.md` comparing baseline vs candidate.
`sample.py` outputs now include an MP4 alongside the npy/PNG/GIF.
Tests: `tests/test_evaluate.py` (4 tests, including an end-to-end
evaluation of a real tiny checkpoint).

## Vidgin's own audio brain (`vidgen/audio/`)

Vidgin generates silent video. Rather than relying on external APIs, closed services, or third-party pretrained weights, Vidgin incorporates its **OWN audio brain**, engineered directly inside this repository, so generated videos gain background sound and a speaking voice that are 100% Vidgin's.

### The Ownership Rule
The paramount rule of this architecture: **NO third-party AI models. No SadTalker, no Wav2Lip, no AudioLDM, no TTS APIs or cloud services (no edge-tts, no Gemini audio), and NO pretrained weights of any kind, anywhere.**

Every audio model is defined in this repo's PyTorch code (`vidgen/audio/`), randomly initialized, and trained by this repo's own training code (`scripts/train_audio.py`) on free, legally clean public data. Standard libraries and system tools (PyTorch for tensors, ffmpeg for muxing) are used for infrastructure; the models, architectures, and weights are ours.

### What Each Model Learns

1. **Log-Mel Spectrogram Utilities (`vidgen/audio/mel.py`)**:
   Standard acoustic representation across all models: sample rate 22050 Hz, 80 mel frequency bins, 1024-point FFT with Hann window, and hop length 256. Constructed in pure PyTorch with triangular mel filterbank basis and Griffin-Lim fallback phase reconstruction.
2. **Audio Diffusion DiT (`vidgen/audio/sound_model.py`) — ~70.86M Parameters**:
   A small text-conditioned audio diffusion transformer (Audio DiT) that generates 80-channel log-mel spectrograms from text captions. Reuses Vidgin's exact block patterns: 2D spacetime/frequency-time patchify, 2D partitioned RoPE (frequency, time) for spatial and temporal translation invariance, AdaLN-Zero timestep modulation, self-attention, and text cross-attention. Learns rectified flow velocity fields ($x_1 - x_0$) with shifted logit-normal scheduling ($s=5.0$) and 10% caption dropout for classifier-free guidance (CFG).
3. **HiFi-GAN Vocoder (`vidgen/audio/vocoder.py`) — ~2.64M Parameters**:
   Vidgin's in-repo neural vocoder mapping log-mel spectrograms back to time-domain audio waveforms. Generator employs 4 transposed convolutions (upsample factors $[8, 8, 2, 2]$ for $256\times$ total upsampling) coupled with Multi-Receptive Field Fusion (MRF) ResNet blocks with kernel sizes $[3, 7, 11]$ and dilations $[1, 3, 5]$. Trained against Multi-Period Discriminators (MPD) and Multi-Scale Discriminators (MSD) using feature matching loss, adversarial loss, and $45\times$ mel spectral L1 loss.
4. **Single-Speaker Voice TTS (`vidgen/audio/voice_model.py`) — ~11.42M Parameters**:
   A compact single-speaker Transformer text-to-mel TTS model. Features a character-level English tokenizer, 6-layer Transformer text encoder ($\text{dim}=256$), 2-layer 1D Conv duration predictor and length regulator alignment mechanism, 6-layer Transformer mel decoder, and 5-layer convolutional Post-Net refinement. Outputs 80-channel log-mel spectrograms ready for synthesis by the vocoder.

### Training Data Sources & Licenses (100% Free, No-Key Required)

Data is prepared using `scripts/prepare_audio_data.py`:
- **Voice**: **LJSpeech-1.1** (~24 hours of audio, 13,100 clips, single English female speaker).
  - *License*: **Public Domain (LibriVox / CC0)**.
  - Downloaded and resampled to 22050 Hz mono, producing `manifest_voice.jsonl` with `{"wav": ..., "text": ...}`.
- **Background Sound**: **ESC-50** (2,000 environmental clips in 50 categories) and **UrbanSound8K** subset.
  - *License*: **Creative Commons Attribution-NonCommercial (CC-BY-NC 3.0)**.
  - Each clip's class label is mapped to a rich diffusion caption via a curated template map (e.g. `"rain"` $\to$ `"steady rainfall with gentle falling water drops"`, `"traffic"` $\to$ `"busy city traffic and passing cars"`), producing `manifest_sound.jsonl`.
- **Vocoder Ground Truth**: Ground-truth log-mels and waveforms combined from voice + sound datasets (`manifest_vocoder.jsonl`).
- **Synthetic CPU Smoke**: `scripts/prepare_audio_data.py --synthetic` generates deterministic synthetic audio and manifests instantly for local and CI testing without internet access.

### Colab Training Sequence (Exact Commands in Order)

All three models train within a single free Google Colab T4 GPU instance (16 GB VRAM). Train in this exact sequence:

```bash
# 0. Prepare training data (or use --synthetic for smoke test)
python scripts/prepare_audio_data.py --target all --out-dir data/audio

# Step 1: Train the Vocoder (mel -> waveform)
# Learns audio reconstruction on ground-truth mels from both voice and sound clips
python scripts/train_audio.py \
    --target vocoder \
    --steps 20000 \
    --batch-size 8 \
    --lr 2e-4 \
    --device cuda \
    --data-manifest data/audio/manifest_vocoder.jsonl \
    --ckpt-dir checkpoints/vocoder \
    --log checkpoints/vocoder/train.jsonl

# Step 2: Train the Sound DiT Model (caption -> mel)
# Learns rectified flow diffusion for ambient scene sound
python scripts/train_audio.py \
    --target sound \
    --steps 15000 \
    --batch-size 8 \
    --lr 1e-4 \
    --device cuda \
    --data-manifest data/audio/manifest_sound.jsonl \
    --ckpt-dir checkpoints/sound \
    --log checkpoints/sound/train.jsonl

# Step 3: Train the Voice TTS Model (text -> mel)
# Learns character alignment and speech mel synthesis for ONE speaker
python scripts/train_audio.py \
    --target voice \
    --steps 15000 \
    --batch-size 8 \
    --lr 2e-4 \
    --device cuda \
    --data-manifest data/audio/manifest_voice.jsonl \
    --ckpt-dir checkpoints/voice \
    --log checkpoints/voice/train.jsonl
```

### Running the Post-Processing Tool (`scripts/add_audio.py`)

Takes a generated silent video and equips it with background audio and optional spoken dialogue:

```bash
# Video with background ambience matching the video's scene prompt
python scripts/add_audio.py \
    --video outputs/nature_clip.mp4 \
    --caption "steady rainfall with gentle drops on green leaves" \
    --out outputs/nature_with_audio.mp4

# Video with both spoken dialogue and ducked background sound
python scripts/add_audio.py \
    --video outputs/street_clip.mp4 \
    --caption "city traffic and passing cars" \
    --dialogue "we are heading downtown now" \
    --out outputs/street_with_dialogue.mp4
```

- **Ducking**: Background sound plays at 100% volume, automatically ducking to 35% during speech activity with smooth gain transitions.
- **Normalization & Mux**: Peak loudness normalized to -1 dB (0.891), multiplexed into MP4 with AAC audio via `ffmpeg`.
- **Strict Verification**: If any required checkpoint is missing, the tool prints the exact `train_audio.py` command needed to train it and terminates. It NEVER falls back to any external model or third-party service.

### Honest v1 Limits in Plain Words
- **v1 Voice is ONE voice**: Single English speaker (LJSpeech), modest quality. It does not do multi-speaker or multilingual voice cloning in v1.
- **v1 Sound knows a small world**: Atmospheric ambience focused on Vidgin's video themes (city, street, traffic, rain, storm, ocean, beach, forest, nature, crowd, campfire).
- **Growth path**: Both models scale with data volume and training steps, following the exact same ladder as the video DiT models.

### Architectural Decision: Lip-Sync and Facial Expressions
Lip-sync and facial expressions are **NOT tools and will NOT be built from outside models** (no Wav2Lip, SadTalker, or third-party mesh warping). 
In Vidgin's philosophy, natural talking faces, jaw motion, and facial expressions come directly from the **video brain itself**, trained on real video footage of people speaking (a designated future data theme). This ensures faces move with natural physics, lighting, and anatomy, rather than pasted-on mouth patches.

## Verification evidence (2026-10-18, this VM: 2 CPU cores, no GPU)

- `pytest tests/ -q`: **49 passed, 2 skipped** (GPU-only tests), 0 failed (39 original + 10 new: real-clip dataset format/parity/errors, trainer `data_dir` smoke steps, and `prepare_clips.py` pure helpers).
- Overfit sanity (`tests/test_trainability.py`): DiT flow loss on a fixed
  batch fell to < 60% of its initial value in 20 steps; VAE reconstruction
  loss decreased monotonically over 35 steps.
- Checkpoint tests: resume reproduces the next-step loss **bit-for-bit** for
  both DiT and VAE trainers.
- Smoke runs (CPU, this VM):
  - `train_vae.py` 12 steps, batch 2: loss 0.9989 → 0.8825, monotonic.
  - `train.py` 12 steps, batch 2, on trained-VAE latents: flow loss
    5.5393 → 2.4754, monotonic.
  - `--resume` from `latest.pt` for 6 more steps: continued 12 → 18,
    loss 2.4754 → 1.7742 (checkpoints `step_000006/12/18.pt` + `latest.pt`).
  - `sample.py` from the trained DiT checkpoint (20 Euler steps, CFG 7.5):
    `checkpoints_smoke/sample1/frames.npy`, shape `(17, 64, 64, 3)`,
    range `[-0.358, 0.019]` — correct shape/range (variance still tiny after
    only 18 steps, as expected; structure emerges with longer training).

## Master improvement programme (2026-10-22, second prompt)

A second audit-driven pass over the same weakest parts, one day after
the targeted programme above. Everything is backward compatible:
defaults reproduce historical behaviour unless a caller opts in.

- **Truncated-download fix (critical, found live).** `prepare_clips.py`
  `download()` streamed with `copyfileobj` and treated connection-close
  as success, so a proxy closing early silently saved partial videos;
  they then failed downstream as `too_short`/`decode_error` (215 of
  250 clips lost in one local chunk; Kaggle's network rarely triggered
  it). Downloads now verify the byte count against `Content-Length`
  and retry (3 attempts, `IncompleteRead` included). Regression tests
  cover both the truncated and complete cases.
- **Prep quality.** Resize is no longer nearest-neighbour by default:
  `--resize area|lanczos|nearest` (`area` = INTER_AREA down / LANCZOS
  up via cv2, Pillow fallback). Quality filters are configurable
  (`passes_quality(clip, thresholds)`; prep flags `--min-sharpness`,
  `--min/max-motion`, `--min-aesthetic`, `--min/max-brightness` —
  brightness bounds default OFF so valid dark/static scenes are not
  rejected by a universal rule). Prep records `caption_source`
  (`search_query`/`filename`) and `source_url` in the manifest, and
  writes `prep_report.json` (config, seed, rejection reasons) plus a
  `contact_sheet.png` QA preview into every output folder.
- **Caption audit.** `vidgen.data.dataset_tools.audit_captions` +
  helper `captions-audit`: counts missing/empty/duplicated/generic/
  very-short captions and the `caption_source` mix — the mechanical
  failure modes of search-query captions, flagged before training.
- **BPE subword tokenizer.** `vidgen/text/bpe.py`: byte-pair encoding
  implemented from scratch (deterministic merges: frequency, then
  lexicographic), trained on Vidgin's own captions, mirroring the
  word-level interface (same special ids, encode/decode/save/load).
  Unseen words split into known subwords instead of collapsing to one
  `<unk>` ("skateboarding" -> s-k-a-t-e-b-o-ar-d-ing; the typo
  "walkking" keeps its "walk" stem). Checkpoints store either kind;
  `tokenizer_from_dict` dispatches on the payload format tag in
  train/sample/evaluate. `build_tokenizer.py --method bpe` prints the
  word-vs-BPE coverage comparison for the same corpus. Which one wins
  for Vidgin is an empirical question — word-level remains the default.
- **Training hardening.** Checkpoint saves are atomic
  (`vidgen/checkpoint.py`: temp file + `os.replace`, so a killed
  session can no longer truncate the only resume point). Both trainers'
  step records add `gpu_mem_mb` on CUDA. The VAE loss gains a
  `--recon-weight` term (default 1.0 = historical loss).
- **Leakage tools.** `find_near_duplicates` (perceptual 16x16
  signature distance; helper `near-duplicates`) catches
  re-encodes/crops of one source that byte hashes miss, and
  `verify_splits` (helper `verify-splits`) proves written splits are
  group-disjoint, complete, and warns when a dataset is too small for
  meaningful validation (<50 clips).
- **Reproducible evaluation.** `scripts/run_evaluation.py` runs the
  whole evidence chain in one command — dataset health, caption audit,
  tokenizer coverage, split leakage, VAE reconstruction (via
  `measure_latents.py`), DiT sample evaluation (via
  `evaluate_checkpoint.py`) — and writes one JSON report with
  provenance (git commit, seed, Python/torch versions, dataset size)
  and pass/warn/fail/skipped per step. `--compare` diffs metric
  deltas against a previous report.
