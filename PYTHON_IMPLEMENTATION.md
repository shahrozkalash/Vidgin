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
| `vidgen/text/encoder.py` | Part 5 | `TinyTextEncoder`: learned word embeddings over a fixed caption vocab → `[B, L, Dt]` + padding mask; cached CFG null embedding; `FrozenEncoderStub` documents the umT5-XXL swap path (interface: `encode(captions) -> ([B,L,Dt], mask)`, frozen, null cached) |
| `vidgen/dit/model.py` | Part 4+6 | Exact-unfold 3D patchify (bit-exact inverse), partitioned 3D RoPE (Tiny: 22/22/20 over head-dim 64), DiT blocks (full 3D self-attn + text cross-attn + FFN) with shared AdaLN-Zero modulation; `forward(latent, t, text, mask) -> velocity` |
| `vidgen/flow.py` | Part 7 | Rectified flow `x_t = (1-t)x_0 + t x_1`, velocity target `x_1 - x_0`; shifted logit-normal timesteps (s=5.0); 10-bin diagnostic losses |
| `vidgen/sampler.py` | Part 8 | Euler / Midpoint / RK4 ODE samplers with classifier-free guidance; oracle recovery < 1e-5 |
| `vidgen/checkpoint.py` | Part 7 | Save/load with python+numpy+torch RNG states → resume reproduces next-step loss bit-for-bit |
| `vidgen/train.py` | Part 7 | `DiTTrainer` + CLI: rectified-flow DiT training in VAE latent space on Moving MNIST; AdamW, grad-clip 1.0, 10% caption dropout, `--vae-ckpt` for a trained VAE; `--data-dir` trains on a prepared real-clip directory instead |
| `vidgen/train_vae.py` | Part 3 | `VAETrainer` + CLI: VAE training (recon MSE + KL), same checkpointing; `--data-dir` for real clips |
| `vidgen/data/clips.py` | Part 1 | `FolderClipDataset`: reads a prepared directory (`manifest.jsonl` + uint8 `(T,H,W,3)` `.npy` clips); yields the exact Moving MNIST sample format; deterministic manifest-order indexing; `max_clips` cap |
| `vidgen/sample.py` | Part 8 | Sample a clip from a DiT checkpoint → VAE decode → `.npy` (+ PNG/GIF if Pillow installed) |
| `scripts/prepare_clips.py` | Part 1 | Colab-oriented CLI: downloads stock videos (Pixabay API via `PIXABAY_API_KEY` env var — recommended, free key issued instantly; Pexels API via `PEXELS_API_KEY` — only if you already have a key, since Pexels has paused issuing new ones; or `--source urls` with direct URLs) → decode (cv2, else imageio) → center 17-frame window → square crop → 64×64 → quality/dedupe filters → `.npy` + captioned manifest; per-video temp-dir cleanup; prints kept/skipped/MB stats |
| `notebooks/prepare_clips_drive.ipynb` | — | Colab notebook: Drive mount → repo → config ("City and people" queries, 500/query) → `getpass` API-key collector (Pixabay/Pexels) → verify cell (loads `FolderClipDataset`, prints stats, renders a filmstrip) → exact real-data `train_vae.py`/`train.py --data-dir` commands |
| `scripts/harvest_urls.py` | Part 1 | No-API-key URL harvester (stdlib only): fetches stock search pages with a browser User-Agent and extracts direct `.mp4` file URLs from the embedded HTML (Pexels: `videos.pexels.com/video-files/<id>/<id>-<variant>.mp4`, pagination via `?page=N`, one sd variant per video; Pixabay: optional secondary source, `_tiny`→`_medium` suffix swap kept only when HEAD returns 200). Pure-function parsing (`extract_pexels_urls`, `choose_variant`) unit-tested without network. Writes `urls.txt` for `prepare_clips.py --source urls` |
| `requirements.txt` | — | `torch`, `numpy`, `pytest` (CPU index for torch; Pillow optional) — no new deps (prep script is stdlib + torch/numpy; decoders preinstalled on Colab) |
| `tests/` | — | 49 tests mirroring the README's CPU suite (see below), incl. real-clip dataset + prep helpers |

## Notable design decisions / spec deviations

- **Text encoder**: Track A uses the learned `TinyTextEncoder` (small caption
  vocabulary) instead of frozen umT5-XXL. The DiT consumes `[B, L, Dt]` +
  padding mask either way, so a frozen umT5 can replace the module later
  (`FrozenEncoderStub` documents the exact interface contract). This keeps the
  whole pipeline trainable on CPU with zero downloads.
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

## What remains GPU/cluster-only (from GPU_CHECKLIST.md)

These are documentation-only — they need hardware nobody here has:

- **100k web-video ingest + cluster VLM captioning** — 8× A100 (80 GB).
- **3D VAE pretraining on 10M clips** — 8× H100 (80 GB), bf16.
- **Frozen umT5-XXL (4.7B) embedding extraction** — ≥ 24 GB VRAM.
- **Vidgen 1.3B DiT training** — 4 nodes × 8 H100, FSDP2 + Ulysses sequence
  parallel (the config dimensions exist in `get_preset("vidgen480p")`; the
  distributed launcher does not).
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
