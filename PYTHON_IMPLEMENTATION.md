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
| `vidgen/train.py` | Part 7 | `DiTTrainer` + CLI: rectified-flow DiT training in VAE latent space on Moving MNIST; AdamW, grad-clip 1.0, 10% caption dropout, `--vae-ckpt` for a trained VAE |
| `vidgen/train_vae.py` | Part 3 | `VAETrainer` + CLI: VAE training (recon MSE + KL), same checkpointing |
| `vidgen/sample.py` | Part 8 | Sample a clip from a DiT checkpoint → VAE decode → `.npy` (+ PNG/GIF if Pillow installed) |
| `requirements.txt` | — | `torch`, `numpy`, `pytest` (CPU index for torch; Pillow optional) |
| `tests/` | — | 39 tests mirroring the README's CPU suite (see below) |

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

## Verification evidence (2026-10-18, this VM: 2 CPU cores, no GPU)

- `pytest tests/ -q`: **39 passed, 2 skipped** (GPU-only tests), 0 failed.
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
