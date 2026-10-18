# Vidgen GPU Acceptance & Execution Checklist

Comprehensive guide for all GPU-dependent validation runs, scaling benchmarks, and cluster training jobs across Parts 0 to 13.

---

## Overview

While the core algorithmic invariants, token counting, causal masks, RoPE re-indexing, loss equations, and sampler oracle recovery are verified on CPU in `scripts/run_cpu_audit_tests.py` and reflected in `AUDIT.md`, high-throughput training, 4.7B text model inference, and multi-node sequence parallelism require discrete GPU clusters (NVIDIA A100/H100).

---

## GPU Checklist & Execution Commands

### 1. Part 1: Full 100k Web Video Ingest & Multi-Node VLM Captioning
* **Hardware Requirement**: 8x A100 (80GB) or cluster endpoint
* **Objective**: Process 100k raw clips, filter via Laplacian sharpness, motion vector range ($0.5 \le v \le 80$), and generate 5-axis dense captions.
* **Command**:
  ```bash
  torchrun --nproc_per_node=8 vidgen/data/curation_cluster.py \
    --manifest data/web_100k_raw.parquet \
    --output-dir data/curated_manifests/ \
    --vlm-endpoint http://vlm-cluster:8000/v1/chat/completions \
    --min-duration 2.0 --max-duration 10.0 \
    --motion-range 0.5 80.0 \
    --aesthetic-min 4.0 \
    --num-workers 32
  ```
* **Acceptance Criteria**: Yields valid curated Parquet manifest with zero corrupt records, valid crop boxes, and paired short + dense captions.

---

### 2. Part 3: 3D Causal VAE Pre-Training on 10M Video Clips
* **Hardware Requirement**: 8x H100 (80GB) SXM5, bf16
* **Objective**: Train 4x8x8 temporal-spatial causal VAE with 16 latent channels on 10M video clips.
* **Command**:
  ```bash
  torchrun --nproc_per_node=8 vidgen/vae/train.py \
    --config vidgen/configs/vae_10m.yaml \
    --batch-size 4 \
    --grad-accum 2 \
    --precision bf16 \
    --checkpoint-dir /checkpoints/vae/ \
    --log-wandb vidgen-vae-10m
  ```
* **Acceptance Criteria**: Mean reconstruction PSNR $> 31.0\text{ dB}$, SSIM $> 0.90$ across validation set; causal temporal receptive field maintained ($t' \le t$).

---

### 3. Part 5: Frozen umT5-XXL (4.7B) Batch Embedding Extraction
* **Hardware Requirement**: 1x A100/H100 (80GB) or 2x 24GB GPUs (Device Map)
* **Objective**: Precompute text embeddings ($[B, L=512, D_t=4096]$) for the 70-prompt validation set and dataset captions with CFG null token caching.
* **Command**:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python3 vidgen/text/embed_extract.py \
    --model google/umt5-xxl \
    --precision bf16 \
    --device cuda:0 \
    --prompt-file data/eval_prompts_70.json \
    --output-dir data/text_embeddings/ \
    --max-seq-len 512
  ```
* **Acceptance Criteria**: Text encoder forward produces normalized embeddings; zero gradients (`requires_grad=False`); null-prompt embedding cached.

---

### 4. Part 6 & 7: Vidgen 1.3B DiT Distributed Backbone Training
* **Hardware Requirement**: 4 nodes $\times$ 8 H100 (32 GPUs), PyTorch 2.4+ FSDP2
* **Objective**: Train Vidgen 1.3B parameter model on 480p 81-frame video latents with shifted logit-normal schedule ($s=5.0$) and sequence packing.
* **Command**:
  ```bash
  torchrun --nnodes=4 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=node-0:29500 \
    vidgen/train.py \
    --config vidgen/configs/vidgen1p3b.yaml \
    --fsdp-sharding-strategy FULL_SHARD \
    --precision bf16 \
    --sequence-parallel ulysses \
    --sp-degree 4 \
    --gradient-checkpointing \
    --batch-token-budget 65536 \
    --checkpoint-dir /checkpoints/vidgen1p3b/
  ```
* **Acceptance Criteria**: Per-timestep binned losses decrease monotonically; gradient norms remain stable ($< 1.0$ with clipping); checkpoint resumption yields exact bit-for-bit loss match.

---

### 5. Part 7: Large-Scale Foundation Model Run (Vidgen 14B, 500k Steps)
* **Hardware Requirement**: 8 nodes $\times$ 8 H100 (64 GPUs)
* **Objective**: Full-scale pretraining of the 14B parameter flagship DiT backbone over 500,000 steps.
* **Command**:
  ```bash
  torchrun --nnodes=8 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=node-0:29500 \
    vidgen/train.py \
    --config vidgen/configs/vidgen14b.yaml \
    --total-steps 500000 \
    --save-interval 5000 \
    --eval-interval 2500 \
    --wandb-project vidgen-scaling-14b
  ```
* **Acceptance Criteria**: Model achieves target MFU $\ge 35\%$; validation loss reaches plateau $< 0.12$.

---

### 6. Part 9: 64x H100 Cluster MFU & Ulysses Throughput Benchmark
* **Hardware Requirement**: 8 nodes $\times$ 8 H100 (64 GPUs) with 800Gbps InfiniBand
* **Objective**: Measure Model FLOPs Utilization (MFU) and All-to-All communication latency for 480p 81-frame batches.
* **Command**:
  ```bash
  torchrun --nnodes=8 --nproc_per_node=8 vidgen/benchmarks/benchmark_cluster_throughput.py \
    --preset 14b \
    --resolution 480x832 \
    --frames 81 \
    --batch-size 1 \
    --flash-attention-v3 \
    --report-mfu
  ```
* **Acceptance Criteria**: Achieved MFU is inside the 30% to 45% target band; sequence parallel communication overhead $< 12\%$ of block runtime.

---

### 7. Part 10: Automated VBench 8-Metric Evaluation Suite (500 Samples)
* **Hardware Requirement**: 4x A100/H100 (80GB)
* **Objective**: Sample 500 clips on standard prompts and compute VBench scores for Subject Consistency, Background Consistency, Motion Smoothness, and Aesthetic Quality.
* **Command**:
  ```bash
  torchrun --nproc_per_node=4 vidgen/eval/vbench_suite.py \
    --checkpoint /checkpoints/vidgen1p3b_final.pt \
    --prompts data/vbench_70.json \
    --num-samples 500 \
    --fps 16 \
    --output-dir /eval_results/vbench/
  ```
* **Acceptance Criteria**: Overall VBench score $> 82.5\%$; Subject Consistency $> 91\%$; Motion Smoothness $> 94\%$.

---

### 8. Part 13: 30-Second Moving MNIST Long-Video Rollout Verification
* **Hardware Requirement**: 1x GPU with $\ge 16\text{ GB}$ VRAM (RTX 4090 or A100)
* **Objective**: Verify that 30s generation (481 frames at 16 fps across 6 consecutive chunks) maintains zero drift on Moving MNIST using rolling window attention and 2 anchor frames.
* **Command**:
  ```bash
  python3 vidgen/sample/pipeline.py \
    --preset tiny \
    --duration 30 \
    --fps 16 \
    --checkpoint /checkpoints/tiny_moving_mnist.pt \
    --rolling-window-chunks 1 \
    --anchor-frames 2 \
    --context-noise-sigma 0.05 \
    --device cuda:0 \
    --output-dir /outputs/long_video/
  ```
* **Acceptance Criteria**: Generated 481-frame video exhibits no digit disintegration, identity drift $< 0.08$, and no border seams.

---

### 9. Part 13: Physical GPU VRAM Profile for $O(1)$ Memory Verification
* **Hardware Requirement**: 1x NVIDIA GPU with `nvidia-smi` / PyTorch CUDA memory profiling
* **Objective**: Measure peak allocated VRAM for 5s, 10s, 20s, and 30s generation to empirically prove flat $O(1)$ memory growth.
* **Command**:
  ```bash
  python3 vidgen/benchmarks/profile_long_video_memory.py \
    --preset small \
    --duration-seconds 30 \
    --fps 16 \
    --chunk-latents 21 \
    --device cuda:0 \
    --log-vram-csv memory_profile.csv
  ```
* **Acceptance Criteria**: Peak VRAM at 30 seconds does not exceed peak VRAM at 5 seconds by more than 5% (flat memory curve, constant $O(1)$).
