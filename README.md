# VidGen: Vidgen-Style Latent Diffusion Text-to-Video Engine

A modular, clean-room PyTorch implementation of a Vidgen-style video diffusion model:
- **Causal 3D VAE**: Spatio-temporal compression (4x temporal, 8x spatial) with strictly causal temporal receptive field.
- **3D Spacetime Patchify**: Spacetime patches projected to transformer token sequence.
- **DiT Backbone**: Full 3D attention with 3D RoPE (partitioned across time, height, width) and text cross-attention.
- **Rectified Flow (Flow Matching)**: Linear trajectory interpolation between Gaussian noise $x_0$ and video latents $x_1$.
- **ODE Sampler**: Euler, Midpoint, and RK4 numerical solvers with Classifier-Free Guidance (CFG).

---

## Part 0: Decisions, Environment & Scale Plan

### 1. Track & Size Presets
* **Selected Track**: **Track A** (Prototyping & verification on CPU/single-device with scaled config definitions).
* **Active Preset**: **Tiny** (64x64, 17 frames, 80 spacetime tokens, ~20M parameters).

| Preset | Resolution | Frames ($T=4k+1$) | Latent $(t, h, w)$ | Patch $(p_t, p_h, p_w)$ | Tokens ($N$) | DiT (L / D / H / FFN) | Params |
|---|---|---|---|---|---|---|---|
| **Tiny** | 64x64 | 17 ($k=4$) | $5 \times 8 \times 8$ | $(1, 2, 2)$ | **80** | 8 / 384 / 6 / 1536 | ~20M |
| **Small** | 256x256 | 33 ($k=8$) | $9 \times 32 \times 32$ | $(1, 2, 2)$ | **2,304** | 16 / 1024 / 16 / 4096 | ~270M |
| **Vidgen 480p** | 480x832 | 81 ($k=20$) | $21 \times 60 \times 104$ | $(1, 2, 2)$ | **32,760** | 30 / 1536 / 12 / 8960 | ~1.3B |
| **Vidgen 14B** | 720x1280 | 81 ($k=20$) | $21 \times 90 \times 160$ | $(1, 2, 2)$ | **75,600** | 40 / 5120 / 40 / 13824 | ~14B |

---

## Part 1: Data Collection & Curation Pipeline

### 1. Dataset Sources & Licensing
* **Open Datasets**:
  * **OpenVid-1M / MiraData**: High-motion benchmark dataset with permissive research terms.
  * **Panda-70M / Koala-36M**: Large-scale curated video-text pairs.
  * *(WebVid-10M is avoided due to stock video scraping takedown & copyright restrictions).*
* **Stock & Licensed Footage**: Pexels / Pixabay API integration with terms permitting AI model training.
* **Image-Text Pairs**: Integrated alongside video clips ($T=1$ repeated temporally). Vidgen jointly trains on images and videos to learn appearance textures cheaply.
* **Track A Synthetic Moving MNIST**: Moving MNIST generator with controllable digit trajectories, speeds, letterboxed black borders, blurs, static clips, and near-duplicates.

### 2. 7 Resumable Parquet Pipeline Stages
1. **Stage 1 (Index & Ingest)**: Ingests raw videos and images into typed Parquet schema: `id`, `media_type`, `path`, `source`, `license`, `duration`, `fps`, `frames`, `height`, `width`.
2. **Stage 2 (Scene-Cut & Duration)**: Enforces clip length between 2.0s and 10.0s (images stay 0.0s).
3. **Stage 3 (Technical Filters & Crop)**: Validates minimum resolution ($64\times 64$), decodability, and detects letterbox/pillarbox black borders (`crop_box`).
4. **Stage 4 (Quality Filters)**:
   * **Sharpness**: Laplacian operator variance $\text{Var}(L * I) \ge 20.0$.
   * **Motion Bounds**: Mean absolute frame difference $0.5 \le \text{motion} \le 80.0$ (drops static and chaotic random noise).
   * **Aesthetic Appeal**: Contrast, dynamic range, and Shannon entropy score $\ge 4.0$.
   * **Watermark & Logo**: Corner high-frequency edge concentration filter $\le 0.20$.
   * **Safety**: NSFW & content classifier.
5. **Stage 5 (Deduplication)**: Multi-frame spatio-temporal video pHash (start, middle, end frames with 16x16 dHash, 768 bits). Removes near-duplicates with Hamming distance $\le 2$.
6. **Stage 6 (VLM Dense Captioning)**: Generates paired **short captions**, **dense long captions**, and **structured JSON metadata** covering subject, action, setting, style, lighting, and camera motion.
7. **Stage 7 (Statistics & Human Review)**:
   * Computes distributions of duration, motion, sharpness, aesthetics, and caption lengths.
   * Exports `statistics_report.json` with filter retention rates.
   * Extracts `review_200_captions.json` (200 random kept captions for quality inspection).
   * Extracts `review_50_rejected.json` (50 rejected clips with rejection reasons for threshold tuning).

---

## Test Execution

```bash
# Run full test suite (CPU tests pass immediately, GPU tests auto-skip)
python3 -m pytest -v
```

Execution output:
```text
tests/test_checkpoint.py::test_checkpoint_reproducibility_next_step_loss PASSED [  5%]
tests/test_checkpoint.py::test_gpu_checkpoint_save_and_resume SKIPPED    [ 10%]
tests/test_config.py::test_tiny_config_loading PASSED                    [ 15%]
tests/test_config.py::test_small_preset_dimensions PASSED                [ 21%]
tests/test_config.py::test_vidgen_480p_preset_dimensions PASSED             [ 26%]
tests/test_config.py::test_device_agnostic_resolution PASSED             [ 31%]
tests/test_config.py::test_frame_rule_validation PASSED                  [ 36%]
tests/test_config.py::test_config_roundtrip PASSED                       [ 42%]
tests/test_config.py::test_gpu_config_resolution SKIPPED                 [ 47%]
tests/test_data_curation.py::test_manifest_schema_and_io PASSED          [ 52%]
tests/test_data_curation.py::test_moving_mnist_generation PASSED         [ 57%]
tests/test_data_curation.py::test_quality_and_signal_filters PASSED      [ 63%]
tests/test_data_curation.py::test_bucket_sampler PASSED                  [ 68%]
tests/test_data_curation.py::test_curated_video_dataset PASSED           [ 73%]
tests/test_data_curation.py::test_end_to_end_1000_clips_acceptance PASSED [ 78%]
tests/test_logging.py::test_structured_logger PASSED                     [ 84%]
tests/test_seed.py::test_seed_everything_determinism PASSED              [ 89%]
tests/test_seed.py::test_seed_everything_divergence PASSED               [ 94%]
tests/test_seed.py::test_gpu_seed_determinism SKIPPED                    [100%]
======================== 16 passed, 3 skipped in 11.49s ========================
```
