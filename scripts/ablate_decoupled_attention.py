#!/usr/bin/env python3
"""
Ablation Script: Seedance Batch 1 Decoupled & Hybrid Attention
Compares Full Spacetime Attention vs Decoupled Attention vs Hybrid Attention.
Evaluates:
1. Analytical FLOPs and peak VRAM memory scaling at 5s (81f) and 30s (481f).
2. Runtime forward pass latency, memory allocation, and token throughput on synthetic dataset.
3. Temporal smoothness, trajectory preservation, and seam continuity.
"""

import sys
import os
import time
import math
import random
from typing import List, Dict, Any, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vidgen.config import AttentionConfig, TINY_PRESET
from vidgen.attention import (
    DecoupledDiTModel,
    estimate_attention_flops_and_memory,
)
from vidgen.oracle import SyntheticSmoothVideo
from vidgen.continuity import LatentFrame, compute_seam_metric


def run_flops_and_memory_comparison():
    """
    Prints comparative analytical table for 5s (81f) and 30s (481f) on Vidgen 480p and Tiny presets.
    """
    print("=" * 80)
    print("PART 1: THEORETICAL FLOPS & MEMORY SCALING ANALYSIS (Seedance Decoupled Attention)")
    print("=" * 80)

    for preset_name, res, dim, depth, heads in [
        ("Vidgen 480p (Production)", (480, 832), 1536, 30, 12),
        ("Tiny Preset (Prototyping)", (64, 64), 256, 6, 4),
    ]:
        print(f"\n--- Preset: {preset_name} [Res: {res[0]}x{res[1]}, Dim: {dim}, Depth: {depth}, Heads: {heads}] ---")
        est_5s = estimate_attention_flops_and_memory(
            duration_seconds=5.0, fps=16, height=res[0], width=res[1],
            dim=dim, depth=depth, num_heads=heads, hybrid_interval=4
        )
        est_30s = estimate_attention_flops_and_memory(
            duration_seconds=30.0, fps=16, height=res[0], width=res[1],
            dim=dim, depth=depth, num_heads=heads, hybrid_interval=4
        )

        headers = f"{'Metric':<32} | {'5s (81f / 21 latents)':<22} | {'30s (481f / 121 latents)':<22}"
        print(headers)
        print("-" * len(headers))
        print(f"{'Total Visual Tokens':<32} | {est_5s['total_visual_tokens']:<22,d} | {est_30s['total_visual_tokens']:<22,d}")
        print(f"{'Full Attention FLOPs':<32} | {est_5s['full_attention']['flops_gflops']:<18.2f} GFLOPs | {est_30s['full_attention']['flops_tflops']:<18.2f} TFLOPs")
        print(f"{'Decoupled Attention FLOPs':<32} | {est_5s['decoupled_attention']['flops_gflops']:<18.2f} GFLOPs | {est_30s['decoupled_attention']['flops_tflops']:<18.2f} TFLOPs")
        print(f"{'Hybrid Attention FLOPs':<32} | {est_5s['hybrid_attention']['flops_gflops']:<18.2f} GFLOPs | {est_30s['hybrid_attention']['flops_tflops']:<18.2f} TFLOPs")
        print(f"{'Full Attn Peak Matrix VRAM':<32} | {est_5s['full_attention']['peak_attn_matrix_mb']:<18.1f} MB     | {est_30s['full_attention']['peak_attn_matrix_gb']:<18.2f} GB (OOM!)")
        print(f"{'Decoupled Peak Matrix VRAM':<32} | {est_5s['decoupled_attention']['peak_attn_matrix_mb']:<18.1f} MB     | {est_30s['decoupled_attention']['peak_attn_matrix_mb']:<18.1f} MB (Flat O(1))")
        print(f"{'30s FLOPs Reduction Ratio':<32} | {'--':<22} | {est_30s['decoupled_attention']['flops_reduction_ratio']:<18.2f}x speedup")
        print(f"{'30s Memory Reduction Ratio':<32} | {'--':<22} | {est_30s['decoupled_attention']['memory_reduction_ratio']:<18.2f}x less VRAM")


def run_synthetic_benchmark(
    duration_frames: int = 21,
    spatial_grid: Tuple[int, int] = (2, 2),
    dim: int = 64,
    depth: int = 4,
    num_heads: int = 4,
    trials: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    Runs actual forward passes on synthetic video dataset comparing Full vs Decoupled vs Hybrid.
    """
    print("\n" + "=" * 80)
    print(f"PART 2: SYNTHETIC DATASET BENCHMARK (T={duration_frames} frames, Grid={spatial_grid[0]}x{spatial_grid[1]}, Dim={dim}, Depth={depth})")
    print("=" * 80)

    H, W = spatial_grid
    S = H * W
    spatial_coords = [(h, w) for h in range(H) for w in range(W)]
    head_dim = dim // num_heads

    # Generate synthetic smooth ground truth trajectory
    gt_video = SyntheticSmoothVideo(total_frames=duration_frames, dim=dim, seed=42)
    frames_visual: List[List[List[float]]] = []
    for t in range(duration_frames):
        f = gt_video.get_frame(t)
        # Replicate frame across spatial coordinates with spatial harmonic offset
        frame_tokens = []
        for s in range(S):
            h, w = spatial_coords[s]
            tok = [f[d] + 0.05 * math.sin(h * 0.5 + w * 0.7 + d * 0.1) for d in range(dim)]
            frame_tokens.append(tok)
        frames_visual.append(frame_tokens)

    # 3 Text conditioning tokens
    rng = random.Random(101)
    text_tokens = [[rng.gauss(0, 0.5) for _ in range(dim)] for _ in range(3)]

    modes = ["full", "decoupled", "hybrid"]
    results = {}

    for mode in modes:
        cfg = AttentionConfig(
            attention_mode=mode,
            enable_decoupled_attention=(mode != "full"),
            hybrid_full_attn_interval=2,
        )
        model = DecoupledDiTModel(
            dim=dim, depth=depth, num_heads=num_heads, head_dim=head_dim, config=cfg, seed=1337
        )

        # Warmup
        _ = model.forward(frames_visual, spatial_coords, text_tokens=text_tokens)

        # Timed trials
        times = []
        out_frames = None
        for _ in range(trials):
            start = time.perf_counter()
            out_frames, _, _ = model.forward(frames_visual, spatial_coords, text_tokens=text_tokens)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        tokens_count = duration_frames * S
        tok_per_sec = tokens_count / max(1e-6, avg_time)

        # Compute trajectory smoothness / continuity metric
        latent_frames_out = [LatentFrame(out_frames[t][0]) for t in range(duration_frames)]
        boundaries = [duration_frames // 2]
        seam_report = compute_seam_metric(latent_frames_out, boundaries)

        # Reconstruction correlation with ground truth
        l2_err = 0.0
        for t in range(duration_frames):
            gt_f = gt_video.get_frame(t)
            out_f = out_frames[t][0]
            l2_err += math.sqrt(sum((gt_f[d] - out_f[d]) ** 2 for d in range(dim)))
        mean_l2 = l2_err / duration_frames

        results[mode] = {
            "avg_latency_ms": avg_time * 1000.0,
            "tokens_per_sec": tok_per_sec,
            "seam_metric": seam_report["mean_seam_metric"],
            "mean_l2_dist": mean_l2,
        }

        print(f"\n[Mode: {mode.upper()}]")
        print(f"  • Latency:          {avg_time * 1000.0:.2f} ms")
        print(f"  • Throughput:       {tok_per_sec:,.0f} tokens/sec")
        print(f"  • Seam Continuity:  {seam_report['mean_seam_metric']:.4f}")
        print(f"  • Trajectory Dist:  {mean_l2:.4f}")

    # Print summary comparison
    print("\n" + "-" * 60)
    print("SUMMARY COMPARISON (Flag ON vs Flag OFF)")
    print("-" * 60)
    speedup = results["full"]["avg_latency_ms"] / max(0.001, results["decoupled"]["avg_latency_ms"])
    print(f"Decoupled vs Full Attention Speedup: {speedup:.2f}x")
    print(f"Full Attention Continuity Metric:   {results['full']['seam_metric']:.4f}")
    print(f"Decoupled Continuity Metric:        {results['decoupled']['seam_metric']:.4f}")
    print(f"Hybrid Continuity Metric:           {results['hybrid']['seam_metric']:.4f}")
    print("=" * 80)
    return results


def main():
    run_flops_and_memory_comparison()
    results = run_synthetic_benchmark()
    return 0


if __name__ == "__main__":
    sys.exit(main())
