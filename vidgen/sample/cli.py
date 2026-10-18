"""Command-Line Interface (CLI) for Wan Video Latent Diffusion Inference.

Usage Example:
    python -m vidgen.sample.cli \\
        --prompt "A cinematic red sports car drifting on a neon wet road at night" \\
        --negative_prompt "blurry, low quality, static, artifact, flicker" \\
        --height 480 --width 832 --frames 81 \\
        --steps 30 --guidance 5.0 --shift 5.0 \\
        --seed 42 --solver euler --output output.mp4
"""

from __future__ import annotations
import argparse
import sys
import torch

from vidgen.sample.pipeline import (
    WanInferencePipeline,
    validate_video_dimensions,
    write_video_mp4,
)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wan Video Latent Diffusion: Prompt in, MP4 out CLI pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core generation prompts
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the target video.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt for unconditional guidance branch.",
    )

    # Spatiotemporal resolution
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Video frame height in pixels (must be multiple of 16).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Video frame width in pixels (must be multiple of 16).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=17,
        help="Number of video frames. Must strictly satisfy T = 4k + 1 (e.g. 17, 33, 49, 65, 81).",
    )

    # Sampling parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of flow matching ODE solver steps (30 to 50 recommended).",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=5.0,
        dest="guidance_scale",
        help="Classifier-Free Guidance (CFG) scale g (typical: 4.0 to 7.0).",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=5.0,
        help="Time schedule shift factor s (Wan default is ~5.0).",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="euler",
        choices=["euler", "heun", "unipc", "dpm_solver"],
        help="Numerical ODE solver scheme.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic latent sampling reproducibility.",
    )

    # Architecture & Weights
    parser.add_argument(
        "--preset",
        type=str,
        default="tiny",
        choices=["tiny", "small", "wan1p3b"],
        help="Model preset architecture.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint (.pt or .bin).",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=True,
        help="Sample from Exponential Moving Average (EMA) weights if available in checkpoint.",
    )
    parser.add_argument(
        "--no_ema",
        action="store_false",
        dest="use_ema",
        help="Disable EMA weights and use raw training weights.",
    )

    # Memory Optimization
    parser.add_argument(
        "--offload_text_encoder",
        action="store_true",
        default=True,
        help="Offload text conditioner to CPU immediately after embedding.",
    )
    parser.add_argument(
        "--offload_dit",
        action="store_true",
        default=True,
        help="Offload DiT backbone to CPU before running VAE decode.",
    )
    parser.add_argument(
        "--tile_spatial",
        action="store_true",
        default=False,
        help="Enable VAE spatial tiling for high-resolution video frames.",
    )
    parser.add_argument(
        "--no_chunked_vae",
        action="store_false",
        dest="chunked_vae",
        default=True,
        help="Disable VAE causal temporal chunking.",
    )
    parser.add_argument(
        "--rewrite_prompt",
        action="store_true",
        default=False,
        help="Use LLM prompt expander to match dense training caption style.",
    )

    # Output & Hardware
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.mp4",
        help="Output destination path for MP4 video.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Playback framerate in FPS (Wan standard: 16 FPS).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda, cpu, or mps).",
    )

    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    parsed = parse_args(args)

    print("=" * 70)
    print("🎬 Wan Video Latent Diffusion: Part 8 Inference Pipeline")
    print("=" * 70)
    print(f"Prompt:          '{parsed.prompt}'")
    if parsed.negative_prompt:
        print(f"Negative Prompt: '{parsed.negative_prompt}'")
    print(f"Resolution:      {parsed.width}x{parsed.height} @ {parsed.frames} frames ({parsed.fps} fps)")
    print(f"Sampling:        {parsed.steps} steps | Solver: {parsed.solver} | Shift: {parsed.shift} | CFG: {parsed.guidance_scale}")
    print(f"Seed:            {parsed.seed} | Device: {parsed.device} | Use EMA: {parsed.use_ema}")

    # 1. Validate frames and dimensions
    try:
        validate_video_dimensions(
            frames=parsed.frames,
            height=parsed.height,
            width=parsed.width,
        )
    except ValueError as e:
        print(f"\n❌ Validation Error: {e}", file=sys.stderr)
        return 1

    # 2. Build Pipeline
    print("\n⏳ Initializing pipeline...")
    device = torch.device(parsed.device)
    pipeline = WanInferencePipeline.from_pretrained(
        checkpoint_path=parsed.checkpoint,
        preset=parsed.preset,
        device=device,
        use_ema=parsed.use_ema,
    )

    # 3. Generate Video
    print(f"🚀 Running reverse flow ODE sampling ({parsed.steps} steps)...")
    def progress_callback(step: int, latents: torch.Tensor, sigma: float):
        if (step + 1) % max(1, parsed.steps // 5) == 0 or step == parsed.steps - 1:
            print(f"   [Step {step + 1:2d}/{parsed.steps}] sigma = {sigma:.4f}")

    video_tensor = pipeline(
        prompt=parsed.prompt,
        negative_prompt=parsed.negative_prompt,
        height=parsed.height,
        width=parsed.width,
        frames=parsed.frames,
        steps=parsed.steps,
        guidance_scale=parsed.guidance_scale,
        shift=parsed.shift,
        seed=parsed.seed,
        solver=parsed.solver,
        offload_text_encoder=parsed.offload_text_encoder,
        offload_dit=parsed.offload_dit,
        use_prompt_rewriter=parsed.rewrite_prompt,
        tile_spatial_vae=parsed.tile_spatial,
        chunked_vae=parsed.chunked_vae,
        callback=progress_callback,
    )

    print(f"\n✨ Generated video shape: {tuple(video_tensor.shape)} (range: [{video_tensor.min():.2f}, {video_tensor.max():.2f}])")

    # 4. Write to MP4
    print(f"💾 Writing MP4 video at {parsed.fps} FPS to '{parsed.output}'...")
    saved_path = write_video_mp4(
        video_tensor=video_tensor,
        output_path=parsed.output,
        fps=parsed.fps,
    )
    print(f"✅ Video successfully saved to {saved_path}!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
