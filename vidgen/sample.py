"""Part 8: generate a clip from a trained DiT checkpoint.

Pipeline: sample latents with an ODE sampler (+CFG) -> VAE decode -> save
frames (.npy always; PNG/GIF too when Pillow is installed).

Usage:
  python -m vidgen.sample --ckpt checkpoints/tiny-dit/latest.pt \\
      --prompt "digit 3 moving left" --steps 20 --outdir outputs/clip1
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from vidgen.checkpoint import inference_weights, load_checkpoint
from vidgen.config import VidgenConfig, get_preset, resolve_device
from vidgen.depth import depth_to_latent_grid
from vidgen.dit.model import DiT
from vidgen.extend import generate_long, new_frames_per_segment, segments_for_seconds
from vidgen.i2v import image_latent, load_image
from vidgen.sampler import sample_latents
from vidgen.seed import seed_everything
from vidgen.text.encoder import build_text_encoder
from vidgen.text.tokenizer import CaptionTokenizer, tokenizer_from_dict
from vidgen.vae.model import CausalVAE


def save_frames(frames: np.ndarray, outdir: str, duration_ms: int = 62) -> dict:
    """frames: [T, H, W, 3] float32 in [-1, 1]. Returns paths written.

    ``duration_ms`` sets the GIF frame duration; 62 keeps the original fast
    preview, 290 matches the project's standard playback rate (~5 s for a
    17-frame clip; the long-clip path uses 290).
    """
    os.makedirs(outdir, exist_ok=True)
    out = {}
    npy_path = os.path.join(outdir, "frames.npy")
    np.save(npy_path, frames)
    out["npy"] = npy_path
    try:
        from vidgen.video_io import write_mp4

        fps = max(1.0, round(1000.0 / max(duration_ms, 1)))
        out["mp4"] = write_mp4(frames, os.path.join(outdir, "clip.mp4"),
                               fps=fps)
    except Exception as exc:  # MP4 is an export nicety, never fatal
        print(f"MP4 export skipped ({exc}); npy/PNG/GIF still written")
    try:
        from PIL import Image

        uint8 = ((frames * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        png_paths = []
        for i, f in enumerate(uint8):
            p = os.path.join(outdir, f"frame_{i:03d}.png")
            Image.fromarray(f).save(p)
            png_paths.append(p)
        gif_path = os.path.join(outdir, "clip.gif")
        imgs = [Image.fromarray(f) for f in uint8]
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                     duration=duration_ms)
        out["png_frames"] = png_paths
        out["gif"] = gif_path
    except ImportError:
        print("Pillow not installed: wrote .npy only (pip install pillow for PNG/GIF)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="digit 3 moving left")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--method", default="euler", choices=["euler", "midpoint", "rk4"])
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default="outputs/clip")
    ap.add_argument("--vae-ckpt", default=None)
    ap.add_argument("--no-ema", action="store_true",
                    help="sample from the raw weights even when the "
                         "checkpoint carries EMA weights (--ema-decay)")
    # Stage 1 upgrade: long video via segment chaining (vidgen.extend).
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="target duration; segments are chained to reach it "
                         "(~290 ms/frame). 0 = single native window (old behavior)")
    ap.add_argument("--segments", type=int, default=0,
                    help="explicit segment count; overrides --seconds")
    ap.add_argument("--ctx-latent", type=int, default=2,
                    help="latent frames of past used as chaining context")
    ap.add_argument("--blend", type=int, default=2,
                    help="pixel frames crossfaded at each segment join")
    ap.add_argument("--frame-ms", type=int, default=0,
                    help="GIF frame duration ms; 0 = 62 for single clips, "
                         "290 for chained clips")
    ap.add_argument("--depth-npy", default=None,
                    help="optional uint8 [T,H,W] depth map .npy (larger = "
                         "nearer) to steer a depth-trained checkpoint with "
                         "a fixed 3D layout (vidgen/depth.py); single "
                         "window only, not with --seconds/--segments")
    # Image-to-video (vidgen/i2v.py): the clip starts from this image.
    ap.add_argument("--image", default=None,
                    help="start image (any PIL-readable file): generate a "
                         "video that begins from it; center-cropped and "
                         "resized to the model's resolution. Works with "
                         "--seconds/--segments too.")
    # Voice for the generated video (vidgen/audio/dialogue.py +
    # scripts/add_audio.py): writes clip_voiced.mp4 next to clip.mp4.
    ap.add_argument("--dialogue", default=None,
                    help="one spoken line, starting at 0.5 s")
    ap.add_argument("--line", action="append", default=[],
                    metavar="START:TEXT",
                    help="timed spoken line, repeatable, e.g. "
                         "--line '1.5: Where are we going?'")
    ap.add_argument("--dialogue-file", default=None,
                    help="timed lines, one 'START: TEXT' per row")
    ap.add_argument("--voice-ckpt", default="checkpoints/voice/latest.pt")
    ap.add_argument("--vocoder-ckpt", default="checkpoints/vocoder/latest.pt")
    ap.add_argument("--ambience", action="store_true",
                    help="also add background sound under the voice "
                         "(needs --sound-ckpt)")
    ap.add_argument("--sound-ckpt", default="checkpoints/sound/latest.pt")
    ap.add_argument("--min-speech-speed", type=float, default=0.6,
                    help="fastest allowed speech when fitting a line into "
                         "its slot (1.0 = never speed up)")
    args = ap.parse_args()

    # Parse the dialogue BEFORE generating, so a typo fails in seconds,
    # not after a full sampling run.
    add_audio = None
    lines = []
    if args.dialogue or args.line or args.dialogue_file:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        import scripts.add_audio as add_audio

        try:
            lines = add_audio.collect_dialogue(args.dialogue, args.line,
                                               args.dialogue_file)
        except ValueError as e:
            raise SystemExit(str(e))

    seed_everything(args.seed)
    device = resolve_device(args.device)
    payload = load_checkpoint(args.ckpt, map_location=device)
    assert payload["kind"] == "dit", f"not a DiT checkpoint: {args.ckpt}"
    # The checkpoint's stored config is authoritative (it records e.g.
    # use_depth, which decides whether the DiT has a depth stem at all);
    # older checkpoints without one fall back to the named preset.
    if payload.get("config"):
        cfg = VidgenConfig.from_dict(payload["config"])
    else:
        cfg = get_preset(payload["preset"])

    vae = CausalVAE().to(device).eval()
    vae_path = args.vae_ckpt or payload.get("vae_ckpt")
    if vae_path:
        vp = load_checkpoint(vae_path, map_location=device)
        vae.load_state_dict(vp["vae"])
    else:
        print("WARNING: no --vae-ckpt given and the checkpoint names no "
              "VAE: decoding with a RANDOMLY INITIALIZED VAE. Output is "
              "only meaningful for smoke tests trained the same way.",
              file=sys.stderr)
    # Phase 2: caption-mode checkpoints carry their exact tokenizer;
    # inference must tokenize exactly as training did.
    sample_tokenizer = (tokenizer_from_dict(payload["tokenizer"])
                        if payload.get("tokenizer") is not None else None)
    if getattr(cfg, "text_tokenizer", "hash") == "caption" and sample_tokenizer is None:
        raise SystemExit(
            "this checkpoint was trained with a caption tokenizer but "
            "carries no 'tokenizer' payload; it is incomplete and "
            "cannot be sampled safely")
    text_enc = build_text_encoder(cfg, tokenizer=sample_tokenizer).to(device)
    dit = DiT(cfg).to(device)
    dit_sd, text_sd, used_ema = inference_weights(payload, not args.no_ema)
    dit.load_state_dict(dit_sd)
    text_enc.load_state_dict(text_sd)
    print(f"sampling with {'EMA' if used_ema else 'raw'} weights")

    first_prefix = None
    if args.image:
        first_prefix = image_latent(
            vae, load_image(args.image, cfg.height, cfg.width), cfg, device)
        print(f"image-to-video: starting from {args.image}")

    segments = args.segments
    if segments <= 0 and args.seconds > 0:
        segments = segments_for_seconds(cfg, args.seconds, args.ctx_latent)
    chained = segments > 1
    if chained and args.depth_npy:
        raise SystemExit("--depth-npy steers a single native window; "
                         "it cannot be combined with --seconds/--segments.")
    if chained:
        frames = generate_long(dit, vae, text_enc, cfg, args.prompt,
                               segments=segments, steps=args.steps,
                               method=args.method, cfg_scale=args.cfg,
                               seed=args.seed, device=device,
                               ctx_latent=args.ctx_latent, blend=args.blend,
                               first_prefix=first_prefix)
        expect_t = cfg.frames + (segments - 1) * new_frames_per_segment(
            cfg, args.ctx_latent)
        assert frames.shape == (expect_t, cfg.height, cfg.width, 3), frames.shape
    else:
        text, text_mask = text_enc([args.prompt])
        null_text, null_mask = text_enc.null_embedding()
        depth = None
        if args.depth_npy:
            if not cfg.use_depth:
                raise SystemExit(
                    "--depth-npy given, but this checkpoint was trained "
                    "without depth conditioning (use_depth=False); the map "
                    "would have no input to enter through.")
            arr = np.load(args.depth_npy)
            want = (cfg.frames, cfg.height, cfg.width)
            if arr.dtype != np.uint8 or arr.shape != want:
                raise SystemExit(
                    f"--depth-npy must be uint8 {want} (T,H,W), got "
                    f"{arr.dtype} {arr.shape}")
            d = torch.from_numpy(arr).to(torch.float32) / 255.0
            depth = depth_to_latent_grid(
                d.view(1, 1, *want).to(device),
                cfg.latent_t, cfg.latent_h, cfg.latent_w)
        lat_shape = (1, cfg.latent_channels, cfg.latent_t, cfg.latent_h, cfg.latent_w)
        start = ({} if first_prefix is None
                 else {"prefix_latent": first_prefix, "prefix_t": 1})
        z = sample_latents(dit, lat_shape, text.to(device), text_mask.to(device),
                           null_text.to(device), null_mask.to(device),
                           method=args.method, num_steps=args.steps,
                           cfg_scale=args.cfg, seed=args.seed, device=device,
                           depth=depth, **start)
        with torch.no_grad():
            clip = vae.decode(z / cfg.latent_scale)  # [1,3,T,H,W] in [-1,1]
        frames = clip[0].permute(1, 2, 3, 0).cpu().numpy()  # [T,H,W,3]
        assert frames.shape == (cfg.frames, cfg.height, cfg.width, 3)
    assert frames.min() >= -1.0 - 1e-5 and frames.max() <= 1.0 + 1e-5
    frame_ms = args.frame_ms or (290 if chained else 62)
    out = save_frames(frames, args.outdir, duration_ms=frame_ms)
    dur = frames.shape[0] * frame_ms / 1000.0
    print(f"sampled clip: shape={frames.shape} (~{dur:.1f}s at {frame_ms}ms/frame) "
          f"range=[{frames.min():.3f},{frames.max():.3f}]")
    for k, v in out.items():
        print(f"  {k}: {v}")
    if lines:
        if "mp4" not in out:
            raise SystemExit("voicing needs the MP4 export, which failed above")
        voiced = os.path.join(args.outdir, "clip_voiced.mp4")
        add_audio.add_audio_to_video(
            out["mp4"], voiced, args.prompt, lines,
            sound_ckpt=args.sound_ckpt, vocoder_ckpt=args.vocoder_ckpt,
            voice_ckpt=args.voice_ckpt, device=device,
            ambience=args.ambience, min_speed_scale=args.min_speech_speed)
        print(f"  voiced_mp4: {voiced}")


if __name__ == "__main__":
    main()
