#!/usr/bin/env python3
"""Vidgin's in-repo audio post-processing pipeline for silent videos.

Adds:
1. Spoken dialogue via Vidgin's own Voice TTS model + HiFi-GAN vocoder.
2. Background sound via Vidgin's own Audio DiT model + HiFi-GAN vocoder.
3. Audio mixing with ducking (100% background, 35% under speech) + peak normalization.
4. Muxing via ffmpeg and dual-stream duration verification with ffprobe.

OWNERSHIP RULE:
Uses ONLY in-repo checkpoints trained by scripts/train_audio.py.
Zero external AI models. Zero third-party TTS, audio diffusion, or lip-sync tools.
If a required checkpoint does not exist, this tool prints the exact train_audio.py
command required to train it and exits cleanly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from vidgen.audio.mel import (
    AudioConfig,
    DEFAULT_AUDIO_CFG,
    denormalize_mel,
    wav_to_mel,
)
from vidgen.audio.sound_model import AudioDiT, AudioDiTConfig
from vidgen.audio.vocoder import HiFiGANGenerator, VocoderConfig
from vidgen.audio.voice_model import VidginVoiceTTS, VoiceModelConfig
from vidgen.checkpoint import load_checkpoint

SAMPLE_RATE = 22050
VOICE_START_OFFSET_SEC = 0.5
DUCKING_BG_LEVEL = 0.35
NORMAL_BG_LEVEL = 1.0

SCENE_SOUND_MAP = {
    "city": "city traffic and passing cars",
    "street": "city traffic and passing cars",
    "traffic": "busy city traffic with horns and passing cars",
    "rain": "steady rainfall with gentle falling water drops",
    "thunder": "heavy thunder and pouring rain",
    "beach": "ocean waves crashing against the sandy beach",
    "ocean": "ocean waves and distant sea breeze",
    "sea": "ocean waves crashing against the shore",
    "forest": "wild birds singing in the morning forest",
    "nature": "wind whistling and blowing through trees",
    "wind": "wind whistling and rustling through trees",
    "crowd": "ambient crowd chatter and street murmur",
    "people": "ambient crowd chatter and footsteps",
    "fire": "warm crackling campfire flames in the open air",
    "campfire": "warm crackling campfire flames in the open air",
}


# ----------------------------------------------------------- System Dependencies
def check_ffmpeg_installed() -> None:
    """Ensure ffmpeg and ffprobe are available on PATH."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "ffmpeg and ffprobe must be installed on your system to run add_audio.py."
        )


# ------------------------------------------------------------- Wave Helpers
def write_wav_file(path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write 1D float32 numpy array in [-1.0, 1.0] to a 16-bit PCM WAV."""
    audio_clipped = np.clip(audio, -1.0, 1.0)
    int16_data = (audio_clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())


def read_wav_file(path: str) -> Tuple[np.ndarray, int]:
    """Read WAV file and return (float32_array, sample_rate)."""
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        raw_bytes = wf.readframes(nframes)

    if sampwidth == 2:
        data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    return data, framerate


# ----------------------------------------------------------- Video Inspection
def get_video_info(video_path: str) -> dict:
    """Inspect duration, fps, and resolution via ffprobe."""
    check_ffmpeg_installed()
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration:format=duration",
        "-of", "json",
        video_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    meta = json.loads(proc.stdout)
    streams = meta.get("streams", [])
    if not streams:
        raise ValueError(f"No video streams found in {video_path}")

    stream = streams[0]
    duration = 0.0
    if stream.get("duration"):
        try:
            duration = float(stream["duration"])
        except ValueError:
            duration = 0.0
    if duration <= 0.0 and "format" in meta and meta["format"].get("duration"):
        try:
            duration = float(meta["format"]["duration"])
        except ValueError:
            duration = 0.0

    return {
        "duration": duration,
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
    }


# ------------------------------------------------------------ Sound Planning
def plan_background_sound(caption: str, custom_prompt: Optional[str] = None) -> str:
    """Plan background sound description using in-repo scene keyword map."""
    if custom_prompt and custom_prompt.strip():
        return custom_prompt.strip()

    cap_lower = caption.lower().strip()
    for keyword, sound_desc in SCENE_SOUND_MAP.items():
        if keyword in cap_lower:
            return sound_desc

    return f"ambient sound of {caption.strip()}"


# ------------------------------------------------------ Checkpoint Verification
def ensure_checkpoint(ckpt_path: str, target: str, suggested_cmd: str) -> None:
    """Verify that a trained in-repo checkpoint exists.

    If missing, prints exact training instructions and exits. Never falls back to external services.
    """
    if not os.path.isfile(ckpt_path):
        print(f"\n[ERROR] Required Vidgin {target} checkpoint NOT found: '{ckpt_path}'")
        print("Vidgin strictly obeys the OWNERSHIP RULE: no external models or downloaded weights are allowed.")
        print(f"Please train Vidgin's own {target} model first using:\n")
        print(f"    {suggested_cmd}\n")
        sys.exit(1)


# --------------------------------------------------------- Pipeline Generation
def generate_speech(
    dialogue: str,
    voice_ckpt: str,
    vocoder_ckpt: str,
    target_duration: float,
    device: str = "cpu",
    start_offset: float = VOICE_START_OFFSET_SEC,
) -> np.ndarray:
    """Synthesize dialogue using Vidgin's Voice TTS + HiFi-GAN Vocoder."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to run Vidgin audio inference.")

    dev = torch.device(device)
    # 1. Load Voice TTS model
    voice_payload = load_checkpoint(voice_ckpt, map_location=str(dev))
    voice_cfg = voice_payload.get("cfg", VoiceModelConfig())
    voice_model = VidginVoiceTTS(voice_cfg).to(dev)
    voice_model.load_state_dict(voice_payload["model"])
    voice_model.eval()

    # 2. Load Vocoder model
    vocoder_payload = load_checkpoint(vocoder_ckpt, map_location=str(dev))
    vocoder_cfg = vocoder_payload.get("cfg", VocoderConfig())
    vocoder = HiFiGANGenerator(vocoder_cfg).to(dev)
    vocoder.load_state_dict(vocoder_payload["generator"])
    vocoder.eval()

    # 3. Generate mel spectrogram from dialogue
    with torch.no_grad():
        mel = voice_model.synthesize(dialogue, device=str(dev))  # [1, 80, frames]
        wav_tensor = vocoder(mel).squeeze().cpu().numpy()  # 1D array

    # 4. Align to target duration with start offset
    total_samples = max(1, int(round(target_duration * SAMPLE_RATE)))
    offset_samples = int(round(start_offset * SAMPLE_RATE))
    result = np.zeros(total_samples, dtype=np.float32)

    available = max(0, total_samples - offset_samples)
    to_copy = min(len(wav_tensor), available)
    if to_copy > 0:
        result[offset_samples : offset_samples + to_copy] = wav_tensor[:to_copy]

    return result


def generate_ambience(
    sound_prompt: str,
    sound_ckpt: str,
    vocoder_ckpt: str,
    target_duration: float,
    steps: int = 25,
    device: str = "cpu",
) -> np.ndarray:
    """Generate ambient sound using Vidgin's Audio DiT + HiFi-GAN Vocoder."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to run Vidgin audio inference.")

    dev = torch.device(device)
    # 1. Load Audio DiT
    sound_payload = load_checkpoint(sound_ckpt, map_location=str(dev))
    sound_cfg = sound_payload.get("cfg", AudioDiTConfig())
    sound_model = AudioDiT(sound_cfg).to(dev)
    sound_model.load_state_dict(sound_payload["model"])
    sound_model.eval()

    # 2. Load Vocoder
    vocoder_payload = load_checkpoint(vocoder_ckpt, map_location=str(dev))
    vocoder_cfg = vocoder_payload.get("cfg", VocoderConfig())
    vocoder = HiFiGANGenerator(vocoder_cfg).to(dev)
    vocoder.load_state_dict(vocoder_payload["generator"])
    vocoder.eval()

    # 3. Sample mel via rectified flow diffusion
    with torch.no_grad():
        mel = sound_model.sample(
            caption=sound_prompt,
            duration_sec=target_duration,
            steps=steps,
            device=str(dev),
        )  # [1, 80, frames]
        wav_tensor = vocoder(mel).squeeze().cpu().numpy()

    # Match exact duration
    total_samples = max(1, int(round(target_duration * SAMPLE_RATE)))
    if len(wav_tensor) < total_samples:
        repeats = math.ceil(total_samples / max(1, len(wav_tensor)))
        wav_tensor = np.tile(wav_tensor, repeats)[:total_samples]
    else:
        wav_tensor = wav_tensor[:total_samples]

    return wav_tensor


# --------------------------------------------------------- Mixing & Ducking
def mix_with_ducking(
    voice_audio: Optional[np.ndarray],
    bg_audio: np.ndarray,
    ducking_bg_level: float = DUCKING_BG_LEVEL,
    normal_bg_level: float = NORMAL_BG_LEVEL,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Mix speech over background sound with 35% ducking during speech."""
    if voice_audio is None:
        peak = np.max(np.abs(bg_audio))
        target_peak = 0.89125  # -1 dB
        return bg_audio * (target_peak / max(peak, 1e-5))

    length = max(len(voice_audio), len(bg_audio))
    v = np.pad(voice_audio, (0, length - len(voice_audio)))
    b = np.pad(bg_audio, (0, length - len(bg_audio)))

    # Compute voice activity envelope
    window = max(1, int(0.05 * sample_rate))  # 50 ms
    abs_v = np.abs(v)
    kernel = np.ones(window) / window
    envelope = np.convolve(abs_v, kernel, mode="same")
    is_speaking = envelope > 0.015

    # Ducking multiplier
    bg_gain = np.where(is_speaking, ducking_bg_level, normal_bg_level).astype(np.float32)

    # Smooth gain transitions
    smooth_win = max(1, int(0.10 * sample_rate))
    smooth_kernel = np.ones(smooth_win) / smooth_win
    bg_gain = np.convolve(bg_gain, smooth_kernel, mode="same")
    bg_gain = np.clip(bg_gain, ducking_bg_level, normal_bg_level)

    mixed = (b * bg_gain) + v

    # Peak normalization to -1 dB (0.89125)
    peak = np.max(np.abs(mixed))
    target_peak = 0.89125
    if peak > 1e-5:
        normalized = mixed * (target_peak / peak)
    else:
        normalized = mixed

    return normalized.astype(np.float32)


# ------------------------------------------------------- Mux & Verification
def mux_and_verify(video_path: str, audio_wav_path: str, out_path: str) -> dict:
    """Mux audio and video via ffmpeg and verify dual streams with ffprobe."""
    check_ffmpeg_installed()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    cmd_mux = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_wav_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd_mux, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    cmd_probe = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=index,codec_type,duration:format=duration",
        "-of", "json",
        out_path,
    ]
    proc = subprocess.run(cmd_probe, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    meta = json.loads(proc.stdout)
    streams = meta.get("streams", [])

    v_streams = [s for s in streams if s.get("codec_type") == "video"]
    a_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if len(v_streams) != 1 or len(a_streams) != 1:
        raise ValueError(
            f"Verification failed: expected 1 video and 1 audio stream, got "
            f"{len(v_streams)} video and {len(a_streams)} audio in {out_path}."
        )

    v_dur = float(v_streams[0].get("duration") or meta.get("format", {}).get("duration", 0.0))
    a_dur = float(a_streams[0].get("duration") or meta.get("format", {}).get("duration", 0.0))

    return {
        "video_streams": 1,
        "audio_streams": 1,
        "video_duration": v_dur,
        "audio_duration": a_dur,
    }


# ------------------------------------------------------------- CLI & Main
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add in-repo background sound and dialogue to a silent video using Vidgin's audio brain."
    )
    parser.add_argument("--video", required=True, help="Input silent video .mp4 file")
    parser.add_argument("--caption", required=True, help="Text caption used to generate the video")
    parser.add_argument("--dialogue", default=None, help="Optional spoken dialogue for voice synthesis")
    parser.add_argument("--sound-prompt", default=None, help="Custom sound prompt overriding auto-plan")
    parser.add_argument("--sound-ckpt", default="checkpoints/sound/latest.pt", help="Sound DiT checkpoint")
    parser.add_argument("--vocoder-ckpt", default="checkpoints/vocoder/latest.pt", help="Vocoder checkpoint")
    parser.add_argument("--voice-ckpt", default="checkpoints/voice/latest.pt", help="Voice TTS checkpoint")
    parser.add_argument("--steps", type=int, default=25, help="ODE diffusion steps for sound generation")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda)")
    parser.add_argument("--out", required=True, help="Output .mp4 video path with audio")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_ffmpeg_installed()

    info = get_video_info(args.video)
    duration = max(info["duration"], 1.0)

    # 1. Checkpoint Verification - Strictly enforced!
    ensure_checkpoint(
        args.vocoder_ckpt,
        "vocoder",
        "python scripts/train_audio.py --target vocoder --steps 1000 --batch-size 4",
    )
    ensure_checkpoint(
        args.sound_ckpt,
        "sound model",
        "python scripts/train_audio.py --target sound --steps 1000 --batch-size 4",
    )
    if args.dialogue and args.dialogue.strip():
        ensure_checkpoint(
            args.voice_ckpt,
            "voice model",
            "python scripts/train_audio.py --target voice --steps 1000 --batch-size 4",
        )

    print(f"\n[add_audio] Processing video: {args.video} ({duration:.2f}s)")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Voice Dialogue
        voice_audio = None
        if args.dialogue and args.dialogue.strip():
            print(f"[1/4] VOICE: Synthesizing dialogue with VidginVoiceTTS: \"{args.dialogue.strip()}\" ...")
            voice_audio = generate_speech(
                dialogue=args.dialogue.strip(),
                voice_ckpt=args.voice_ckpt,
                vocoder_ckpt=args.vocoder_ckpt,
                target_duration=duration,
                device=args.device,
            )
        else:
            print("[1/4] VOICE: No dialogue provided, skipping voice synthesis.")

        # Step 2: Background Ambience Plan & Generate
        sound_plan = plan_background_sound(args.caption, custom_prompt=args.sound_prompt)
        print(f"[2/4] SOUND: Generating background ambience with AudioDiT -> \"{sound_plan}\" ...")
        bg_audio = generate_ambience(
            sound_prompt=sound_plan,
            sound_ckpt=args.sound_ckpt,
            vocoder_ckpt=args.vocoder_ckpt,
            target_duration=duration,
            steps=args.steps,
            device=args.device,
        )

        # Step 3: Mixing with Ducking
        print("[3/4] MIXING: Mixing voice over background with 35% ducking & -1 dB normalization ...")
        mixed_audio = mix_with_ducking(voice_audio, bg_audio)
        mixed_wav = os.path.join(tmpdir, "mixed.wav")
        write_wav_file(mixed_wav, mixed_audio, sample_rate=SAMPLE_RATE)

        # Step 4: Muxing and ffprobe verification
        print(f"[4/4] MUXING: Muxing with ffmpeg -> {args.out} ...")
        meta = mux_and_verify(args.video, mixed_wav, args.out)
        print(
            f"[add_audio] SUCCESS! Verified with ffprobe: {meta['video_streams']} video "
            f"({meta['video_duration']:.2f}s) and {meta['audio_streams']} audio ({meta['audio_duration']:.2f}s)."
        )


if __name__ == "__main__":
    main()
