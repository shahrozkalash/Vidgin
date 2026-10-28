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
from typing import List, Optional, Tuple

import numpy as np

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from vidgen.audio import CHECKPOINT_CLASSES
from vidgen.audio.dialogue import (
    DialogueLine,
    build_speech_track,
    describe_report,
    load_dialogue_file,
    load_voice_synthesizer,
    parse_dialogue_line,
)
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
    """Synthesize one dialogue line using Vidgin's Voice TTS + HiFi-GAN Vocoder.

    Kept for compatibility; timed multi-line dialogue goes through
    :func:`generate_dialogue_track`.
    """
    track, _ = generate_dialogue_track(
        [DialogueLine(start_offset, dialogue)], voice_ckpt, vocoder_ckpt,
        target_duration, device=device)
    return track


def generate_dialogue_track(
    lines: List[DialogueLine],
    voice_ckpt: str,
    vocoder_ckpt: str,
    target_duration: float,
    device: str = "cpu",
    min_speed_scale: float = 0.6,
) -> Tuple[np.ndarray, list]:
    """Voice timed dialogue lines onto one track (models loaded once).

    Returns ``(track, report)``; see vidgen/audio/dialogue.py for the
    placement rules (no overlaps, speed-fit before cutting, full report).
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to run Vidgin audio inference.")
    synthesize = load_voice_synthesizer(voice_ckpt, vocoder_ckpt, device)
    return build_speech_track(lines, synthesize, target_duration,
                              sample_rate=SAMPLE_RATE,
                              min_speed_scale=min_speed_scale)


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
    sound_payload = load_checkpoint(sound_ckpt, map_location=str(dev),
        safe_classes=CHECKPOINT_CLASSES)
    sound_cfg = sound_payload.get("cfg", AudioDiTConfig())
    sound_model = AudioDiT(sound_cfg).to(dev)
    sound_model.load_state_dict(sound_payload["model"])
    sound_model.eval()

    # 2. Load Vocoder
    vocoder_payload = load_checkpoint(vocoder_ckpt, map_location=str(dev),
        safe_classes=CHECKPOINT_CLASSES)
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
    parser.add_argument("--line", action="append", default=[], metavar="START:TEXT",
                        help="timed dialogue line, repeatable, e.g. --line '1.5: Where are we going?'")
    parser.add_argument("--dialogue-file", default=None,
                        help="text file of timed dialogue lines ('START: TEXT' per row, '#' comments)")
    parser.add_argument("--no-ambience", action="store_true",
                        help="voice only: skip the background-sound model (no sound checkpoint needed)")
    parser.add_argument("--min-speech-speed", type=float, default=0.6,
                        help="fastest allowed speech when fitting a line into its slot "
                             "(voice-model duration scale; 1.0 = never speed up)")
    return parser.parse_args()


def collect_dialogue(dialogue: Optional[str] = None, lines=(),
                     dialogue_file: Optional[str] = None) -> List[DialogueLine]:
    """All dialogue from the CLI forms: --dialogue (one untimed line at the
    default offset), repeatable --line 'START: TEXT', and --dialogue-file."""
    out: List[DialogueLine] = []
    if dialogue and dialogue.strip():
        out.append(DialogueLine(VOICE_START_OFFSET_SEC, dialogue.strip()))
    out.extend(parse_dialogue_line(spec) for spec in lines)
    if dialogue_file:
        out.extend(load_dialogue_file(dialogue_file))
    return out


def add_audio_to_video(
    video: str,
    out: str,
    caption: str,
    lines: List[DialogueLine],
    sound_prompt: Optional[str] = None,
    sound_ckpt: str = "checkpoints/sound/latest.pt",
    vocoder_ckpt: str = "checkpoints/vocoder/latest.pt",
    voice_ckpt: str = "checkpoints/voice/latest.pt",
    steps: int = 25,
    device: str = "cpu",
    ambience: bool = True,
    min_speed_scale: float = 0.6,
) -> dict:
    """Voice and/or score a silent video; returns the ffprobe metadata plus
    the dialogue placement report under ``"dialogue"``."""
    if not lines and not ambience:
        raise ValueError("nothing to add: no dialogue lines and ambience is off")
    check_ffmpeg_installed()

    info = get_video_info(video)
    duration = max(info["duration"], 1.0)

    # 1. Checkpoint Verification - Strictly enforced!
    ensure_checkpoint(
        vocoder_ckpt,
        "vocoder",
        "python scripts/train_audio.py --target vocoder --steps 1000 --batch-size 4",
    )
    if ambience:
        ensure_checkpoint(
            sound_ckpt,
            "sound model",
            "python scripts/train_audio.py --target sound --steps 1000 --batch-size 4",
        )
    if lines:
        ensure_checkpoint(
            voice_ckpt,
            "voice model",
            "python scripts/train_audio.py --target voice --steps 1000 --batch-size 4",
        )

    print(f"\n[add_audio] Processing video: {video} ({duration:.2f}s)")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Voice Dialogue
        voice_audio = None
        report: list = []
        if lines:
            print(f"[1/4] VOICE: Synthesizing {len(lines)} dialogue line(s) with VidginVoiceTTS ...")
            voice_audio, report = generate_dialogue_track(
                lines, voice_ckpt, vocoder_ckpt, duration,
                device=device, min_speed_scale=min_speed_scale)
            for r in report:
                if not r["skipped"]:
                    print(f"      {r['start']:6.2f}s-{r['end']:6.2f}s  {r['text']}")
            for note in describe_report(report):
                print(f"      WARNING: {note}")
        else:
            print("[1/4] VOICE: No dialogue provided, skipping voice synthesis.")

        # Step 2: Background Ambience Plan & Generate
        if ambience:
            sound_plan = plan_background_sound(caption, custom_prompt=sound_prompt)
            print(f"[2/4] SOUND: Generating background ambience with AudioDiT -> \"{sound_plan}\" ...")
            bg_audio = generate_ambience(
                sound_prompt=sound_plan,
                sound_ckpt=sound_ckpt,
                vocoder_ckpt=vocoder_ckpt,
                target_duration=duration,
                steps=steps,
                device=device,
            )
        else:
            print("[2/4] SOUND: Ambience off, voice-only track.")
            bg_audio = np.zeros(max(1, int(round(duration * SAMPLE_RATE))), dtype=np.float32)

        # Step 3: Mixing with Ducking
        print("[3/4] MIXING: Mixing voice over background with 35% ducking & -1 dB normalization ...")
        mixed_audio = mix_with_ducking(voice_audio, bg_audio)
        mixed_wav = os.path.join(tmpdir, "mixed.wav")
        write_wav_file(mixed_wav, mixed_audio, sample_rate=SAMPLE_RATE)

        # Step 4: Muxing and ffprobe verification
        print(f"[4/4] MUXING: Muxing with ffmpeg -> {out} ...")
        meta = mux_and_verify(video, mixed_wav, out)
        print(
            f"[add_audio] SUCCESS! Verified with ffprobe: {meta['video_streams']} video "
            f"({meta['video_duration']:.2f}s) and {meta['audio_streams']} audio ({meta['audio_duration']:.2f}s)."
        )
    meta["dialogue"] = report
    return meta


def main() -> None:
    args = parse_args()
    try:
        lines = collect_dialogue(args.dialogue, args.line, args.dialogue_file)
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")
    if not lines and args.no_ambience:
        sys.exit("[ERROR] --no-ambience needs dialogue (--dialogue, --line or --dialogue-file)")
    add_audio_to_video(
        args.video, args.out, args.caption, lines,
        sound_prompt=args.sound_prompt, sound_ckpt=args.sound_ckpt,
        vocoder_ckpt=args.vocoder_ckpt, voice_ckpt=args.voice_ckpt,
        steps=args.steps, device=args.device, ambience=not args.no_ambience,
        min_speed_scale=args.min_speech_speed,
    )


if __name__ == "__main__":
    main()
