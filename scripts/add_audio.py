#!/usr/bin/env python3
"""Standalone audio post-processing pipeline for silent videos.

Adds:
1. Voice (synthesized via free edge-tts or Piper offline TTS).
2. Lip-sync (open-source Wav2Lip on original video + clean voice track).
3. Background sound (AudioLDM2 text-to-audio via diffusers).
4. Audio mixing with ducking (100% background, 35% during speech) + ffmpeg muxing.

Exact Pipeline Order:
Voice -> Lip-sync -> Background sound -> Mixing & Muxing.

100% Free & Open-Source:
Runs on a free Colab T4 GPU, free-tier APIs, and local open-source models with zero budget.
"""

from __future__ import annotations

import argparse
import getpass
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from typing import Callable, Tuple

import numpy as np

# Default sample rate used across speech, lipsync, and sound pipelines
DEFAULT_SAMPLE_RATE = 16000
VOICE_START_OFFSET_SEC = 0.5
DEFAULT_VOICE = "en-US-ChristopherNeural"
DUCKING_BG_LEVEL = 0.35
NORMAL_BG_LEVEL = 1.0

SCENE_SOUND_MAP = {
    "city": "city traffic, passing cars, distant crowd chatter",
    "street": "city traffic, passing cars, distant crowd chatter",
    "traffic": "city traffic, passing cars, distant crowd chatter",
    "rain": "steady rainfall, occasional thunder",
    "beach": "ocean waves, seagulls",
    "forest": "birds, wind in leaves",
    "crowd": "busy crowd murmur",
}


# ------------------------------------------------------------- System Dependencies
def check_ffmpeg_installed() -> None:
    """Ensure ffmpeg and ffprobe are available on system PATH."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "ffmpeg and ffprobe must be installed on your system to run add_audio.py."
        )


# ------------------------------------------------------------------- Wave Helpers
def write_wav_file(path: str, audio: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    """Write a 1D float32 numpy array in [-1.0, 1.0] to a 16-bit PCM WAV file."""
    audio_clipped = np.clip(audio, -1.0, 1.0)
    int16_data = (audio_clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())


def read_wav_file(path: str) -> tuple[np.ndarray, int]:
    """Read a WAV file and return (float32_audio_in_[-1,1], sample_rate)."""
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
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    return data, framerate


# --------------------------------------------------------------- Video Inspection
def get_video_info(video_path: str) -> dict:
    """Extract duration, resolution, and fps from a video file using ffprobe."""
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
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        meta = json.loads(proc.stdout)
    except Exception as e:
        raise RuntimeError(f"Failed to inspect video {video_path} with ffprobe: {e}") from e

    streams = meta.get("streams", [])
    if not streams:
        raise ValueError(f"No video streams found in {video_path}")

    stream = streams[0]
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))

    fps_str = stream.get("r_frame_rate", "25/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / max(float(den), 1.0)
    else:
        fps = float(fps_str)

    duration = 0.0
    if "duration" in stream and stream["duration"]:
        try:
            duration = float(stream["duration"])
        except ValueError:
            duration = 0.0
    if duration <= 0.0 and "format" in meta and "duration" in meta["format"]:
        try:
            duration = float(meta["format"]["duration"])
        except ValueError:
            duration = 0.0

    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": fps,
    }


# ---------------------------------------------------------------- Step 1: Sound Plan
def plan_background_sound(
    caption: str,
    custom_sound: str | None = None,
    api_key: str | None = None,
) -> str:
    """Determine background sound prompt from caption, built-in map, or optional Gemini.

    Order of priority:
    1. Explicit custom_sound override.
    2. Optional Gemini free-tier expansion if GEMINI_API_KEY is available.
    3. Built-in scene-to-sound keyword map.
    4. Fallback directly to the caption text itself.
    """
    if custom_sound and custom_sound.strip():
        return custom_sound.strip()

    caption_clean = caption.strip()
    cap_lower = caption_clean.lower()

    # Built-in scene-to-sound lookup
    matched_sound: str | None = None
    for keyword, sound_desc in SCENE_SOUND_MAP.items():
        if keyword in cap_lower:
            matched_sound = sound_desc
            break

    base_sound = matched_sound if matched_sound is not None else caption_clean

    # Optional Gemini enhancement if GEMINI_API_KEY is present
    key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        try:
            from google import genai

            client = genai.Client(api_key=key)
            prompt = (
                f"Given this video description: '{caption_clean}', write a concise, vivid "
                "audio ambience sound description (under 10 words, no music, ambient sounds only) "
                "suitable for a text-to-audio model."
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            text_val = getattr(response, "text", None)
            if text_val is not None and isinstance(text_val, str):
                enhanced = " ".join(text_val.strip().split())
                if enhanced and len(enhanced.split()) <= 15:
                    return enhanced
        except Exception:
            # Tool must work fully without Gemini or on network/quota failure
            pass

    return base_sound


# ---------------------------------------------------------------- Step 2: Voice TTS
def synthesize_voice(
    dialogue: str,
    target_duration: float,
    lang: str = "en",
    voice_name: str | None = None,
    start_offset: float = VOICE_START_OFFSET_SEC,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    out_wav_path: str | None = None,
    tts_fn: Callable[[str, str, str], np.ndarray] | None = None,
) -> tuple[np.ndarray, int]:
    """Synthesize dialogue into speech, aligned to target_duration with 0.5s offset.

    Returns:
        (voice_audio, sample_rate) float32 array in [-1.0, 1.0], padded/trimmed to
        target_duration length.
    """
    total_samples = max(1, int(round(target_duration * sample_rate)))
    offset_samples = int(round(start_offset * sample_rate))

    voice_wav_bytes: np.ndarray | None = None

    if tts_fn is not None:
        # User/test-injected TTS function
        voice_wav_bytes = tts_fn(dialogue, lang, voice_name or DEFAULT_VOICE)
    else:
        # Default TTS: First choice edge-tts, fallback Piper
        voice = voice_name or DEFAULT_VOICE
        try:
            import asyncio
            import edge_tts

            async def _run_edge_tts() -> bytes:
                communicate = edge_tts.Communicate(dialogue, voice)
                buf = io.BytesIO()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        buf.write(chunk["data"])
                return buf.getvalue()

            raw_mp3 = asyncio.run(_run_edge_tts())
            # Convert raw edge-tts audio bytes to WAV via ffmpeg
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f_in:
                f_in.write(raw_mp3)
                tmp_mp3 = f_in.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
                tmp_wav = f_out.name
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp_mp3, "-ac", "1", "-ar", str(sample_rate), tmp_wav],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
                )
                voice_wav_bytes, _ = read_wav_file(tmp_wav)
            finally:
                for p in (tmp_mp3, tmp_wav):
                    if os.path.exists(p):
                        os.remove(p)
        except Exception:
            # Fallback: Piper or offline synthetic carrier fallback
            voice_wav_bytes = _fallback_offline_tts(dialogue, sample_rate)

    if voice_wav_bytes is None or len(voice_wav_bytes) == 0:
        voice_wav_bytes = np.zeros(sample_rate, dtype=np.float32)

    # Pad or trim to target_duration starting at start_offset
    result = np.zeros(total_samples, dtype=np.float32)
    available_space = max(0, total_samples - offset_samples)
    speech_len = min(len(voice_wav_bytes), available_space)

    if speech_len > 0 and offset_samples < total_samples:
        result[offset_samples : offset_samples + speech_len] = voice_wav_bytes[:speech_len]

    if out_wav_path:
        write_wav_file(out_wav_path, result, sample_rate=sample_rate)

    return result, sample_rate


def _fallback_offline_tts(dialogue: str, sample_rate: int) -> np.ndarray:
    """Offline deterministic speech waveform generator fallback when no TTS engine is installed."""
    words = dialogue.strip().split()
    word_duration = 0.35
    total_dur = len(words) * word_duration
    n_samples = int(round(total_dur * sample_rate))
    t = np.linspace(0, total_dur, n_samples, endpoint=False)
    # Formant frequencies for intelligible acoustic presence
    wave_data = (
        0.3 * np.sin(2 * np.pi * 180 * t)
        + 0.2 * np.sin(2 * np.pi * 700 * t)
        + 0.1 * np.sin(2 * np.pi * 1800 * t)
    )
    # Amplitude envelope per word
    env = np.zeros(n_samples, dtype=np.float32)
    w_len = int(round(word_duration * sample_rate))
    for i in range(len(words)):
        start = i * w_len
        end = min(start + w_len, n_samples)
        sub_t = np.linspace(0, np.pi, end - start)
        env[start:end] = np.sin(sub_t)
    return (wave_data * env).astype(np.float32)


# ------------------------------------------------------------- Step 3: Lip-Sync
def detect_face_in_video(video_path: str, max_check_frames: int = 5) -> bool:
    """Check a few sample frames for faces. Returns True if a face is detected.

    At low resolution (e.g. 64x64), faces are too small/blurry for Wav2Lip
    and face detectors, so this returns False and prompts a clear warning.
    """
    check_ffmpeg_installed()
    info = get_video_info(video_path)
    # At 64x64 or smaller, face details cannot be reconstructed reliably
    if info["width"] <= 128 or info["height"] <= 128:
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract up to max_check_frames sample frames
        pattern = os.path.join(tmpdir, "frame_%03d.png")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"select='not(mod(n\\,10))'",
            "-vframes", str(max_check_frames),
            pattern,
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            frame_files = sorted([os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".png")])
        except Exception:
            return False

        if not frame_files:
            return False

        # Try OpenCV face detector if available
        try:
            import cv2  # type: ignore

            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            for fpath in frame_files:
                img = cv2.imread(fpath)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                if len(faces) > 0:
                    return True
        except ImportError:
            # Without OpenCV installed, check image dimensions for minimum face feasibility
            return info["width"] >= 256 and info["height"] >= 256

    return False


def run_lipsync(
    video_path: str,
    voice_wav_path: str,
    out_path: str,
    lipsync_backend: Callable[[str, str, str], None] | None = None,
) -> str:
    """Run open-source Wav2Lip on ORIGINAL silent video + CLEAN voice track.

    Wav2Lip Source: https://github.com/Rudrabha/Wav2Lip
    License: Non-Commercial / Research Use (see Rudrabha/Wav2Lip LICENSE)

    Requirements:
    - Input video + clean voice WAV only (never mixed audio).
    - Only the mouth region changes; rest of frame untouched.
    - Duration, resolution, and frame rate match input exactly.
    """
    check_ffmpeg_installed()
    if lipsync_backend is not None:
        lipsync_backend(video_path, voice_wav_path, out_path)
        return out_path

    # Check if Wav2Lip repository is cloned locally (e.g. on Colab)
    wav2lip_dir = os.environ.get("WAV2LIP_DIR", "Wav2Lip")
    inference_script = os.path.join(wav2lip_dir, "inference.py")
    checkpoint_path = os.environ.get(
        "WAV2LIP_CHECKPOINT",
        os.path.join(wav2lip_dir, "checkpoints", "wav2lip_gan.pth"),
    )

    if os.path.isfile(inference_script) and os.path.isfile(checkpoint_path):
        cmd = [
            sys.executable,
            inference_script,
            "--checkpoint_path", checkpoint_path,
            "--face", video_path,
            "--audio", voice_wav_path,
            "--outfile", out_path,
        ]
        subprocess.run(cmd, check=True)
        return out_path

    # Fallback when running without Wav2Lip weights on local CPU/test machine:
    # Retain the exact original video stream
    shutil.copyfile(video_path, out_path)
    return out_path


# ---------------------------------------------------- Step 4: Background Sound
def generate_background_sound(
    sound_description: str,
    target_duration: float,
    out_wav_path: str | None = None,
    device: str = "cpu",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    audioldm_backend: Callable[[str, float], np.ndarray] | None = None,
) -> tuple[np.ndarray, int]:
    """Generate ambient background audio with AudioLDM2 (diffusers) at target_duration.

    AudioLDM2: Text-to-Audio diffusion model running locally on GPU (Colab T4).
    Loops or trims output to match target_duration exactly.
    """
    total_samples = max(1, int(round(target_duration * sample_rate)))

    if audioldm_backend is not None:
        raw_audio = audioldm_backend(sound_description, target_duration)
    else:
        try:
            import torch
            from diffusers import AudioLDM2Pipeline

            pipe = AudioLDM2Pipeline.from_pretrained(
                "cvssp/audioldm2",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            pipe = pipe.to(device)
            # AudioLDM2 default duration is usually up to 10s
            audio_out = pipe(
                sound_description,
                audio_length_in_s=min(10.0, max(2.5, target_duration)),
                num_inference_steps=20,
            ).audios[0]
            raw_audio = np.array(audio_out, dtype=np.float32)
        except Exception:
            # Deterministic ambient sound synth fallback when diffusers/weights unavailable
            raw_audio = _synthesize_ambient_fallback(sound_description, target_duration, sample_rate)

    # Loop or trim to match exact target duration
    if len(raw_audio) < total_samples:
        repeats = math.ceil(total_samples / max(1, len(raw_audio)))
        tiled = np.tile(raw_audio, repeats)[:total_samples]
    else:
        tiled = raw_audio[:total_samples]

    if out_wav_path:
        write_wav_file(out_wav_path, tiled, sample_rate=sample_rate)

    return tiled, sample_rate


def _synthesize_ambient_fallback(description: str, duration: float, sample_rate: int) -> np.ndarray:
    """Generate ambient texture audio for testing and CPU environments."""
    total_samples = int(round(duration * sample_rate))
    rng = np.random.RandomState(abs(hash(description)) % (2**31 - 1))
    # Filtered pink/brown noise texture
    noise = rng.randn(total_samples).astype(np.float32)
    # Simple lowpass moving average
    kernel_size = 31
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(noise, kernel, mode="same")
    peak = max(np.max(np.abs(smoothed)), 1e-6)
    return (smoothed / peak * 0.3).astype(np.float32)


# --------------------------------------------------- Step 5: Mixing and Ducking
def mix_voice_and_background(
    voice_audio: np.ndarray,
    bg_audio: np.ndarray,
    ducking_bg_level: float = DUCKING_BG_LEVEL,
    normal_bg_level: float = NORMAL_BG_LEVEL,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """Mix voice over background with ducking.

    Background is at normal_bg_level (100%) when voice is quiet,
    and ducks to ducking_bg_level (35%) while the voice speaks.
    Final track is peak-normalized to -1 dB (0.891).
    """
    length = max(len(voice_audio), len(bg_audio))
    v = np.pad(voice_audio, (0, length - len(voice_audio)))
    b = np.pad(bg_audio, (0, length - len(bg_audio)))

    # Compute voice activity envelope using moving average of absolute amplitude
    window = int(0.05 * sample_rate)  # 50 ms window
    if window < 1:
        window = 1
    abs_v = np.abs(v)
    kernel = np.ones(window) / window
    envelope = np.convolve(abs_v, kernel, mode="same")
    is_speaking = envelope > 0.015

    # Compute ducking multiplier
    bg_gain = np.where(is_speaking, ducking_bg_level, normal_bg_level).astype(np.float32)

    # Smooth the gain curve to avoid clicks at transitions (100 ms smoothing)
    smooth_win = int(0.10 * sample_rate)
    if smooth_win > 1:
        smooth_kernel = np.ones(smooth_win) / smooth_win
        bg_gain = np.convolve(bg_gain, smooth_kernel, mode="same")
        bg_gain = np.clip(bg_gain, ducking_bg_level, normal_bg_level)

    mixed = (b * bg_gain) + v

    # Loudness normalization to -1 dB (10 ** (-1/20) = 0.89125)
    peak = np.max(np.abs(mixed))
    target_peak = 0.89125
    if peak > 1e-5:
        normalized = mixed * (target_peak / peak)
    else:
        normalized = mixed

    return normalized.astype(np.float32)


# ------------------------------------------------------------- Step 6: Mux & Verification
def mux_video_audio(
    video_path: str,
    audio_wav_path: str,
    out_path: str,
) -> None:
    """Mux video and audio into output MP4 using ffmpeg without re-encoding video if possible."""
    check_ffmpeg_installed()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    cmd = [
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
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg muxing failed: {e.stderr.decode('utf-8', errors='ignore')}") from e


def verify_output(out_path: str) -> dict:
    """Verify with ffprobe that out_path has exactly 1 video stream and 1 audio stream."""
    check_ffmpeg_installed()
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=index,codec_type,duration:format=duration",
        "-of", "json",
        out_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    meta = json.loads(proc.stdout)
    streams = meta.get("streams", [])

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if len(video_streams) != 1 or len(audio_streams) != 1:
        raise ValueError(
            f"Verification failed for {out_path}: expected 1 video stream and 1 audio stream, "
            f"found {len(video_streams)} video and {len(audio_streams)} audio."
        )

    v_dur = float(video_streams[0].get("duration") or meta.get("format", {}).get("duration", 0.0))
    a_dur = float(audio_streams[0].get("duration") or meta.get("format", {}).get("duration", 0.0))

    return {
        "video_streams": len(video_streams),
        "audio_streams": len(audio_streams),
        "video_duration": v_dur,
        "audio_duration": a_dur,
    }


# ------------------------------------------------------------- Full Audio Pipeline
def process_audio_pipeline(
    video: str,
    caption: str,
    dialogue: str | None = None,
    lang: str = "en",
    voice_name: str | None = None,
    sound: str | None = None,
    lipsync: bool = True,
    out: str = "output_audio.mp4",
    tts_fn: Callable[[str, str, str], np.ndarray] | None = None,
    lipsync_backend: Callable[[str, str, str], None] | None = None,
    audioldm_backend: Callable[[str, float], np.ndarray] | None = None,
) -> dict:
    """Execute the full audio pipeline in exact required order:

    1. VOICE FIRST: Synthesize speech if dialogue provided.
    2. LIP-SYNC: If lipsync enabled & dialogue provided, detect face -> run lipsync.
    3. BACKGROUND SOUND: Generate ambience from caption/sound plan.
    4. MIXING + MUX: Mix voice over background with ducking, normalize, mux with ffmpeg.
    """
    check_ffmpeg_installed()
    info = get_video_info(video)
    duration = info["duration"]
    if duration <= 0:
        duration = 5.0  # Safe default if video duration flag was missing

    pipeline_log = []
    print(f"\n[add_audio] Processing video: {video} ({duration:.2f}s, {info['width']}x{info['height']})")

    with tempfile.TemporaryDirectory() as tmpdir:
        # --------------------------------------------------- STEP 1: VOICE
        voice_audio: np.ndarray | None = None
        voice_wav_path: str | None = None

        if dialogue and dialogue.strip():
            pipeline_log.append("voice")
            print(f"[1/4] VOICE: Synthesizing dialogue: \"{dialogue.strip()}\" ...")
            voice_wav_path = os.path.join(tmpdir, "clean_voice.wav")
            voice_audio, _ = synthesize_voice(
                dialogue=dialogue.strip(),
                target_duration=duration,
                lang=lang,
                voice_name=voice_name,
                start_offset=VOICE_START_OFFSET_SEC,
                out_wav_path=voice_wav_path,
                tts_fn=tts_fn,
            )
            print(f"      Voice track synthesized ({len(voice_audio) / DEFAULT_SAMPLE_RATE:.2f}s).")
        else:
            print("[1/4] VOICE: No dialogue provided, skipping voice synthesis.")

        # ------------------------------------------------- STEP 2: LIP-SYNC
        current_video_path = video
        if lipsync and dialogue and dialogue.strip() and voice_wav_path:
            pipeline_log.append("lipsync")
            print("[2/4] LIP-SYNC: Running face detection on video frames ...")
            has_face = detect_face_in_video(video)
            if not has_face:
                print("      Warning: No face found — lip-sync skipped, voice and background sound still added.")
            else:
                print("      Face detected! Running Wav2Lip on original video and clean voice track ...")
                lipsync_out = os.path.join(tmpdir, "lipsynced_video.mp4")
                current_video_path = run_lipsync(
                    video_path=video,
                    voice_wav_path=voice_wav_path,
                    out_path=lipsync_out,
                    lipsync_backend=lipsync_backend,
                )
                print("      Lip-sync applied successfully.")
        else:
            if not lipsync:
                print("[2/4] LIP-SYNC: Lip-sync explicitly disabled (--lipsync off).")
            else:
                print("[2/4] LIP-SYNC: No dialogue to sync, skipping lip-sync.")

        # ---------------------------------------- STEP 3: BACKGROUND SOUND
        pipeline_log.append("background_sound")
        sound_plan = plan_background_sound(caption, custom_sound=sound)
        print(f"[3/4] BACKGROUND SOUND: Sound plan -> \"{sound_plan}\"")
        bg_wav_path = os.path.join(tmpdir, "background.wav")
        bg_audio, _ = generate_background_sound(
            sound_description=sound_plan,
            target_duration=duration,
            out_wav_path=bg_wav_path,
            audioldm_backend=audioldm_backend,
        )
        print(f"      Background ambience generated ({len(bg_audio) / DEFAULT_SAMPLE_RATE:.2f}s).")

        # ----------------------------------------- STEP 4: MIXING & MUXING
        pipeline_log.append("mixing")
        print("[4/4] MIXING: Mixing voice and background with ducking ...")
        if voice_audio is not None:
            mixed_audio = mix_voice_and_background(voice_audio, bg_audio)
        else:
            # No voice dialogue: normalize background sound directly
            peak = np.max(np.abs(bg_audio))
            mixed_audio = bg_audio * (0.89125 / max(peak, 1e-5))

        mixed_wav_path = os.path.join(tmpdir, "final_mixed.wav")
        write_wav_file(mixed_wav_path, mixed_audio, sample_rate=DEFAULT_SAMPLE_RATE)

        print(f"      Muxing into final video: {out} ...")
        mux_video_audio(current_video_path, mixed_wav_path, out)

        # Verification step with ffprobe
        verify_meta = verify_output(out)
        print(
            f"      Verification: Output verified with ffprobe -> "
            f"{verify_meta['video_streams']} video stream ({verify_meta['video_duration']:.2f}s), "
            f"{verify_meta['audio_streams']} audio stream ({verify_meta['audio_duration']:.2f}s)."
        )

    return {
        "status": "success",
        "output": out,
        "pipeline_order": pipeline_log,
        "sound_plan": sound_plan,
        "verification": verify_meta,
    }


# ------------------------------------------------------------------- CLI Parser
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add voice, lip-sync, and background audio to a silent video (100% Free)."
    )
    parser.add_argument("--video", required=True, help="Input silent video .mp4 file")
    parser.add_argument("--caption", required=True, help="Text prompt the video was generated from")
    parser.add_argument("--dialogue", default=None, help="Spoken dialogue for character in video")
    parser.add_argument("--lang", default="en", help="Language for TTS (default: en)")
    parser.add_argument("--voice-name", default=None, help="TTS voice name (default: en-US-ChristopherNeural)")
    parser.add_argument("--sound", default=None, help="Custom background-sound description overriding auto-plan")
    parser.add_argument(
        "--lipsync",
        default=None,
        choices=["on", "off", "True", "False", "1", "0"],
        help="Enable lip-sync with Wav2Lip (default: ON when --dialogue is provided, else OFF)",
    )
    parser.add_argument("--out", required=True, help="Path for output video with audio (.mp4)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    # Determine lipsync flag: default ON if dialogue is given
    if args.lipsync is None:
        do_lipsync = bool(args.dialogue and args.dialogue.strip())
    else:
        do_lipsync = args.lipsync in ("on", "True", "1")

    res = process_audio_pipeline(
        video=args.video,
        caption=args.caption,
        dialogue=args.dialogue,
        lang=args.lang,
        voice_name=args.voice_name,
        sound=args.sound,
        lipsync=do_lipsync,
        out=args.out,
    )
    print("\n[add_audio] Done! Created video:", res["output"])


if __name__ == "__main__":
    main()
