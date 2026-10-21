#!/usr/bin/env python3
"""Dataset preparation for Vidgin's audio models.

Builds training manifests from 100% FREE, no-key sources only:
1. Voice: LJSpeech-1.1
   - Speaker: Single English female speaker (Linda Johnson, LibriVox)
   - Duration: ~24 hours, 13,100 clips
   - License: Public Domain (CC0 / LibriVox)
   - URL: https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
   - Manifest: {"wav": path, "text": normalized_text, "duration": float}

2. Background Sounds:
   - ESC-50: 2,000 environmental recordings across 50 classes (5s each)
     License: Creative Commons Attribution-NonCommercial (CC-BY-NC 3.0)
     URL: https://github.com/karolpiczak/ESC-50/archive/master.zip
   - UrbanSound8K / FSD50K subset: urban environmental sounds
     License: Creative Commons (CC-BY-NC 3.0)
   - Labels are mapped to descriptive diffusion captions using a curated scene template map
   - Manifest: {"wav": path, "caption": caption, "duration": float}

3. Synthetic smoke mode:
   - `--synthetic`: Generates clean synthetic audio + captions on CPU with zero downloads
     for instant pipeline verification.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tarfile
import urllib.request
import wave
import zipfile
from typing import Dict, List, Optional, Tuple

TARGET_SAMPLE_RATE = 22050

# Curated scene and sound template map for class-to-caption expansion
SOUND_TEMPLATE_MAP: Dict[str, str] = {
    "rain": "steady rainfall with gentle falling water drops",
    "thunderstorm": "heavy thunder and pouring rain",
    "sea_waves": "ocean waves crashing against the sandy beach",
    "waves": "ocean waves and distant sea breeze",
    "wind": "wind whistling and blowing through trees",
    "crackling_fire": "warm crackling campfire flames in the open air",
    "fire": "crackling fire and burning wood embers",
    "chirping_birds": "wild birds singing in the morning forest",
    "birds": "birds chirping in the trees",
    "crowd": "ambient crowd chatter and street murmur",
    "street_music": "live street music performance in the city",
    "car_horn": "city traffic with car horns and passing vehicles",
    "engine_idling": "idling car engine and city street noise",
    "engine": "engine rumbling and mechanical hum",
    "traffic": "busy city traffic and passing cars",
    "siren": "distant emergency siren echoing through city streets",
    "dog": "dog barking outdoors",
    "dog_bark": "dog barking outdoors in the distance",
    "cat": "cat meowing gently indoors",
    "footsteps": "rhythmic footsteps walking on pavement",
    "clock_tick": "clock ticking steadily in a quiet room",
    "train": "train moving along the railway tracks",
    "airplane": "airplane jet engine soaring overhead",
    "crying_baby": "baby crying softly in a room",
    "laughing": "people laughing and talking happily",
    "applause": "enthusiastic audience applause and clapping",
}


def label_to_caption(label: str) -> str:
    """Map audio classification label to a descriptive prompt for audio diffusion."""
    norm = label.lower().strip().replace("-", "_").replace(" ", "_")
    if norm in SOUND_TEMPLATE_MAP:
        return SOUND_TEMPLATE_MAP[norm]
    # Check partial substring matches
    for k, v in SOUND_TEMPLATE_MAP.items():
        if k in norm or norm in k:
            return v
    # Clean fallback
    clean_name = norm.replace("_", " ")
    return f"ambient sound of {clean_name}"


def write_wav(path: str, samples: List[float], sr: int = TARGET_SAMPLE_RATE) -> None:
    """Write float samples in [-1.0, 1.0] to a 16-bit mono WAV file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    raw = bytearray()
    for s in samples:
        clamped = max(-1.0, min(1.0, s))
        val = int(clamped * 32767.0)
        raw.extend(struct.pack("<h", val))
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(raw)


def read_wav_info(path: str) -> Tuple[int, int, float]:
    """Return (sample_rate, num_samples, duration_seconds)."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        dur = float(n) / max(sr, 1)
        return sr, n, dur


def resample_wav_to_22050(in_path: str, out_path: str) -> float:
    """Resample any WAV/audio file to 22050 Hz mono 16-bit PCM.

    Uses ffmpeg if available, otherwise native wave stream copy if already 22050.
    Returns duration in seconds.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    sr, n, dur = read_wav_info(in_path)
    if sr == TARGET_SAMPLE_RATE:
        if in_path != out_path:
            shutil.copyfile(in_path, out_path)
        return dur

    # Try ffmpeg
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-i", in_path,
            "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
            out_path,
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        _, _, dur_out = read_wav_info(out_path)
        return dur_out

    # Simple linear interpolation fallback in pure python
    with wave.open(in_path, "rb") as wf:
        n_ch = wf.getnchannels()
        w_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    fmt = f"<{n_frames * n_ch}h" if w_width == 2 else None
    if fmt is None:
        raise RuntimeError(f"Cannot resample {in_path} without ffmpeg (sampwidth={w_width})")

    unpacked = struct.unpack(fmt, raw)
    if n_ch > 1:
        # Average channels to mono
        mono = [
            sum(unpacked[i * n_ch : (i + 1) * n_ch]) / (n_ch * 32768.0)
            for i in range(n_frames)
        ]
    else:
        mono = [v / 32768.0 for v in unpacked]

    # Linear resample
    out_len = int(round(len(mono) * (TARGET_SAMPLE_RATE / sr)))
    resampled = []
    ratio = float(len(mono) - 1) / max(1, out_len - 1)
    for i in range(out_len):
        pos = i * ratio
        idx = int(pos)
        frac = pos - idx
        if idx + 1 < len(mono):
            val = mono[idx] * (1.0 - frac) + mono[idx + 1] * frac
        else:
            val = mono[idx]
        resampled.append(val)

    write_wav(out_path, resampled, sr=TARGET_SAMPLE_RATE)
    return float(out_len) / TARGET_SAMPLE_RATE


# ----------------------------------------------------------- LJSpeech Download
LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


def prepare_ljspeech(out_dir: str, max_clips: Optional[int] = None) -> str:
    """Download and prepare LJSpeech-1.1 dataset (Public Domain).

    Returns path to manifest_voice.jsonl.
    """
    voice_dir = os.path.join(out_dir, "ljspeech")
    os.makedirs(voice_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest_voice.jsonl")

    tar_path = os.path.join(voice_dir, "LJSpeech-1.1.tar.bz2")
    extract_marker = os.path.join(voice_dir, "LJSpeech-1.1", "metadata.csv")

    if not os.path.isfile(extract_marker):
        print(f"[prepare_audio] Downloading LJSpeech-1.1 from {LJSPEECH_URL} ...")
        print("  License: Public Domain (CC0 / LibriVox recordings)")
        urllib.request.urlretrieve(LJSPEECH_URL, tar_path)
        print("[prepare_audio] Extracting LJSpeech-1.1 archive ...")
        with tarfile.open(tar_path, "r:bz2") as tar:
            tar.extractall(path=voice_dir)

    metadata_path = extract_marker
    wav_base_dir = os.path.join(voice_dir, "LJSpeech-1.1", "wavs")
    out_wav_dir = os.path.join(voice_dir, "resampled_wavs")
    os.makedirs(out_wav_dir, exist_ok=True)

    items = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 3:
                continue
            clip_id = row[0].strip()
            # row[2] has normalized text (numbers expanded as words)
            norm_text = row[2].strip() or row[1].strip()
            orig_wav = os.path.join(wav_base_dir, f"{clip_id}.wav")
            if not os.path.isfile(orig_wav):
                continue

            target_wav = os.path.join(out_wav_dir, f"{clip_id}.wav")
            dur = resample_wav_to_22050(orig_wav, target_wav)
            items.append({
                "id": clip_id,
                "wav": os.path.abspath(target_wav),
                "text": norm_text,
                "duration": round(dur, 3),
                "license": "Public Domain (LibriVox / CC0)",
            })

            if max_clips and len(items) >= max_clips:
                break

    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    print(f"[prepare_audio] Voice manifest created: {manifest_path} ({len(items)} clips)")
    return manifest_path


# ------------------------------------------------------------- ESC-50 Download
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"


def prepare_esc50(out_dir: str, max_clips: Optional[int] = None) -> str:
    """Download and prepare ESC-50 dataset (CC-BY-NC 3.0).

    Returns path to manifest_sound.jsonl.
    """
    esc_dir = os.path.join(out_dir, "esc50")
    os.makedirs(esc_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest_sound.jsonl")

    zip_path = os.path.join(esc_dir, "esc50.zip")
    csv_marker = os.path.join(esc_dir, "ESC-50-master", "meta", "esc50.csv")

    if not os.path.isfile(csv_marker):
        print(f"[prepare_audio] Downloading ESC-50 from {ESC50_URL} ...")
        print("  License: Creative Commons Attribution-NonCommercial (CC-BY-NC 3.0)")
        urllib.request.urlretrieve(ESC50_URL, zip_path)
        print("[prepare_audio] Extracting ESC-50 archive ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(esc_dir)

    metadata_path = csv_marker
    audio_base_dir = os.path.join(esc_dir, "ESC-50-master", "audio")
    out_wav_dir = os.path.join(esc_dir, "resampled_wavs")
    os.makedirs(out_wav_dir, exist_ok=True)

    items = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            category = row["category"]
            orig_wav = os.path.join(audio_base_dir, filename)
            if not os.path.isfile(orig_wav):
                continue

            caption = label_to_caption(category)
            target_wav = os.path.join(out_wav_dir, filename)
            dur = resample_wav_to_22050(orig_wav, target_wav)

            items.append({
                "id": filename,
                "wav": os.path.abspath(target_wav),
                "caption": caption,
                "category": category,
                "duration": round(dur, 3),
                "license": "Creative Commons Attribution-NonCommercial (CC-BY-NC 3.0)",
            })

            if max_clips and len(items) >= max_clips:
                break

    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    print(f"[prepare_audio] Sound manifest created: {manifest_path} ({len(items)} clips)")
    return manifest_path


# ------------------------------------------------------------- Synthetic Data
def prepare_synthetic_data(out_dir: str, num_samples: int = 16) -> Tuple[str, str, str]:
    """Generate deterministic synthetic audio clips and manifests for CPU testing."""
    os.makedirs(out_dir, exist_ok=True)
    voice_wav_dir = os.path.join(out_dir, "synth_voice")
    sound_wav_dir = os.path.join(out_dir, "synth_sound")
    os.makedirs(voice_wav_dir, exist_ok=True)
    os.makedirs(sound_wav_dir, exist_ok=True)

    # 1. Voice samples
    sample_texts = [
        "the quick brown fox jumps over the lazy dog",
        "welcome to vidgin video latent diffusion studio",
        "sound and speech are generated in real time",
        "every model is defined in this repository",
        "we build and train our own models from scratch",
        "zero third party models and zero downloaded weights",
        "the vocoder turns mel spectrograms into waveforms",
        "rectified flow diffusion creates beautiful sounds",
    ]
    voice_items = []
    sr = TARGET_SAMPLE_RATE

    for i in range(num_samples):
        text = sample_texts[i % len(sample_texts)]
        words = text.split()
        dur = max(1.5, len(words) * 0.35)
        n_samples = int(round(dur * sr))

        # Harmonics for speech formant synthesis
        samples = []
        for s_idx in range(n_samples):
            t = s_idx / sr
            val = (
                0.3 * math.sin(2.0 * math.pi * 140.0 * t)
                + 0.2 * math.sin(2.0 * math.pi * 650.0 * t)
                + 0.1 * math.sin(2.0 * math.pi * 1800.0 * t)
            )
            # Envelope modulation per word
            env = 0.5 + 0.5 * math.sin(2.0 * math.pi * (len(words) / dur) * t)
            samples.append(val * env)

        wav_path = os.path.join(voice_wav_dir, f"voice_{i:03d}.wav")
        write_wav(wav_path, samples, sr=sr)
        voice_items.append({
            "id": f"voice_{i:03d}",
            "wav": os.path.abspath(wav_path),
            "text": text,
            "duration": round(dur, 3),
            "license": "Vidgin Synthetic (In-repo Generator)",
        })

    voice_manifest = os.path.join(out_dir, "manifest_voice.jsonl")
    with open(voice_manifest, "w", encoding="utf-8") as f:
        for it in voice_items:
            f.write(json.dumps(it) + "\n")

    # 2. Sound samples
    sound_prompts = [
        ("city", "city traffic and passing cars"),
        ("rain", "steady rainfall with gentle water drops"),
        ("ocean", "ocean waves crashing against the sandy beach"),
        ("forest", "wild birds singing in the morning forest"),
        ("thunder", "heavy thunder and pouring rain"),
        ("crowd", "ambient crowd chatter and street murmur"),
        ("fire", "warm crackling campfire flames in the open air"),
        ("wind", "wind whistling and blowing through trees"),
    ]
    sound_items = []
    for i in range(num_samples):
        cat, caption = sound_prompts[i % len(sound_prompts)]
        dur = 3.0
        n_samples = int(round(dur * sr))

        # Synthesize colored noise texture
        samples = []
        state = (i + 1) * 1103515245 + 12345
        for s_idx in range(n_samples):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            noise = (state / 0x7FFFFFFF) * 2.0 - 1.0
            t = s_idx / sr
            # Low frequency modulation for waves / wind / rumble
            mod = 0.5 + 0.5 * math.sin(2.0 * math.pi * 0.5 * t)
            samples.append(noise * 0.25 * mod)

        wav_path = os.path.join(sound_wav_dir, f"sound_{i:03d}.wav")
        write_wav(wav_path, samples, sr=sr)
        sound_items.append({
            "id": f"sound_{i:03d}",
            "wav": os.path.abspath(wav_path),
            "caption": caption,
            "category": cat,
            "duration": round(dur, 3),
            "license": "Vidgin Synthetic (In-repo Generator)",
        })

    sound_manifest = os.path.join(out_dir, "manifest_sound.jsonl")
    with open(sound_manifest, "w", encoding="utf-8") as f:
        for it in sound_items:
            f.write(json.dumps(it) + "\n")

    # 3. Combined vocoder manifest
    vocoder_manifest = os.path.join(out_dir, "manifest_vocoder.jsonl")
    with open(vocoder_manifest, "w", encoding="utf-8") as f:
        for it in voice_items + sound_items:
            f.write(json.dumps({"wav": it["wav"], "duration": it["duration"]}) + "\n")

    print(f"[prepare_audio] Created synthetic datasets in {out_dir}:")
    print(f"  - Voice:   {voice_manifest} ({len(voice_items)} clips)")
    print(f"  - Sound:   {sound_manifest} ({len(sound_items)} clips)")
    print(f"  - Vocoder: {vocoder_manifest} ({len(voice_items) + len(sound_items)} clips)")

    return voice_manifest, sound_manifest, vocoder_manifest


# ------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare training data for Vidgin's audio models from FREE, no-key sources."
    )
    parser.add_argument(
        "--target",
        choices=["all", "voice", "sound", "vocoder"],
        default="all",
        help="Dataset to prepare: voice (LJSpeech), sound (ESC-50), vocoder (both), or all.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/audio",
        help="Directory to save downloads and manifests (default: data/audio).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic in-repo test data on CPU with zero downloads.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Limit number of clips to process per dataset (useful for testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.synthetic:
        prepare_synthetic_data(args.out_dir, num_samples=args.max_clips or 16)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    voice_manifest: Optional[str] = None
    sound_manifest: Optional[str] = None

    if args.target in ("all", "voice", "vocoder"):
        voice_manifest = prepare_ljspeech(args.out_dir, max_clips=args.max_clips)

    if args.target in ("all", "sound", "vocoder"):
        sound_manifest = prepare_esc50(args.out_dir, max_clips=args.max_clips)

    if args.target in ("all", "vocoder") and voice_manifest and sound_manifest:
        # Build unified manifest for vocoder training
        vocoder_manifest = os.path.join(args.out_dir, "manifest_vocoder.jsonl")
        count = 0
        with open(vocoder_manifest, "w", encoding="utf-8") as out_f:
            for m_path in (voice_manifest, sound_manifest):
                with open(m_path, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        row = json.loads(line)
                        out_f.write(json.dumps({"wav": row["wav"], "duration": row["duration"]}) + "\n")
                        count += 1
        print(f"[prepare_audio] Combined vocoder manifest: {vocoder_manifest} ({count} clips)")


if __name__ == "__main__":
    main()
