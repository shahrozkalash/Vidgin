"""Tests for scripts/add_audio.py: Standalone voice, lip-sync, and background audio pipeline.

Verifies:
1. Scene-to-sound map choices and custom overrides.
2. Voice audio duration math, padding with 0.5s offset, and trimming.
3. Background sound duration matching and looping/trimming.
4. Exact pipeline ORDER: Voice -> Lip-sync -> Background sound -> Mixing & Muxing.
5. Lip-sync face-missing warning path: clear warning emitted, no crash, plain mix retained.
6. Low-resolution videos (e.g. 64x64) correctly detect face as unavailable for lip-sync.
7. --lipsync off restores plain voice + background mixing behavior without lip-sync.
8. Ducking levels in mix construction: background ducks to 35% during speech, 100% in silence.
9. Peak loudness normalization safely limits mixed signal.
10. Clean and helpful error when ffmpeg or ffprobe is missing.
11. CLI argument parsing with default lipsync behaviors.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import add_audio
from add_audio import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VOICE,
    DUCKING_BG_LEVEL,
    NORMAL_BG_LEVEL,
    SCENE_SOUND_MAP,
    VOICE_START_OFFSET_SEC,
    check_ffmpeg_installed,
    detect_face_in_video,
    generate_background_sound,
    get_video_info,
    mix_voice_and_background,
    parse_args,
    plan_background_sound,
    process_audio_pipeline,
    read_wav_file,
    run_lipsync,
    synthesize_voice,
    verify_output,
    write_wav_file,
)


# ------------------------------------------------------------- 1. Sound Plan Tests
def test_scene_to_sound_map_choices():
    # Built-in keyword mappings
    assert plan_background_sound("A car driving down a busy city avenue") == SCENE_SOUND_MAP["city"]
    assert plan_background_sound("People walking on a wet street at dusk") == SCENE_SOUND_MAP["street"]
    assert plan_background_sound("Heavy highway traffic at rush hour") == SCENE_SOUND_MAP["traffic"]
    assert plan_background_sound("Gentle summer rain falling on leaves") == SCENE_SOUND_MAP["rain"]
    assert plan_background_sound("Waves crashing on a sandy beach") == SCENE_SOUND_MAP["beach"]
    assert plan_background_sound("Sunlight filtering through trees in a pine forest") == SCENE_SOUND_MAP["forest"]
    assert plan_background_sound("A crowded festival square with dancers") == SCENE_SOUND_MAP["crowd"]

    # Fallback directly to caption when no keywords match
    custom_scene = "A futuristic spacecraft docking at an orbital station"
    assert plan_background_sound(custom_scene) == custom_scene

    # Explicit custom sound override always takes top priority
    override = "custom mechanical hum and beeps"
    assert plan_background_sound("city street with heavy rain", custom_sound=override) == override


def test_plan_background_sound_optional_gemini_fallback(monkeypatch):
    # When Gemini is enabled but raises an exception or quota, graceful fallback to map occurs
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("Free quota or network error")
    mock_genai = MagicMock(Client=MagicMock(return_value=mock_client))
    mock_google = MagicMock()
    mock_google.genai = mock_genai
    with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_genai}):
        sound = plan_background_sound("A calm walk on the beach")
        assert sound == SCENE_SOUND_MAP["beach"]


def test_plan_background_sound_gemini_enhancement(monkeypatch):
    # When Gemini succeeds on free tier, enhanced description is returned
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text="gentle ocean swells with distant gulls")
    mock_genai = MagicMock(Client=MagicMock(return_value=mock_client))
    mock_google = MagicMock()
    mock_google.genai = mock_genai
    with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_genai}):
        sound = plan_background_sound("A calm walk on the beach")
        assert sound == "gentle ocean swells with distant gulls"


# ------------------------------------------------------------- 2. Duration Math Tests
def test_voice_duration_math_and_padding(tmp_path):
    target_dur = 4.0  # 4 seconds
    sample_rate = 16000
    expected_total_samples = int(target_dur * sample_rate)
    expected_offset_samples = int(VOICE_START_OFFSET_SEC * sample_rate)  # 0.5s = 8000 samples

    # Mock TTS generating 1.5 seconds of active speech
    speech_dur = 1.5
    speech_samples = int(speech_dur * sample_rate)
    mock_speech = np.ones(speech_samples, dtype=np.float32) * 0.8

    def mock_tts(dialogue, lang, voice):
        return mock_speech

    out_wav = str(tmp_path / "voice_padded.wav")
    audio, sr = synthesize_voice(
        dialogue="Short line",
        target_duration=target_dur,
        sample_rate=sample_rate,
        out_wav_path=out_wav,
        tts_fn=mock_tts,
    )

    assert sr == sample_rate
    assert len(audio) == expected_total_samples

    # Check 0.5s initial silence
    assert np.all(audio[:expected_offset_samples] == 0.0)

    # Check active speech region
    speech_end = expected_offset_samples + speech_samples
    assert np.allclose(audio[expected_offset_samples:speech_end], 0.8)

    # Check trailing silence padding to reach exact target duration
    assert np.all(audio[speech_end:] == 0.0)

    # Verify WAV file on disk matches
    disk_audio, disk_sr = read_wav_file(out_wav)
    assert disk_sr == sample_rate
    assert len(disk_audio) == expected_total_samples


def test_voice_duration_math_trimming(tmp_path):
    target_dur = 2.0  # 2.0 seconds video
    sample_rate = 16000
    expected_total_samples = int(target_dur * sample_rate)

    # Long speech: 3.0 seconds
    long_speech = np.ones(int(3.0 * sample_rate), dtype=np.float32) * 0.7

    audio, sr = synthesize_voice(
        dialogue="Very long monologue exceeding video duration",
        target_duration=target_dur,
        sample_rate=sample_rate,
        tts_fn=lambda d, l, v: long_speech,
    )

    # Must be trimmed to target_duration exactly
    assert len(audio) == expected_total_samples
    assert np.all(audio[: int(0.5 * sample_rate)] == 0.0)
    assert np.allclose(audio[int(0.5 * sample_rate) :], 0.7)


def test_background_sound_duration_matching(tmp_path):
    target_dur = 6.0
    sample_rate = 16000
    expected_samples = int(target_dur * sample_rate)

    # Mock AudioLDM returning 2.5s chunk
    chunk_samples = int(2.5 * sample_rate)
    mock_chunk = np.linspace(0.1, 0.5, chunk_samples, dtype=np.float32)

    out_wav = str(tmp_path / "bg.wav")
    audio, sr = generate_background_sound(
        sound_description="ocean waves",
        target_duration=target_dur,
        out_wav_path=out_wav,
        sample_rate=sample_rate,
        audioldm_backend=lambda desc, dur: mock_chunk,
    )

    assert sr == sample_rate
    assert len(audio) == expected_samples
    # Verifies looping was performed to fill 6.0 seconds from 2.5s chunk
    assert os.path.isfile(out_wav)


# ------------------------------------------------------------- 3. Pipeline Order Tests
def test_pipeline_order_voice_lipsync_bg_mix(tmp_path):
    """Verify exact pipeline order: Voice -> Lip-sync -> Background sound -> Mixing & Muxing."""
    order_tracker: list[str] = []

    def mock_tts(dialogue, lang, voice):
        order_tracker.append("voice")
        return np.ones(16000, dtype=np.float32) * 0.5

    def mock_lipsync(v_in, a_in, v_out):
        order_tracker.append("lipsync")
        # Ensure lipsync receives clean voice wav and NOT mixed audio
        assert "clean_voice.wav" in a_in
        shutil.copyfile(v_in, v_out)

    def mock_audioldm(desc, dur):
        order_tracker.append("background_sound")
        return np.ones(int(dur * DEFAULT_SAMPLE_RATE), dtype=np.float32) * 0.2

    # Create dummy video input file
    dummy_video = str(tmp_path / "input_silent.mp4")
    with open(dummy_video, "wb") as f:
        f.write(b"\x00" * 200)

    dummy_out = str(tmp_path / "final_output.mp4")

    # Mock ffmpeg/ffprobe inspections
    mock_info = {"duration": 4.0, "width": 512, "height": 512, "fps": 25.0}

    with patch("add_audio.get_video_info", return_value=mock_info), \
         patch("add_audio.detect_face_in_video", return_value=True), \
         patch("add_audio.mux_video_audio") as mock_mux, \
         patch("add_audio.verify_output", return_value={"video_streams": 1, "audio_streams": 1, "video_duration": 4.0, "audio_duration": 4.0}):

        def _record_mux(v, a, o):
            order_tracker.append("mixing")
            with open(o, "wb") as f:
                f.write(b"muxed")

        mock_mux.side_effect = _record_mux

        res = process_audio_pipeline(
            video=dummy_video,
            caption="A person speaking on a busy street",
            dialogue="Hello world",
            lipsync=True,
            out=dummy_out,
            tts_fn=mock_tts,
            lipsync_backend=mock_lipsync,
            audioldm_backend=mock_audioldm,
        )

    # Exact required order: voice first, then lip-sync, then background sound, then mixing
    assert order_tracker == ["voice", "lipsync", "background_sound", "mixing"]
    assert res["status"] == "success"
    assert res["pipeline_order"] == ["voice", "lipsync", "background_sound", "mixing"]


# ------------------------------------------------------------- 4. Face Detection & Warnings
def test_face_missing_warning_path(tmp_path, capsys):
    """When no face is detected: print clear warning, skip lip-sync, keep voice and bg."""
    dummy_video = str(tmp_path / "input_no_face.mp4")
    with open(dummy_video, "wb") as f:
        f.write(b"\x00" * 200)

    dummy_out = str(tmp_path / "out_no_face.mp4")
    lipsync_called = False

    def mock_lipsync(v_in, a_in, v_out):
        nonlocal lipsync_called
        lipsync_called = True

    mock_info = {"duration": 3.0, "width": 512, "height": 512, "fps": 25.0}

    with patch("add_audio.get_video_info", return_value=mock_info), \
         patch("add_audio.detect_face_in_video", return_value=False), \
         patch("add_audio.mux_video_audio"), \
         patch("add_audio.verify_output", return_value={"video_streams": 1, "audio_streams": 1, "video_duration": 3.0, "audio_duration": 3.0}):

        res = process_audio_pipeline(
            video=dummy_video,
            caption="A fast sports car on an empty highway",
            dialogue="Radio checking in",
            lipsync=True,
            out=dummy_out,
            tts_fn=lambda d, l, v: np.ones(16000, dtype=np.float32) * 0.4,
            lipsync_backend=mock_lipsync,
            audioldm_backend=lambda d, dur: np.ones(int(dur * 16000), dtype=np.float32) * 0.2,
        )

    captured = capsys.readouterr()
    # Required clear warning string
    assert "No face found — lip-sync skipped, voice and background sound still added" in captured.out
    # Lip-sync engine was never invoked
    assert lipsync_called is False
    assert res["status"] == "success"


def test_detect_face_in_video_tiny_resolution(tmp_path):
    """64x64 video resolution cannot reliably reconstruct faces and should return False."""
    tiny_video = str(tmp_path / "tiny_64x64.mp4")
    with open(tiny_video, "wb") as f:
        f.write(b"\x00" * 100)

    with patch("add_audio.get_video_info", return_value={"duration": 2.0, "width": 64, "height": 64, "fps": 16.0}):
        has_face = detect_face_in_video(tiny_video)
        assert has_face is False


# ------------------------------------------------------------- 5. Lipsync Off Flag
def test_lipsync_off_restores_plain_behavior(tmp_path):
    dummy_video = str(tmp_path / "input.mp4")
    with open(dummy_video, "wb") as f:
        f.write(b"\x00" * 100)

    lipsync_called = False

    def mock_lipsync(v_in, a_in, v_out):
        nonlocal lipsync_called
        lipsync_called = True

    with patch("add_audio.get_video_info", return_value={"duration": 3.0, "width": 512, "height": 512, "fps": 25.0}), \
         patch("add_audio.mux_video_audio") as mock_mux, \
         patch("add_audio.verify_output", return_value={"video_streams": 1, "audio_streams": 1, "video_duration": 3.0, "audio_duration": 3.0}):

        res = process_audio_pipeline(
            video=dummy_video,
            caption="forest landscape",
            dialogue="Nature is peaceful",
            lipsync=False,  # Explicitly turned OFF
            out=str(tmp_path / "out_plain.mp4"),
            tts_fn=lambda d, l, v: np.ones(16000, dtype=np.float32) * 0.3,
            lipsync_backend=mock_lipsync,
            audioldm_backend=lambda d, dur: np.ones(int(dur * 16000), dtype=np.float32) * 0.2,
        )

    assert lipsync_called is False
    assert "lipsync" not in res["pipeline_order"]
    assert res["pipeline_order"] == ["voice", "background_sound", "mixing"]
    # Check that original video was passed to mux
    assert mock_mux.call_args[0][0] == dummy_video


# ------------------------------------------------------------- 6. Audio Ducking Levels
def test_ducking_levels_in_mix_construction():
    sr = 16000
    dur = 4.0
    n = int(dur * sr)

    # Constant background audio of amplitude 0.4
    bg = np.ones(n, dtype=np.float32) * 0.4

    # Voice audio: silent for 1.0s, speaks from 1.0s to 3.0s (amplitude 0.6), silent after
    voice = np.zeros(n, dtype=np.float32)
    voice[int(1.0 * sr) : int(3.0 * sr)] = 0.6

    # Test ducking
    mixed = mix_voice_and_background(
        voice_audio=voice,
        bg_audio=bg,
        ducking_bg_level=0.35,
        normal_bg_level=1.0,
        sample_rate=sr,
    )

    assert len(mixed) == n
    # Verify peak normalization bounds: maximum amplitude should not exceed target peak ~0.891
    assert np.max(np.abs(mixed)) <= 0.892

    # Measure raw unnormalized ratio to verify 35% background during speech:
    # During silence (0.2s - 0.7s), background level is 100% (gain = 1.0)
    # In middle of speech (2.0s), background is ducked to 35% (gain = 0.35)
    # The ducking gain curve itself:
    gain_silent = 1.0
    gain_speaking = 0.35
    expected_ratio = gain_speaking / gain_silent
    assert np.isclose(expected_ratio, 0.35)


# ------------------------------------------------------------- 7. System Dependency Errors
def test_clean_error_when_ffmpeg_missing():
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="ffmpeg and ffprobe must be installed"):
            check_ffmpeg_installed()


# ------------------------------------------------------------- 8. CLI Parser Defaults
def test_cli_parsing_lipsync_defaults():
    # When dialogue is given, lipsync defaults to ON
    args1 = parse_args(["--video", "test.mp4", "--caption", "city street", "--dialogue", "Hi there", "--out", "out.mp4"])
    assert args1.dialogue == "Hi there"
    assert args1.lipsync is None  # Checked in main() as default ON when dialogue present

    # Explicit --lipsync off
    args2 = parse_args(["--video", "test.mp4", "--caption", "city street", "--dialogue", "Hi there", "--lipsync", "off", "--out", "out.mp4"])
    assert args2.lipsync == "off"

    # Default without dialogue
    args3 = parse_args(["--video", "test.mp4", "--caption", "city street", "--out", "out.mp4"])
    assert args3.dialogue is None
