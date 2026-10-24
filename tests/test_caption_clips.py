"""Tests for scripts/caption_clips.py: Upgraded two-backend clip captioning system.

Verifies:
1. Frame sampling and resize to <= 512 px max side.
2. Manifest rewriting (captions updated, caption_weak kept, caption_source='gemini' or 'local').
3. Preservation of "depth" keys completely untouched.
4. Resume-skip behavior for both caption_source='gemini' and 'local'.
5. Overwrite flag forces re-captioning.
6. Limit option caps inspected clips.
7. Pacing stop: stops cleanly when --max-calls is reached and saves progress.
8. Free quota exit: on HTTP 429 / quota exhaustion across all keys, saves progress,
   prints message, and exits cleanly.
9. Two-key rotation on HTTP 429: rotates from GEMINI_API_KEY to GEMINI_API_KEY_2.
10. Model fallback order on 503/overload:
    gemini-3.8-flash -> gemini-3.5-flash -> gemini-3.6-flash -> gemini-3.7-flash -> gemini-flash-latest.
11. Auto-backend switch on quota exhaustion: continues on local backend when all keys hit quota.
12. Quality-gate rejections: rejects empty, <8 words, fps, resolution, or bare filenames;
    retries once; marks failed if retry also fails.
13. Local backend direct captioning and caption post-processing (15-30 words, single sentence).
14. Failure logging to caption_failures.log and per-backend summary reporting.
"""

from __future__ import annotations

import io
import json
import os
import sys
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import caption_clips
from caption_clips import (
    CAPTION_INSTRUCTION,
    DEFAULT_MAX_CALLS,
    DEFAULT_MODEL,
    FAILURES_LOG,
    FALLBACK_MODELS,
    QuotaExceededError,
    caption_dataset,
    extract_three_frames,
    frame_to_jpeg_bytes,
    generate_clip_caption,
    generate_clip_caption_local,
    is_overload_error,
    is_quota_error,
    load_manifest,
    normalize_frame,
    parse_args,
    postprocess_local_caption,
    resolve_api_key,
    resolve_api_keys,
    resolve_local_model_source,
    sample_frames,
    save_manifest_atomic,
    validate_caption_detailed,
    validate_caption_quality,
)


def _create_mock_dataset(
    tmp_path,
    clips: list[dict],
    shape: tuple[int, int, int, int] = (17, 64, 64, 3),
) -> str:
    """Create a temporary dataset with .npy clips and manifest.jsonl."""
    data_dir = str(tmp_path)
    rng = np.random.RandomState(42)

    for item in clips:
        arr = rng.randint(0, 256, size=shape, dtype=np.uint8)
        np.save(os.path.join(data_dir, item["file"]), arr)
        if "depth" in item and item["depth"]:
            depth_arr = rng.randint(0, 256, size=shape[:3], dtype=np.uint8)
            np.save(os.path.join(data_dir, item["depth"]), depth_arr)

    manifest_path = os.path.join(data_dir, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in clips:
            f.write(json.dumps(item) + "\n")

    return data_dir


# ------------------------------------------------------------- Basic frame tests
def test_extract_three_frames():
    clip_17 = np.zeros((17, 32, 32, 3), dtype=np.uint8)
    clip_17[0, 0, 0, 0] = 10
    clip_17[8, 0, 0, 0] = 20
    clip_17[16, 0, 0, 0] = 30

    sampled = extract_three_frames(clip_17)
    assert len(sampled) == 3
    assert sampled[0][0, 0, 0] == 10
    assert sampled[1][0, 0, 0] == 20
    assert sampled[2][0, 0, 0] == 30

    clip_1 = np.ones((1, 10, 10, 3), dtype=np.uint8) * 5
    sampled_1 = extract_three_frames(clip_1)
    assert len(sampled_1) == 3
    assert sampled_1[0][0, 0, 0] == 5

    clip_2 = np.zeros((2, 10, 10, 3), dtype=np.uint8)
    clip_2[0, 0, 0, 0] = 1
    clip_2[1, 0, 0, 0] = 2
    sampled_2 = extract_three_frames(clip_2)
    assert len(sampled_2) == 3
    assert sampled_2[0][0, 0, 0] == 1
    assert sampled_2[2][0, 0, 0] == 2

    with pytest.raises(ValueError, match="0 frames"):
        extract_three_frames(np.zeros((0, 10, 10, 3), dtype=np.uint8))


def test_frame_to_jpeg_bytes_resizing():
    wide_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    jpeg_bytes = frame_to_jpeg_bytes(wide_frame, max_side=512)
    assert isinstance(jpeg_bytes, bytes)
    assert len(jpeg_bytes) > 0

    img = Image.open(io.BytesIO(jpeg_bytes))
    w, h = img.size
    assert max(w, h) == 512
    assert w == 512
    assert h == round(720 * 512 / 1280)


# ------------------------------------------------------------- Manifest & Depth
def test_caption_dataset_updates_manifest_and_fields(tmp_path):
    clips = [
        {"file": "clip_000001.npy", "caption": "busy street"},
        {
            "file": "clip_000002.npy",
            "caption": "car driving",
            "depth": "clip_000002_depth.npy",
        },
        {"file": "clip_000003.npy", "caption": "pedestrians crossing"},
    ]
    data_dir = _create_mock_dataset(tmp_path, clips)

    generated_captions = [
        "A red double-decker bus drives along a busy city street past walking pedestrians, the camera tracking beside it under bright midday sun.",
        "A sleek silver sedan cruises down a coastal highway beside crashing waves, the camera panning smoothly in golden afternoon light.",
        "Pedestrians with umbrellas cross a rain-slicked city intersection as yellow taxis idle, the camera static under overcast skies.",
    ]

    mock_client = MagicMock()
    mock_responses = [MagicMock(text=c) for c in generated_captions]
    mock_client.models.generate_content.side_effect = mock_responses

    processed, skipped, failed = caption_dataset(
        data_dir=data_dir,
        pace_seconds=0.0,
        client=mock_client,
    )

    assert processed == 3
    assert skipped == 0
    assert failed == 0
    assert mock_client.models.generate_content.call_count == 3

    # Check instruction was passed in contents
    call_args = mock_client.models.generate_content.call_args_list[0]
    contents = call_args.kwargs.get("contents") or call_args.args[1]
    prompt_part = contents[-1]
    assert prompt_part.text == CAPTION_INSTRUCTION

    # Verify manifest was updated correctly
    _, updated_entries = load_manifest(data_dir)
    assert len(updated_entries) == 3

    # Clip 1
    assert updated_entries[0]["file"] == "clip_000001.npy"
    assert updated_entries[0]["caption"] == generated_captions[0]
    assert updated_entries[0]["caption_weak"] == "busy street"
    assert updated_entries[0]["caption_source"] == "gemini"

    # Clip 2: verify depth key survived completely untouched!
    assert updated_entries[1]["file"] == "clip_000002.npy"
    assert updated_entries[1]["caption"] == generated_captions[1]
    assert updated_entries[1]["caption_weak"] == "car driving"
    assert updated_entries[1]["caption_source"] == "gemini"
    assert updated_entries[1]["depth"] == "clip_000002_depth.npy"

    # Clip 3
    assert updated_entries[2]["file"] == "clip_000003.npy"
    assert updated_entries[2]["caption"] == generated_captions[2]
    assert updated_entries[2]["caption_weak"] == "pedestrians crossing"
    assert updated_entries[2]["caption_source"] == "gemini"


# ------------------------------------------------------------- Resume safety for both backends
def test_resume_skip_both_gemini_and_local(tmp_path):
    """Resume-safe: skip entries whose caption_source is already 'gemini' or 'local'."""
    clips = [
        {
            "file": "clip_000001.npy",
            "caption": "A red bus drives along a street under noon sun with camera static.",
            "caption_weak": "city bus",
            "caption_source": "gemini",
        },
        {
            "file": "clip_000002.npy",
            "caption": "A dog runs across a lawn with the camera tracking smoothly under bright daylight.",
            "caption_weak": "dog running",
            "caption_source": "local",
        },
        {"file": "clip_000003.npy", "caption": "horse galloping"},
    ]
    data_dir = _create_mock_dataset(tmp_path, clips)

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(
        text="A brown horse gallops across an open fenced pasture, the camera tracking steadily in morning light."
    )

    # 1. Without overwrite: both clip 1 and clip 2 are skipped; only clip 3 is captioned
    processed, skipped, failed = caption_dataset(
        data_dir=data_dir,
        overwrite=False,
        pace_seconds=0.0,
        client=mock_client,
    )
    assert processed == 1
    assert skipped == 2
    assert failed == 0
    assert mock_client.models.generate_content.call_count == 1

    _, entries_run1 = load_manifest(data_dir)
    assert entries_run1[0]["caption_source"] == "gemini"
    assert entries_run1[1]["caption_source"] == "local"
    assert entries_run1[2]["caption_source"] == "gemini"

    # 2. With overwrite: all three clips are re-captioned
    mock_client.reset_mock()
    mock_client.models.generate_content.side_effect = [
        MagicMock(text="A cyclist rides past tall glass skyscrapers along a designated bike lane, camera panning under morning sun."),
        MagicMock(text="A golden retriever chases a red ball through green park grass, camera tracking low in afternoon light."),
        MagicMock(text="A brown stallion trots across a grassy field near wooden stables, camera static in clear daylight."),
    ]

    processed2, skipped2, failed2 = caption_dataset(
        data_dir=data_dir,
        overwrite=True,
        pace_seconds=0.0,
        client=mock_client,
    )
    assert processed2 == 3
    assert skipped2 == 0
    assert mock_client.models.generate_content.call_count == 3


def test_resume_skip_and_overwrite(tmp_path):
    clips = [
        {
            "file": "clip_000001.npy",
            "caption": "A red bus drives along a street under noon sun with camera tracking smoothly.",
            "caption_weak": "city bus",
            "caption_source": "gemini",
        },
        {"file": "clip_000002.npy", "caption": "dog running"},
    ]
    data_dir = _create_mock_dataset(tmp_path, clips)

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(
        text="A golden retriever sprints across a green meadow towards a wooden fence, the camera tracking low in warm evening light."
    )

    # 1. Without overwrite: clip 1 is skipped, only clip 2 is processed
    processed, skipped, failed = caption_dataset(
        data_dir=data_dir,
        overwrite=False,
        pace_seconds=0.0,
        client=mock_client,
    )
    assert processed == 1
    assert skipped == 1
    assert failed == 0
    assert mock_client.models.generate_content.call_count == 1

    # 2. With overwrite: both clips are re-captioned with valid >=8 word captions
    mock_client.reset_mock()
    mock_client.models.generate_content.side_effect = [
        MagicMock(text="An express train speeds across a mountain bridge under clear skies, camera tracking alongside smoothly."),
        MagicMock(text="A cargo ship navigates into a busy industrial harbor channel, camera static in foggy morning light."),
    ]

    processed, skipped, failed = caption_dataset(
        data_dir=data_dir,
        overwrite=True,
        pace_seconds=0.0,
        client=mock_client,
    )
    assert processed == 2
    assert skipped == 0
    assert mock_client.models.generate_content.call_count == 2


# ------------------------------------------------------------- Pacing & Max calls
def test_max_calls_pacing_stop(tmp_path):
    """Verify that reaching --max-calls stops cleanly and saves progress."""
    clips = [
        {"file": f"clip_{i:06d}.npy", "caption": f"weak label {i}"}
        for i in range(1, 6)
    ]
    data_dir = _create_mock_dataset(tmp_path, clips)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        MagicMock(text=f"A scenic landscape view number {i} under morning sun, the camera static beside the road.")
        for i in range(1, 6)
    ]

    # Run with max_calls=2
    processed, skipped, failed = caption_dataset(
        data_dir=data_dir,
        max_calls=2,
        pace_seconds=0.0,
        client=mock_client,
    )

    assert processed == 2
    assert failed == 0
    assert mock_client.models.generate_content.call_count == 2

    # Verify manifest saved the 2 processed clips
    _, entries = load_manifest(data_dir)
    assert entries[0]["caption_source"] == "gemini"
    assert entries[1]["caption_source"] == "gemini"
    assert "caption_source" not in entries[2]


# ------------------------------------------------------------- Two-Key Rotation on HTTP 429
def test_two_key_rotation_on_429(tmp_path, monkeypatch):
    """Verify rotation to GEMINI_API_KEY_2 when key 1 encounters HTTP 429 quota."""
    monkeypatch.setattr(caption_clips.time, "sleep", lambda s: None)

    clips = [
        {"file": "clip_000001.npy", "caption": "first clip"},
        {"file": "clip_000002.npy", "caption": "second clip"},
    ]
    data_dir = _create_mock_dataset(tmp_path, clips)

    class Mock429(Exception):
        status_code = 429

    client_key1 = MagicMock()
    # Client 1 fails on clip 1 with 429
    client_key1.models.generate_content.side_effect = Mock429("Quota exceeded on key 1")

    client_key2 = MagicMock()
    # Client 2 succeeds on clip 1 and clip 2
    client_key2.models.generate_content.side_effect = [
        MagicMock(text="A sailboat glides across calm blue ocean waters towards a lighthouse, camera tracking steadily in morning sun."),
        MagicMock(text="A cyclist rides past autumn trees in a quiet park, camera tracking smoothly beside the path in golden afternoon light."),
    ]

    processed, skipped, failed = caption_dataset(
        data_dir=data_dir,
        pace_seconds=0.0,
        clients=[client_key1, client_key2],
    )

    assert processed == 2
    assert failed == 0
    assert client_key1.models.generate_content.call_count == 1
    assert client_key2.models.generate_content.call_count == 2

    _, entries = load_manifest(data_dir)
    assert entries[0]["caption_source"] == "gemini"
    assert entries[1]["caption_source"] == "gemini"


def test_free_quota_error_exit_when_all_keys_exhausted(tmp_path, capsys, monkeypatch):
    """Verify HTTP 429 quota error across all keys saves progress, prints message, and exits cleanly."""
    monkeypatch.setattr(caption_clips.time, "sleep", lambda s: None)

    clips = [
        {"file": "clip_000001.npy", "caption": "first clip"},
        {"file": "clip_000002.npy", "caption": "quota trigger clip"},
        {"file": "clip_000003.npy", "caption": "unreached clip"},
    ]
    data_dir = _create_mock_dataset(tmp_path, clips)

    class Mock429Error(Exception):
        status_code = 429

    client1 = MagicMock()
    client1.models.generate_content.side_effect = [
        MagicMock(text="A red bicycle rests against a stone wall in an alley, the camera static in morning light."),
        Mock429Error("Key 1 quota exceeded"),
    ]

    client2 = MagicMock()
    client2.models.generate_content.side_effect = Mock429Error("Key 2 quota exceeded")

    processed, skipped, failed = caption_dataset(
        data_dir=data_dir,
        pace_seconds=0.0,
        clients=[client1, client2],
        backend="gemini",
    )

    captured = capsys.readouterr()
    assert "Free quota reached — run again tomorrow; it will continue where it stopped" in captured.out
    assert processed == 1
    assert failed == 0

    _, entries = load_manifest(data_dir)
    assert entries[0]["caption_source"] == "gemini"
    assert "caption_source" not in entries[1]
    assert "caption_source" not in entries[2]


# ------------------------------------------------------------- Model Fallback on 503
def test_model_fallback_order_on_503(monkeypatch):
    """Verify fallback chain on 503/overload:
    gemini-3.8-flash -> gemini-3.5-flash -> gemini-3.6-flash -> gemini-3.7-flash -> gemini-flash-latest
    with backoff delays.
    """
    sleep_calls = []
    monkeypatch.setattr(caption_clips.time, "sleep", lambda s: sleep_calls.append(s))

    class Mock503Error(Exception):
        status_code = 503

    client = MagicMock()

    # Model 1 (gemini-3.8-flash) fails 3 times on 503
    # Model 2 (gemini-3.5-flash) fails 3 times on 503
    # Model 3 (gemini-3.6-flash) succeeds!
    called_models = []

    def mock_generate_content(model, contents):
        called_models.append(model)
        if model in ("gemini-3.8-flash", "gemini-3.5-flash"):
            raise Mock503Error(f"Model {model} overloaded (503 Service Unavailable)")
        return MagicMock(
            text="A speedboat cuts through turquoise coastal waters leaving a white wake, camera panning under midday sun."
        )

    client.models.generate_content.side_effect = mock_generate_content

    caption = generate_clip_caption(
        client,
        model="gemini-3.8-flash",
        jpeg_frames=[b"dummy_jpeg"],
    )

    assert "speedboat" in caption
    # Verify exact models tried in order
    unique_models_in_order = []
    for m in called_models:
        if not unique_models_in_order or unique_models_in_order[-1] != m:
            unique_models_in_order.append(m)

    assert unique_models_in_order == [
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.6-flash",
    ]
    # Verify 503 backoff delays were passed (10s, 25s)
    assert 10.0 in sleep_calls
    assert 25.0 in sleep_calls


# ------------------------------------------------------------- Auto-Backend Switch
def test_auto_backend_switch_on_quota_exhaustion(tmp_path, monkeypatch):
    """Verify --backend auto: starts on Gemini, and when quota is exhausted across all keys,
    automatically continues the SAME run on the local backend.
    """
    monkeypatch.setattr(caption_clips.time, "sleep", lambda s: None)

    clips = [
        {"file": "clip_000001.npy", "caption": "first clip"},
        {"file": "clip_000002.npy", "caption": "second clip"},
        {"file": "clip_000003.npy", "caption": "third clip"},
    ]
    data_dir = _create_mock_dataset(tmp_path, clips)

    class Mock429(Exception):
        status_code = 429

    mock_gemini_client = MagicMock()
    # Clip 1 succeeds on Gemini; Clip 2 triggers 429
    mock_gemini_client.models.generate_content.side_effect = [
        MagicMock(text="A steam locomotive rolls through snowy pine forests, the camera tracking steadily under winter sunlight."),
        Mock429("Daily quota exhausted (429)"),
    ]

    # Mock local captioner function
    local_responses = [
        "a silver airplane landing on a wet runway with the camera panning smoothly in the fog",
        "a group of friends hiking up a rocky hill trail under clear blue skies",
    ]
    mock_local = MagicMock(side_effect=local_responses)

    processed, skipped, failed = caption_dataset(
        data_dir=data_dir,
        backend="auto",
        pace_seconds=0.0,
        client=mock_gemini_client,
        local_captioner=mock_local,
    )

    assert processed == 3
    assert skipped == 0
    assert failed == 0

    _, entries = load_manifest(data_dir)
    # Clip 1 produced by Gemini
    assert entries[0]["caption_source"] == "gemini"
    assert "steam locomotive" in entries[0]["caption"]

    # Clip 2 produced by Local after auto-switch
    assert entries[1]["caption_source"] == "local"
    assert "airplane" in entries[1]["caption"].lower()

    # Clip 3 also produced by Local
    assert entries[2]["caption_source"] == "local"
    assert "hiking" in entries[2]["caption"].lower()


# ------------------------------------------------------------- Local Backend Direct
def test_local_backend_direct(tmp_path):
    """Verify --backend local functions completely offline with zero Gemini dependencies."""
    clips = [
        {"file": "clip_000001.npy", "caption": "weak tag 1"},
        {"file": "clip_000002.npy", "caption": "weak tag 2"},
    ]
    data_dir = _create_mock_dataset(tmp_path, clips)

    mock_local = MagicMock(
        side_effect=[
            "a surfer riding a tall ocean wave at sunrise",
            "a deer standing quietly in a misty autumn forest",
        ]
    )

    processed, skipped, failed = caption_dataset(
        data_dir=data_dir,
        backend="local",
        pace_seconds=0.0,
        local_captioner=mock_local,
    )

    assert processed == 2
    assert skipped == 0
    assert failed == 0

    _, entries = load_manifest(data_dir)
    assert entries[0]["caption_source"] == "local"
    assert entries[1]["caption_source"] == "local"
    # Honest local captions: the model's own words, cleaned — never padded
    # with invented camera/lighting phrases to reach a word count.
    assert entries[0]["caption"] == "A surfer riding a tall ocean wave at sunrise."
    assert entries[0]["caption"].endswith(".")
    assert "tracking" not in entries[0]["caption"].lower()
    assert "camera" not in entries[0]["caption"].lower()


# ------------------------------------------------------------- Quality Gate Rejections
def test_quality_gate_rejections(tmp_path, monkeypatch):
    """Quality gate: reject empty, near-empty, fps, resolution, or bare filenames.
    Retry once; if retry fails, mark clip failed and log to failures file.

    Layered gate (upgraded): a short but real caption is VALID with a
    warning — length alone is not a failure.
    """
    monkeypatch.setattr(caption_clips.time, "sleep", lambda s: None)

    # Unit validation checks
    assert not validate_caption_quality("")[0]
    assert not validate_caption_quality("   ")[0]
    assert not validate_caption_quality("Too short")[0]  # 2 words: no information
    ok_short, _r, warns_short = validate_caption_detailed("A car driving on road")
    assert ok_short is True  # 5 real words: valid for a simple scene
    assert any("short" in w for w in warns_short)
    assert not validate_caption_quality("A video shot at 60fps in the park under sun")[0]  # fps
    assert not validate_caption_quality("A footage rendered at 1920x1080 resolution in daylight")[0]  # resolution
    assert not validate_caption_quality("clip_000124.mp4")[0]  # bare filename
    assert not validate_caption_quality("A street scene rendered in 1080p under cloudy skies today")[0]  # resolution

    # Valid sentence passes
    valid_text = "A red bus drives along the city street under bright midday sun, camera tracking steadily."
    ok, reason = validate_caption_quality(valid_text)
    assert ok is True
    assert reason == ""

    # Test integration: Clip 1 fails on attempt 1, passes on retry attempt 2.
    # Clip 2 fails on both attempts 1 and 2 -> marked failed.
    clips = [
        {"file": "clip_000001.npy", "caption": "recovering clip"},
        {"file": "clip_000002.npy", "caption": "always failing clip"},
    ]
    data_dir = _create_mock_dataset(tmp_path, clips)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        # Clip 1 attempt 1: rejected (< 8 words)
        MagicMock(text="A car driving"),
        # Clip 1 attempt 2 (retry): valid
        MagicMock(text="A sleek sports car cruises along a winding mountain highway, camera tracking smoothly in bright afternoon sunlight."),
        # Clip 2 attempt 1: rejected (fps metadata)
        MagicMock(text="A train moving fast captured at 30 fps under morning sun."),
        # Clip 2 attempt 2 (retry): rejected (resolution metadata)
        MagicMock(text="A train moving fast at 1920x1080 resolution in the countryside."),
    ]

    processed, skipped, failed = caption_dataset(
        data_dir=data_dir,
        pace_seconds=0.0,
        client=mock_client,
    )

    assert processed == 1
    assert failed == 1

    _, entries = load_manifest(data_dir)
    assert entries[0]["caption_source"] == "gemini"
    assert "caption_source" not in entries[1]

    failures_log = os.path.join(data_dir, FAILURES_LOG)
    assert os.path.isfile(failures_log)
    with open(failures_log, "r", encoding="utf-8") as f:
        failed_files = [line.strip() for line in f if line.strip()]
    assert "clip_000002.npy" in failed_files


# ------------------------------------------------------------- Postprocessing & Helpers
def test_postprocess_local_caption():
    """REGRESSION (fabrication): post-processing must never invent details.

    The previous implementation appended camera-movement and lighting
    clauses (e.g. ", with the camera tracking smoothly under bright
    natural daylight") plus a setting phrase to pad captions to 15-30
    words. BLIP observed none of that. The honest contract: clean the
    model's own words only.
    """
    raw_short = "a dog running on the grass"
    processed = postprocess_local_caption(raw_short)
    assert processed == "A dog running on the grass."
    assert "camera" not in processed.lower()
    assert "daylight" not in processed.lower()
    assert "tracking" not in processed.lower()
    assert "open setting" not in processed.lower()

    raw_already_good = (
        "A red vintage sports car cruises down an empty desert highway, "
        "the camera tracking smoothly under bright midday sunlight."
    )
    processed_good = postprocess_local_caption(raw_already_good)
    # The model's own words are preserved verbatim (cleaned only).
    assert processed_good == raw_already_good
    assert processed_good.endswith(".")


def test_resolve_api_keys_two_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "primary_key_abc")
    monkeypatch.setenv("GEMINI_API_KEY_2", "secondary_key_xyz")

    keys = resolve_api_keys()
    assert keys == ["primary_key_abc", "secondary_key_xyz"]
    assert resolve_api_key() == "primary_key_abc"


def test_resolve_api_keys_single_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "single_key_123")
    monkeypatch.delenv("GEMINI_API_KEY_2", raising=False)

    keys = resolve_api_keys()
    assert keys == ["single_key_123"]
    assert resolve_api_key() == "single_key_123"


def test_resolve_api_keys_from_getpass(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_2", raising=False)
    monkeypatch.setattr(caption_clips.getpass, "getpass", lambda prompt: "prompted_key_789")

    keys = resolve_api_keys()
    assert keys == ["prompted_key_789"]


def test_resolve_api_key_empty_exits(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_2", raising=False)
    monkeypatch.setattr(caption_clips.getpass, "getpass", lambda prompt: "")
    with pytest.raises(SystemExit):
        resolve_api_key()


def test_is_quota_and_overload_error():
    class ErrWithCode(Exception):
        code = 429

    class Err503(Exception):
        status_code = 503

    assert is_quota_error(ErrWithCode())
    assert is_quota_error(RuntimeError("HTTP 429: Too Many Requests"))
    assert is_quota_error(RuntimeError("RESOURCE_EXHAUSTED quota reached"))

    assert is_overload_error(Err503())
    assert is_overload_error(RuntimeError("503 Service Unavailable: The model is overloaded. Please try again later."))
    assert is_overload_error(RuntimeError("Model is temporarily overloaded"))
    assert not is_overload_error(RuntimeError("Connection timeout"))


def test_parse_args():
    args = parse_args([
        "--data-dir", "/content/clips",
        "--backend", "auto",
        "--limit", "50",
        "--max-calls", "100",
        "--model", "gemini-3.5-flash",
        "--overwrite",
        "--pace", "8.0",
    ])
    assert args.data_dir == "/content/clips"
    assert args.backend == "auto"
    assert args.limit == 50
    assert args.max_calls == 100
    assert args.model == "gemini-3.5-flash"
    assert args.overwrite is True
    assert args.pace == 8.0

    # Defaults
    args_default = parse_args(["--data-dir", "/content/clips"])
    assert args_default.backend == "gemini"
    assert args_default.limit is None
    assert args_default.max_calls == DEFAULT_MAX_CALLS
    assert args_default.model == "gemini-3.8-flash"
    assert args_default.overwrite is False
    assert args_default.pace == 6.0
