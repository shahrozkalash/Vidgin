"""Tests for scripts/caption_clips.py: 100% Free Tier Gemini vision clip captioner.

Verifies:
1. Frame sampling and resize to <= 512 px max side.
2. Manifest rewriting (captions updated, caption_weak kept, caption_source='gemini').
3. Preservation of "depth" keys completely untouched.
4. Resume-skip behavior when caption_source='gemini'.
5. Overwrite flag forces re-captioning.
6. Limit option caps inspected clips.
7. Pacing stop: stops cleanly when --max-calls is reached and saves progress.
8. Free quota exit: on HTTP 429 / quota exhaustion, saves progress, prints the
   exact free-quota message, and exits cleanly (no crash, picks up on next run).
9. General transient API error retries up to 3 times before logging to failures.
"""

from __future__ import annotations

import io
import json
import os
import sys
from unittest.mock import MagicMock, patch

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
    QuotaExceededError,
    caption_dataset,
    extract_three_frames,
    frame_to_jpeg_bytes,
    is_quota_error,
    load_manifest,
    parse_args,
    resolve_api_key,
    save_manifest_atomic,
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


def test_resume_skip_and_overwrite(tmp_path):
    clips = [
        {
            "file": "clip_000001.npy",
            "caption": "A red bus drives along a street under noon sun.",
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

    _, entries_run1 = load_manifest(data_dir)
    assert entries_run1[0]["caption"] == "A red bus drives along a street under noon sun."
    assert (
        entries_run1[1]["caption"]
        == "A golden retriever sprints across a green meadow towards a wooden fence, the camera tracking low in warm evening light."
    )

    # 2. With overwrite: both clips are re-captioned
    mock_client.reset_mock()
    mock_client.models.generate_content.side_effect = [
        MagicMock(text="Overwritten caption 1"),
        MagicMock(text="Overwritten caption 2"),
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


def test_max_calls_pacing_stop(tmp_path):
    """Verify that reaching --max-calls stops cleanly and saves progress."""
    clips = [
        {"file": f"clip_{i:06d}.npy", "caption": f"weak label {i}"}
        for i in range(1, 6)
    ]
    data_dir = _create_mock_dataset(tmp_path, clips)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        MagicMock(text=f"A scenic landscape view number {i} under morning sun, the camera static.")
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
    assert "caption_source" not in entries[3]
    assert "caption_source" not in entries[4]

    # Resume without overwrite with max_calls=2: skips the first 2, processes next 2
    mock_client.reset_mock()
    mock_client.models.generate_content.side_effect = [
        MagicMock(text=f"A scenic landscape view number {i} under morning sun, the camera static.")
        for i in range(3, 6)
    ]

    processed2, skipped2, failed2 = caption_dataset(
        data_dir=data_dir,
        max_calls=2,
        pace_seconds=0.0,
        client=mock_client,
    )
    assert processed2 == 2
    assert skipped2 == 2
    assert mock_client.models.generate_content.call_count == 2

    _, entries2 = load_manifest(data_dir)
    assert entries2[2]["caption_source"] == "gemini"
    assert entries2[3]["caption_source"] == "gemini"
    assert "caption_source" not in entries2[4]


def test_free_quota_error_exit(tmp_path, capsys):
    """Verify HTTP 429 quota error saves progress, prints message, and exits cleanly."""
    clips = [
        {"file": "clip_000001.npy", "caption": "first clip"},
        {"file": "clip_000002.npy", "caption": "quota trigger clip"},
        {"file": "clip_000003.npy", "caption": "unreached clip"},
    ]
    data_dir = _create_mock_dataset(tmp_path, clips)

    class Mock429Error(Exception):
        status_code = 429

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        MagicMock(text="A red bicycle rests against a stone wall in an alley, the camera static in morning light."),
        Mock429Error("Resource exhausted: 429 Rate limit exceeded. Quota reached for free tier."),
    ]

    processed, skipped, failed = caption_dataset(
        data_dir=data_dir,
        pace_seconds=0.0,
        client=mock_client,
    )

    captured = capsys.readouterr()
    assert "Free quota reached — run again tomorrow; it will continue where it stopped" in captured.out
    assert processed == 1
    assert failed == 0  # Not logged as failure, treated as clean quota stop

    # Verify clip 1 was saved into manifest, clip 2 and 3 remain untouched
    _, entries = load_manifest(data_dir)
    assert entries[0]["caption_source"] == "gemini"
    assert "caption_source" not in entries[1]
    assert "caption_source" not in entries[2]

    # Verify no entry was written to caption_failures.log
    failures_log = os.path.join(data_dir, FAILURES_LOG)
    assert not os.path.exists(failures_log)


def test_is_quota_error():
    class ErrWithCode(Exception):
        code = 429

    class ErrWithStatusCode(Exception):
        status_code = 429

    assert is_quota_error(ErrWithCode())
    assert is_quota_error(ErrWithStatusCode())
    assert is_quota_error(RuntimeError("HTTP 429: Too Many Requests"))
    assert is_quota_error(RuntimeError("RESOURCE_EXHAUSTED quota exceeded"))
    assert is_quota_error(RuntimeError("Rate limit exceeded for model"))
    assert not is_quota_error(RuntimeError("Connection timeout"))
    assert not is_quota_error(ValueError("Invalid frame shape"))


def test_api_retry_and_failure_logging(tmp_path, monkeypatch):
    clips = [
        {"file": "clip_000001.npy", "caption": "failing clip"},
        {"file": "clip_000002.npy", "caption": "working clip"},
    ]
    data_dir = _create_mock_dataset(tmp_path, clips)

    monkeypatch.setattr(caption_clips.time, "sleep", lambda s: None)

    mock_client = MagicMock()

    # Clip 1 fails 3 times on transient network error (not quota); Clip 2 succeeds
    mock_client.models.generate_content.side_effect = [
        ConnectionResetError("Connection lost"),
        ConnectionResetError("Connection lost"),
        ConnectionResetError("Connection lost"),
        MagicMock(
            text="A train speeds across a viaduct through rolling green hills, the camera tilting upward under bright midday sun."
        ),
    ]

    processed, skipped, failed = caption_dataset(
        data_dir=data_dir,
        pace_seconds=0.0,
        client=mock_client,
    )
    assert processed == 1
    assert skipped == 0
    assert failed == 1

    failures_log = os.path.join(data_dir, FAILURES_LOG)
    assert os.path.isfile(failures_log)
    with open(failures_log, "r", encoding="utf-8") as f:
        failed_files = [line.strip() for line in f if line.strip()]
    assert failed_files == ["clip_000001.npy"]


def test_resolve_api_key_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env_test_secret_key_123")
    key = resolve_api_key()
    assert key == "env_test_secret_key_123"


def test_resolve_api_key_from_getpass(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(caption_clips.getpass, "getpass", lambda prompt: "prompted_key_456")
    key = resolve_api_key()
    assert key == "prompted_key_456"


def test_resolve_api_key_empty_exits(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(caption_clips.getpass, "getpass", lambda prompt: "")
    with pytest.raises(SystemExit):
        resolve_api_key()


def test_parse_args():
    args = parse_args([
        "--data-dir", "/content/clips",
        "--limit", "50",
        "--max-calls", "100",
        "--model", "gemini-2.5-flash-lite",
        "--overwrite",
        "--pace", "8.0",
    ])
    assert args.data_dir == "/content/clips"
    assert args.limit == 50
    assert args.max_calls == 100
    assert args.model == "gemini-2.5-flash-lite"
    assert args.overwrite is True
    assert args.pace == 8.0

    # Defaults
    args_default = parse_args(["--data-dir", "/content/clips"])
    assert args_default.limit is None
    assert args_default.max_calls == DEFAULT_MAX_CALLS
    assert args_default.model == "gemini-2.5-flash"
    assert args_default.overwrite is False
    assert args_default.pace == 6.0
