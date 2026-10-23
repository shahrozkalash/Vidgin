#!/usr/bin/env python3
"""Replace weak stock-site captions with descriptive captions from Gemini vision (100% Free Tier).

Paced for the Gemini API free tier ($0 cost):
- Waits at least 6 seconds between requests to respect RPM limits.
- Supports --max-calls (default 240) to stay safely within daily free-tier limits.
- On HTTP 429 / quota exhaustion, saves progress, prints a clear message, and exits cleanly.
- Resume-safe: clips already marked caption_source: gemini are skipped on subsequent runs.
- Preserves all existing keys (including "depth") untouched.
"""

from __future__ import annotations

import argparse
import getpass
import io
import json
import os
import sys
import tempfile
import time

import numpy as np
from PIL import Image

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MANIFEST_NAME = "manifest.jsonl"
FAILURES_LOG = "caption_failures.log"
DEFAULT_MODEL = "gemini-2.5-flash"
MAX_SIDE = 512
FLUSH_INTERVAL = 25
DEFAULT_PACE_SECONDS = 6.0
DEFAULT_MAX_CALLS = 240
BACKOFF_DELAYS = (1.0, 2.0, 4.0)

CAPTION_INSTRUCTION = (
    "Describe this video clip in ONE sentence of 15 to 30 words, present tense. "
    "Cover: the main subject, what it is doing and how it moves, the setting, "
    "the camera movement (static, pan, tilt, zoom, or tracking), and the "
    "lighting or time of day. Do not use the words video, clip, or footage. "
    "Do not mention audio. Do not guess at text or signs you cannot read. "
    "Do not hedge with words like appears or seems."
)


class QuotaExceededError(Exception):
    """Raised when free-tier request quota (RPM/RPD, HTTP 429) is encountered."""
    pass


def is_quota_error(exc: Exception) -> bool:
    """Detect if an exception is an HTTP 429 / rate limit / quota error."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return True
    msg = str(exc).lower()
    return (
        "429" in msg
        or "resource_exhausted" in msg
        or "rate limit" in msg
        or "quota" in msg
    )


# ------------------------------------------------------------- Frame processing
def extract_three_frames(clip: np.ndarray) -> list[np.ndarray]:
    """Sample first, middle, last frames from uint8 (T, H, W, 3) clip."""
    t = len(clip)
    if t == 0:
        raise ValueError("clip has 0 frames")
    if t == 1:
        return [clip[0], clip[0], clip[0]]
    if t == 2:
        return [clip[0], clip[0], clip[1]]
    return [clip[0], clip[t // 2], clip[t - 1]]


def frame_to_jpeg_bytes(frame: np.ndarray, max_side: int = MAX_SIDE) -> bytes:
    """Resize frame so widest side <= max_side px; return encoded JPEG bytes."""
    img = Image.fromarray(frame)
    w, h = img.size
    if max(w, h) > max_side:
        if w >= h:
            new_w = max_side
            new_h = max(1, int(round(h * max_side / w)))
        else:
            new_h = max_side
            new_w = max(1, int(round(w * max_side / h)))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ------------------------------------------------------------ Manifest handling
def load_manifest(data_dir: str) -> tuple[str, list[dict]]:
    """Read manifest.jsonl from data_dir and return (manifest_path, entries)."""
    manifest_path = os.path.join(data_dir, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{manifest_path}:{lineno}: invalid JSON: {e}") from e
            if not isinstance(rec, dict) or "file" not in rec or "caption" not in rec:
                raise ValueError(
                    f"{manifest_path}:{lineno}: each line must contain 'file' and 'caption'"
                )
            entries.append(rec)
    return manifest_path, entries


def save_manifest_atomic(manifest_path: str, entries: list[dict]) -> None:
    """Atomically rewrite manifest.jsonl via a temporary file and rename."""
    dirname = os.path.dirname(os.path.abspath(manifest_path))
    fd, tmp_path = tempfile.mkstemp(prefix="manifest_tmp_", dir=dirname, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, manifest_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def log_failure(data_dir: str, file_name: str) -> None:
    """Record a failed clip filename to caption_failures.log."""
    log_path = os.path.join(data_dir, FAILURES_LOG)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{file_name}\n")


# ------------------------------------------------------------------- Gemini API
def resolve_api_key() -> str:
    """Obtain Gemini API key exclusively from env or getpass prompt."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        try:
            key = getpass.getpass("Enter GEMINI_API_KEY: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nAPI key prompt cancelled.")
    if not key:
        sys.exit("error: GEMINI_API_KEY is not set.")
    return key


def generate_clip_caption(
    client: any,
    model: str,
    jpeg_frames: list[bytes],
    max_retries: int = 3,
) -> str:
    """Send 3 JPEG frames to Gemini. Fast-fails on 429 quota, retries transient errors."""
    try:
        from google.genai import types
        contents = [
            types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in jpeg_frames
        ]
        contents.append(types.Part.from_text(text=CAPTION_INSTRUCTION))
    except (ImportError, AttributeError):
        class _DummyPart:
            def __init__(self, data=None, mime_type=None, text=None):
                self.data = data
                self.mime_type = mime_type
                self.text = text
            @classmethod
            def from_bytes(cls, data, mime_type):
                return cls(data=data, mime_type=mime_type)
            @classmethod
            def from_text(cls, text):
                return cls(text=text)
        contents = [
            _DummyPart.from_bytes(data=b, mime_type="image/jpeg") for b in jpeg_frames
        ]
        contents.append(_DummyPart.from_text(text=CAPTION_INSTRUCTION))

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
            )
            raw = (response.text or "").strip()
            # Strip outer quotes if returned
            if (raw.startswith('"') and raw.endswith('"')) or (
                raw.startswith("'") and raw.endswith("'")
            ):
                raw = raw[1:-1].strip()
            # Collapse any internal newlines or multiple spaces into one clean sentence
            caption = " ".join(raw.split())
            if not caption:
                raise ValueError("Model returned empty text response")
            return caption
        except Exception as err:
            last_error = err
            # Immediately raise QuotaExceededError on 429/quota error without retrying
            if is_quota_error(err):
                raise QuotaExceededError(str(err)) from err

            if attempt < max_retries - 1:
                delay = BACKOFF_DELAYS[attempt] if attempt < len(BACKOFF_DELAYS) else 4.0
                time.sleep(delay)

    raise RuntimeError(
        f"API call failed after {max_retries} attempts: {last_error}"
    ) from last_error


# ------------------------------------------------------------- Main processing
def caption_dataset(
    data_dir: str,
    limit: int | None = None,
    max_calls: int | None = DEFAULT_MAX_CALLS,
    model: str = DEFAULT_MODEL,
    overwrite: bool = False,
    pace_seconds: float = DEFAULT_PACE_SECONDS,
    client: any = None,
) -> tuple[int, int, int]:
    """Process clips in data_dir, updating captions with Gemini vision.

    Paced for free-tier usage:
    - Stops cleanly when max_calls is reached.
    - Exits normally on 429 quota exhaustion after saving progress.
    - Waits pace_seconds between API calls.

    Returns (processed_count, skipped_count, failed_count).
    """
    manifest_path, entries = load_manifest(data_dir)
    total_entries = len(entries)
    if total_entries == 0:
        print(f"Manifest in {data_dir} is empty.", flush=True)
        return 0, 0, 0

    if client is None:
        from google import genai

        api_key = resolve_api_key()
        client = genai.Client(api_key=api_key)

    processed_count = 0
    skipped_count = 0
    failed_count = 0
    dirty_since_flush = 0

    target_limit = (
        min(total_entries, int(limit)) if limit is not None else total_entries
    )

    for i, entry in enumerate(entries, start=1):
        if limit is not None and processed_count >= target_limit:
            break

        # Resume safety check
        if not overwrite and entry.get("caption_source") == "gemini":
            skipped_count += 1
            continue

        # Check max-calls limit before making another API call
        if max_calls is not None and processed_count >= max_calls:
            print(
                f"Reached --max-calls limit ({max_calls}) for this session. "
                "Saving progress and stopping cleanly.",
                flush=True,
            )
            break

        clip_file = entry["file"]
        clip_path = os.path.join(data_dir, clip_file)
        if not os.path.isfile(clip_path):
            print(f"[{i}/{total_entries}] file missing: {clip_file}, skipping", flush=True)
            failed_count += 1
            log_failure(data_dir, clip_file)
            continue

        try:
            clip = np.load(clip_path)
            three_frames = extract_three_frames(clip)
            jpeg_frames = [frame_to_jpeg_bytes(f) for f in three_frames]
            new_caption = generate_clip_caption(client, model, jpeg_frames)
        except QuotaExceededError:
            # Save all progress immediately and exit cleanly per free tier requirement
            save_manifest_atomic(manifest_path, entries)
            dirty_since_flush = 0
            print(
                "Free quota reached — run again tomorrow; it will continue where it stopped",
                flush=True,
            )
            return processed_count, skipped_count, failed_count
        except Exception as e:
            failed_count += 1
            log_failure(data_dir, clip_file)
            print(f"[{i}/{total_entries}] error on {clip_file}: {e}", flush=True)
            continue

        # Preserve previous caption
        if "caption_weak" not in entry:
            entry["caption_weak"] = entry.get("caption", "")
        entry["caption"] = new_caption
        entry["caption_source"] = "gemini"

        processed_count += 1
        dirty_since_flush += 1

        # Print progress and captions for first 10 newly captioned clips
        if processed_count <= 10:
            print(f"[{i}/{total_entries}] {new_caption}", flush=True)
        else:
            print(f"[{i}/{total_entries}]", flush=True)

        # Flush manifest atomically every FLUSH_INTERVAL processed clips
        if dirty_since_flush >= FLUSH_INTERVAL:
            save_manifest_atomic(manifest_path, entries)
            dirty_since_flush = 0

        # Check if max-calls was just reached after this caption
        if max_calls is not None and processed_count >= max_calls:
            print(
                f"Reached --max-calls limit ({max_calls}) for this session. "
                "Saving progress and stopping cleanly.",
                flush=True,
            )
            break

        # Free tier pacing: wait between requests to prevent RPM rate limiting
        if pace_seconds > 0:
            time.sleep(pace_seconds)

    # Final atomic flush if any changes were made
    if dirty_since_flush > 0:
        save_manifest_atomic(manifest_path, entries)

    return processed_count, skipped_count, failed_count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace weak captions in manifest.jsonl with Gemini vision captions (100% Free Tier)."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to prepared clips directory containing .npy files and manifest.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of clips to inspect in this run (default: all)",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=DEFAULT_MAX_CALLS,
        help=f"Maximum API calls before cleanly stopping for the day (default: {DEFAULT_MAX_CALLS})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini vision model identifier (default: {DEFAULT_MODEL}; or gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-caption clips already marked with caption_source: gemini",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=DEFAULT_PACE_SECONDS,
        help=f"Minimum seconds between API requests (default: {DEFAULT_PACE_SECONDS}s for free tier)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    print(f"Captioning clips in {args.data_dir} using model {args.model} (Free Tier) ...", flush=True)
    processed, skipped, failed = caption_dataset(
        data_dir=args.data_dir,
        limit=args.limit,
        max_calls=args.max_calls,
        model=args.model,
        overwrite=args.overwrite,
        pace_seconds=args.pace,
    )
    print("=" * 50)
    print(f"Captioned: {processed}")
    print(f"Skipped:   {skipped} (already gemini)")
    print(f"Failed:    {failed} (see {FAILURES_LOG} if > 0)")


if __name__ == "__main__":
    main()
