#!/usr/bin/env python3
"""Replace weak stock-site captions with descriptive captions from Gemini vision or local BLIP.

Upgraded two-backend captioning system (100% Free Tier):
- Gemini backend:
  - Default model: gemini-3.8-flash.
  - Fallback chain on 503/overload: gemini-3.5-flash, gemini-3.6-flash, gemini-3.7-flash, gemini-flash-latest.
  - 503 retries with backoff delays (10s, 25s, 50s).
  - Two API keys support (GEMINI_API_KEY and GEMINI_API_KEY_2) with automatic rotation on HTTP 429.
  - Paced at >=6s between calls with --max-calls cap.
- Local backend (--backend local):
  - Locally running IMAGE captioner (Salesforce/blip-image-captioning-base via transformers).
    BLIP captions single frames; it cannot judge motion, camera movement, or
    lighting changes. Local captions are therefore honest frame-grounded
    descriptions of the sampled middle frame — nothing is appended or
    invented to make them sound richer.
  - Works on Colab/Kaggle GPU if available, else CPU, with $0 cost and zero API quota.
  - Lazy import guard so Gemini runs on machines without transformers.
  - Weights come from --local-model-dir / VIDGIN_LOCAL_CAPTION_MODEL when set
    (a local folder is loaded offline); otherwise the HuggingFace model id is
    downloaded on first use.
- Auto mode (--backend auto):
  - Tries Gemini first, rotates keys on 429, and on full quota exhaustion continues on local backend.
- Robustness:
  - Resume-safe: skips entries whose caption_source is "gemini" or "local".
  - Atomic manifest saves flushed every 5 captions and on exit.
  - Layered quality validation: hard failures (empty, metadata, model error
    text, wrappers) reject the caption; softer issues (very generic wording,
    duplicate of another clip's caption) are warnings recorded in the run
    report / review export, never silent edits.
  - Preserves all existing keys (especially "depth" and "caption_weak").
  - Final SUMMARY line printing per-backend counts.
  - --dry-run reports what would change without writing anything.
"""

from __future__ import annotations

import argparse
import getpass
import io
import json
import os
import re
import sys
import tempfile
import time

import numpy as np
from PIL import Image

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MANIFEST_NAME = "manifest.jsonl"
FAILURES_LOG = "caption_failures.log"
DEFAULT_MODEL = "gemini-3.8-flash"
FALLBACK_MODELS = (
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-flash-latest",
)
MAX_SIDE = 512
FLUSH_INTERVAL = 5
DEFAULT_PACE_SECONDS = 6.0
DEFAULT_MAX_CALLS = 240
TRANSIENT_BACKOFF_DELAYS = (1.0, 2.0, 4.0)
OVERLOAD_BACKOFF_DELAYS = (10.0, 25.0, 50.0)
LOCAL_MODEL_NAME = "Salesforce/blip-image-captioning-base"

CAPTION_INSTRUCTION = (
    "You are given several frames from one short video clip, in chronological "
    "order. Write a factual caption of one or two sentences, present tense, "
    "normally 20 to 60 words when the content justifies it; shorter is fine "
    "for a simple scene. Describe only what the frames actually show: the "
    "main subject and its visible appearance, its action and how it changes "
    "across the frames, the environment and background, and interactions "
    "between subjects. Mention camera movement (static, pan, tilt, zoom, "
    "tracking) ONLY when comparing the frames gives clear evidence of it; "
    "do not confuse subject movement with camera movement. Mention lighting "
    "or weather only when clearly visible. For a static scene, describe the "
    "composition instead of inventing motion. Do not invent actions, "
    "objects, locations, identities, emotions, or motives. Do not use the "
    "words video, clip, or footage. Do not mention audio, subtitles, or "
    "watermarks. Return ONLY the caption, with no introduction or quotes."
)

_FPS_PATTERN = re.compile(r"\b\d*\.?\d*\s*fps\b", re.IGNORECASE)
_RES_PATTERN = re.compile(
    r"\b\d{3,4}\s*[xX*×]\s*\d{3,4}\b|\b(1080p|720p|4k|2k|480p|360p)\b",
    re.IGNORECASE,
)
_EXT_PATTERN = re.compile(
    r"\.(mp4|mov|avi|mkv|npy|jpg|jpeg|png|webm)$",
    re.IGNORECASE,
)
_FILENAME_PREFIX_PATTERN = re.compile(
    r"^(clip|video|img|dsc|mov|vid)[\w\-]*(\.[a-z0-9]+)?$",
    re.IGNORECASE,
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


def is_overload_error(exc: Exception) -> bool:
    """Detect if an exception is an HTTP 503 / model overload / temporarily unavailable error."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 503:
        return True
    msg = str(exc).lower()
    return (
        "503" in msg
        or "overloaded" in msg
        or "overload" in msg
        or "service unavailable" in msg
        or "unavailable" in msg
    )


# ------------------------------------------------------------- Frame processing
def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Return one frame as uint8 (H, W, 3).

    Handles grayscale (H, W) and (H, W, 1), RGBA (drops alpha), and float
    arrays (assumed 0..1 or 0..255 range, clipped and scaled safely).
    Raises ValueError for empty or unsupported shapes.
    """
    arr = np.asarray(frame)
    if arr.size == 0:
        raise ValueError("frame is empty")
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.concatenate([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif not (arr.ndim == 3 and arr.shape[-1] == 3):
        raise ValueError(f"unsupported frame shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float64)
        if arr.max(initial=0) <= 1.5:  # 0..1 float range
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _frames_nearly_identical(a: np.ndarray, b: np.ndarray,
                             threshold: float = 2.0) -> bool:
    """Cheap duplicate check: mean absolute pixel difference below threshold."""
    if a.shape != b.shape:
        return False
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)))) < threshold


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


def sample_frames(clip: np.ndarray, max_frames: int = 3) -> list[np.ndarray]:
    """Deterministically sample up to ``max_frames`` frames across a clip.

    Frames stay in chronological order and are spread evenly. Consecutive
    picks that are (nearly) identical pixels are dropped when the clip has
    other distinct frames to offer, so static duplicates do not waste the
    model's context. Every frame is normalized to uint8 (H, W, 3).
    With max_frames=3 this returns first/middle/last, matching
    extract_three_frames on distinct frames.
    """
    arr = np.asarray(clip)
    if arr.ndim < 3 or len(arr) == 0:
        raise ValueError("clip has 0 frames")
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    t = len(arr)
    if max_frames == 3:
        picked = extract_three_frames(arr)
    elif t == 1:
        picked = [arr[0]]
    else:
        idx = sorted({int(round(i * (t - 1) / (max_frames - 1)))
                      for i in range(max_frames)} if max_frames > 1 else {t // 2})
        picked = [arr[i] for i in idx]
    frames = [normalize_frame(f) for f in picked]
    if len(frames) > 1:
        distinct = [frames[0]]
        for f in frames[1:]:
            if not _frames_nearly_identical(distinct[-1], f):
                distinct.append(f)
        # A fully static clip legitimately collapses to one frame.
        frames = distinct
    return frames


def frame_to_jpeg_bytes(frame: np.ndarray, max_side: int = MAX_SIDE) -> bytes:
    """Resize frame so widest side <= max_side px; return encoded JPEG bytes."""
    img = Image.fromarray(normalize_frame(frame))
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


def _write_json_atomic(path: str, obj: any) -> None:
    """Write JSON to path atomically (temp file + rename)."""
    dirname = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(prefix="report_tmp_", dir=dirname, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def _write_jsonl_atomic(path: str, rows: list[dict]) -> None:
    """Write JSONL rows to path atomically (temp file + rename)."""
    dirname = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(prefix="review_tmp_", dir=dirname, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
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


# ------------------------------------------------------------- Quality Gate
_MIN_WORDS_HARD = 4        # below this a caption carries almost no information
_MAX_WORDS_WARN = 90       # above this it is probably rambling, not describing
_GENERIC_PATTERNS = (
    "a video of", "video of something", "something happening",
    "some stuff", "a scene with", "an image of something",
)
_INTRO_PATTERNS = re.compile(
    r"^(here is|here's|this is (a|the) caption|sure[,!.]|the caption is|"
    r"in this (video|clip|image)|the (video|image) (shows|depicts|displays))\b",
    re.IGNORECASE,
)
_ERROR_PATTERNS = re.compile(
    r"(as an ai|i cannot|i can't|unable to (process|view|see)|"
    r"error[:\s]|failed to|no image (was )?(provided|given)|api (key|error))",
    re.IGNORECASE,
)
_MARKDOWN_PATTERN = re.compile(r"(^|\s)(#{1,6}\s|\*\*|```|^\s*[-*]\s)", re.MULTILINE)


def validate_caption_detailed(caption: str) -> tuple[bool, str, list[str]]:
    """Layered caption validation.

    Returns (is_valid, hard_failure_reason, warnings).

    Hard failures reject the caption outright: empty, extremely short,
    file metadata (fps / resolution / bare filename), model error text,
    markdown or JSON wrappers, introductory phrases, or a repeated
    sentence. Warnings never reject: generic wording, unusual length, or
    other soft issues a human may want to review. A valid short caption
    for a simple scene passes — length alone is not a failure.
    """
    warnings: list[str] = []
    if not caption or not caption.strip():
        return False, "empty caption", warnings

    clean = " ".join(caption.strip().split())
    words = clean.split()

    if len(words) < _MIN_WORDS_HARD:
        return False, f"too short ({len(words)} words)", warnings
    if _FPS_PATTERN.search(clean):
        return False, "contains fps metadata", warnings
    if _RES_PATTERN.search(clean):
        return False, "contains resolution metadata", warnings
    if _EXT_PATTERN.search(clean) or _FILENAME_PREFIX_PATTERN.match(clean):
        return False, "looks like file metadata or bare filename", warnings
    if len(words) == 1 and ("." in clean or "_" in clean or "-" in clean):
        return False, "looks like a bare filename", warnings
    if clean.startswith(("{", "[", "```")) or clean.endswith("```"):
        return False, "wrapped in JSON or markdown", warnings
    if _MARKDOWN_PATTERN.search(clean):
        return False, "contains markdown formatting", warnings
    if _INTRO_PATTERNS.search(clean):
        return False, "starts with an introductory phrase", warnings
    if _ERROR_PATTERNS.search(clean.lower()):
        return False, "looks like a model error message", warnings
    sentences = [s.strip().lower() for s in re.split(r"[.!?]+", clean) if s.strip()]
    if len(sentences) != len(set(sentences)):
        return False, "repeats a sentence", warnings
    if "\n" in caption.strip():
        return False, "multiple paragraphs", warnings

    lower = clean.lower()
    if any(p in lower for p in _GENERIC_PATTERNS):
        warnings.append("generic wording")
    if len(words) > _MAX_WORDS_WARN:
        warnings.append(f"unusually long ({len(words)} words)")
    if len(words) < 8:
        warnings.append(f"short caption ({len(words)} words)")
    return True, "", warnings


def validate_caption_quality(caption: str) -> tuple[bool, str]:
    """Backward-compatible hard gate: (is_valid, rejection_reason)."""
    is_valid, reason, _warnings = validate_caption_detailed(caption)
    return is_valid, reason


# ------------------------------------------------------------------- Gemini API
def resolve_api_keys() -> list[str]:
    """Obtain Gemini API keys from GEMINI_API_KEY and GEMINI_API_KEY_2 (or getpass prompt)."""
    key1 = os.environ.get("GEMINI_API_KEY", "").strip()
    key2 = os.environ.get("GEMINI_API_KEY_2", "").strip()

    if not key1 and not key2:
        try:
            key1 = getpass.getpass("Enter GEMINI_API_KEY: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nAPI key prompt cancelled.")
        if not key1:
            sys.exit("error: GEMINI_API_KEY is not set.")

    keys = [k for k in [key1, key2] if k]
    return keys


def resolve_api_key() -> str:
    """Obtain primary Gemini API key (backwards compatibility)."""
    keys = resolve_api_keys()
    return keys[0]


def generate_clip_caption(
    client: any,
    model: str = DEFAULT_MODEL,
    jpeg_frames: list[bytes] | None = None,
    max_retries: int = 3,
    fallback_chain: tuple[str, ...] = FALLBACK_MODELS,
) -> str:
    """Send 3 JPEG frames to Gemini.
    
    - Fast-fails on 429 quota with QuotaExceededError.
    - On 503/overload, retries with backoff (10s, 25s, 50s), then tries fallback models in chain:
      gemini-3.5-flash, gemini-3.6-flash, gemini-3.7-flash, gemini-flash-latest.
    - Retries transient network errors up to max_retries.
    """
    from google.genai import types

    if jpeg_frames is None:
        jpeg_frames = []

    contents = [
        types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in jpeg_frames
    ]
    contents.append(types.Part.from_text(text=CAPTION_INSTRUCTION))

    # Construct ordered sequence of models to try
    models_to_try = [model]
    for fm in fallback_chain:
        if fm not in models_to_try:
            models_to_try.append(fm)

    last_error: Exception | None = None

    for m_idx, current_model in enumerate(models_to_try):
        max_attempts = len(OVERLOAD_BACKOFF_DELAYS)
        for attempt in range(max_attempts):
            try:
                response = client.models.generate_content(
                    model=current_model,
                    contents=contents,
                )
                raw = (getattr(response, "text", "") or "").strip()
                # Strip outer quotes if returned
                if (raw.startswith('"') and raw.endswith('"')) or (
                    raw.startswith("'") and raw.endswith("'")
                ):
                    raw = raw[1:-1].strip()
                caption = " ".join(raw.split())
                if not caption:
                    raise ValueError("Model returned empty text response")
                return caption
            except Exception as err:
                last_error = err
                # Immediately raise QuotaExceededError on 429 without retrying or model fallback
                if is_quota_error(err):
                    raise QuotaExceededError(str(err)) from err

                if is_overload_error(err):
                    if attempt < len(OVERLOAD_BACKOFF_DELAYS) - 1:
                        delay = OVERLOAD_BACKOFF_DELAYS[attempt]
                        time.sleep(delay)
                        continue
                    else:
                        # Overloaded after retries for current_model; fall through to next fallback model
                        if m_idx < len(models_to_try) - 1:
                            next_model = models_to_try[m_idx + 1]
                            print(
                                f"Model {current_model} overloaded (503). Falling back to {next_model}...",
                                flush=True,
                            )
                        break

                # Transient non-overload error
                if attempt < max_retries - 1:
                    delay = (
                        TRANSIENT_BACKOFF_DELAYS[attempt]
                        if attempt < len(TRANSIENT_BACKOFF_DELAYS)
                        else 4.0
                    )
                    time.sleep(delay)
                else:
                    # If this was not 503, don't try other models
                    raise RuntimeError(
                        f"API call failed on {current_model} after {attempt + 1} attempts: {err}"
                    ) from err

    raise RuntimeError(
        f"API call failed across all models ({models_to_try}): {last_error}"
    ) from last_error


# ------------------------------------------------------------- Local Backend (BLIP)
_LOCAL_CAPTIONER = None


LOCAL_MODEL_ENV = "VIDGIN_LOCAL_CAPTION_MODEL"


def resolve_local_model_source(model_dir: str | None = None) -> str:
    """Where local weights come from: explicit arg > env var > HF model id."""
    if model_dir:
        return model_dir
    env = os.environ.get(LOCAL_MODEL_ENV, "").strip()
    if env:
        return env
    return LOCAL_MODEL_NAME


def get_local_captioner(device: str | None = None,
                        model_dir: str | None = None):
    """Lazily load the local BLIP captioner (processor + model + device).

    ``model_dir`` (or the VIDGIN_LOCAL_CAPTION_MODEL env var) may point at a
    local folder holding the weights; a folder is loaded with
    local_files_only=True so no network access is attempted. Without one,
    the HuggingFace model id is used and weights download on first use.
    The additional dependencies for this backend are transformers + torch:
    ``pip install transformers torch``.
    """
    global _LOCAL_CAPTIONER
    if _LOCAL_CAPTIONER is not None:
        return _LOCAL_CAPTIONER

    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
    except ImportError as e:
        raise ImportError(
            "Local captioning requires 'transformers' and 'torch'. "
            "Install with: pip install transformers torch"
        ) from e

    import torch

    if device in (None, "auto"):
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested but CUDA is not available in this "
            "PyTorch build / machine; use --device cpu or auto."
        )
    else:
        target_device = device

    source = resolve_local_model_source(model_dir)
    if os.path.isdir(source):
        processor = BlipProcessor.from_pretrained(source, local_files_only=True)
        model = BlipForConditionalGeneration.from_pretrained(
            source, local_files_only=True)
    else:
        processor = BlipProcessor.from_pretrained(source)
        model = BlipForConditionalGeneration.from_pretrained(source)
    model.to(target_device)
    model.eval()

    _LOCAL_CAPTIONER = (processor, model, target_device)
    return _LOCAL_CAPTIONER


def postprocess_local_caption(raw: any, frames: list[np.ndarray] | None = None) -> str:
    """Clean a raw local-model caption WITHOUT adding information.

    The local model is a single-frame image captioner: it cannot observe
    camera movement, lighting changes, or motion. This function therefore
    only normalizes the model's own words — strips boilerplate lead-ins,
    collapses whitespace, capitalizes, and ends with one period. It must
    NEVER append camera movement, lighting, weather, setting, or any other
    detail the model did not produce (an earlier version did exactly that,
    which fabricated training data; see the regression tests).
    """
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raw = str(raw)
    if not raw.strip():
        return ""

    # Clean leading boilerplate
    clean = raw.strip().strip('"\'')
    clean = re.sub(
        r"^(a photo of|an image of|a video of|there is|a view of|a picture of)\s+",
        "",
        clean,
        flags=re.IGNORECASE,
    ).strip()

    if not clean:
        return ""

    # Standardize whitespace, capitalize, single trailing period.
    clean = " ".join(clean.split())
    clean = clean[0].upper() + clean[1:]
    clean = clean.rstrip(".!?,; ") + "."
    return clean


def generate_clip_caption_local(
    three_frames: list[np.ndarray],
    local_captioner: any = None,
    device: str | None = None,
) -> str:
    """Generate caption using local vision model (BLIP) fed 3 frames across the clip."""
    if local_captioner is None:
        processor, model, dev = get_local_captioner(device=device, model_dir=local_model_dir)
    elif isinstance(local_captioner, tuple) and len(local_captioner) == 3:
        processor, model, dev = local_captioner
    elif callable(local_captioner):
        raw = local_captioner(three_frames)
        raw_text = getattr(raw, "text", raw)
        return postprocess_local_caption(raw_text, three_frames)
    elif hasattr(local_captioner, "generate_content"):
        # Mock client compatibility
        resp = local_captioner.generate_content(three_frames)
        raw_text = getattr(resp, "text", str(resp))
        return postprocess_local_caption(raw_text, three_frames)
    else:
        # Generic object with generate/decode or similar
        processor = getattr(local_captioner, "processor", None)
        model = getattr(local_captioner, "model", local_captioner)
        dev = getattr(local_captioner, "device", "cpu")

    import torch

    pil_frames = [
        Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in three_frames
    ]

    # Feed frames to processor and generate
    inputs = processor(images=pil_frames, return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(dev)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=45)

    raw_captions = [
        processor.decode(out, skip_special_tokens=True).strip() for out in outputs
    ]

    # Select representative frame caption (middle frame anchor)
    mid_idx = len(raw_captions) // 2
    raw = raw_captions[mid_idx] if raw_captions else ""
    if not raw and raw_captions:
        raw = max(raw_captions, key=len)

    return postprocess_local_caption(raw, three_frames)


# ------------------------------------------------------------- Main processing
def caption_dataset(
    data_dir: str,
    limit: int | None = None,
    max_calls: int | None = DEFAULT_MAX_CALLS,
    model: str = DEFAULT_MODEL,
    overwrite: bool = False,
    pace_seconds: float = DEFAULT_PACE_SECONDS,
    client: any = None,
    clients: list[any] | None = None,
    backend: str = "gemini",
    local_captioner: any = None,
    device: str | None = None,
    dry_run: bool = False,
    max_frames: int = 3,
    local_model_dir: str | None = None,
    report_path: str | None = None,
    review_export_path: str | None = None,
    verbose: bool = False,
) -> tuple[int, int, int]:
    """Process clips in data_dir, updating captions with Gemini vision or local BLIP.

    Supports:
    - --backend gemini: Gemini API with two-key rotation on 429, stopping cleanly on quota exhaustion.
    - --backend local: Local vision captioner (Salesforce/blip-image-captioning-base).
    - --backend auto: Gemini first, automatically continuing on local backend when all keys hit quota.
    - Resume safety: skips entries with caption_source in ("gemini", "local").
    - Layered validation: hard failures reject (retry once, then log failure);
      warnings are collected into the run report / review export.
    - Atomic manifest saves flushed every FLUSH_INTERVAL (5) captions.
    - dry_run: generate and validate captions but write nothing (no manifest
      saves, no failure log); counts describe what WOULD change.
    - Final SUMMARY line printing per-backend counts.

    Returns (processed_count, skipped_count, failed_count).
    """
    manifest_path, entries = load_manifest(data_dir)
    total_entries = len(entries)
    if total_entries == 0:
        print(f"Manifest in {data_dir} is empty.", flush=True)
        return 0, 0, 0

    active_backend = backend.lower() if backend else "gemini"
    if active_backend not in ("gemini", "local", "auto"):
        raise ValueError(f"Unknown backend: {backend}. Choose 'gemini', 'local', or 'auto'.")

    # Setup clients for Gemini
    gemini_clients: list[any] = []
    current_key_idx = 0

    if active_backend in ("gemini", "auto"):
        if clients is not None and len(clients) > 0:
            gemini_clients = list(clients)
        elif client is not None:
            gemini_clients = [client]
        else:
            from google import genai

            keys = resolve_api_keys()
            gemini_clients = [genai.Client(api_key=k) for k in keys]

    # Pre-load local captioner if local backend is selected
    if active_backend == "local" and local_captioner is None:
        try:
            local_captioner = get_local_captioner(device=device, model_dir=local_model_dir)
        except Exception as e:
            print(f"Failed to initialize local captioner: {e}", flush=True)
            raise

    processed_count = 0
    skipped_count = 0
    failed_count = 0
    dirty_since_flush = 0

    processed_gemini = 0
    skipped_gemini = 0
    failed_gemini = 0

    processed_local = 0
    skipped_local = 0
    failed_local = 0

    gemini_calls_made = 0

    # Run-report state: warnings, duplicates, review items, caption lengths.
    report_warnings: list[dict] = []
    review_items: list[dict] = []
    seen_captions: dict[str, str] = {}
    caption_lengths: list[int] = []

    def finish() -> tuple[int, int, int]:
        """Shared exit path: summary print + optional report/review files."""
        _print_summary(
            processed_count, skipped_count, failed_count,
            processed_gemini, skipped_gemini, failed_gemini,
            processed_local, skipped_local, failed_local,
        )
        if dry_run:
            print("DRY RUN: no files were written.", flush=True)
        if report_path:
            report = {
                "data_dir": data_dir,
                "backend_requested": backend,
                "backend_final": active_backend,
                "dry_run": dry_run,
                "processed": processed_count,
                "skipped": skipped_count,
                "failed": failed_count,
                "by_backend": {
                    "gemini": {"processed": processed_gemini,
                               "skipped": skipped_gemini,
                               "failed": failed_gemini},
                    "local": {"processed": processed_local,
                              "skipped": skipped_local,
                              "failed": failed_local},
                },
                "gemini_calls_made": gemini_calls_made,
                "warnings": report_warnings,
                "review_count": len(review_items),
                "caption_word_lengths": caption_lengths,
            }
            _write_json_atomic(report_path, report)
        if review_export_path:
            _write_jsonl_atomic(review_export_path, review_items)
        return processed_count, skipped_count, failed_count

    target_limit = (
        min(total_entries, int(limit)) if limit is not None else total_entries
    )

    for i, entry in enumerate(entries, start=1):
        if limit is not None and processed_count >= target_limit:
            break

        # Resume safety check: skip both gemini and local unless overwrite is set
        existing_source = entry.get("caption_source")
        if not overwrite and existing_source in ("gemini", "local"):
            skipped_count += 1
            if existing_source == "gemini":
                skipped_gemini += 1
            else:
                skipped_local += 1
            continue

        clip_file = entry["file"]
        clip_path = os.path.join(data_dir, clip_file)
        if not os.path.isfile(clip_path):
            print(f"[{i}/{total_entries}] file missing: {clip_file}, skipping", flush=True)
            failed_count += 1
            if active_backend in ("gemini", "auto"):
                failed_gemini += 1
            else:
                failed_local += 1
            if not dry_run:
                log_failure(data_dir, clip_file)
            review_items.append({'file': clip_file, 'caption': entry.get('caption', ''), 'source': 'failed', 'issues': ['captioning failed']})
            continue

        # Load clip frames
        try:
            clip = np.load(clip_path)
            three_frames = sample_frames(clip, max_frames)
        except Exception as e:
            failed_count += 1
            if active_backend in ("gemini", "auto"):
                failed_gemini += 1
            else:
                failed_local += 1
            if not dry_run:
                log_failure(data_dir, clip_file)
            review_items.append({'file': clip_file, 'caption': entry.get('caption', ''), 'source': 'failed', 'issues': ['captioning failed']})
            print(f"[{i}/{total_entries}] error reading {clip_file}: {e}", flush=True)
            continue

        # Generate caption with quality gate (retry once)
        new_caption: str | None = None
        current_source: str = ""
        pending_warnings: list[str] = []

        quality_attempts = 2
        clip_failed = False

        for q_attempt in range(quality_attempts):
            # Check max-calls limit before attempting Gemini call
            if active_backend in ("gemini", "auto") and max_calls is not None and gemini_calls_made >= max_calls:
                if active_backend == "auto":
                    print(
                        f"Reached --max-calls limit ({max_calls}) for Gemini API. "
                        "Switching to local backend for remaining clips.",
                        flush=True,
                    )
                    active_backend = "local"
                    if local_captioner is None:
                        try:
                            local_captioner = get_local_captioner(device=device, model_dir=local_model_dir)
                        except Exception as err:
                            print(
                                f"Local backend unavailable ({err}). Saving progress and stopping.",
                                flush=True,
                            )
                            if not dry_run:
                                save_manifest_atomic(manifest_path, entries)
                            return finish()
                else:
                    print(
                        f"Reached --max-calls limit ({max_calls}) for this session. "
                        "Saving progress and stopping cleanly.",
                        flush=True,
                    )
                    if not dry_run:
                        save_manifest_atomic(manifest_path, entries)
                    return finish()

            try:
                if active_backend in ("gemini", "auto"):
                    jpeg_frames = [frame_to_jpeg_bytes(f) for f in three_frames]
                    active_client = (
                        gemini_clients[current_key_idx] if gemini_clients else None
                    )
                    candidate = generate_clip_caption(
                        active_client,
                        model=model,
                        jpeg_frames=jpeg_frames,
                    )
                    gemini_calls_made += 1
                    current_source = "gemini"
                else:
                    candidate = generate_clip_caption_local(
                        three_frames,
                        local_captioner=local_captioner,
                        device=device,
                    )
                    current_source = "local"

            except QuotaExceededError:
                # HTTP 429 quota reached on current Gemini key
                if current_key_idx + 1 < len(gemini_clients):
                    current_key_idx += 1
                    print(
                        f"Key {current_key_idx} quota reached (HTTP 429). Rotating to key {current_key_idx + 1}...",
                        flush=True,
                    )
                    # Retry current clip with new key
                    continue
                else:
                    # All Gemini keys exhausted
                    if active_backend == "auto":
                        print(
                            "All Gemini API keys exhausted (HTTP 429). "
                            "Automatically switching to local backend...",
                            flush=True,
                        )
                        active_backend = "local"
                        if local_captioner is None:
                            try:
                                local_captioner = get_local_captioner(device=device, model_dir=local_model_dir)
                            except Exception as err:
                                print(
                                    f"Local backend unavailable ({err}). Saving progress and stopping cleanly.",
                                    flush=True,
                                )
                                if not dry_run:
                                    save_manifest_atomic(manifest_path, entries)
                                return finish()
                        # Retry current clip on local backend
                        continue
                    else:
                        # gemini backend only: save progress and exit cleanly
                        if not dry_run:
                            save_manifest_atomic(manifest_path, entries)
                        print(
                            "Free quota reached — run again tomorrow; it will continue where it stopped",
                            flush=True,
                        )
                        return finish()

            except Exception as e:
                failed_count += 1
                if active_backend in ("gemini", "auto"):
                    failed_gemini += 1
                else:
                    failed_local += 1
                if not dry_run:
                    log_failure(data_dir, clip_file)
                review_items.append({'file': clip_file, 'caption': entry.get('caption', ''), 'source': 'failed', 'issues': ['captioning failed']})
                print(f"[{i}/{total_entries}] error on {clip_file}: {e}", flush=True)
                clip_failed = True
                break

            # Validate against the layered quality gate
            is_valid, reason, cand_warnings = validate_caption_detailed(candidate)
            if is_valid:
                new_caption = candidate
                pending_warnings = cand_warnings
                break
            else:
                print(
                    f"[{i}/{total_entries}] Quality gate rejected caption ({reason}): '{candidate}'",
                    flush=True,
                )
                if q_attempt == 0:
                    print(f"[{i}/{total_entries}] Retrying caption for {clip_file}...", flush=True)
                    continue
                else:
                    print(
                        f"[{i}/{total_entries}] Quality gate failed twice for {clip_file}. Marking failed.",
                        flush=True,
                    )
                    failed_count += 1
                    if current_source == "gemini" or active_backend in ("gemini", "auto"):
                        failed_gemini += 1
                    else:
                        failed_local += 1
                    if not dry_run:
                        log_failure(data_dir, clip_file)
                    review_items.append({'file': clip_file, 'caption': entry.get('caption', ''), 'source': 'failed', 'issues': ['captioning failed']})
                    clip_failed = True
                    break

        if clip_failed or new_caption is None:
            continue

        # Dataset-level checks: duplicate captions across clips are a
        # warning for human review, never a silent rejection or edit.
        clip_warnings = list(pending_warnings)
        normalized = " ".join(new_caption.lower().split())
        first_seen = seen_captions.get(normalized)
        if first_seen is not None and first_seen != clip_file:
            clip_warnings.append(f"duplicate caption of {first_seen}")
        else:
            seen_captions[normalized] = clip_file
        caption_lengths.append(len(new_caption.split()))
        if clip_warnings:
            review_items.append({
                "file": clip_file,
                "caption": new_caption,
                "source": current_source,
                "issues": clip_warnings,
            })
            for w in clip_warnings:
                report_warnings.append({"file": clip_file, "warning": w})

        if not dry_run:
            # Preserve previous caption into caption_weak
            if "caption_weak" not in entry:
                entry["caption_weak"] = entry.get("caption", "")
            entry["caption"] = new_caption
            entry["caption_source"] = current_source

        processed_count += 1
        dirty_since_flush += 1

        if current_source == "gemini":
            processed_gemini += 1
        else:
            processed_local += 1

        # Print progress and captions for first 10 newly captioned clips
        if verbose or processed_count <= 10:
            print(f"[{i}/{total_entries}] [{current_source}] {new_caption}", flush=True)
        else:
            print(f"[{i}/{total_entries}] [{current_source}]", flush=True)

        # Flush manifest atomically every FLUSH_INTERVAL processed clips
        if dirty_since_flush >= FLUSH_INTERVAL:
            if not dry_run:
                save_manifest_atomic(manifest_path, entries)
            dirty_since_flush = 0

        # Check if max-calls was reached
        if active_backend in ("gemini", "auto") and max_calls is not None and gemini_calls_made >= max_calls:
            if active_backend == "auto":
                print(
                    f"Reached --max-calls limit ({max_calls}) for Gemini API. "
                    "Switching to local backend for remaining clips.",
                    flush=True,
                )
                active_backend = "local"
                if local_captioner is None:
                    try:
                        local_captioner = get_local_captioner(device=device, model_dir=local_model_dir)
                    except Exception as err:
                        print(
                            f"Local backend unavailable ({err}). Saving progress and stopping.",
                            flush=True,
                        )
                        break
            else:
                print(
                    f"Reached --max-calls limit ({max_calls}) for this session. "
                    "Saving progress and stopping cleanly.",
                    flush=True,
                )
                break

        # Free tier pacing for Gemini requests
        if current_source == "gemini" and pace_seconds > 0:
            time.sleep(pace_seconds)

    # Final atomic flush if any changes were made
    if dirty_since_flush > 0:
        if not dry_run:
            save_manifest_atomic(manifest_path, entries)

    return finish()


def _print_summary(
    processed_count: int,
    skipped_count: int,
    failed_count: int,
    processed_gemini: int,
    skipped_gemini: int,
    failed_gemini: int,
    processed_local: int,
    skipped_local: int,
    failed_local: int,
) -> None:
    """Print standard summary block and single-line summary with per-backend counts."""
    print("=" * 50, flush=True)
    print(
        f"Captioned: {processed_count} (gemini: {processed_gemini}, local: {processed_local})",
        flush=True,
    )
    print(f"Skipped:   {skipped_count} (already captioned)", flush=True)
    print(f"Failed:    {failed_count} (see {FAILURES_LOG} if > 0)", flush=True)
    print(
        f"SUMMARY: gemini: {processed_gemini} processed, {skipped_gemini} skipped, {failed_gemini} failed | "
        f"local: {processed_local} processed, {skipped_local} skipped, {failed_local} failed",
        flush=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace weak captions in manifest.jsonl with Gemini vision or local BLIP captions."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to prepared clips directory containing .npy files and manifest.jsonl",
    )
    parser.add_argument(
        "--backend",
        choices=["gemini", "local", "auto"],
        default="gemini",
        help="Captioning backend: 'gemini' (Free Tier API), 'local' (BLIP vision model), or 'auto' (Gemini first, fallback to local on quota)",
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
        help=f"Maximum Gemini API calls before cleanly stopping for the day (default: {DEFAULT_MAX_CALLS})",
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
        help="Re-caption clips already marked with caption_source: gemini or local",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=DEFAULT_PACE_SECONDS,
        help=f"Minimum seconds between Gemini API requests (default: {DEFAULT_PACE_SECONDS}s for free tier)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Generate and validate captions but write nothing (no manifest or log changes)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for the local backend (default: auto)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=3,
        help="Maximum frames sampled per clip (default: 3 = first/middle/last)",
    )
    parser.add_argument(
        "--local-model-dir",
        default=None,
        help="Local folder with BLIP weights (loaded offline); "
             f"overrides the {LOCAL_MODEL_ENV} env var and the HuggingFace download",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Write a JSON run report (counts, warnings, duplicates) to this path",
    )
    parser.add_argument(
        "--review-export",
        default=None,
        help="Write a JSONL export of clips needing human review to this path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print every generated caption, not just the first 10",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    print(
        f"Captioning clips in {args.data_dir} using backend '{args.backend}' "
        f"(model: {args.model if args.backend != 'local' else LOCAL_MODEL_NAME}) ...",
        flush=True,
    )
    caption_dataset(
        data_dir=args.data_dir,
        backend=args.backend,
        limit=args.limit,
        max_calls=args.max_calls,
        model=args.model,
        overwrite=args.overwrite,
        pace_seconds=args.pace,
        dry_run=args.dry_run,
        device=None if args.device == "auto" else args.device,
        max_frames=args.max_frames,
        local_model_dir=args.local_model_dir,
        report_path=args.report,
        review_export_path=args.review_export,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
