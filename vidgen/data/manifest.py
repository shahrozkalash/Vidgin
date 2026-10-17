"""Parquet Manifest Management for VidGen Video & Image Curation.

Provides typed schema, atomic write, reading, and stage-by-stage resumption.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd


MANIFEST_COLUMNS = [
    "id",
    "media_type",        # "video" or "image"
    "path",
    "source",            # "moving_mnist", "openvid", "panda70m", "stock_pexels", "synthetic"
    "license",           # "MIT", "Apache-2.0", "CC-BY", "Pexels-Terms"
    "duration",          # float seconds (0.0 for images)
    "fps",               # float
    "frames",            # int frame count (1 for images)
    "height",            # int
    "width",             # int
    "crop_box",          # string "[top, left, bottom, right]" or empty
    "sharpness_score",   # float (Laplacian variance)
    "motion_score",      # float (mean frame difference)
    "aesthetic_score",   # float (estimated visual appeal)
    "text_overlay_ratio",# float (fraction of frame with detected text/logo)
    "is_safe",           # bool
    "phash",             # string perceptual hash
    "filter_status",     # "passed" or "rejected:<reason>"
    "caption_short",     # short text prompt
    "caption_long",      # dense descriptive text prompt
    "caption_json",      # detailed structured JSON
]


def create_empty_manifest() -> pd.DataFrame:
    """Create an empty DataFrame with standard manifest schema."""
    dtypes = {
        "id": "string",
        "media_type": "string",
        "path": "string",
        "source": "string",
        "license": "string",
        "duration": "float64",
        "fps": "float64",
        "frames": "int64",
        "height": "int64",
        "width": "int64",
        "crop_box": "string",
        "sharpness_score": "float64",
        "motion_score": "float64",
        "aesthetic_score": "float64",
        "text_overlay_ratio": "float64",
        "is_safe": "bool",
        "phash": "string",
        "filter_status": "string",
        "caption_short": "string",
        "caption_long": "string",
        "caption_json": "string",
    }
    df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtypes.items()})
    return df


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read a Parquet manifest file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return pd.read_parquet(path)


def write_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    """Atomically write a DataFrame to a Parquet file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp.parquet")
    df.to_parquet(temp_path, index=False, engine="pyarrow")
    os.replace(temp_path, path)
    return path


def resume_or_init_manifest(path: str | Path) -> pd.DataFrame:
    """Load existing manifest if present for resumption, else create empty."""
    path = Path(path)
    if path.exists():
        return read_manifest(path)
    return create_empty_manifest()
