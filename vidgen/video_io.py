"""Playable MP4 export for generated frames.

Phase 7 of the targeted improvement programme (2026-10-22). Uses
OpenCV's VideoWriter (mp4v) when available and falls back to the
system ``ffmpeg`` binary; either way the file is re-opened and its
frame count verified before success is reported — an unplayable or
truncated export is an error, never a silent artifact.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np


def frames_to_uint8(frames: np.ndarray) -> np.ndarray:
    """Float frames [T, H, W, 3] in [-1, 1] -> uint8 [T, H, W, 3]."""
    arr = np.asarray(frames)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected frames [T,H,W,3], got shape {arr.shape}")
    return ((arr * 0.5 + 0.5).clip(0, 1) * 255).round().astype(np.uint8)


def probe_mp4(path: str) -> tuple[int, int, int]:
    """(frame_count, width, height) of an MP4, read with OpenCV or ffprobe."""
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            count = 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            while True:
                ok, _frame = cap.read()
                if not ok:
                    break
                count += 1
            cap.release()
            if count > 0:
                return count, width, height
    except Exception:
        pass

    # ffprobe fallback
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=width,height,nb_read_packets",
        "-of", "csv=p=0", path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0 and proc.stdout.strip():
        parts = [p.strip() for p in proc.stdout.strip().split(",") if p.strip()]
        if len(parts) >= 3:
            try:
                width = int(parts[0])
                height = int(parts[1])
                count = int(parts[2])
                return count, width, height
            except ValueError:
                pass
    raise ValueError(f"cannot open or probe video file: {path}")


def _write_mp4_cv2(uint8: np.ndarray, path: str, fps: float) -> None:
    import cv2

    t, h, w, _ = uint8.shape
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open (mp4v codec)")
    try:
        for frame in uint8:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _write_mp4_ffmpeg(uint8: np.ndarray, path: str, fps: float) -> None:
    t, h, w, _ = uint8.shape
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(float(fps)),
        "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        path,
    ]
    proc = subprocess.run(cmd, input=uint8.tobytes(),
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg MP4 export failed: {proc.stderr.decode(errors='replace')[:300]}")


def write_mp4(frames: np.ndarray, path: str, fps: float = 16.0) -> str:
    """Write float frames [T,H,W,3] in [-1,1] to a verified MP4.

    Returns the path. Raises if no writer works or the written file
    does not contain exactly the input's frame count.
    """
    uint8 = frames_to_uint8(frames)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    errors = []
    for writer in (_write_mp4_cv2, _write_mp4_ffmpeg):
        try:
            writer(uint8, path, fps)
            break
        except Exception as exc:  # try the next backend
            errors.append(f"{writer.__name__}: {exc}")
    else:
        raise RuntimeError("MP4 export failed: " + "; ".join(errors))
    count, _w, _h = probe_mp4(path)
    if count != uint8.shape[0]:
        raise RuntimeError(
            f"MP4 verification failed: wrote {uint8.shape[0]} frames, "
            f"read back {count}")
    return path
