#!/usr/bin/env python3
"""Turn stock videos into a FolderClipDataset directory.

Reads videos from Pexels (free API key) or from a list of direct video URLs,
then for each video: download -> decode -> center 17-frame temporal window ->
center-crop to square -> resize to 64x64 -> quality/dedupe filters ->
<uint8 (T,H,W,3) .npy> + one line in manifest.jsonl with a caption.

Usage (Pixabay, "City and people" theme — recommended: free key issued instantly):
  export PIXABAY_API_KEY=...   # free key shown right after signup at https://pixabay.com/api/docs/
  python scripts/prepare_clips.py --source pixabay \\
      --queries "people walking city street,crowd crossing street,city traffic aerial" \\
      --per-query 500 --out /content/drive/MyDrive/vidgin-data/tiny-clips

Usage (Pexels; only if you already have a key — Pexels has paused issuing new ones):
  export PEXELS_API_KEY=...   # free key from https://www.pexels.com/api/
  python scripts/prepare_clips.py --source pexels \\
      --queries "people walking city street,crowd crossing street,city traffic aerial" \\
      --per-query 500 --out /content/drive/MyDrive/vidgin-data/tiny-clips

Usage (no API key — direct video file URLs, one per line):
  python scripts/prepare_clips.py --source urls --urls-file urls.txt \\
      --out ./my-clips

Downloads live in a temp dir that is cleaned after every video, so disk use
stays small (only the finished .npy clips + manifest accumulate in --out).
The API key is read from the PIXABAY_API_KEY (or PEXELS_API_KEY) environment
variable only: it is never hardcoded and never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

# Make `vidgen` importable no matter where the script is run from
# (Colab runs it as `python scripts/prepare_clips.py` after chdir'ing).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidgen.data.filters import passes_quality

PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"
PEXELS_PER_PAGE = 80  # API maximum
PIXABAY_SEARCH_URL = "https://pixabay.com/api/videos/"
PIXABAY_PER_PAGE = 200  # API maximum (totalHits is capped at 500 per query)
PIXABAY_REQUEST_SLEEP = 0.35  # keep well under the 100 requests/60s rate limit
TIMEOUT = 60
MIN_WIDTH = 480  # prefer the smallest video file at least this wide
MEAN_BRIGHTNESS_MIN = 25.0  # 0-255 scale; below this the clip is "too dark"


# ------------------------------------------------------------- pure helpers
def caption_from_filename(url: str) -> str:
    """Turn a video URL into a simple human caption from its file name."""
    path = urllib.parse.urlparse(url).path
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    name = re.sub(r"[_\-\.]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "video clip"


def content_hash(clip: np.ndarray) -> str:
    """SHA-256 of the raw bytes; used to drop exact-duplicate clips."""
    return hashlib.sha256(clip.tobytes()).hexdigest()


def select_frame_indices(n_total: int, frames: int) -> list[int] | None:
    """Center temporal window of `frames` frames at an even stride.

    Returns None when the video is too short.
    """
    if n_total < frames:
        return None
    stride = max(1, (n_total - 1) // max(frames - 1, 1))
    span = (frames - 1) * stride
    if span >= n_total:  # stride too big for a centered window
        stride = 1
        span = frames - 1
    start = (n_total - 1 - span) // 2
    return [start + t * stride for t in range(frames)]


def center_crop_square(frame: np.ndarray) -> np.ndarray:
    """frame (H, W, 3) -> (S, S, 3) center crop, S = min(H, W)."""
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return frame[y0 : y0 + s, x0 : x0 + s]


def resize_nearest(frame: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbor resize (S, S, 3) -> (size, size, 3). Pure numpy."""
    s = frame.shape[0]
    ys = (np.linspace(0, s, size, endpoint=False)).astype(int)
    xs = (np.linspace(0, s, size, endpoint=False)).astype(int)
    return frame[ys[:, None], xs[None, :]]


def choose_video_file(video_files: list[dict]) -> dict | None:
    """Pick the cheapest file to download: smallest width >= MIN_WIDTH,
    else the smallest available. Prefers mp4."""
    mp4s = [f for f in video_files if f.get("file_type") == "video/mp4"] or video_files
    if not mp4s:
        return None
    big_enough = [f for f in mp4s if (f.get("width") or 0) >= MIN_WIDTH]
    pool = big_enough or mp4s
    return min(pool, key=lambda f: (f.get("width") or 10**9, f.get("height") or 10**9))


def pexels_caption(video: dict, query: str) -> str:
    """Pexels video objects carry no titles; the search query is the caption."""
    return query.strip() or "stock video clip"


# ------------------------------------------------------------------ decode
def decode_video(path: str, want_frames: int | None = None) -> np.ndarray:
    """Decode a video file to uint8 (N, H, W, 3). Tries cv2, then imageio.

    Memory-bounded: when want_frames is given and the total frame count is
    known, only the frames build_clip will actually sample are kept during
    the single decode pass. The previous accumulate-everything version put
    whole videos in RAM and was OOM-killed (exit 137) by long/high-res
    sources at random positions in a URL list.
    """
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        keep_idx: set[int] | None = None
        if want_frames and total > want_frames:
            idx = select_frame_indices(total, want_frames)
            if idx is not None:
                keep_idx = {int(i) for i in idx}
        frames = []
        n = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if keep_idx is None or n in keep_idx:
                frames.append(bgr[:, :, ::-1])  # BGR -> RGB
            n += 1
        cap.release()
        if not frames:
            raise ValueError("no frames decoded")
        return np.stack(frames).astype(np.uint8)
    except ImportError:
        pass
    try:
        import imageio.v2 as imageio  # type: ignore

        frames = [f for f in imageio.get_reader(path)]
        if not frames:
            raise ValueError("no frames decoded")
        return np.stack([np.asarray(f, dtype=np.uint8) for f in frames])
    except ImportError:
        pass
    raise RuntimeError(
        "No video decoder available. Install one of:\n"
        "  pip install opencv-python-headless   (recommended)\n"
        "  pip install imageio imageio-ffmpeg"
    )


# ------------------------------------------------------------- per-video io
def download(url: str, dest: str) -> None:
    """Download a URL to dest with a timeout; one retry on failure."""
    last = None
    for attempt in range(2):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "vidgin-prepare/1.0"}
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(
                dest, "wb"
            ) as f:
                shutil.copyfileobj(r, f, length=1024 * 256)
            return
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"download failed: {url}: {last}")


def pexels_search(api_key: str, query: str, per_page: int, page: int) -> list[dict]:
    """One page of Pexels video search results."""
    params = urllib.parse.urlencode(
        {"query": query, "per_page": per_page, "page": page}
    )
    req = urllib.request.Request(
        f"{PEXELS_SEARCH_URL}?{params}",
        headers={"Authorization": api_key, "User-Agent": "vidgin-prepare/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:200]
        raise RuntimeError(f"Pexels API error {e.code}: {body}") from e
    return data.get("videos", [])


def collect_pexels(api_key: str, queries: list[str], per_query: int) -> list[tuple[str, str]]:
    """Return [(download_url, caption)] across all queries, paginating."""
    items: list[tuple[str, str]] = []
    for query in queries:
        got = 0
        page = 1
        while got < per_query:
            want = min(PEXELS_PER_PAGE, per_query - got)
            videos = pexels_search(api_key, query, want, page)
            if not videos:
                break
            for v in videos:
                f = choose_video_file(v.get("video_files", []))
                if f and f.get("link"):
                    items.append((f["link"], pexels_caption(v, query)))
                    got += 1
                    if got >= per_query:
                        break
            page += 1
            if len(videos) < want:
                break
        print(f"  query {query!r}: {got} videos found", flush=True)
    return items


def choose_pixabay_video_file(videos: dict) -> dict | None:
    """Pick a cheap variant to download: prefer medium, else large, else small.

    We downscale to 64x64 anyway, so the smallest variant that is at least
    256px wide wins; if none is, take the first available variant in
    preference order. Mirrors choose_video_file.
    """
    for name in ("medium", "large", "small"):
        v = videos.get(name)
        if v and v.get("url") and (v.get("width") or 0) >= 256:
            return v
    for name in ("medium", "large", "small"):
        v = videos.get(name)
        if v and v.get("url"):
            return v
    return None


def pixabay_caption(hit: dict, query: str) -> str:
    """Pixabay hits carry comma-separated tags; caption = first few tags,
    falling back to the search query (mirrors pexels_caption)."""
    tags = [t.strip() for t in str(hit.get("tags", "")).split(",") if t.strip()]
    if tags:
        return ", ".join(tags[:6])
    return query.strip() or "stock video clip"


def pixabay_search(api_key: str, query: str, per_page: int, page: int) -> dict:
    """One page of Pixabay video search results (returns the raw payload)."""
    params = urllib.parse.urlencode(
        {"key": api_key, "q": query, "per_page": per_page, "page": page}
    )
    req = urllib.request.Request(
        f"{PIXABAY_SEARCH_URL}?{params}",
        headers={"User-Agent": "vidgin-prepare/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:200]
        raise RuntimeError(f"Pixabay API error {e.code}: {body}") from e


def collect_pixabay(api_key: str, queries: list[str], per_query: int) -> list[tuple[str, str]]:
    """Return [(download_url, caption)] across all queries, paginating.

    The API caps totalHits at 500 per query; collect up to per_query items,
    stopping early when pages are exhausted. Only a handful of requests per
    query, spaced by PIXABAY_REQUEST_SLEEP.
    """
    items: list[tuple[str, str]] = []
    for query in queries:
        got = 0
        page = 1
        while got < per_query:
            want = min(PIXABAY_PER_PAGE, per_query - got)
            payload = pixabay_search(api_key, query, want, page)
            hits = payload.get("hits", [])
            if page == 1:
                print(f"  query {query!r}: {payload.get('totalHits', '?')} total hits on Pixabay", flush=True)
            if not hits:
                break
            for h in hits:
                f = choose_pixabay_video_file(h.get("videos", {}))
                if f and f.get("url"):
                    items.append((f["url"], pixabay_caption(h, query)))
                    got += 1
                    if got >= per_query:
                        break
            if len(hits) < want:
                break  # last page
            page += 1
            time.sleep(PIXABAY_REQUEST_SLEEP)
        print(f"  query {query!r}: {got} videos found", flush=True)
    return items


# ------------------------------------------------------------ clip building
def build_clip(frames: np.ndarray, n_frames: int, size: int) -> tuple[np.ndarray | None, str | None]:
    """frames (N,H,W,3) uint8 -> (clip (T,H,W,3) uint8, None) or (None, reason)."""
    idx = select_frame_indices(len(frames), n_frames)
    if idx is None:
        return None, "too_short"
    window = frames[idx]
    out = np.stack(
        [resize_nearest(center_crop_square(f), size) for f in window]
    ).astype(np.uint8)
    if float(out.mean()) < MEAN_BRIGHTNESS_MIN:
        return None, "too_dark"
    # reuse the trainer's own quality filters (expects [3,T,H,W] in [-1,1])
    norm = out.transpose(3, 0, 1, 2).astype(np.float32) / 127.5 - 1.0
    q = passes_quality(norm)
    if not q["pass"]:
        why = ",".join(
            k for k, ok in
            (("sharp", q["sharpness_ok"]), ("motion", q["motion_ok"]),
             ("aesthetic", q["aesthetic_ok"]))
            if not ok
        )
        return None, f"quality_fail({why})"
    return out, None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download stock videos and prepare a FolderClipDataset directory.")
    ap.add_argument("--source", choices=["pexels", "pixabay", "urls"], required=True,
                    help="pexels: Pexels API (needs PEXELS_API_KEY env var — "
                         "Pexels has paused issuing new keys); "
                         "pixabay: Pixabay API (needs PIXABAY_API_KEY env var — "
                         "free key issued instantly at https://pixabay.com/api/docs/); "
                         "urls: one direct video-file URL per line in --urls-file")
    ap.add_argument("--queries", default="",
                    help="comma-separated search phrases (with --source pexels/pixabay)")
    ap.add_argument("--per-query", type=int, default=250,
                    help="videos to fetch per query (with --source pexels)")
    ap.add_argument("--urls-file", default=None,
                    help="path to a text file with one video URL per line "
                         "(with --source urls)")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--frames", type=int, default=17,
                    help="frames per clip (keep (T-1) %% 4 == 0 for the VAE)")
    ap.add_argument("--size", type=int, default=64,
                    help="square clip size in pixels (multiple of 16)")
    args = ap.parse_args()

    if (args.frames - 1) % 4 != 0:
        sys.exit(f"error: --frames must satisfy (T-1) % 4 == 0, got {args.frames}")
    if args.size % 16 != 0:
        sys.exit(f"error: --size must be a multiple of 16, got {args.size}")

    if args.source == "pexels":
        env_var, api_name, signup = "PEXELS_API_KEY", "Pexels", "https://www.pexels.com/api/"
    elif args.source == "pixabay":
        env_var, api_name, signup = "PIXABAY_API_KEY", "Pixabay", "https://pixabay.com/api/docs/"
    if args.source in ("pexels", "pixabay"):
        api_key = os.environ.get(env_var)
        if not api_key:
            extra = ""
            if args.source == "pexels":
                extra = ("\nNOTE: Pexels has paused issuing new API keys. If you don't "
                         "already have one, use --source pixabay instead —\n"
                         "its free key is issued instantly at https://pixabay.com/api/docs/.\n")
            print(
                f"error: {env_var} is not set.\n\n"
                f"Get a free API key (no card required):\n"
                f"  1. Sign up at {signup}\n"
                f"  2. Copy your key from the dashboard/docs page\n"
                f"  3. In this terminal run:  export {env_var}=<your key>\n"
                f"     In Colab, use the notebook's getpass cell instead —\n"
                f"     never paste the key into chat or the notebook text.\n"
                f"{extra}\n"
                "Or use --source urls with a list of direct video URLs (no key).",
                file=sys.stderr,
            )
            sys.exit(2)
        queries = [q.strip() for q in args.queries.split(",") if q.strip()]
        if not queries:
            sys.exit(f"error: --queries is required with --source {args.source}")
        print(f"collecting up to {args.per_query} videos per query from {api_name} ...")
        items = (collect_pexels(api_key, queries, args.per_query)
                 if args.source == "pexels"
                 else collect_pixabay(api_key, queries, args.per_query))
    else:
        if not args.urls_file or not os.path.isfile(args.urls_file):
            sys.exit(f"error: --urls-file not found: {args.urls_file}")
        items = []
        with open(args.urls_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    items.append((line, caption_from_filename(line)))
        print(f"read {len(items)} URLs from {args.urls_file}")

    if not items:
        sys.exit("error: no videos to process")

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.jsonl")
    kept = downloaded = skipped = 0
    reasons: dict[str, int] = {}
    seen_hashes: set[str] = set()
    bytes_written = 0
    kept_offset = 0
    existing_count = 0

    # Resume support: when --out already holds clips from an earlier run,
    # continue numbering after the highest existing clip index instead of
    # overwriting clip_000001.npy on every run, and treat the existing
    # clips' content as already seen so re-running a list appends safely.
    existing_indices: list[int] = []
    for fname in os.listdir(args.out):
        if fname.startswith("clip_") and fname.endswith(".npy"):
            try:
                existing_indices.append(int(fname[5:-4]))
            except ValueError:
                continue
    if existing_indices:
        kept_offset = max(existing_indices)
        existing_count = len(existing_indices)
        for idx in existing_indices:
            arr = np.load(os.path.join(args.out, f"clip_{idx:06d}.npy"))
            seen_hashes.add(content_hash(arr))
        print(
            f"resuming in {args.out}: {existing_count} clips already present "
            f"(highest index {kept_offset}); new clips continue from "
            f"clip_{kept_offset + 1:06d}.npy",
            flush=True,
        )

    with open(manifest_path, "a", encoding="utf-8") as mf:
        for n, (url, caption) in enumerate(items, start=1):
            tmpdir = tempfile.mkdtemp(prefix="vidgin_clip_")
            try:
                tmp_path = os.path.join(tmpdir, "video")
                try:
                    download(url, tmp_path)
                except Exception as e:  # noqa: BLE001 - per-video, keep going
                    skipped += 1
                    reasons["download_error"] = reasons.get("download_error", 0) + 1
                    print(f"[{n}/{len(items)}] skip ({e})", flush=True)
                    continue
                downloaded += 1
                try:
                    frames = decode_video(tmp_path, want_frames=args.frames)
                except Exception as e:  # noqa: BLE001
                    skipped += 1
                    reasons["decode_error"] = reasons.get("decode_error", 0) + 1
                    print(f"[{n}/{len(items)}] skip decode: {e}", flush=True)
                    continue
                clip, reason = build_clip(frames, args.frames, args.size)
                if clip is None:
                    skipped += 1
                    reasons[reason] = reasons.get(reason, 0) + 1  # type: ignore[arg-type]
                    continue
                h = content_hash(clip)
                if h in seen_hashes:
                    skipped += 1
                    reasons["duplicate"] = reasons.get("duplicate", 0) + 1
                    continue
                seen_hashes.add(h)
                name = f"clip_{kept_offset + kept + 1:06d}.npy"
                np.save(os.path.join(args.out, name), clip)
                mf.write(json.dumps({"file": name, "caption": caption}) + "\n")
                mf.flush()
                bytes_written += os.path.getsize(os.path.join(args.out, name))
                kept += 1
                if kept % 25 == 0:
                    print(f"[{n}/{len(items)}] kept {kept} clips so far", flush=True)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

    mb = bytes_written / (1024 * 1024)
    print("=" * 50)
    print(f"videos seen:      {len(items)}")
    print(f"downloaded:       {downloaded}")
    print(f"kept as clips:    {kept}")
    print(f"skipped:          {skipped} {reasons if reasons else ''}")
    print(f"dataset on disk:  {mb:.1f} MB in {args.out}")
    print(f"manifest:         {manifest_path} ({existing_count + kept} lines)")


if __name__ == "__main__":
    main()
