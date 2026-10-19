#!/usr/bin/env python3
"""Harvest direct stock-video file URLs from search pages (no API key needed).

Pexels (primary source): the raw HTML of a search page
(https://www.pexels.com/search/videos/<query>/?page=N) embeds direct
https://videos.pexels.com/video-files/<id>/<id>-<variant>.mp4 URLs — about
a dozen unique video IDs per page, each with several resolution variants.
Pagination works via ?page=N.

Pixabay (optional secondary source): search pages embed tiny-preview URLs
https://cdn.pixabay.com/video/..._tiny.mp4. For each, we attempt a suffix
swap _tiny -> _medium and keep the URL only if a HEAD request returns 200
(following redirects). Disabled by default (default --sources pexels).

Usage (no API key):
  python scripts/harvest_urls.py \\
      --queries "people walking city street,crowd crossing street" \\
      --pages 12 --out urls.txt

Then feed the list into the collector:
  python scripts/prepare_clips.py --source urls --urls-file urls.txt \\
      --out /content/drive/MyDrive/vidgin-data/tiny-clips

Stdlib only (urllib, re, argparse, time). All parsing/choosing logic is in
pure functions so it can be unit-tested without network access.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
TIMEOUT = 30

# Pexels video file URL: .../video-files/<vid>/<vid>-<variant>.mp4
PEXELS_RE = re.compile(
    r"https://videos\.pexels\.com/video-files/(\d+)/(\d+)-([A-Za-z0-9_]+)\.mp4"
)
# Pixabay embedded previews: .../video/<date>/<pageid>-<id>_tiny.mp4
PIXABAY_TINY_RE = re.compile(
    r"(https://cdn\.pixabay\.com/video/\d{4}/\d{2}/\d{2}/\d+-(\d+)_tiny\.mp4)"
)


def extract_pexels_urls(html: str) -> dict[str, list[str]]:
    """Group Pexels video-file URLs by video ID: {id: [variant, ...]}.

    Each list holds the variant names (e.g. 'sd_360_640_30fps'); the full
    URL is rebuildable as
    https://videos.pexels.com/video-files/<id>/<id>-<variant>.mp4.
    """
    grouped: dict[str, set[str]] = {}
    for vid, fid, variant in PEXELS_RE.findall(html):
        grouped.setdefault(vid, set()).add(variant)
    return {vid: sorted(variants) for vid, variants in grouped.items()}


def choose_variant(variants: list[str]) -> str | None:
    """Pick one variant per video, preferring small files (training is 64x64).

    Preference order: anything containing "360" (e.g. sd_360_640_30fps),
    then "540", then any "sd_" variant, then the first variant.
    """
    if not variants:
        return None

    def rank(v: str) -> int:
        if "360" in v:
            return 0
        if "540" in v:
            return 1
        if v.startswith("sd_"):
            return 2
        return 3

    return sorted(variants, key=lambda v: (rank(v), v))[0]


def pexels_page_url(query: str, page: int) -> str:
    q = urllib.parse.quote(query.strip())
    return f"https://www.pexels.com/search/videos/{q}/?page={page}"


def fetch(url: str, timeout: int = TIMEOUT) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


# ---------------------------------------------------------------- Pixabay
def extract_pixabay_tiny_urls(html: str) -> dict[str, str]:
    """Find Pixabay tiny-preview URLs: {video id: tiny url}."""
    return {vid: url for url, vid in PIXABAY_TINY_RE.findall(html)}


def pixabay_medium_url(tiny_url: str) -> str:
    """Swap the tiny-preview suffix for the medium variant."""
    return tiny_url.replace("_tiny.mp4", "_medium.mp4")


def url_head_ok(url: str, timeout: int = TIMEOUT) -> bool:
    """True if a HEAD request on url returns 200 (following redirects)."""
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": BROWSER_UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def pixabay_search_url(query: str, page: int) -> str:
    q = urllib.parse.quote(query.strip())
    return f"https://pixabay.com/videos/search/{q}/?pagi={page}"


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--queries", required=True,
                    help="comma-separated search queries")
    ap.add_argument("--pages", type=int, default=12,
                    help="search-result pages per query (default 12)")
    ap.add_argument("--out", required=True,
                    help="output file, one URL per line")
    ap.add_argument("--sources", default="pexels",
                    help="comma-separated subset of: pexels, pixabay "
                         "(default 'pexels')")
    ap.add_argument("--delay", type=float, default=0.5,
                    help="seconds between page fetches (default 0.5)")
    ap.add_argument("--max-urls", type=int, default=None,
                    help="optional cap on total URLs written")
    args = ap.parse_args()

    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    if not queries:
        raise SystemExit("--queries must contain at least one query")
    sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    for s in sources:
        if s not in ("pexels", "pixabay"):
            raise SystemExit(f"unknown source: {s} (use pexels and/or pixabay)")

    seen: dict[tuple[str, str], str] = {}  # (source, video id) -> chosen url
    total_counts: dict[str, int] = {}

    for qi, query in enumerate(queries):
        before = len(seen)
        for page in range(1, args.pages + 1):
            try:
                if "pexels" in sources:
                    html = fetch(pexels_page_url(query, page))
                    for vid, variants in extract_pexels_urls(html).items():
                        key = ("pexels", vid)
                        if key not in seen:
                            v = choose_variant(variants)
                            if v:
                                seen[key] = (
                                    f"https://videos.pexels.com/video-files/"
                                    f"{vid}/{vid}-{v}.mp4"
                                )
                if "pixabay" in sources:
                    html = fetch(pixabay_search_url(query, page))
                    for vid, tiny in extract_pixabay_tiny_urls(html).items():
                        key = ("pixabay", vid)
                        if key not in seen:
                            medium = pixabay_medium_url(tiny)
                            if url_head_ok(medium):
                                seen[key] = medium
                if args.max_urls and len(seen) >= args.max_urls:
                    break
            except Exception as e:  # per-page tolerance: log and continue
                print(f"[warn] {query!r} page {page}: {type(e).__name__}: {e}",
                      file=sys.stderr)
            time.sleep(args.delay)
        total_counts[query] = len(seen) - before
        print(f"{query!r}: +{total_counts[query]} new urls "
              f"(total {len(seen)})", flush=True)
        if args.max_urls and len(seen) >= args.max_urls:
            break

    urls = list(seen.values())
    if args.max_urls:
        urls = urls[: args.max_urls]
    with open(args.out, "w", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")
    print(f"wrote {len(urls)} urls -> {args.out}")
    for q, c in total_counts.items():
        print(f"  {q!r}: {c}")


if __name__ == "__main__":
    main()
