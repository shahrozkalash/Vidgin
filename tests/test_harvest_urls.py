"""No-key URL harvester: pure-function tests (no network access).

Synthetic HTML strings exercise the Pexels video-files extractor, the
per-video variant chooser, dedupe behaviour and the Pixabay tiny->medium
suffix swap with a mocked HEAD check.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from harvest_urls import (  # noqa: E402
    choose_variant,
    extract_pexels_urls,
    extract_pixabay_tiny_urls,
    pixabay_medium_url,
    url_head_ok,
)

BASE = "https://videos.pexels.com/video-files"

FAKE_HTML = f"""
<html><body>
<!-- video A: hd variant, two sd variants, one repeated occurrence -->
<script>window.X="{BASE}/11111/11111-hd_1920_1080_30fps.mp4"</script>
<script>window.X="{BASE}/11111/11111-sd_540_960_30fps.mp4"</script>
<script>window.X="{BASE}/11111/11111-sd_360_640_30fps.mp4"</script>
<script>window.X="{BASE}/11111/11111-sd_360_640_30fps.mp4"</script>
<!-- video B: hd-only video -->
<script>window.X="{BASE}/22222/22222-hd_1920_1080_30fps.mp4"</script>
<!-- video C: sd without 360/540 in the name -->
<script>window.X="{BASE}/33333/33333-sd_640_360_30fps.mp4"</script>
<!-- look-alikes that must NOT match -->
<img src="{BASE}/44444/44444-sd_360_640_30fps.webm">
<a href="https://videos.example.com/video-files/55555/55555-sd_360.mp4">
</body></html>
"""


def test_extract_pexels_urls_groups_by_video_id():
    grouped = extract_pexels_urls(FAKE_HTML)
    assert set(grouped) == {"11111", "22222", "33333"}
    assert grouped["11111"] == ["hd_1920_1080_30fps", "sd_360_640_30fps",
                                "sd_540_960_30fps"]
    assert grouped["22222"] == ["hd_1920_1080_30fps"]
    assert grouped["33333"] == ["sd_640_360_30fps"]
    # repeated occurrence must not duplicate the variant
    assert len(grouped["11111"]) == 3


def test_choose_variant_prefers_small_sd():
    assert choose_variant(["hd_1920_1080_30fps", "sd_540_960_30fps",
                           "sd_360_640_30fps"]) == "sd_360_640_30fps"
    assert choose_variant(["hd_1920_1080_30fps",
                           "sd_540_960_30fps"]) == "sd_540_960_30fps"
    assert choose_variant(["uhd_3840_2160_30fps",
                           "sd_640_360_30fps"]) == "sd_640_360_30fps"
    assert choose_variant(["uhd_3840_2160_30fps",
                           "hd_1920_1080_30fps"]) in ("hd_1920_1080_30fps",
                                                      "uhd_3840_2160_30fps")
    assert choose_variant([]) is None


def test_dedupe_key_is_source_and_video_id():
    # same video appearing twice must collapse to one (source, id) entry
    seen: dict[tuple[str, str], str] = {}
    for vid, variants in extract_pexels_urls(FAKE_HTML).items():
        key = ("pexels", vid)
        if key not in seen:
            seen[key] = choose_variant(variants)
    # re-running the same extraction must not add anything
    for vid, variants in extract_pexels_urls(FAKE_HTML).items():
        key = ("pexels", vid)
        if key not in seen:
            seen[key] = choose_variant(variants)
    assert len(seen) == 3


PIXABAY_HTML = """
<video><source src="https://cdn.pixabay.com/video/2023/10/22/185729-876543210_tiny.mp4" type="video/mp4">
<video><source src="https://cdn.pixabay.com/video/2023/10/22/185729-876543210_tiny.mp4" type="video/mp4">
<video><source src="https://cdn.pixabay.com/video/2022/05/01/116167-123456789_tiny.mp4" type="video/mp4">
"""


def test_extract_pixabay_tiny_urls_and_suffix_swap():
    found = extract_pixabay_tiny_urls(PIXABAY_HTML)
    assert set(found) == {"876543210", "123456789"}
    medium = pixabay_medium_url(found["876543210"])
    assert medium == ("https://cdn.pixabay.com/video/2023/10/22/"
                      "185729-876543210_medium.mp4")
    assert "_tiny.mp4" not in medium


def test_pixabay_head_check_is_mockable(monkeypatch):
    import harvest_urls

    calls = []
    monkeypatch.setattr(
        harvest_urls, "url_head_ok",
        lambda url, timeout=30: calls.append(url) or url.endswith("_medium.mp4"),
    )
    found = extract_pixabay_tiny_urls(PIXABAY_HTML)
    ok = [v for v in found.values()
          if harvest_urls.url_head_ok(pixabay_medium_url(v))]
    assert len(ok) == 2
    assert all("_medium.mp4" in c for c in calls)


def test_url_head_ok_tolerates_network_errors(monkeypatch):
    import urllib.request
    import harvest_urls

    def boom(req, timeout=30):
        raise OSError("no network in tests")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert url_head_ok("https://example.invalid/x.mp4") is False
