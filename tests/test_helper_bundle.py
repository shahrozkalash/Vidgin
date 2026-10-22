"""Tests for the Vidgin helper bundle.

Covers the local dataset tools in ``vidgen/data/dataset_tools.py`` and
their CLI in ``scripts/vidgin_helper.py``:

1. A complete clip + depth dataset checks clean.
2. Missing, corrupt, wrong-shape, dark, and mismatched-depth files are
   reported with useful counts.
3. Orphan files and duplicate manifest references are reported.
4. Packing merges same-named chunks without collisions, rewrites depth
   names, preserves captions, and verifies the output.
5. Packing refuses bad sources, mixed shapes, and non-empty outputs.
6. The CLI wires check/pack results to exit codes and delegates the
   captions command to the existing caption tool.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import vidgin_helper
from vidgen.data.clips import FolderClipDataset
from vidgen.data.dataset_tools import check_dataset, pack_datasets


def _write_dataset(
    root,
    *,
    count: int = 2,
    shape: tuple[int, int, int, int] = (5, 16, 16, 3),
    with_depth: bool = True,
    start: int = 1,
    value: int = 120,
) -> str:
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for offset in range(count):
        idx = start + offset
        name = f"clip_{idx:06d}.npy"
        clip = np.full(shape, value + offset, dtype=np.uint8)
        # Add a little variation so sampled frames are not a flat colour.
        rows = ((np.arange(shape[1], dtype=np.int16) + value) % 256).astype(np.uint8)
        clip[..., 0] = rows[None, :, None]
        np.save(root / name, clip)
        entry = {"file": name, "caption": f"caption {idx}"}
        if with_depth:
            depth_name = f"clip_{idx:06d}_depth.npy"
            np.save(root / depth_name, np.full(shape[:3], 90, dtype=np.uint8))
            entry["depth"] = depth_name
        entries.append(entry)
    with open(root / "manifest.jsonl", "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return str(root)


def _read_manifest(data_dir: str) -> list[dict]:
    with open(os.path.join(data_dir, "manifest.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_check_dataset_good_with_depth(tmp_path):
    data_dir = _write_dataset(tmp_path / "good", count=3)

    report = check_dataset(
        data_dir,
        expected_frames=5,
        expected_height=16,
        expected_width=16,
        require_depth=True,
    )

    assert report.ok
    assert report.manifest_entries == 3
    assert report.clean_clips == 3
    assert report.depth_referenced == 3
    assert report.depth_present == 3
    assert report.depth_valid == 3
    assert report.clip_shape == [5, 16, 16, 3]
    assert report.depth_shape == [5, 16, 16]
    assert report.errors == []
    assert report.warnings == []


def test_check_dataset_reports_missing_corrupt_dark_and_depth_mismatch(tmp_path):
    root = tmp_path / "bad"
    root.mkdir()
    entries = [
        {"file": "missing.npy", "caption": "missing"},
        {"file": "dark.npy", "caption": "dark"},
        {"file": "corrupt.npy", "caption": "corrupt"},
        {
            "file": "wrong_depth.npy",
            "caption": "wrong depth",
            "depth": "wrong_depth_depth.npy",
        },
    ]
    np.save(root / "dark.npy", np.zeros((5, 16, 16, 3), dtype=np.uint8))
    (root / "corrupt.npy").write_bytes(b"not an npy file")
    np.save(root / "wrong_depth.npy", np.full((5, 16, 16, 3), 150, dtype=np.uint8))
    np.save(root / "wrong_depth_depth.npy", np.zeros((5, 8, 16), dtype=np.uint8))
    with open(root / "manifest.jsonl", "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    report = check_dataset(str(root), require_depth=False)
    codes = {issue.code for issue in report.errors}
    warning_codes = {issue.code for issue in report.warnings}

    assert not report.ok
    assert report.missing_clips == 1
    assert "clip_missing" in codes
    assert "clip_unreadable" in codes
    assert "depth_shape_mismatch" in codes
    assert "clip_too_dark" in warning_codes
    assert report.dark_clips == ["dark.npy"]
    assert report.error_clips == 3  # missing, corrupt, depth mismatch
    assert report.warning_clips == 1  # dark


def test_check_dataset_reports_orphans_and_duplicate_references(tmp_path):
    data_dir = _write_dataset(tmp_path / "dupes", count=1, with_depth=False)
    entries = _read_manifest(data_dir)
    entries.append(dict(entries[0]))
    with open(os.path.join(data_dir, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    np.save(tmp_path / "dupes" / "clip_999999.npy", np.full((5, 16, 16, 3), 140, np.uint8))
    np.save(tmp_path / "dupes" / "clip_999999_depth.npy", np.zeros((5, 16, 16), np.uint8))

    report = check_dataset(data_dir, detect_duplicates=True)

    assert not report.ok
    assert report.duplicate_file_references == {"clip_000001.npy": [1, 2]}
    assert report.orphan_files == ["clip_999999.npy"]
    assert report.unreferenced_depth_files == ["clip_999999_depth.npy"]
    assert {issue.code for issue in report.warnings} >= {
        "orphan_clip_file",
        "unreferenced_depth_file",
    }


def test_pack_merges_same_named_chunks_and_preserves_depth(tmp_path):
    first = _write_dataset(tmp_path / "chunk1", count=2, value=110)
    second = _write_dataset(tmp_path / "chunk2", count=2, value=170)
    out = tmp_path / "packed"

    report = pack_datasets([first, second], str(out), labels=["group1", "group2"])

    assert report.entries == 4
    assert report.clip_files == 4
    assert report.depth_files == 4
    entries = _read_manifest(str(out))
    assert [entry["file"] for entry in entries] == [
        "group1__clip_000001.npy",
        "group1__clip_000002.npy",
        "group2__clip_000001.npy",
        "group2__clip_000002.npy",
    ]
    assert entries[0]["depth"] == "group1__clip_000001_depth.npy"
    assert entries[0]["caption"] == "caption 1"
    assert entries[0]["pack_source"] == "group1"
    assert entries[-1]["pack_source"] == "group2"

    checked = check_dataset(str(out), require_depth=True)
    assert checked.ok
    dataset = FolderClipDataset(str(out), frames=5, height=16, width=16)
    assert len(dataset) == 4
    tensor, caption = dataset[3]
    assert tensor.shape == (3, 5, 16, 16)
    assert caption == "caption 2"
    assert dataset.get_depth(3) is not None


def test_pack_refuses_bad_source_mixed_shapes_and_nonempty_output(tmp_path):
    good = _write_dataset(tmp_path / "good", count=1)
    other_shape = _write_dataset(
        tmp_path / "other", count=1, shape=(9, 16, 16, 3), value=150
    )
    with pytest.raises(ValueError, match="different clip shapes"):
        pack_datasets([good, other_shape], str(tmp_path / "mixed"))

    bad = tmp_path / "bad"
    bad.mkdir()
    with open(bad / "manifest.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"file": "ghost.npy", "caption": "ghost"}) + "\n")
    with pytest.raises(ValueError, match="not ready"):
        pack_datasets([str(bad)], str(tmp_path / "badpack"))

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        pack_datasets([good], str(occupied))

    replaced = pack_datasets([good], str(occupied), overwrite=True)
    assert replaced.entries == 1
    assert not (occupied / "keep.txt").exists()
    assert check_dataset(str(occupied)).ok


def test_helper_cli_check_and_pack_exit_codes(tmp_path, capsys):
    source = _write_dataset(tmp_path / "source", count=2)
    out = tmp_path / "out"

    assert vidgin_helper.main(["check", source, "--require-depth"]) == 0
    check_out = capsys.readouterr().out
    assert "Result: OK - ready" in check_out

    assert vidgin_helper.main(
        ["pack", source, "--out", str(out), "--labels", "ready"]
    ) == 0
    pack_out = capsys.readouterr().out
    assert "Total: 2 clips, 2 depth maps" in pack_out

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "manifest.jsonl").write_text(
        json.dumps({"file": "ghost.npy", "caption": "ghost"}) + "\n",
        encoding="utf-8",
    )
    assert vidgin_helper.main(["check", str(bad)]) == 1


def test_helper_cli_captions_delegates_to_existing_tool(tmp_path, monkeypatch):
    calls = {}

    def fake_caption_dataset(**kwargs):
        calls.update(kwargs)
        return 2, 1, 0

    monkeypatch.setattr(vidgin_helper, "_caption_dataset", fake_caption_dataset)

    rc = vidgin_helper.main(
        [
            "captions",
            "--data-dir",
            str(tmp_path),
            "--limit",
            "3",
            "--max-calls",
            "9",
            "--pace",
            "0",
        ]
    )

    assert rc == 0
    assert calls["data_dir"] == str(tmp_path)
    assert calls["limit"] == 3
    assert calls["max_calls"] == 9
    assert calls["pace_seconds"] == 0.0


def test_helper_parser_requires_a_command():
    with pytest.raises(SystemExit):
        vidgin_helper.main([])


# ---------------------------------------------------------------------------
# Split command: deterministic, grouped, leak-free
# ---------------------------------------------------------------------------

def _write_grouped_dataset(root):
    """10 clips from 5 sources (2 clips share each source group)."""
    entries = []
    for g in range(5):
        for j in range(2):
            name = f"clip_{g}{j}.npy"
            arr = np.full((17, 64, 64, 3), 100 + g, dtype=np.uint8)
            np.save(root / name, arr)
            entries.append({"file": name, "caption": f"group {g} clip {j}",
                            "source": f"video-{g}"})
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")


def test_make_splits_grouped_deterministic(tmp_path):
    from vidgen.data.dataset_tools import make_splits

    _write_grouped_dataset(tmp_path)
    report = make_splits(str(tmp_path), ratios=(0.6, 0.2, 0.2), seed=7)
    assert sum(report.counts.values()) == 10
    seen_files = set()
    group_split = {}
    for name in ("train", "val", "test"):
        path = tmp_path / f"manifest_{name}.jsonl"
        lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        assert len(lines) == report.counts[name]
        for entry in lines:
            assert entry["file"] not in seen_files  # disjoint
            seen_files.add(entry["file"])
            # a source group lives in exactly one split (no leakage)
            assert group_split.setdefault(entry["source"], name) == name
    assert len(seen_files) == 10
    # Same seed -> byte-identical split files.
    first = {n: (tmp_path / f"manifest_{n}.jsonl").read_bytes() for n in ("train", "val", "test")}
    make_splits(str(tmp_path), ratios=(0.6, 0.2, 0.2), seed=7, overwrite=True)
    for n, blob in first.items():
        assert (tmp_path / f"manifest_{n}.jsonl").read_bytes() == blob
    # Refuses to overwrite without the flag.
    with pytest.raises(FileExistsError):
        make_splits(str(tmp_path), ratios=(0.6, 0.2, 0.2), seed=7)


def test_split_consumed_by_dataset_and_cli(tmp_path):
    from vidgen.data.clips import FolderClipDataset
    from vidgen.data.dataset_tools import make_splits

    _write_grouped_dataset(tmp_path)
    report = make_splits(str(tmp_path), ratios=(0.6, 0.2, 0.2), seed=3)
    ds = FolderClipDataset(str(tmp_path), manifest_name="manifest_val.jsonl")
    assert len(ds) == report.counts["val"]
    clip, caption = ds[0]
    assert clip.shape == (3, 17, 64, 64) and caption

    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "scripts/vidgin_helper.py", "split", str(tmp_path),
         "--ratios", "0.6,0.2,0.2", "--seed", "3", "--overwrite"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True, text=True)
    assert out.returncode == 0 and "train" in out.stdout


def test_require_real_guard(tmp_path):
    """--require-real must fail loudly instead of synthetic fallback."""
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for mod in ("vidgen.train", "vidgen.train_vae"):
        out = subprocess.run(
            [sys.executable, "-m", mod, "--require-real", "--steps", "1"],
            cwd=root, capture_output=True, text=True)
        assert out.returncode != 0
        assert "refusing to fall back" in (out.stderr + out.stdout)
