"""Caption engine vs Gemini eval pack (scripts/build_caption_eval_pack.py).

Offline: a synthetic 120-clip, Gemini-captioned chunk; the engine runs with
its stub backend (test-only), which is enough to assemble and check the
pack. Nothing here measures caption quality.
"""
import contextlib
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import build_caption_eval_pack as bep
import caption_engine as engine

N = 120


def _chunk(root: Path, n=N, name="chunk3", baseline=True, extra_rows=0):
    """A Stage-2-style chunk: n Gemini-captioned clips with depth maps, and
    the immutable baseline snapshotted on the Stage 2 tree (the parent)."""
    d = root / "stage2" / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    lines = []
    for i in range(n + extra_rows):
        f = f"clip_{i + 1:06d}.npy"
        np.save(d / f, rng.integers(0, 255, (9, 24, 24, 3), dtype=np.uint8))
        np.save(d / f"clip_{i + 1:06d}_depth.npy",
                rng.random((9, 24, 24), dtype=np.float32))
        rec = {"file": f, "depth": f"clip_{i + 1:06d}_depth.npy",
               "source_url": f"https://videos.example/{i}.mp4"}
        if i < n:
            rec.update(caption=f"Scene {i}: a street with people walking "
                               f"past shops.",
                       caption_short=f"street {i}", subject="people",
                       action="walking", setting="a street",
                       camera_motion="static", lighting="even",
                       caption_source="gemini", caption_weak="city clip")
        else:
            rec.update(caption="weak only", caption_source="weak")
        lines.append(json.dumps(rec, separators=(",", ":")) + "\n")
    (d / "manifest.jsonl").write_bytes("".join(lines).encode())
    if baseline:
        with contextlib.redirect_stdout(io.StringIO()):
            engine.snapshot_baseline(d.parent)
    return d


def _tree_hash(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def _caption_with_stub(out: Path):
    with contextlib.redirect_stdout(io.StringIO()):
        for m in engine.manifests(out / "engine"):
            engine.process_manifest(m, engine.StubBackend(), quiet=True,
                                    flush_every=0)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("arena")
    src = _chunk(root)
    before = _tree_hash(src.parent)
    out = root / "pack"
    info = bep.build(str(src), str(out))
    _caption_with_stub(out)
    ab = bep.make_ab(str(out))
    return {"root": root, "src": src, "out": out, "info": info, "ab": ab,
            "before": before}


def test_pack_has_exactly_120_pairs(built):
    out = built["out"]
    ref = [json.loads(l) for l in open(out / "reference" / "chunk3" / "manifest.jsonl")]
    eng = [json.loads(l) for l in open(out / "engine" / "chunk3" / "manifest.jsonl")]
    assert len(ref) == len(eng) == N
    assert all(r["caption_source"] == "gemini" for r in ref)
    e, r = bep.pair_keys(str(out))
    assert len(e) == len(r) == len(e & r) == N
    # after the engine captioned its copy, ITS OWN pairing finds all 120
    pairs = engine.benchmark_metrics(out / "engine", out / "reference")
    assert len(pairs) == N
    clips = sorted(p.name for p in (out / "engine" / "chunk3").glob("clip_*.npy"))
    assert len(clips) == 2 * N                  # 120 clips + their depth maps
    assert built["info"]["clips"] == N


def test_baselines_are_byte_identical_to_the_source(built):
    src, out = built["src"], built["out"]
    src_rows = (src / "manifest.jsonl").read_bytes().splitlines(keepends=True)
    ref_rows = (out / "reference" / "chunk3" / "manifest.jsonl").read_bytes() \
        .splitlines(keepends=True)
    assert ref_rows == src_rows[:N]
    # the source's immutable baseline was checked, and the pack's own
    # baseline holds the same digests
    assert built["info"]["source_baseline_rows_checked"] == N
    src_base = json.loads((src.parent / ".gemini_baseline.json").read_text())
    ref_base = json.loads((out / "reference" / ".gemini_baseline.json").read_text())
    assert sorted(ref_base.values()) == sorted(src_base.values())
    with contextlib.redirect_stdout(io.StringIO()):
        assert engine.verify(out / "reference")
    # the source was only read
    assert _tree_hash(src.parent) == built["before"]
    # the engine's copy has no caption of any kind
    eng0 = json.loads((out / "engine" / "chunk3" / "manifest.jsonl.bak")
                      .read_text().splitlines()[0])
    assert not any(k.startswith(("caption", "gemini")) for k in eng0)
    assert not set(engine.FIELDS) & set(eng0)


def test_ab_key_is_sealed_away_from_the_judge(built):
    out, ab = built["out"], built["ab"]
    judge = out / "judge"
    files = sorted(p.relative_to(judge).as_posix() for p in judge.rglob("*")
                   if p.is_file())
    assert files == sorted(["README.md", "vidgin_ab_pack.md"] +
                           [f"frames/pair_{i:02d}.png" for i in range(1, 31)])
    assert not list(out.joinpath("engine").rglob("vidgin_ab_*"))
    key = json.loads((out / "sealed" / "vidgin_ab_key.json").read_text())
    assert ab["pairs"] == len(key["pairs"]) == 30
    pack = (judge / "vidgin_ab_pack.md").read_text(encoding="utf-8")
    assert len(re.findall(r"^## Pair \d+:", pack, re.M)) == 30
    for p in judge.rglob("*"):
        if p.suffix in (".md", ".json", ".txt"):
            t = p.read_text(encoding="utf-8").lower()
            for leak in ("gemini", "vidgin_ab_key", '"tool"', "engine",
                         "caption_source", "seed"):
                assert leak not in t, (p.name, leak)
    # The key is right: position A/B per pair matches who wrote that caption.
    ref = {json.loads(l)["file"]: json.loads(l)["caption"]
           for l in open(out / "reference" / "chunk3" / "manifest.jsonl")}
    tool_side = {"A": 0, "B": 0}
    for m in re.finditer(r"^## Pair (\d+): chunk3/(\S+)\n\nCaption A: (.*)\n"
                         r"(?:.*\n){4}Caption B: (.*)$", pack, re.M):
        i, f, a, b = m.groups()
        gemini_side = "A" if a == ref[f] else "B"
        assert {a, b} & {ref[f]} and a != b
        assert key["pairs"][i]["tool"] != gemini_side
        tool_side[key["pairs"][i]["tool"]] += 1
    assert sum(tool_side.values()) == 30 and min(tool_side.values()) > 0


def test_score_opens_the_key_only_after_judging(built, tmp_path):
    out = built["out"]
    key = json.loads((out / "sealed" / "vidgin_ab_key.json").read_text())["pairs"]
    pack = (out / "judge" / "vidgin_ab_pack.md").read_text(encoding="utf-8")

    def rate(m):                     # a judge who prefers the engine 20 of 30
        i = m.group(1)
        tool = key[i]["tool"]
        win = int(i) <= 20
        a, b = ((5, 2) if (tool == "A") == win else (2, 5))
        return m.group(0).replace("Rating A (1-5): ____", f"Rating A (1-5): {a}") \
            .replace("Rating B (1-5): ____", f"Rating B (1-5): {b}")
    rated = re.sub(r"^## Pair (\d+):.*?(?=^## Pair |\Z)", rate, pack,
                   flags=re.M | re.S)
    path = tmp_path / "rated.md"
    path.write_text(rated, encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        result = bep.score(str(out), str(path))
    assert result == {"rated": 30, "wins": 20, "ties": 0, "losses": 10}
    assert json.loads((out / "ab_result.json").read_text()) == result
    # wired to the engine's eval suite: it pairs all 120 and prints the A/B
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ready = engine.eval_suite(out / "engine", out / "reference",
                                  ab_result=out / "ab_result.json")
    text = buf.getvalue()
    assert "Pairs=120" in text and '"wins": 20' in text
    assert not ready                 # stub provenance never counts as READY


def test_same_seed_same_pack_and_no_rebuild_in_place(built):
    root = built["root"]
    out2 = root / "pack2"
    bep.build(str(built["src"]), str(out2))
    _caption_with_stub(out2)
    bep.make_ab(str(out2))
    for sub in ("reference", "judge", "sealed"):
        assert _tree_hash(out2 / sub) == _tree_hash(built["out"] / sub), sub
    assert (out2 / "pack.json").read_bytes() == (built["out"] / "pack.json").read_bytes()
    # a different seed draws a different blind pack
    out3 = root / "pack3"
    bep.build(str(built["src"]), str(out3), seed=7)
    _caption_with_stub(out3)
    bep.make_ab(str(out3))
    assert (out3 / "sealed" / "vidgin_ab_key.json").read_bytes() != \
        (built["out"] / "sealed" / "vidgin_ab_key.json").read_bytes()
    # never over an existing pack, never a second draw
    with pytest.raises(FileExistsError):
        bep.build(str(built["src"]), str(out2))
    with pytest.raises(FileExistsError):
        bep.make_ab(str(out2))


def test_refuses_a_changed_baseline_too_few_rows_and_uncaptioned(tmp_path):
    src = _chunk(tmp_path / "a")
    rows = (src / "manifest.jsonl").read_bytes().splitlines(keepends=True)
    rows[5] = rows[5].replace(b"Scene 5:", b"Edited 5:")
    (src / "manifest.jsonl").write_bytes(b"".join(rows))
    with pytest.raises(ValueError, match="row 6 no longer matches"):
        bep.build(str(src), str(tmp_path / "out_a"))

    few = _chunk(tmp_path / "b", n=100, extra_rows=20)
    with pytest.raises(ValueError, match="only 100 Gemini-captioned rows"):
        bep.build(str(few), str(tmp_path / "out_b"))

    ok = _chunk(tmp_path / "c", baseline=False)
    info = bep.build(str(ok), str(tmp_path / "out_c"))
    assert info["source_baseline"] is None
    with pytest.raises(ValueError, match="only 0 clips have an engine caption"):
        bep.make_ab(str(tmp_path / "out_c"))


def test_more_gemini_rows_than_needed_are_sampled_by_seed(tmp_path):
    src = _chunk(tmp_path, n=130)
    a = bep.build(str(src), str(tmp_path / "a"))
    b = bep.build(str(src), str(tmp_path / "b"))
    assert [r["row"] for r in a["rows"]] == [r["row"] for r in b["rows"]]
    assert len(a["rows"]) == N and len({r["row"] for r in a["rows"]}) == N


def test_baseline_can_live_higher_up(tmp_path):
    src = _chunk(tmp_path, baseline=False)
    with contextlib.redirect_stdout(io.StringIO()):
        base = engine.snapshot_baseline(tmp_path)      # two levels up
    info = bep.build(str(src), str(tmp_path / "p"), source_baseline=str(base))
    assert info["source_baseline_rows_checked"] == N


def test_cli(tmp_path, capsys):
    src = _chunk(tmp_path)
    assert bep.main(["build", str(src), str(tmp_path / "p")]) == 0
    out = capsys.readouterr().out
    assert "120 clips, 120 engine/reference pairs" in out
    assert "120 rows checked" in out
