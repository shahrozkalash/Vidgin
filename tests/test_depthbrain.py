"""DepthBrain (vidgen/depthbrain.py) — Vidgin's own depth estimator, to
replace MiDaS (a borrowed pretrained model) in prep once it has proved
itself.

Covered: default size within the 5-20M spec, the U-Net on any frame size,
the scale/shift-invariant loss (and that it still punishes an upside-down
map), the teacher-agreement metrics and the pass/fail agreement check,
real learning on synthetic clips whose depth is visible in the pixels,
bit-identical resume, checkpoint -> estimate_depth (the prep contract),
label_depth.py, and the gates: prep defaults to MiDaS, and refuses a
missing or unverified DepthBrain before any download. Nothing downloads.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from vidgen import depth as depth_mod
from vidgen import depthbrain as brain_mod
from vidgen.depth import depth_source_id, estimate_depth
from vidgen.depthbrain import (AGREEMENT_GATE, DepthBrain, DepthBrainConfig,
                               agreement_check, count_params, depth_loss,
                               depth_metrics, load_depthbrain)

from label_depth import label_dataset
from train_depthbrain import (DepthBrainTrainer, collect_items, is_val,
                              split_key)

SMALL = dict(base=8, levels=3, size=32, frames=3, batch_size=4,
             warmup_steps=5, lr=3e-3, ema_decay=0.9)
# The real gate wants >= 50 held-out clips; the synthetic sets are smaller.
FEW_CLIPS_GATE = dict(AGREEMENT_GATE, min_val_clips=5)


def _scene(rng, t=9, s=32):
    """A clip whose depth is readable from its pixels: a near bright square
    over a background that recedes upward (darker = farther)."""
    yy = np.linspace(0.0, 1.0, s)[:, None].repeat(s, 1)          # 0 top
    depth = np.empty((t, s, s), np.float32)
    rgb = np.empty((t, s, s, 3), np.uint8)
    x0, y0 = rng.integers(2, s // 2, size=2)
    for i in range(t):
        d = 0.6 * yy
        x = (x0 + i) % (s - 10)
        d[y0:y0 + 10, x:x + 10] = 1.0
        depth[i] = d
        tint = rng.uniform(0.7, 1.0, size=3)
        rgb[i] = (np.clip(d[..., None] * tint * 255, 0, 255)).astype(np.uint8)
    lo, hi = depth.min(), depth.max()
    return rgb, ((depth - lo) / (hi - lo) * 255).round().astype(np.uint8)


def _dataset(path, n=24, with_depth=True, teacher=None, seed=0, urls=None):
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(seed)
    with open(os.path.join(path, "manifest.jsonl"), "w") as f:
        for i in range(n):
            rgb, dep = _scene(rng)
            name = f"clip_{i + 1:06d}.npy"
            np.save(os.path.join(path, name), rgb)
            rec = {"file": name, "caption": "scene"}
            if urls:
                rec["source_url"] = urls[i]
            if with_depth:
                np.save(os.path.join(path, name[:-4] + "_depth.npy"), dep)
                rec["depth"] = name[:-4] + "_depth.npy"
                if teacher:
                    rec["depth_source"] = teacher
            f.write(json.dumps(rec) + "\n")
    return str(path)


# ------------------------------------------------------------------ model
@pytest.mark.parametrize("hw", [(32, 32), (30, 45), (17, 64)])
def test_any_frame_size_round_trips(hw):
    m = DepthBrain(DepthBrainConfig(base=8, levels=3)).eval()
    with torch.no_grad():
        out = m(torch.rand(2, 3, *hw) * 2 - 1)
    assert out.shape == (2, *hw) and torch.isfinite(out).all()


def test_loss_ignores_scale_and_shift_but_not_sign():
    torch.manual_seed(0)
    tgt = torch.rand(2, 3, 16, 16)
    perfect = depth_loss(tgt * 7.0 + 3.0, tgt)["loss"]
    assert float(perfect) < 1e-4
    assert torch.allclose(depth_loss(tgt * 0.5 - 2.0, tgt)["loss"], perfect,
                          atol=1e-4)
    assert float(depth_loss(-tgt, tgt)["loss"]) > 1.0   # upside-down map
    noisy = depth_loss(tgt + 0.3 * torch.rand_like(tgt), tgt)["loss"]
    assert float(perfect) < float(noisy)


def test_loss_skips_flat_teacher_maps():
    pred = torch.rand(2, 3, 8, 8, requires_grad=True)
    out = depth_loss(pred, torch.zeros(2, 3, 8, 8))
    assert float(out["loss"].detach()) == 0.0
    out["loss"].backward()                    # still differentiable
    mixed = torch.stack([torch.zeros(3, 8, 8), torch.rand(3, 8, 8)])
    a = depth_loss(pred.detach(), mixed)["loss"]
    b = depth_loss(pred.detach()[1:], mixed[1:])["loss"]
    assert torch.allclose(a, b)


def test_metrics_perfect_inverted_and_chance():
    torch.manual_seed(0)
    tgt = torch.rand(3, 4, 16, 16)
    good = depth_metrics(tgt * 2 + 1, tgt)
    assert good["mae"] < 1e-6 and good["order"] == 1.0 and good["flicker"] < 1e-6
    assert good["pearson"] == pytest.approx(1.0)
    assert good["pearson_per_clip"] == pytest.approx([1.0] * 3)
    flipped = depth_metrics(-tgt, tgt)
    assert flipped["order"] == 0.0
    assert flipped["mae"] > 0.2               # cannot be "fixed" by a negative scale
    assert flipped["pearson"] == pytest.approx(-1.0)
    assert 0.4 < depth_metrics(torch.rand_like(tgt), tgt)["order"] < 0.6


# ---------------------------------------------------------------- training
def test_default_size_is_within_spec():
    n = count_params(DepthBrain())
    assert 5e6 <= n <= 20e6, n
    assert DepthBrainConfig().base == 64
    with pytest.raises(ValueError, match="multiple of groups"):
        DepthBrainConfig(base=20)


def test_agreement_check_pass_and_every_failure_reason():
    good = {"val_clips": 60, "val_order": 0.93, "val_mae": 0.04,
            "val_flicker": 0.01, "val_pearson": 0.95}
    assert agreement_check(good, step=7) == {
        "passed": True, "reasons": [], "gate": AGREEMENT_GATE,
        "measured": good, "step": 7}
    bad = agreement_check({"val_clips": 10, "val_order": 0.7, "val_mae": 0.2,
                           "val_flicker": 0.05, "val_pearson": 0.5})
    assert not bad["passed"] and len(bad["reasons"]) == 5
    nothing = agreement_check({"val_clips": 0})
    assert not nothing["passed"] and "only 0 held-out" in nothing["reasons"][0]
    nan = agreement_check(dict(good, val_order=float("nan")))
    assert not nan["passed"]                   # NaN never passes
    assert not agreement_check(dict(good, val_pearson=float("nan")))["passed"]


def test_gate_keeps_every_v1_threshold_and_adds_pearson():
    assert AGREEMENT_GATE == {"min_val_clips": 50, "min_order": 0.85,
                              "max_mae": 0.08, "max_flicker": 0.03,
                              "min_pearson": 0.90}


def test_high_order_and_mae_but_low_pearson_is_refused():
    """Every v1 metric comfortably passes; Pearson alone fails the gate."""
    ev = {"val_clips": 120, "val_order": 0.97, "val_mae": 0.01,
          "val_flicker": 0.005, "val_pearson": 0.80}
    check = agreement_check(ev)
    assert not check["passed"]
    assert check["reasons"] == ["pearson 0.800 < 0.9"]
    assert agreement_check(dict(ev, val_pearson=0.90))["passed"]   # >= holds


def _ckpt_with_check(tmp_path, name, measured):
    """A real (tiny) DepthBrain checkpoint carrying a given agreement
    record for its own step."""
    data = _dataset(tmp_path / f"teach_{name}", n=4)
    tr = DepthBrainTrainer([data], ckpt_dir=str(tmp_path / "ck"), val_pct=0.0,
                           device="cpu", **SMALL)
    tr.train_step()
    tr.agreement = (agreement_check(measured, step=tr.step)
                    if "val_pearson" in measured else
                    {"passed": True, "reasons": [], "measured": measured,
                     "gate": {k: v for k, v in AGREEMENT_GATE.items()
                              if k != "min_pearson"}, "step": tr.step})
    path = str(tmp_path / "ck" / f"{name}.pt")
    tr.save(path)
    return path


@pytest.mark.parametrize("case", ["low_pearson", "no_pearson_v1_check"])
def test_prep_refuses_a_checkpoint_that_fails_only_on_pearson(
        tmp_path, monkeypatch, case):
    """(a) high order / low MAE but Pearson < 0.90 and (c) a v1 checkpoint
    whose check passed before Pearson existed: both refused by label_depth,
    estimate_depth and prep (before any download)."""
    import prepare_clips

    v1 = {"val_clips": 120, "val_order": 0.97, "val_mae": 0.01,
          "val_flicker": 0.005}
    measured = dict(v1, val_pearson=0.80) if case == "low_pearson" else v1
    ckpt = _ckpt_with_check(tmp_path, case, measured)
    want = ("pearson 0.800 < 0.9" if case == "low_pearson"
            else "no Pearson correlation recorded")
    depth_mod._BRAINS.clear()
    rgb, _ = _scene(np.random.default_rng(1))
    with pytest.raises(ValueError, match=want):
        estimate_depth(rgb, source="depthbrain", ckpt=ckpt)
    data = _dataset(tmp_path / "new", n=2, with_depth=False, seed=3)
    with pytest.raises(ValueError, match=want):
        label_dataset(data, source="depthbrain", ckpt=ckpt)

    def no_network(*a, **k):
        raise AssertionError("prep reached the download stage")

    monkeypatch.setattr(prepare_clips, "download", no_network)
    urls = tmp_path / "urls.txt"
    urls.write_text("https://example.invalid/a.mp4\n")
    monkeypatch.setattr(sys, "argv", [
        "prepare_clips.py", "--source", "urls", "--urls-file", str(urls),
        "--out", str(tmp_path / "out"), "--depth",
        "--depth-source", "depthbrain", "--depthbrain-ckpt", ckpt])
    with pytest.raises(SystemExit) as e:
        prepare_clips.main()
    assert "not passed its agreement check" in str(e.value.code)
    assert want in str(e.value.code)

    # The same weights with a Pearson >= 0.90 are accepted: Pearson was the
    # only reason for the refusals above.
    ok = _ckpt_with_check(tmp_path, "ok", dict(v1, val_pearson=0.95))
    depth_mod._BRAINS.clear()
    assert estimate_depth(rgb, source="depthbrain", ckpt=ok).shape == (9, 32, 32)


def test_depthbrain_learns_depth_from_pixels_and_passes_its_check(
        tmp_path, monkeypatch):
    data = _dataset(tmp_path / "d", n=40)
    tr = DepthBrainTrainer([data], ckpt_dir=str(tmp_path / "ck"),
                           total_steps=120, val_pct=20.0, device="cpu", **SMALL)
    assert tr.val_items and tr.train_items
    before = tr.evaluate()["val_order"]
    first = [tr.train_step()["loss"] for _ in range(10)]
    for _ in range(110):
        rec = tr.train_step()
    after = tr.evaluate()
    assert np.mean(first) > rec["loss"]
    assert after["val_order"] > 0.9 > before
    check = tr.final_check(FEW_CLIPS_GATE)
    assert check["passed"], check["reasons"]
    assert check["step"] == 120
    assert check["measured"]["val_pearson"] >= 0.9

    path = str(tmp_path / "ck" / "depthbrain.pt")
    tr.save(path)
    report = json.load(open(tr.write_report()))
    assert report["teachers"] == {"midas_small": 40}
    assert report["agreement"]["passed"] is True
    # (b) the report holds Pearson per held-out clip and as the mean.
    per = report["agreement"]["pearson_per_clip"]
    val_names = {f"d/{it[1]}" for it in tr.val_items}
    assert set(per) == val_names and len(per) == 7
    assert report["agreement"]["measured"]["val_pearson"] == pytest.approx(
        np.mean(list(per.values())))
    assert report["agreement"]["gate"]["min_pearson"] == 0.90

    # Prep re-scores the stored numbers against the OFFICIAL gate, so a pass
    # under a looser gate (7 held-out clips < 50) does not count ...
    with pytest.raises(ValueError, match="only 7 held-out clips"):
        estimate_depth(_scene(np.random.default_rng(5))[0],
                       source="depthbrain", ckpt=path)
    # ... it counts only where the official gate itself would pass.
    monkeypatch.setattr(brain_mod, "AGREEMENT_GATE", FEW_CLIPS_GATE)
    depth_mod._BRAINS.clear()

    # A passed checkpoint is accepted as-is: estimate_depth keeps the contract.
    brain = load_depthbrain(path)
    assert brain.step == 120 and not brain.training and brain.agreement["passed"]
    rgb, dep = _scene(np.random.default_rng(99))
    out = estimate_depth(rgb, source="depthbrain", ckpt=path)
    assert out.dtype == np.uint8 and out.shape == dep.shape
    assert out.min() == 0 and out.max() == 255
    corr = np.corrcoef(out.ravel().astype(float), dep.ravel().astype(float))[0, 1]
    assert corr > 0.8                          # nearer is larger, like MiDaS
    big = np.repeat(np.repeat(rgb, 2, axis=1), 2, axis=2)   # 64x64 clip
    assert estimate_depth(big, source="depthbrain", ckpt=path).shape == (9, 64, 64)
    assert depth_source_id("depthbrain", path) == "depthbrain:depthbrain.pt@step120"


def test_a_check_only_counts_for_the_weights_it_measured(tmp_path, monkeypatch):
    monkeypatch.setattr(brain_mod, "AGREEMENT_GATE",
                        dict(FEW_CLIPS_GATE, min_val_clips=1, min_order=0.0,
                             max_mae=1.0, max_flicker=1.0, min_pearson=-1.0))
    data = _dataset(tmp_path / "d", n=12)
    tr = DepthBrainTrainer([data], ckpt_dir=str(tmp_path / "ck"),
                           val_pct=30.0, device="cpu", **SMALL)
    tr.train_step()
    assert tr.final_check()["passed"]
    passed = str(tmp_path / "ck" / "passed.pt")
    tr.save(passed)
    assert depth_source_id("depthbrain", passed).endswith("@step1")

    tr.train_step()                            # weights moved on
    assert tr.agreement is None
    tr.save(str(tmp_path / "ck" / "moved.pt"))
    with pytest.raises(ValueError, match="no agreement check recorded"):
        depth_source_id("depthbrain", str(tmp_path / "ck" / "moved.pt"))

    # Resuming a passed checkpoint and training more drops the stale pass.
    res = DepthBrainTrainer.resume(passed)
    assert res.agreement["passed"]
    res.train_step()
    assert res.agreement is None
    # A hand-edited step mismatch is caught too.
    from vidgen.checkpoint import load_checkpoint, save_checkpoint

    payload = load_checkpoint(passed)
    payload["step"] = 99
    forged = str(tmp_path / "ck" / "forged.pt")
    save_checkpoint(forged, payload)
    with pytest.raises(ValueError, match="measured step 1, but these weights "
                                         "are step 99"):
        depth_source_id("depthbrain", forged)


def test_resume_refuses_to_move_the_held_out_split(tmp_path):
    data = _dataset(tmp_path / "d", n=6)
    tr = DepthBrainTrainer([data], ckpt_dir=str(tmp_path / "ck"),
                           device="cpu", val_pct=5.0, **SMALL)
    tr.train_step()
    path = str(tmp_path / "ck" / "latest.pt")
    tr.save(path)
    with pytest.raises(ValueError, match="differs from the run's 5.0"):
        DepthBrainTrainer.resume(path, val_pct=20.0)
    assert DepthBrainTrainer.resume(path, val_pct=5.0).step == 1


def test_resume_is_bit_identical(tmp_path):
    data = _dataset(tmp_path / "d", n=12)
    kw = dict(total_steps=8, device="cpu", **SMALL)
    a = DepthBrainTrainer([data], ckpt_dir=str(tmp_path / "a"), **kw)
    for _ in range(6):
        a.train_step()
    b = DepthBrainTrainer([data], ckpt_dir=str(tmp_path / "b"), **kw)
    for _ in range(3):
        b.train_step()
    path = str(tmp_path / "b" / "latest.pt")
    b.save(path)
    c = DepthBrainTrainer.resume(path)
    assert c.step == 3 and c.ckpt_dir == str(tmp_path / "b")
    for _ in range(3):
        c.train_step()
    for k, v in a.model.state_dict().items():
        assert torch.equal(v, c.model.state_dict()[k]), k
    assert torch.equal(a.ema.shadow["depthbrain"]["head.weight"],
                       c.ema.shadow["depthbrain"]["head.weight"])


def test_cli_trains_and_records_a_failed_check(tmp_path, capsys):
    from train_depthbrain import main

    data = _dataset(tmp_path / "d", n=8)
    ck = tmp_path / "ck"
    main(["--data", data, "--steps", "4", "--batch-size", "2", "--frames", "2",
          "--size", "32", "--base", "8", "--device", "cpu",
          "--ckpt-dir", str(ck), "--eval-every", "2", "--log-every", "2",
          "--log", str(ck / "train.jsonl")])
    brain = load_depthbrain(str(ck / "depthbrain.pt"))
    assert brain.step == 4
    assert brain.agreement["passed"] is False  # 8 clips, 4 steps: of course
    report = json.load(open(ck / "depthbrain_report.json"))
    assert report["step"] == 4 and report["agreement"]["passed"] is False
    assert "NOT passed" in capsys.readouterr().out


def test_training_needs_maps_and_warns_on_own_maps(tmp_path, capsys):
    bare = _dataset(tmp_path / "bare", n=4, with_depth=False)
    with pytest.raises(ValueError, match="no clips with depth maps"):
        DepthBrainTrainer([bare], device="cpu", **SMALL)
    own = _dataset(tmp_path / "own", n=4, teacher="depthbrain:x.pt@step5")
    DepthBrainTrainer([own], device="cpu", val_pct=0.0, **SMALL)
    assert "made by DepthBrain itself" in capsys.readouterr().err


def test_val_split_is_stable_and_disjoint(tmp_path):
    data = _dataset(tmp_path / "d", n=30)
    t1, v1, _ = collect_items([data], 20.0)
    t2, v2, _ = collect_items([data], 20.0)
    assert t1 == t2 and v1 == v2 and v1
    assert not {i[1] for i in t1} & {i[1] for i in v1}
    assert is_val("round1/clip_000001.npy", 100.0)
    assert not is_val("round1/clip_000001.npy", 0.0)


def test_one_source_video_never_lands_on_both_sides(tmp_path):
    """Two rounds holding clips of the same videos: each video's clips
    are all in training or all held out, never split across."""
    urls = [f"https://videos.example/{i % 10}.mp4" for i in range(20)]
    a = _dataset(tmp_path / "round1", n=20, urls=urls)
    b = _dataset(tmp_path / "round2", n=20, urls=urls[::-1], seed=1)
    train, val, _ = collect_items([a, b], 30.0)
    url_of = {}
    for d in (a, b):
        for line in open(os.path.join(d, "manifest.jsonl")):
            r = json.loads(line)
            url_of[(d, r["file"])] = r["source_url"]
    side_t = {url_of[(i[0], i[1])] for i in train}
    side_v = {url_of[(i[0], i[1])] for i in val}
    assert side_v and side_t and not side_t & side_v
    assert split_key("x", {"file": "f.npy"}) == "x/f.npy"
    assert split_key("x", {"file": "f.npy", "source_url": "u"}) == "url:u"


# ------------------------------------------------------------ prep / label
def _tiny_ckpt(tmp_path, data):
    """An UNVERIFIED DepthBrain (one step, no agreement check)."""
    tr = DepthBrainTrainer([data], ckpt_dir=str(tmp_path / "ck"),
                           device="cpu", val_pct=0.0, **SMALL)
    tr.train_step()
    path = str(tmp_path / "ck" / "depthbrain.pt")
    tr.save(path)
    return path


def test_midas_stays_the_default_everywhere():
    import inspect

    assert depth_mod.DEFAULT_DEPTH_SOURCE == "midas"
    assert inspect.signature(estimate_depth).parameters["source"].default == "midas"
    assert inspect.signature(label_dataset).parameters["source"].default == "midas"
    assert depth_source_id() == "midas_small"   # no download just to name it


def test_unverified_depthbrain_is_refused_unless_overridden(tmp_path):
    ckpt = _tiny_ckpt(tmp_path, _dataset(tmp_path / "teach", n=4))
    data = _dataset(tmp_path / "new", n=3, with_depth=False, seed=3)
    with pytest.raises(ValueError, match="not passed its agreement check"):
        label_dataset(data, source="depthbrain", ckpt=ckpt)
    assert not any(n.endswith("_depth.npy") for n in os.listdir(data))
    rgb, _ = _scene(np.random.default_rng(1))
    with pytest.raises(ValueError, match="no agreement check recorded"):
        estimate_depth(rgb, source="depthbrain", ckpt=ckpt)
    out = label_dataset(data, source="depthbrain", ckpt=ckpt,
                        allow_unverified=True)
    assert out["source"] == "depthbrain:depthbrain.pt@step1,unverified"


def test_label_depth_adds_maps_then_skips(tmp_path):
    ckpt = _tiny_ckpt(tmp_path, _dataset(tmp_path / "teach", n=4))
    data = _dataset(tmp_path / "new", n=5, with_depth=False, seed=3)
    kw = dict(source="depthbrain", ckpt=ckpt, allow_unverified=True)
    out = label_dataset(data, flush_every=2, **kw)
    assert out == {"labelled": 5, "skipped": 0,
                   "source": "depthbrain:depthbrain.pt@step1,unverified"}
    recs = [json.loads(l) for l in open(os.path.join(data, "manifest.jsonl"))]
    for r in recs:
        assert r["depth_source"] == out["source"]
        d = np.load(os.path.join(data, r["depth"]))
        assert d.dtype == np.uint8 and d.shape == (9, 32, 32)
    assert label_dataset(data, **kw)["skipped"] == 5
    assert label_dataset(data, overwrite=True, limit=2, **kw)["labelled"] == 2


def test_missing_checkpoint_fails_fast_with_guidance(tmp_path, monkeypatch):
    monkeypatch.delenv(depth_mod.DEPTHBRAIN_ENV, raising=False)
    monkeypatch.chdir(tmp_path)                # no default checkpoint here
    with pytest.raises(FileNotFoundError, match="train_depthbrain.py"):
        depth_source_id("depthbrain")
    data = _dataset(tmp_path / "d", n=2, with_depth=False)
    with pytest.raises(FileNotFoundError):
        label_dataset(data, source="depthbrain")
    assert json.loads(open(os.path.join(data, "manifest.jsonl"))
                      .readline()).get("depth") is None


def test_env_var_picks_the_checkpoint(tmp_path, monkeypatch):
    ckpt = _tiny_ckpt(tmp_path, _dataset(tmp_path / "teach", n=4))
    monkeypatch.setenv(depth_mod.DEPTHBRAIN_ENV, ckpt)
    assert depth_mod.depthbrain_path() == ckpt
    rgb, _ = _scene(np.random.default_rng(1))
    assert estimate_depth(rgb, source="depthbrain",
                          allow_unverified=True).shape == (9, 32, 32)


def test_unknown_source_and_bad_checkpoint_kind(tmp_path):
    rgb, _ = _scene(np.random.default_rng(1))
    with pytest.raises(ValueError, match="unknown depth source"):
        estimate_depth(rgb, source="zoe")
    with pytest.raises(ValueError, match="unknown depth source"):
        depth_source_id("zoe")
    from vidgen.checkpoint import save_checkpoint

    bad = str(tmp_path / "vae.pt")
    save_checkpoint(bad, {"kind": "vae", "model": {}})
    with pytest.raises(ValueError, match="not a DepthBrain checkpoint"):
        load_depthbrain(bad)


def _run_prep(tmp_path, monkeypatch, *extra):
    import prepare_clips

    def no_network(*a, **k):
        raise RuntimeError("offline test: no downloads")

    monkeypatch.setattr(prepare_clips, "download", no_network)
    urls = tmp_path / "urls.txt"
    urls.write_text("https://example.invalid/a.mp4\n")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prepare_clips.py", "--source", "urls", "--urls-file", str(urls),
        "--out", str(out), "--no-contact-sheet", "--depth", *extra])
    prepare_clips.main()
    return json.load(open(out / "prep_report.json"))


def test_prep_depth_defaults_to_midas(tmp_path, monkeypatch):
    def brain_used(*a, **k):
        raise AssertionError("prep touched DepthBrain without being asked")

    monkeypatch.setattr(depth_mod, "load_depthbrain_cached", brain_used)
    report = _run_prep(tmp_path, monkeypatch)
    assert report["config"]["depth_source"] == "midas_small"
    assert report["rejection_reasons"] == {"download_error": 1}


@pytest.mark.parametrize("trained", [False, True])
def test_prep_refuses_missing_or_unverified_depthbrain_before_downloading(
        tmp_path, monkeypatch, trained):
    import prepare_clips

    ckpt = (_tiny_ckpt(tmp_path, _dataset(tmp_path / "teach", n=4))
            if trained else str(tmp_path / "missing.pt"))

    def no_network(*a, **k):
        raise AssertionError("prep reached the download stage")

    monkeypatch.setattr(prepare_clips, "download", no_network)
    urls = tmp_path / "urls.txt"
    urls.write_text("https://example.invalid/a.mp4\n")
    monkeypatch.setattr(sys, "argv", [
        "prepare_clips.py", "--source", "urls", "--urls-file", str(urls),
        "--out", str(tmp_path / "out"), "--depth",
        "--depth-source", "depthbrain", "--depthbrain-ckpt", ckpt])
    with pytest.raises(SystemExit) as e:
        prepare_clips.main()
    want = ("not passed its agreement check" if trained
            else "No DepthBrain checkpoint")
    assert want in str(e.value.code)
