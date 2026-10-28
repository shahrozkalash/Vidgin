"""Image-to-video (vidgen/i2v.py) and timed voice dialogue for videos
(vidgen/audio/dialogue.py, scripts/add_audio.py, sample.py flags)."""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from vidgen.audio.dialogue import (DialogueLine, build_speech_track,
                                   describe_report, load_dialogue_file,
                                   load_voice_synthesizer,
                                   parse_dialogue_line)
from vidgen.checkpoint import save_checkpoint
from vidgen.config import get_preset
from vidgen.dit.model import DiT
from vidgen.extend import generate_long
from vidgen.i2v import generate_from_image, image_latent, load_image
from vidgen.sampler import sample_latents
from vidgen.text.encoder import build_text_encoder
from vidgen.vae.model import CausalVAE

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 22050


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    cfg = get_preset("tiny")
    return (cfg, DiT(cfg).eval(), CausalVAE().eval(),
            build_text_encoder(cfg).eval())


def _image(cfg, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (cfg.height, cfg.width, 3), dtype=np.uint8)


# ------------------------------------------------------------ image-to-video
def test_image_latent_is_exactly_the_videos_first_latent_frame(models):
    cfg, _, vae, _ = models
    img = _image(cfg)
    video = torch.rand(1, 3, cfg.frames, cfg.height, cfg.width) * 2 - 1
    video[:, :, 0] = (torch.from_numpy(img).float() / 127.5 - 1).permute(2, 0, 1)
    with torch.no_grad():
        mu_video, _ = vae.encode(video)
    lat = image_latent(vae, img, cfg)
    assert lat.shape == (1, cfg.latent_channels, 1, cfg.latent_h, cfg.latent_w)
    # Equal up to float rounding (~1e-7, varies with the CPU thread count).
    torch.testing.assert_close(lat, mu_video[:, :, :1] * cfg.latent_scale,
                               atol=1e-6, rtol=0)


def test_sampling_ends_exactly_on_the_image_latent(models):
    cfg, dit, vae, text_enc = models
    prefix = image_latent(vae, _image(cfg), cfg)
    text, mask = text_enc(["a ball rolls"])
    null, null_mask = text_enc.null_embedding()
    shape = (1, cfg.latent_channels, cfg.latent_t, cfg.latent_h, cfg.latent_w)
    z = sample_latents(dit, shape, text, mask, null, null_mask, num_steps=3,
                       prefix_latent=prefix, prefix_t=1)
    assert torch.equal(z[:, :, :1], prefix)


def test_generate_from_image_shape_and_first_frame(models):
    cfg, dit, vae, text_enc = models
    img = _image(cfg, seed=1)
    frames = generate_from_image(dit, vae, text_enc, cfg, img, "a ball",
                                 steps=2)
    assert frames.shape == (cfg.frames, cfg.height, cfg.width, 3)
    # Frame 0 is the VAE's reconstruction of the image. The decoder's
    # normalization spans the window, so allow ~1 grey level (2/255).
    with torch.no_grad():
        recon = vae.decode(image_latent(vae, img, cfg) / cfg.latent_scale)
    np.testing.assert_allclose(frames[0], recon[0, :, 0].permute(1, 2, 0).numpy(),
                               atol=2 / 255)


def test_long_video_can_start_from_an_image(models):
    cfg, dit, vae, text_enc = models
    prefix = image_latent(vae, _image(cfg), cfg)
    frames = generate_long(dit, vae, text_enc, cfg, "a ball", segments=2,
                           steps=2, first_prefix=prefix)
    assert frames.shape[0] > cfg.frames and np.isfinite(frames).all()


def test_load_image_center_crops_without_stretching(tmp_path):
    from PIL import Image

    wide = np.zeros((100, 300, 3), dtype=np.uint8)
    wide[:, 100:200] = (255, 0, 0)          # red square in the middle
    path = str(tmp_path / "wide.png")
    Image.fromarray(wide).save(path)
    img = load_image(path, 64, 64)
    assert img.shape == (64, 64, 3) and img.dtype == np.uint8
    # The 1:1 center crop is the red square, not the black sides.
    assert img[:, :, 0].mean() > 240 and img[:, :, 1].mean() < 15


def test_image_latent_rejects_wrong_size(models):
    cfg, _, vae, _ = models
    with pytest.raises(ValueError, match="load_image"):
        image_latent(vae, np.zeros((10, 10, 3), np.uint8), cfg)


# ------------------------------------------------------------------ dialogue
def test_parse_dialogue_line():
    assert parse_dialogue_line("1.5: Where are we going?") == \
        DialogueLine(1.5, "Where are we going?")
    for bad in ("no time here", "-1: negative", "x: nan", "2:   "):
        with pytest.raises(ValueError):
            parse_dialogue_line(bad)


def test_dialogue_file(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("# scene 1\n0.5: Hello\n\n2: Bye\n", encoding="utf-8")
    assert load_dialogue_file(str(p)) == [DialogueLine(0.5, "Hello"),
                                          DialogueLine(2.0, "Bye")]
    p.write_text("0.5 Hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":1:"):
        load_dialogue_file(str(p))


def _fake_synth(sec_per_char=0.05):
    """Waveform of len(text) * sec_per_char seconds, scaled by duration."""
    calls = []

    def synth(text, scale):
        calls.append((text, scale))
        n = int(round(len(text) * sec_per_char * scale * SR))
        return np.full(n, 0.5, dtype=np.float32)
    synth.calls = calls
    return synth


def test_lines_are_placed_at_their_times():
    synth = _fake_synth()
    track, rep = build_speech_track(
        [DialogueLine(2.0, "b" * 10), DialogueLine(0.5, "a" * 10)],
        synth, target_duration=5.0)
    assert len(track) == 5 * SR
    assert [r["start"] for r in rep] == [0.5, 2.0]       # sorted by time
    assert all(r["speed_scale"] == 1.0 and r["truncated_sec"] == 0 for r in rep)
    assert track[int(0.5 * SR)] == 0.5 and track[int(0.4 * SR)] == 0
    assert describe_report(rep) == []


def test_overlong_line_is_sped_up_not_cut_for_the_next_line():
    synth = _fake_synth()
    # 40 chars = 2.0 s of speech in a 1.0 s slot: sped to x0.6 = 1.2 s,
    # then it FINISHES and the next line is pushed back instead.
    _, rep = build_speech_track(
        [DialogueLine(0.0, "x" * 40), DialogueLine(1.15, "y")],
        synth, target_duration=3.0, min_speed_scale=0.6)
    assert rep[0]["speed_scale"] == 0.6 and rep[0]["truncated_sec"] == 0
    assert rep[0]["end"] == pytest.approx(1.2, abs=0.01)
    assert synth.calls[1] == ("x" * 40, 0.6)
    assert rep[1]["shifted"] and rep[1]["start"] > rep[0]["end"]
    assert any("faster" in n for n in describe_report(rep))


def test_only_the_video_end_cuts_speech():
    synth = _fake_synth()
    # 2.0 s line at 2.5 s in a 3.0 s video: x0.6 -> 1.2 s, last 0.7 s cut.
    _, rep = build_speech_track([DialogueLine(2.5, "x" * 40)], synth,
                                target_duration=3.0, min_speed_scale=0.6)
    assert rep[0]["speed_scale"] == 0.6
    assert rep[0]["truncated_sec"] == pytest.approx(0.7, abs=0.01)
    assert any("end of the video" in n for n in describe_report(rep))


def test_overlapping_line_is_shifted_and_late_line_skipped():
    synth = _fake_synth()
    _, rep = build_speech_track(
        [DialogueLine(0.0, "a" * 20), DialogueLine(0.2, "b" * 4),
         DialogueLine(9.0, "late")],
        synth, target_duration=3.0, min_speed_scale=1.0)
    assert rep[1]["shifted"] and rep[1]["start"] >= rep[0]["end"]
    assert rep[2]["skipped"]
    assert any("SKIPPED" in n for n in describe_report(rep))


def test_voice_synthesizer_from_checkpoints(tmp_path):
    from vidgen.audio.vocoder import HiFiGANGenerator, VocoderConfig
    from vidgen.audio.voice_model import VidginVoiceTTS, VoiceModelConfig

    torch.manual_seed(0)
    vcfg, gcfg = VoiceModelConfig(), VocoderConfig()
    voice = str(tmp_path / "voice.pt")
    vocoder = str(tmp_path / "vocoder.pt")
    save_checkpoint(voice, {"target": "voice", "cfg": vcfg,
                            "model": VidginVoiceTTS(vcfg).state_dict()})
    save_checkpoint(vocoder, {"target": "vocoder", "cfg": gcfg,
                              "generator": HiFiGANGenerator(gcfg).state_dict()})
    synth = load_voice_synthesizer(voice, vocoder)
    wav = synth("hello there friend", 1.0)
    assert wav.ndim == 1 and len(wav) > 0 and np.isfinite(wav).all()
    track, rep = build_speech_track([DialogueLine(0.1, "hello there")],
                                    synth, target_duration=30.0)
    assert not rep[0]["skipped"] and np.abs(track).max() > 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None
                    or shutil.which("ffprobe") is None,
                    reason="ffmpeg/ffprobe not installed")
def test_add_audio_voice_only_end_to_end(tmp_path):
    sys.path.insert(0, REPO_ROOT)
    import scripts.add_audio as add_audio
    from vidgen.audio.vocoder import HiFiGANGenerator, VocoderConfig
    from vidgen.audio.voice_model import VidginVoiceTTS, VoiceModelConfig
    from vidgen.video_io import write_mp4

    vcfg, gcfg = VoiceModelConfig(), VocoderConfig()
    voice = str(tmp_path / "voice.pt")
    vocoder = str(tmp_path / "vocoder.pt")
    save_checkpoint(voice, {"target": "voice", "cfg": vcfg,
                            "model": VidginVoiceTTS(vcfg).state_dict()})
    save_checkpoint(vocoder, {"target": "vocoder", "cfg": gcfg,
                              "generator": HiFiGANGenerator(gcfg).state_dict()})
    video = write_mp4(np.zeros((32, 64, 64, 3), np.float32),
                      str(tmp_path / "v.mp4"), fps=16)
    meta = add_audio.add_audio_to_video(
        video, str(tmp_path / "out.mp4"), "a street",
        [DialogueLine(0.2, "hello")], voice_ckpt=voice,
        vocoder_ckpt=vocoder, ambience=False)
    assert meta["audio_streams"] == 1 and len(meta["dialogue"]) == 1


# ------------------------------------------------------------- sample.py CLI
def _tiny_ckpt(tmp_path):
    from vidgen.train import DiTTrainer

    tr = DiTTrainer(preset="tiny", seed=0, batch_size=1, num_clips=2,
                    ckpt_dir=str(tmp_path))
    path = str(tmp_path / "dit.pt")
    tr.save(path)
    return path


def _run_sample(*args):
    return subprocess.run([sys.executable, "-m", "vidgen.sample", *args],
                          cwd=REPO_ROOT, capture_output=True, text=True,
                          timeout=600)


def test_sample_cli_image_to_video(tmp_path):
    from PIL import Image

    img = str(tmp_path / "start.png")
    Image.fromarray(_image(get_preset("tiny"))).save(img)
    out = str(tmp_path / "clip")
    r = _run_sample("--ckpt", _tiny_ckpt(tmp_path), "--image", img,
                    "--steps", "2", "--outdir", out)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "image-to-video" in r.stdout
    assert np.load(os.path.join(out, "frames.npy")).shape[0] == 17


def test_sample_cli_bad_dialogue_fails_before_generating(tmp_path):
    r = _run_sample("--ckpt", _tiny_ckpt(tmp_path), "--line", "oops no time",
                    "--outdir", str(tmp_path / "clip"))
    assert r.returncode != 0 and "START: TEXT" in r.stderr
    assert not os.path.exists(tmp_path / "clip" / "frames.npy")
