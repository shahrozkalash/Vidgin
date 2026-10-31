"""GitHub Release weights-only assets must warm-start the trainers.

The Round 1 release (v0.1.0-round1-tiny) ships fp16 weights-only exports
whose payload kind is 'vae-weights-fp16', not the full-checkpoint kind
'vae'. The first Stage 2 launch on Kaggle (2026-10-23) crashed on exactly
this: load_vae_weights rejected the release asset. These tests pin the
acceptance so the published assets stay usable as warm-start donors,
while unrelated kinds are still rejected loudly. (Since the
2026-10-31 VAE growth, the tiny-shaped asset reaches the small
preset through the vidgen.vae.grow transplant — also pinned below.)
"""

import pytest
import torch

from vidgen.train_vae import VAETrainer


def _release_style_asset(tmp_path):
    donor = VAETrainer(preset="tiny", seed=0, num_clips=8, batch_size=2,
                       ckpt_dir=None)
    donor.train_step()
    sd16 = {k: v.detach().clone().half()
            for k, v in donor.vae.state_dict().items()}
    path = str(tmp_path / "vae_weights_fp16.pt")
    torch.save({"kind": "vae-weights-fp16", "vae": sd16,
                "preset": "tiny", "step": 4000}, path)
    expected = {k: v.float() for k, v in sd16.items()}
    return path, expected


def test_warmstart_accepts_release_fp16_asset(tmp_path):
    path, expected = _release_style_asset(tmp_path)
    tr = VAETrainer(preset="small", seed=7, num_clips=4, batch_size=2,
                    ckpt_dir=None)
    # The asset is a 16/32 tiny VAE; since the 2026-10-31 growth the
    # small preset's VAE is 32/64, so the direct strict load refuses —
    # the documented path is to transplant the asset first
    # (vidgen.vae.grow), after which it warm-starts as before.
    with pytest.raises(RuntimeError):
        tr.load_vae_weights(path)
    from vidgen.vae.grow import grow_vae_checkpoint_file, \
        grow_vae_state_dict
    grown_path = str(tmp_path / "grown_fp16.pt")
    grow_vae_checkpoint_file(path, grown_path, "small")
    grown_expected, _ = grow_vae_state_dict(expected)
    tr.load_vae_weights(grown_path)
    after = tr.vae.state_dict()
    assert set(after) == set(grown_expected)
    for k in after:
        assert torch.allclose(after[k].float(),
                              grown_expected[k].float(), atol=1e-3), k
    # still a fresh run, not a resume
    assert tr.step == 0
    assert tr.init_from == grown_path


def test_warmstart_still_rejects_wrong_kind(tmp_path):
    path, _ = _release_style_asset(tmp_path)
    payload = torch.load(path, weights_only=False)
    payload["kind"] = "dit-weights-fp16"
    bad = str(tmp_path / "bad.pt")
    torch.save(payload, bad)
    tr = VAETrainer(preset="tiny", seed=1, num_clips=4, batch_size=2,
                    ckpt_dir=None)
    with pytest.raises(ValueError):
        tr.load_vae_weights(bad)
