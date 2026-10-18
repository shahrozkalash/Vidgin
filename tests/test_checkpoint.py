"""Part 7: checkpoint save/resume reproduces the next-step loss bit-for-bit."""

import torch

from vidgen.train import DiTTrainer
from vidgen.train_vae import VAETrainer


def _losses(trainer_cls, steps, **kw):
    tr = trainer_cls(**kw)
    losses = [tr.train_step()["loss"] for _ in range(steps)]
    return tr, losses


def test_dit_checkpoint_resume_reproducibility(tmp_path):
    kw = dict(preset="tiny", seed=0, batch_size=2, num_clips=8,
              ckpt_dir=str(tmp_path))
    a, losses_a = _losses(DiTTrainer, 4, **kw)
    ckpt = str(tmp_path / "step2.pt")
    # rebuild: train 2 steps fresh, save, continue 2 more
    b, losses_b = _losses(DiTTrainer, 2, **kw)
    b.save(ckpt)
    losses_b += [b.train_step()["loss"] for _ in range(2)]
    # resume from the checkpoint and run the same 2 steps
    c = DiTTrainer.resume(ckpt, **{k: v for k, v in kw.items()})
    losses_c = [c.train_step()["loss"] for _ in range(2)]
    assert losses_b[2] == losses_c[0], (losses_b[2], losses_c[0])
    assert losses_b[3] == losses_c[1], (losses_b[3], losses_c[1])
    # and both match the uninterrupted run
    assert losses_a[2] == losses_c[0]
    assert losses_a[3] == losses_c[1]


def test_vae_checkpoint_resume_reproducibility(tmp_path):
    kw = dict(preset="tiny", seed=0, batch_size=2, num_clips=8,
              ckpt_dir=str(tmp_path))
    a, losses_a = _losses(VAETrainer, 4, **kw)
    ckpt = str(tmp_path / "vae_step2.pt")
    b, losses_b = _losses(VAETrainer, 2, **kw)
    b.save(ckpt)
    losses_b += [b.train_step()["loss"] for _ in range(2)]
    c = VAETrainer.resume(ckpt, **{k: v for k, v in kw.items()})
    losses_c = [c.train_step()["loss"] for _ in range(2)]
    assert losses_a[2] == losses_c[0]
    assert losses_a[3] == losses_c[1]


def test_no_cuda_in_checkpoints(tmp_path):
    tr = DiTTrainer(preset="tiny", seed=0, batch_size=1, num_clips=4,
                    ckpt_dir=str(tmp_path))
    p = str(tmp_path / "x.pt")
    tr.save(p)
    payload = torch.load(p, map_location="cpu", weights_only=False)
    assert payload["kind"] == "dit"
    assert payload["rng"]["torch_cuda"] is None
