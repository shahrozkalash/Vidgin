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
    # The real invariant is device-portability of what save_checkpoint stores:
    # the RNG snapshot must contain no CUDA tensors, on GPU machines and off.
    # (torch.load(map_location="cpu") masks on-disk CUDA tensors anyway, so the
    # snapshot itself is the right thing to assert on.)
    from vidgen.seed import get_rng_states

    tr = DiTTrainer(preset="tiny", seed=0, batch_size=1, num_clips=4,
                    ckpt_dir=str(tmp_path))
    p = str(tmp_path / "x.pt")
    tr.save(p)
    payload = torch.load(p, map_location="cpu", weights_only=False)
    assert payload["kind"] == "dit"

    states = get_rng_states()
    assert isinstance(states["torch"], torch.Tensor)
    assert states["torch"].device.type == "cpu"
    cuda_states = states["torch_cuda"]
    if torch.cuda.is_available():
        assert isinstance(cuda_states, list) and len(cuda_states) > 0
        for s in cuda_states:
            assert isinstance(s, torch.Tensor)
            assert s.device.type == "cpu", f"CUDA RNG state not on CPU: {s.device}"
    else:
        assert cuda_states is None
