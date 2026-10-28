"""Hardening pass (2026-10-28) regression tests.

1. fp16 GradScaler + NonFiniteGuard: the first gradient overflow (routine
   in fp16 training) was skipped without ``scaler.update()``, so the next
   ``unscale_()`` raised and every --amp fp16 CUDA run died.
2. Fully-masked text rows (the CFG null embedding): torch < 2.5 returns
   NaN from SDPA for them; the DiT now yields the same zeros on every torch.
3. DiT checkpoints never recorded their VAE, so sampling/resume without
   --vae-ckpt silently used a randomly initialized VAE.
4. save_checkpoint fsyncs before the atomic rename.
"""
import os

import pytest
import torch
import torch.nn.functional as F

from vidgen.checkpoint import load_checkpoint, save_checkpoint
from vidgen.config import get_preset
from vidgen.dit.model import DiT
from vidgen.train import DiTTrainer
from vidgen.train_utils import NonFiniteGuard, guarded_optimizer_step
from vidgen.vae.model import CausalVAE


# ------------------------------------------------------------ 1. GradScaler
def _scaled_step(p, opt, scaler, guard, mult):
    opt.zero_grad()
    loss = (p * mult).sum()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_([p], 1.0)
    return guarded_optimizer_step(opt, scaler, guard, float(loss.detach()),
                                  float(gn))


def test_scaler_overflow_does_not_wedge_training():
    p = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.SGD([p], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** 16)
    guard = NonFiniteGuard(max_bad=2)
    before = p.detach().clone()
    # 1e38 * 65536 overflows fp32: the scaler must skip and back off.
    assert _scaled_step(p, opt, scaler, guard, 1e38) == "skipped_overflow"
    assert torch.equal(p.detach(), before)
    assert scaler.get_scale() < 2.0 ** 16
    # Previously this raised "unscale_() has already been called ...".
    for _ in range(4):  # more than max_bad: overflow is not a guard failure
        assert _scaled_step(p, opt, scaler, guard, 1.0) == "stepped"
    assert not torch.equal(p.detach(), before)


def test_scaler_nonfinite_loss_is_skipped_and_scaler_stays_usable():
    p = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.SGD([p], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** 4)
    guard = NonFiniteGuard()
    assert _scaled_step(p, opt, scaler, guard, float("nan")) \
        == "skipped_nonfinite"
    assert _scaled_step(p, opt, scaler, guard, 1.0) == "stepped"


def test_no_scaler_nonfinite_grad_is_skipped():
    p = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.SGD([p], lr=0.1)
    guard = NonFiniteGuard()
    p.grad = torch.full((3,), float("inf"))
    assert guarded_optimizer_step(opt, None, guard, 1.0, float("inf")) \
        == "skipped_nonfinite"
    assert torch.equal(p.detach(), torch.ones(3))
    assert p.grad is None or not p.grad.isinf().any()


def test_dit_trainer_runs_with_grad_scaler(tmp_path):
    tr = DiTTrainer(preset="tiny", seed=0, batch_size=1, num_clips=4,
                    ckpt_dir=str(tmp_path))
    tr.scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** 16)
    for _ in range(3):
        rec = tr.train_step()
    assert rec["step"] == 2


# ------------------------------------------------------- 2. masked text rows
def _old_torch_sdpa(real):
    """Emulate torch < 2.5: fully-masked attention rows come back NaN."""
    def sdpa(q, k, v, attn_mask=None, **kw):
        out = real(q, k, v, attn_mask=attn_mask, **kw)
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            dead = ~attn_mask.any(dim=-1, keepdim=True)  # [B,1,N,1]
            out = out.masked_fill(dead, float("nan"))
        return out
    return sdpa


def test_null_text_rows_finite_on_old_torch(monkeypatch):
    torch.manual_seed(0)
    cfg = get_preset("tiny")
    dit = DiT(cfg).eval()
    # Non-zero modulation so the cross-attention branch reaches the output.
    torch.nn.init.normal_(dit.adaLN_shared.weight, std=0.02)
    lat = torch.randn(2, cfg.latent_channels, cfg.latent_t, cfg.latent_h,
                      cfg.latent_w)
    t = torch.rand(2)
    text = torch.randn(2, 4, cfg.text_dim)
    mask = torch.tensor([[True, True, False, False],
                         [False, False, False, False]])  # row 1 = null text
    with torch.no_grad():
        ref = dit(lat, t, text, mask)
        monkeypatch.setattr(F, "scaled_dot_product_attention",
                            _old_torch_sdpa(F.scaled_dot_product_attention))
        old = dit(lat, t, text, mask)
    assert torch.isfinite(old).all()
    assert torch.equal(ref, old)


# --------------------------------------------------------- 3. VAE provenance
def _vae_ckpt(path):
    torch.manual_seed(123)
    save_checkpoint(str(path), {"kind": "vae",
                                "vae": CausalVAE().state_dict()})
    return str(path)


def test_dit_checkpoint_records_vae_and_resume_reuses_it(tmp_path):
    vae_path = _vae_ckpt(tmp_path / "vae.pt")
    kw = dict(preset="tiny", seed=0, batch_size=1, num_clips=4,
              ckpt_dir=str(tmp_path))
    tr = DiTTrainer(vae_ckpt=vae_path, **kw)
    tr.train_step()
    ckpt = str(tmp_path / "dit.pt")
    tr.save(ckpt)
    assert load_checkpoint(ckpt)["vae_ckpt"] == vae_path

    # Resume WITHOUT --vae-ckpt: the stored VAE is used, not a random one.
    res = DiTTrainer.resume(ckpt, **kw)
    assert res.vae_ckpt == vae_path
    for a, b in zip(tr.vae.state_dict().values(),
                    res.vae.state_dict().values()):
        assert torch.equal(a, b)

    # A stored VAE that vanished is a loud error, not a silent fallback.
    os.remove(vae_path)
    with pytest.raises(FileNotFoundError, match="--vae-ckpt"):
        DiTTrainer.resume(ckpt, **kw)


def test_resume_without_stored_vae_is_unchanged(tmp_path):
    kw = dict(preset="tiny", seed=0, batch_size=1, num_clips=4,
              ckpt_dir=str(tmp_path))
    tr = DiTTrainer(**kw)
    ckpt = str(tmp_path / "dit.pt")
    tr.save(ckpt)
    assert load_checkpoint(ckpt)["vae_ckpt"] is None
    assert DiTTrainer.resume(ckpt, **kw).vae_ckpt is None


# ------------------------------------------------- 3b. pooled metadata kept
def _worker_ckpts(tmp_path, tok, n=2):
    from vidgen.pool import load_payload_file

    kw = dict(preset="tiny", seed=0, batch_size=1, num_clips=4,
              ckpt_dir=str(tmp_path), tokenizer=tok)
    paths = []
    for i in range(n):
        p = str(tmp_path / f"w{i}.pt")
        DiTTrainer(**kw).save(p)
        paths.append(p)
    return kw, [load_payload_file(p) for p in paths], paths


def test_pooled_caption_checkpoint_keeps_tokenizer(tmp_path):
    from vidgen.pool import merge_payloads
    from vidgen.text.tokenizer import CaptionTokenizer

    tok = CaptionTokenizer.train(["a red ball", "a blue cube moves left"])
    kw, payloads, paths = _worker_ckpts(tmp_path, tok)
    pooled = merge_payloads(payloads, input_names=paths)
    assert pooled["tokenizer"] == tok.to_dict()
    p = str(tmp_path / "pooled.pt")
    save_checkpoint(p, pooled)
    # Previously: "trained in legacy hash mode" — the round could not start.
    DiTTrainer(**kw).load_dit_weights(p)


def test_pool_rejects_mismatched_tokenizers(tmp_path):
    from vidgen.pool import merge_payloads
    from vidgen.text.tokenizer import CaptionTokenizer

    tok = CaptionTokenizer.train(["a red ball"])
    _, payloads, paths = _worker_ckpts(tmp_path, tok)
    payloads[1]["tokenizer"] = CaptionTokenizer.train(["green tree"]).to_dict()
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        merge_payloads(payloads, input_names=paths)


# ------------------------------------------------------------ 4. durability
def test_save_checkpoint_fsyncs_before_rename(tmp_path, monkeypatch):
    calls = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync",
                        lambda fd: (calls.append("fsync"), real_fsync(fd)))
    monkeypatch.setattr(os, "replace",
                        lambda a, b: (calls.append("replace"),
                                      real_replace(a, b)))
    path = str(tmp_path / "c.pt")
    save_checkpoint(path, {"kind": "x", "w": torch.ones(2)})
    assert calls == ["fsync", "replace"]
    assert not os.path.exists(path + ".tmp")
    assert torch.equal(load_checkpoint(path)["w"], torch.ones(2))
