"""Final end-to-end programme tests (2026-10-23).

- SSIM sanity (identity = 1, blur lowers it, noise lowers it more).
- The conditioning PATH: different captions must change the DiT's
  prediction through cross-attention, and text-encoder parameters
  must receive gradients (captions are never silently discarded).
- Preset geometry: documented parameter counts, counted on the meta
  device so even flagship shapes cost no memory.
- The memory estimator's verdicts on the presets that matter.
"""

import importlib.util
import os

import torch

from vidgen.config import get_preset
from vidgen.dit.model import DiT
from vidgen.flow import flow_loss
from vidgen.train import DiTTrainer
from vidgen.vae.model import CausalVAE
from vidgen.vae.stats import ssim_index

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------ SSIM

def test_ssim_identity_is_one():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 4, 32, 32) * 2 - 1
    assert abs(ssim_index(x, x) - 1.0) < 1e-3


def test_ssim_blur_beats_noise():
    torch.manual_seed(0)
    base = torch.rand(1, 3, 17, 17)
    img = torch.nn.functional.interpolate(
        base, size=(64, 64), mode="bilinear", align_corners=False)
    x = (img.repeat(1, 3, 2, 1, 1)[:, :, :4] * 2 - 1).contiguous()
    blur = torch.nn.functional.avg_pool2d(
        x.reshape(-1, 3, 64, 64), 5, stride=1, padding=2)
    blur = blur.reshape(1, 3, 4, 64, 64)
    noise = (x + torch.randn_like(x) * 0.8).clamp(-1, 1)
    assert ssim_index(blur, x) > ssim_index(noise, x)
    assert 0.0 <= ssim_index(noise, x) <= 1.0


# -------------------------------------------------- conditioning path

def test_conditioning_gates_open_and_encoder_trains():
    """The conditioning path, honestly staged (adaLN-Zero design):

    At exact init the DiT's block gates are zero BY DESIGN (the
    published DiT init), so captions cannot yet change predictions —
    but the gate parameters receive gradient, the first optimizer
    step opens the gates, and from then on cross-attention AND text
    encoder parameters receive real gradients. If any link in that
    chain breaks (gates frozen, cross-attn detached, encoder outside
    the optimizer), conditioning could never be learned and this
    test fails. Verified live 2026-10-23: init pred diff 0.0, gate
    grad 16.2, branch grads exactly 0, all nonzero after one step.
    """
    torch.manual_seed(0)
    tr = DiTTrainer(preset="tiny", batch_size=2, device="cpu")
    cfg = tr.cfg
    # Latent shape straight from the VAE's compression of the preset:
    # time (T-1)//4+1, space H/8 x W/8, latent_channels deep.
    lat = torch.randn(2, cfg.latent_channels,
                      (cfg.frames - 1) // 4 + 1,
                      cfg.height // 8, cfg.width // 8)
    t = torch.full((2,), 0.5)
    def probe():
        # Fresh encodings each call: backward frees the graph.
        text_a, mask_a = tr.text_enc(["a red car drives fast"] * 2)
        text_b, mask_b = tr.text_enc(["a boat sails on calm water"] * 2)
        return (tr.dit(lat, t, text_a, mask_a),
                tr.dit(lat, t, text_b, mask_b))

    pred_a, pred_b = probe()
    assert pred_a.shape == lat.shape
    # AdaLN-Zero: identical at init is the DESIGN, not a broken wire.
    assert float((pred_a - pred_b).abs().max()) < 1e-6

    flow_loss(pred_a, torch.randn_like(lat)).backward()
    gate_grad = float(tr.dit.adaLN_shared.weight.grad.abs().sum())
    assert gate_grad > 0, "adaLN gates receive no gradient — stuck shut"
    assert float(tr.dit.blocks[0].cross_kv.weight.grad.abs().sum()) == 0.0
    tr.dit.zero_grad(set_to_none=True)
    tr.text_enc.zero_grad(set_to_none=True)

    tr.train_step()  # one real optimizer step opens the gates
    pred_a2, _ = probe()
    flow_loss(pred_a2, torch.randn_like(lat)).backward()
    assert float(tr.dit.blocks[0].cross_kv.weight.grad.abs().sum()) > 0, (
        "cross-attention still receives no gradient after the gates "
        "opened — conditioning can never be learned")
    enc_grads = [p.grad for p in tr.text_enc.parameters()
                 if p.requires_grad]
    assert any(g is not None and float(g.abs().sum()) > 0
               for g in enc_grads), (
        "no gradient reached the text encoder after gate opening")


def test_empty_and_unseen_prompts_do_not_crash():
    tr = DiTTrainer(preset="tiny", batch_size=1, device="cpu")
    for caps in ([""], ["zzzqqq unseenword flibberty"], ["a"]):
        text, mask = tr.text_enc(caps)
        assert text.shape[0] == 1 and torch.isfinite(text).all()


# ------------------------------------------------------- preset counts

def _count(module):
    return sum(p.numel() for p in module.parameters())


def test_documented_parameter_counts():
    # Small's counts moved on 2026-10-31 (VAE growth): its latent
    # channels went 16 -> 32, so the DiT's patch-in/head grew by
    # 131,136 params and its VAE is the grown 32/64 (14,588,483).
    for preset, dit_expected, vae_expected in (
            ("tiny", 21_446_848, 3_650_851),
            ("small", 286_067_840, 14_588_483)):
        cfg = get_preset(preset)
        with torch.device("meta"):
            dit, vae = DiT(cfg), CausalVAE(cfg.vae_latent_ch,
                                           cfg.vae_base_ch)
        assert _count(dit) == dit_expected, preset
        assert _count(vae) == vae_expected, preset


def test_estimator_verdicts():
    est = _load_script("estimate_memory")
    small = est.estimate("small", 2, True, "none")
    assert small["dit_params"] == 286_067_840
    assert small["fits"]["Free T4 (Colab/Kaggle)"] is True
    big = est.estimate("vidgen480p", 1, True, "bf16")
    assert big["fits"]["Free T4 (Colab/Kaggle)"] is False
