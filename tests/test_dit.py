"""Parts 4+6: patchify inverse, RoPE invariance, DiT forward shape,
param-count sanity, non-zero gradients."""

import torch
import torch.nn.functional as F

from vidgen.config import get_preset
from vidgen.dit.model import (
    DiT,
    apply_rope_3d,
    count_params,
    patchify_latents,
    token_grid_positions,
    unpatchify_tokens,
)
from vidgen.text.encoder import TinyTextEncoder


def _tiny_dit():
    cfg = get_preset("tiny")
    return cfg, DiT(cfg).eval()


def test_patchify_unpatchify_identity_inverse():
    x = torch.randn(2, 16, 5, 8, 8)
    tok = patchify_latents(x, (1, 2, 2))
    assert tok.shape == (2, 80, 64)
    back = unpatchify_tokens(tok, (1, 2, 2), (5, 4, 4), 16)
    assert (x - back).abs().max() < 1e-6


def test_rope_relative_position_invariance():
    # attention scores must be invariant under a global (t,h,w) translation
    torch.manual_seed(0)
    b, heads, dh, n = 1, 2, 64, 24
    split = (22, 22, 20)
    q = torch.randn(b, heads, n, dh)
    k = torch.randn(b, heads, n, dh)
    pt, ph, pw = (torch.randint(0, 4, (n,)), torch.randint(0, 3, (n,)),
                  torch.randint(0, 3, (n,)))
    qr = apply_rope_3d(q, pt, ph, pw, split)
    kr = apply_rope_3d(k, pt, ph, pw, split)
    s1 = (qr @ kr.transpose(-1, -2)) / (dh ** 0.5)
    qr2 = apply_rope_3d(q, pt + 7, ph + 5, pw + 9, split)
    kr2 = apply_rope_3d(k, pt + 7, ph + 5, pw + 9, split)
    s2 = (qr2 @ kr2.transpose(-1, -2)) / (dh ** 0.5)
    assert (s1 - s2).abs().max() < 1e-5


def test_dit_forward_shape():
    cfg, dit = _tiny_dit()
    enc = TinyTextEncoder(d_t=cfg.text_dim, max_len=cfg.text_max_len).eval()
    lat = torch.randn(1, 16, 5, 8, 8)
    t = torch.tensor([0.5])
    text, mask = enc(["digit 3 moving left"])
    with torch.no_grad():
        v = dit(lat, t, text, mask)
    assert v.shape == lat.shape, "DiT must output a velocity field of latent shape"


def test_tiny_param_count_within_10pct():
    cfg, dit = _tiny_dit()
    n = count_params(dit)
    target = cfg.target_params_m * 1e6
    assert abs(n - target) / target < 0.10, f"{n/1e6:.1f}M vs target 20M"


def _warmup(dit, enc, steps=2):
    """Run a few optimizer steps so AdaLN-Zero gates move off exact zero
    (at init the gates are 0 by design, which mutes every branch)."""
    dit.train()
    opt = torch.optim.SGD(dit.parameters(), lr=1e-3)
    lat = torch.randn(1, 16, 5, 8, 8)
    t = torch.tensor([0.3])
    for _ in range(steps):
        text, mask = enc(["digit 1 moving right"])  # fresh graph each step
        opt.zero_grad()
        dit(lat, t, text, mask).sum().backward()
        opt.step()
    return lat, t


def test_nonzero_gradients_all_blocks():
    cfg, dit = _tiny_dit()
    enc = TinyTextEncoder(d_t=cfg.text_dim, max_len=cfg.text_max_len)
    lat, t = _warmup(dit, enc)
    text, mask = enc(["digit 1 moving right"])
    dit.zero_grad()
    dit(lat, t, text, mask).sum().backward()
    for i, blk in enumerate(dit.blocks):
        g = sum(p.grad.abs().sum().item() for p in blk.parameters() if p.grad is not None)
        assert g > 0, f"block {i} got zero gradients"


def test_text_conditioning_changes_output():
    cfg, dit = _tiny_dit()
    enc = TinyTextEncoder(d_t=cfg.text_dim, max_len=cfg.text_max_len).eval()
    _warmup(dit, enc)  # move AdaLN-Zero gates off exact zero (see above)
    dit.eval()
    lat = torch.randn(1, 16, 5, 8, 8)
    t = torch.tensor([0.5])
    t1, m1 = enc(["digit 3 moving left"])
    t2, m2 = enc(["digit 7 moving right"])
    with torch.no_grad():
        v1 = dit(lat, t, t1, m1)
        v2 = dit(lat, t, t2, m2)
    assert (v1 - v2).abs().max() > 1e-4, "text conditioning must affect the velocity"
