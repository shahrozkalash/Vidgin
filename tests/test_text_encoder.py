"""TinyTextEncoder device-contract regression tests.

Regression for the Colab T4 crash: forward() used to return the padding
mask on CPU while the token embeddings were on CUDA, crashing DiT training
with "Expected all tensors to be on the same device".
Contract: every tensor TinyTextEncoder returns must live on the module's
(embedding weight's) device.
"""

import torch

from vidgen.text.encoder import TinyTextEncoder


def test_forward_returns_single_device_tensors():
    enc = TinyTextEncoder()
    emb, mask = enc(["digit 3 moving left", "digit 7 moving right"])
    dev = enc.embedding.weight.device
    assert emb.shape == (2, enc.max_len, enc.d_t)
    assert mask.shape == (2, enc.max_len)
    assert mask.dtype == torch.bool
    assert emb.device == dev
    assert mask.device == dev
    assert emb.device == mask.device


def test_null_embedding_returns_single_device_tensors():
    enc = TinyTextEncoder()
    emb, mask = enc.null_embedding()
    dev = enc.embedding.weight.device
    assert emb.shape == (1, 1, enc.d_t)
    assert emb.device == dev
    assert mask.device == dev
    assert not mask.any()


def test_forward_routes_embedding_device_into_mask_allocation(monkeypatch):
    # The actual regression: the old forward() built `mask` with a bare
    # torch.zeros(...) (device=None), which lands on CPU even when the
    # module runs on CUDA. Lock in that forward() passes the embedding
    # weight's device into every zeros() it allocates. (Fails on the old
    # code, where the recorded devices are None.)
    enc = TinyTextEncoder()
    seen_devices = []
    orig_zeros = torch.zeros

    def spy_zeros(*args, **kwargs):
        seen_devices.append(kwargs.get("device"))
        return orig_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", spy_zeros)
    enc(["digit 3 moving left"])
    assert seen_devices, "expected forward() to allocate with torch.zeros"
    expected = enc.embedding.weight.device
    assert all(d == expected for d in seen_devices), seen_devices


def test_forward_mask_values_correct():
    enc = TinyTextEncoder()
    _, mask = enc(["digit 3 moving left", ""])
    assert mask[0].sum().item() == len(enc.tokenize("digit 3 moving left"))
    assert mask[1].sum().item() == len(enc.tokenize(""))
    # unknown words are not dropped, just mapped to <unk>
    _, mask2 = enc(["some weird phrase"])
    assert mask2[0].sum().item() == len(enc.tokenize("some weird phrase")) > 0


# --- Flagship TransformerTextEncoder (2026-10-21 upgrade) -----------------

from vidgen.config import DiTConfig, VidgenConfig, get_preset
from vidgen.text.encoder import TransformerTextEncoder, VOCAB, build_text_encoder


def test_factory_keeps_legacy_encoder_for_existing_presets():
    for name in ("tiny", "small", "vidgen480p"):
        enc = build_text_encoder(get_preset(name))
        assert isinstance(enc, TinyTextEncoder)
        assert sum(p.numel() for p in enc.parameters()) == len(VOCAB) * 256


def test_factory_builds_transformer_encoder_for_flagships():
    for name in ("vidgen9b", "vidgen14b"):
        enc = build_text_encoder(get_preset(name))
        assert isinstance(enc, TransformerTextEncoder)


def test_flagship_text_specs():
    c9, c14 = get_preset("vidgen9b"), get_preset("vidgen14b")
    assert (c9.text_dim, c9.text_max_len, c9.text_layers) == (1024, 128, 16)
    assert (c14.text_dim, c14.text_max_len, c14.text_layers) == (1024, 128, 24)
    assert get_preset("tiny").text_layers == 0


def test_transformer_forward_shapes_mask_and_finiteness():
    enc = TransformerTextEncoder(d_t=64, max_len=16, layers=2, heads=4,
                                 ffn=128, vocab_size=512)
    emb, mask = enc(["busy city street", ""])
    assert emb.shape == (2, 16, 64)
    assert mask.shape == (2, 16) and mask.dtype == torch.bool
    assert mask[0].sum().item() == 3
    assert not mask[1].any()
    assert torch.isfinite(emb).all()  # empty caption row must not NaN
    assert emb[0, 3:].abs().max().item() == 0.0  # padded slots zeroed


def test_transformer_null_embedding_contract():
    enc = TransformerTextEncoder(d_t=64, max_len=16, layers=2, heads=4,
                                 ffn=128, vocab_size=512)
    emb, mask = enc.null_embedding()
    assert emb.shape == (1, 1, 64)
    assert not mask.any()
    assert torch.isfinite(emb).all()


def test_transformer_tokenize_deterministic_open_vocab():
    a = TransformerTextEncoder(d_t=32, max_len=8, layers=1, heads=4,
                               ffn=64, vocab_size=256)
    b = TransformerTextEncoder(d_t=32, max_len=8, layers=1, heads=4,
                               ffn=64, vocab_size=256)
    ids = a.tokenize("helicopter zebra quantum")
    assert ids == b.tokenize("helicopter zebra quantum")
    assert len(ids) == 3 and all(1 <= i < 256 for i in ids)
    assert len(set(ids)) == 3  # distinct words land in distinct buckets here


def test_flagship_encoder_param_counts():
    with torch.device("meta"):
        e9 = build_text_encoder(get_preset("vidgen9b"))
        e14 = build_text_encoder(get_preset("vidgen14b"))
    n9 = sum(p.numel() for p in e9.parameters())
    n14 = sum(p.numel() for p in e14.parameters())
    assert 190_000_000 < n9 < 230_000_000
    assert 290_000_000 < n14 < 330_000_000


def test_transformer_encoder_wires_into_dit_forward():
    from vidgen.dit.model import DiT
    cfg = VidgenConfig(
        name="wiring", height=64, width=64, frames=17,
        latent_t=5, latent_h=8, latent_w=8,
        dit=DiTConfig(layers=2, dim=128, heads=4, ffn=256,
                      rope_split=(12, 10, 10)),
        text_dim=64, text_max_len=16, text_layers=2, text_heads=4,
        text_ffn=128, text_vocab_size=512).validate()
    enc = build_text_encoder(cfg)
    dit = DiT(cfg)
    text, mask = enc(["busy city street"])
    latent = torch.randn(1, cfg.latent_channels, cfg.latent_t,
                         cfg.latent_h, cfg.latent_w)
    out = dit(latent, torch.zeros(1), text, mask)
    assert out.shape == latent.shape
    assert torch.isfinite(out).all()
