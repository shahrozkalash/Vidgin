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
