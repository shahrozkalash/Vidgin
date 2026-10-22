"""Tests for the Phase 2 caption-trained tokenizer.

Covers vidgen/text/tokenizer.py (CaptionTokenizer), its integration
into both text encoders, checkpoint storage/compatibility in the DiT
trainer, and the hash-collision audit of the legacy scheme.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vidgen.config import DiTConfig, VidgenConfig
from vidgen.text.encoder import (
    TinyTextEncoder,
    TransformerTextEncoder,
    build_text_encoder,
)
from vidgen.text.tokenizer import (
    BOS_ID,
    EOS_ID,
    UNK_ID,
    CaptionTokenizer,
    audit_hash_collisions,
    hash_bucket,
    normalize_tokens,
)

CAPTIONS = [
    "people walking across a city street",
    "cars driving through a busy intersection",
    "a crowd moves through the city center",
    "people crossing the street at night",
]


@pytest.fixture()
def tokenizer() -> CaptionTokenizer:
    return CaptionTokenizer.train(CAPTIONS, max_vocab=64)


# ---------------------------------------------------------------------------
# CaptionTokenizer core
# ---------------------------------------------------------------------------

def test_train_is_deterministic(tokenizer):
    again = CaptionTokenizer.train(list(reversed(CAPTIONS)), max_vocab=64)
    assert again.words == tokenizer.words


def test_specials_come_first(tokenizer):
    assert tokenizer.words[:4] == ["<pad>", "<unk>", "<bos>", "<eos>"]
    assert len(tokenizer) <= 64


def test_encode_decode_roundtrip(tokenizer):
    ids = tokenizer.encode("people walking across a city street")
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    assert tokenizer.decode(ids) == "people walking across a city street"


def test_unknown_word_maps_to_unk(tokenizer):
    ids = tokenizer.encode("a helicopter lands")
    assert UNK_ID in ids  # helicopter/lands never appear in CAPTIONS


def test_distinct_prompts_distinct_sequences(tokenizer):
    a = tokenizer.encode("people walking across a city street")
    b = tokenizer.encode("cars driving through a busy intersection")
    assert a != b
    c = tokenizer.encode("people crossing the street at night")
    assert a != c  # shares words but differs -> different sequence


def test_oov_rate(tokenizer):
    assert tokenizer.oov_rate(CAPTIONS) == 0.0
    assert tokenizer.oov_rate(["a helicopter lands"]) == pytest.approx(2 / 3)


def test_save_load_roundtrip(tokenizer, tmp_path):
    path = str(tmp_path / "tokenizer.json")
    tokenizer.save(path)
    assert CaptionTokenizer.load(path) == tokenizer
    assert CaptionTokenizer.from_dict(tokenizer.to_dict()) == tokenizer
    with pytest.raises(ValueError):
        CaptionTokenizer.from_dict({"format": "nope", "words": []})


def test_normalize_tokens():
    assert normalize_tokens("People, WALKING! 2 fast cars.") == [
        "people", "walking", "2", "fast", "cars"]


# ---------------------------------------------------------------------------
# Legacy hash audit
# ---------------------------------------------------------------------------

def test_hash_audit_finds_collisions_in_small_bucket_space():
    words = [f"word{i}" for i in range(40)]
    collisions = audit_hash_collisions(words, buckets=8)
    assert collisions  # 40 words into 7 buckets must collide
    for bucket, group in collisions.items():
        assert len(group) >= 2
        assert all(hash_bucket(w, 8) == bucket for w in group)


def test_hash_audit_empty_when_no_collisions():
    assert audit_hash_collisions(["digit"], buckets=8192) == {}


# ---------------------------------------------------------------------------
# Encoder integration
# ---------------------------------------------------------------------------

def test_tiny_encoder_with_tokenizer(tokenizer):
    enc = TinyTextEncoder(d_t=32, max_len=16, tokenizer=tokenizer)
    assert enc.embedding.num_embeddings == len(tokenizer)
    emb_a, mask_a = enc(["people walking across a city street"])
    emb_b, _ = enc(["cars driving through a busy intersection"])
    assert emb_a.shape == (1, 16, 32)
    assert mask_a[0].sum().item() == len(
        tokenizer.encode("people walking across a city street"))
    assert not torch.allclose(emb_a, emb_b)
    loss = emb_a.sum()
    loss.backward()
    assert enc.embedding.weight.grad is not None
    assert enc.embedding.weight.grad.abs().sum() > 0
    null_emb, null_mask = enc.null_embedding()
    assert null_emb.shape == (1, 1, 32) and not null_mask.any()


def test_transformer_encoder_with_tokenizer(tokenizer):
    enc = TransformerTextEncoder(
        d_t=32, max_len=16, layers=2, heads=4, ffn=64, tokenizer=tokenizer)
    assert enc.vocab_size == len(tokenizer)
    assert enc.embedding.num_embeddings == len(tokenizer)
    emb, mask = enc(["people walking", "cars"])
    assert emb.shape == (2, 16, 32) and mask.shape == (2, 16)
    emb.sum().backward()
    assert enc.embedding.weight.grad is not None
    null_emb, _ = enc.null_embedding()
    assert null_emb.shape == (1, 1, 32)


def test_legacy_hash_tokenize_unchanged():
    enc = TransformerTextEncoder(d_t=32, max_len=8, layers=1, heads=4, ffn=64)
    import zlib
    expect = [1 + zlib.crc32(w.encode()) % (8192 - 1)
              for w in "people walking".split()]
    assert enc.tokenize("people walking") == expect


def _small_transformer_cfg(**overrides) -> VidgenConfig:
    base = dict(
        name="tok-test", height=64, width=64, frames=17,
        dit=DiTConfig(2, 64, 4, 128),
        text_dim=32, text_max_len=16, text_layers=2, text_heads=4,
        text_ffn=64, text_vocab_size=8192,
    )
    base.update(overrides)
    return VidgenConfig(**base)


def test_build_text_encoder_modes(tokenizer):
    hash_enc = build_text_encoder(_small_transformer_cfg())
    assert isinstance(hash_enc, TransformerTextEncoder)
    assert hash_enc.tokenizer is None

    cap_cfg = _small_transformer_cfg(text_tokenizer="caption")
    cap_enc = build_text_encoder(cap_cfg, tokenizer=tokenizer)
    assert cap_enc.tokenizer is tokenizer

    with pytest.raises(ValueError, match="caption tokenizer"):
        build_text_encoder(cap_cfg)  # caption mode without the tokenizer
    with pytest.raises(ValueError, match="hash"):
        build_text_encoder(_small_transformer_cfg(), tokenizer=tokenizer)


def test_old_config_dict_without_tokenizer_field_defaults_to_hash():
    d = _small_transformer_cfg().to_dict()
    del d["text_tokenizer"]  # a pre-Phase-2 checkpoint config
    cfg = VidgenConfig.from_dict(d)
    assert cfg.text_tokenizer == "hash"


# ---------------------------------------------------------------------------
# Trainer: checkpoint storage + compatibility errors
# ---------------------------------------------------------------------------

def _trainer(tokenizer=None, ckpt_dir=None):
    from vidgen.train import DiTTrainer

    return DiTTrainer(preset="tiny", seed=0, num_clips=8, batch_size=2,
                      ckpt_dir=ckpt_dir, tokenizer=tokenizer)


def test_trainer_checkpoint_carries_tokenizer(tokenizer, tmp_path):
    tr = _trainer(tokenizer=tokenizer)
    rec = tr.train_step()
    assert math.isfinite(rec["loss"])
    path = str(tmp_path / "dit_tok.pt")
    tr.save(path)

    from vidgen.checkpoint import load_checkpoint
    payload = load_checkpoint(path)
    assert payload["config"]["text_tokenizer"] == "caption"
    assert CaptionTokenizer.from_dict(payload["tokenizer"]) == tokenizer

    from vidgen.train import DiTTrainer
    tr2 = DiTTrainer.resume(path, seed=0, num_clips=8, batch_size=2,
                            ckpt_dir=None)
    assert tr2.tokenizer == tokenizer
    assert tr2.cfg.text_tokenizer == "caption"


def test_trainer_weights_load_compatibility(tokenizer, tmp_path):
    donor = _trainer(tokenizer=tokenizer)
    path = str(tmp_path / "donor.pt")
    donor.save(path)

    # Hash-mode recipient: loud error, not a shape mismatch.
    plain = _trainer()
    with pytest.raises(ValueError, match="caption tokenizer"):
        plain.load_dit_weights(path)

    # Same tokenizer: loads fine.
    same = _trainer(tokenizer=tokenizer)
    same.load_dit_weights(path)

    # Different vocabulary: loud error.
    other = CaptionTokenizer.train(["completely different words here"],
                                   max_vocab=32)
    diff = _trainer(tokenizer=other)
    with pytest.raises(ValueError, match="DIFFERENT caption tokenizer"):
        diff.load_dit_weights(path)


def test_legacy_trainer_checkpoint_has_no_tokenizer_key(tmp_path):
    tr = _trainer()
    path = str(tmp_path / "legacy.pt")
    tr.save(path)
    from vidgen.checkpoint import load_checkpoint
    payload = load_checkpoint(path)
    assert "tokenizer" not in payload
    assert payload["config"]["text_tokenizer"] == "hash"
