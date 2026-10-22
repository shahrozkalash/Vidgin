"""DiT model growth (vidgen/grow.py): transplant a trained smaller
DiT into a larger preset so the big model does not start from scratch.

Covered: depth layer mapping, head-aware width expansion (a target
head inherits a whole source head's q/k/v — never a mix), the
zeroed (old out) x (new in) blocks, strict loading of the grown
weights into the real target architecture, text-encoder growth
(legacy->legacy, transformer->transformer, and the embeddings-only
legacy->transformer hop), payload-level validation, and the CLI
script producing a checkpoint the trainer's --init-weights accepts.

Custom small configs stand in for the real presets so the suite stays
fast; one preset-level test grows the real tiny geometry toward small
at the state-dict level (shapes + copied slices only — a full Small
model is too heavy for a unit test).
"""

import os
import subprocess
import sys
import zlib

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vidgen.config import DiTConfig, VidgenConfig, get_preset  # noqa: E402
from vidgen.dit.model import DiT  # noqa: E402
from vidgen.grow import (  # noqa: E402
    grow_dit_state_dict,
    grow_payload,
    grow_text_state_dict,
    layer_map,
)
from vidgen.text.encoder import VOCAB, build_text_encoder  # noqa: E402
from vidgen.train import DiTTrainer  # noqa: E402


def _cfg(name, dim, layers, heads, ffn, split, text_dim=64, text_layers=0,
         text_heads=4, text_ffn=128, text_max_len=16):
    return VidgenConfig(
        name=name, height=64, width=64, frames=17,
        latent_t=5, latent_h=8, latent_w=8,
        dit=DiTConfig(layers, dim, heads, ffn, split),
        text_dim=text_dim, text_max_len=text_max_len,
        text_layers=text_layers, text_heads=text_heads,
        text_ffn=text_ffn)


@pytest.fixture(scope="module")
def src_cfg():
    return _cfg("src", 128, 2, 4, 256, (8, 8, 16))


@pytest.fixture(scope="module")
def tgt_cfg():
    return _cfg("tgt", 256, 4, 8, 512, (8, 8, 16))


@pytest.fixture(scope="module")
def src_dit(src_cfg):
    torch.manual_seed(0)
    return DiT(src_cfg)


# ---------------------------------------------------------------- mapping
def test_layer_map_endpoints_and_stretch():
    assert layer_map(8, 8) == list(range(8))
    assert layer_map(2, 4) == [0, 0, 1, 1]
    m = layer_map(16, 40)
    assert m[0] == 0 and m[-1] == 15 and len(m) == 40
    assert all(a <= b for a, b in zip(m, m[1:])), "map must be monotone"
    assert layer_map(1, 5) == [0] * 5
    assert layer_map(5, 1) == [0]


# ------------------------------------------------------------------- DiT
def test_grow_shapes_and_strict_load(src_dit, src_cfg, tgt_cfg):
    grown, stats = grow_dit_state_dict(src_dit.state_dict(), src_cfg,
                                       tgt_cfg, seed=42)
    tgt = DiT(tgt_cfg)
    tgt.load_state_dict(grown, strict=True)  # raises on any mismatch
    assert stats["dit_total_params"] == \
        sum(p.numel() for p in tgt.parameters())
    assert 0 < stats["dit_copied_params"] < stats["dit_total_params"]
    assert stats["block_map"] == [0, 0, 1, 1]


def test_grow_depth_mapping_copies_whole_source_blocks(src_dit, src_cfg,
                                                      tgt_cfg):
    src = src_dit.state_dict()
    grown, _ = grow_dit_state_dict(src, src_cfg, tgt_cfg, seed=42)
    # target block 3 maps from source block 1: its 1D params (norms)
    # must equal the source's on the shared slice, 1s beyond (LN fresh).
    assert torch.equal(grown["blocks.3.norm1.weight"][:128],
                       src["blocks.1.norm1.weight"])
    assert torch.equal(grown["blocks.3.norm1.weight"][128:],
                       torch.ones(128))
    # and its FFN first layer: top-left == source, top-right == 0
    g = grown["blocks.3.ffn.0.weight"]
    s = src["blocks.1.ffn.0.weight"]
    assert torch.equal(g[:256, :128], s)
    assert torch.equal(g[:256, 128:], torch.zeros(256, 128))


def test_grow_attention_is_head_aware(src_dit, src_cfg, tgt_cfg):
    src = src_dit.state_dict()
    grown, _ = grow_dit_state_dict(src, src_cfg, tgt_cfg, seed=42)
    gs = grown["blocks.0.qkv.weight"].reshape(3, 8, 32, 256)
    ss = src["blocks.0.qkv.weight"].reshape(3, 4, 32, 128)
    # target head 2 maps to source head 1 (2*4//8): full q/k/v slices.
    for grp in range(3):
        assert torch.equal(gs[grp, 2, :, :128], ss[grp, 1])
    # old heads must not read new token dims at step 0:
    assert torch.equal(gs[:, :, :, 128:], torch.zeros(3, 8, 32, 128))
    # attn_out: input axis head-structured — target head 2's columns in
    # the transplanted rows come from source head 1's columns.
    go = grown["blocks.0.attn_out.weight"]
    so = src["blocks.0.attn_out.weight"]
    assert torch.equal(go[:128, 64:96], so[:, 32:64])


def test_grow_modulation_chunks_stay_aligned(src_dit, src_cfg, tgt_cfg):
    src = src_dit.state_dict()
    grown, _ = grow_dit_state_dict(src, src_cfg, tgt_cfg, seed=42)
    # adaLN_shared bias is 9 stacked dim-chunks: chunk 5 of the target
    # must hold chunk 5 of the source on the shared slice (a naive
    # flat copy would smear chunk boundaries).
    g = grown["adaLN_shared.bias"].reshape(9, 256)
    s = src["adaLN_shared.bias"].reshape(9, 128)
    assert torch.equal(g[5, :128], s[5])
    assert torch.equal(g[5, 128:], torch.zeros(128))


def test_grown_model_forward_is_finite(src_dit, src_cfg, tgt_cfg):
    grown, _ = grow_dit_state_dict(src_dit.state_dict(), src_cfg,
                                   tgt_cfg, seed=42)
    tgt = DiT(tgt_cfg)
    tgt.load_state_dict(grown, strict=True)
    tgt.eval()
    with torch.no_grad():
        out = tgt(torch.randn(1, 16, 5, 8, 8), torch.rand(1),
                  torch.randn(1, 3, src_cfg.text_dim),
                  torch.ones(1, 3, dtype=torch.bool))
    assert out.shape == (1, 16, 5, 8, 8)
    assert torch.isfinite(out).all()


def test_grow_is_deterministic(src_dit, src_cfg, tgt_cfg):
    a, _ = grow_dit_state_dict(src_dit.state_dict(), src_cfg, tgt_cfg,
                               seed=7)
    b, _ = grow_dit_state_dict(src_dit.state_dict(), src_cfg, tgt_cfg,
                               seed=7)
    for k in a:
        assert torch.equal(a[k], b[k]), f"seeded growth diverged on {k}"


# ------------------------------------------------------------------ text
def test_grow_text_legacy_to_legacy(src_cfg, tgt_cfg):
    torch.manual_seed(0)
    enc = build_text_encoder(src_cfg)
    grown, stats = grow_text_state_dict(enc.state_dict(), src_cfg,
                                        tgt_cfg, seed=3)
    tgt_enc = build_text_encoder(tgt_cfg)
    tgt_enc.load_state_dict(grown, strict=True)
    assert stats["text_transfer"] == "full"
    # same vocab, wider embeddings: leading slice is the source table.
    assert torch.equal(grown["embedding.weight"][:, :64],
                       enc.state_dict()["embedding.weight"])


def test_grow_text_legacy_to_transformer(src_cfg):
    torch.manual_seed(0)
    enc = build_text_encoder(src_cfg)
    tgt_cfg = _cfg("tt", 128, 2, 4, 256, (8, 8, 16), text_layers=2)
    grown, stats = grow_text_state_dict(enc.state_dict(), src_cfg,
                                        tgt_cfg, seed=3)
    tgt_enc = build_text_encoder(tgt_cfg)
    tgt_enc.load_state_dict(grown, strict=True)
    assert stats["text_transfer"] == "embeddings_only"
    assert stats["text_words_placed"] == len(VOCAB)
    assert stats["text_bucket_collisions"] == 0
    src_emb = enc.state_dict()["embedding.weight"]
    for i, word in enumerate(VOCAB):
        bucket = 1 + zlib.crc32(word.encode()) % (8192 - 1)
        assert torch.equal(grown["embedding.weight"][bucket, :64],
                           src_emb[i]), f"word {word!r} misplaced"
    emb, _ = tgt_enc(["digit moving left"])
    assert torch.isfinite(emb).all()


def test_grow_text_transformer_to_transformer():
    src_cfg = _cfg("ts", 128, 2, 4, 256, (8, 8, 16), text_dim=64,
                   text_layers=2)
    tgt_cfg = _cfg("tt2", 256, 4, 8, 512, (8, 8, 16), text_dim=128,
                   text_layers=4, text_heads=8, text_ffn=256)
    torch.manual_seed(0)
    enc = build_text_encoder(src_cfg)
    grown, stats = grow_text_state_dict(enc.state_dict(), src_cfg,
                                        tgt_cfg, seed=3)
    tgt_enc = build_text_encoder(tgt_cfg)
    tgt_enc.load_state_dict(grown, strict=True)
    assert stats["text_transfer"] == "full"
    assert stats["text_layer_map"] == [0, 0, 1, 1]
    # target layer 3 <- source layer 1: its norm weight's shared slice.
    assert torch.equal(
        grown["encoder.layers.3.norm1.weight"][:64],
        enc.state_dict()["encoder.layers.1.norm1.weight"])


def test_grow_text_transformer_to_legacy_rejected():
    src_cfg = _cfg("ts", 128, 2, 4, 256, (8, 8, 16), text_dim=64,
                   text_layers=2)
    enc = build_text_encoder(src_cfg)
    with pytest.raises(ValueError, match="downgrade"):
        grow_text_state_dict(enc.state_dict(), src_cfg,
                             _cfg("lg", 128, 2, 4, 256, (8, 8, 16)),
                             seed=3)


# --------------------------------------------------------------- payload
def _tiny_payload(tmp_path):
    tr = DiTTrainer(preset="tiny", seed=0, num_clips=8, batch_size=2,
                    ckpt_dir=None)
    tr.train_step()
    path = os.path.join(str(tmp_path), "donor.pt")
    tr.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return path, payload


def test_grow_payload_rejects_vae_and_same_geometry(tmp_path):
    _, payload = _tiny_payload(tmp_path)
    with pytest.raises(ValueError, match="DiT checkpoint"):
        grow_payload({"kind": "vae", "vae": {}}, "small")
    with pytest.raises(ValueError, match="same geometry"):
        grow_payload(payload, "tiny")


def test_grow_payload_tiny_to_small_shapes(tmp_path):
    _, payload = _tiny_payload(tmp_path)
    grown = grow_payload(payload, "small", seed=5)
    assert grown["kind"] == "dit" and grown["preset"] == "small"
    assert grown["step"] == 0
    assert "opt" not in grown and "rng" not in grown  # weights-only
    meta = grown["grow"]
    assert meta["source_preset"] == "tiny"
    assert meta["block_map"] == layer_map(8, 16)
    assert 0 < meta["copied_fraction"] < 1
    tgt_cfg = get_preset("small")
    with torch.device("meta"):
        tgt = DiT(tgt_cfg)
    tgt_shapes = {k: tuple(v.shape)
                  for k, v in tgt.state_dict().items()}
    assert {k: tuple(v.shape) for k, v in grown["dit"].items()} == \
        tgt_shapes
    # the transplanted slice of block 15 (from tiny block 7) is exact:
    assert torch.equal(grown["dit"]["blocks.15.norm1.weight"][:384],
                       payload["dit"]["blocks.7.norm1.weight"])


def test_grow_cli_and_trainer_init_weights(tmp_path):
    donor_path, _ = _tiny_payload(tmp_path)
    out = os.path.join(str(tmp_path), "grown_custom.pt")
    # CLI against the real presets would build a full Small checkpoint
    # (~1.1 GB); exercise the script end-to-end on tiny->small anyway —
    # it is the exact user path — but only assert on the file + meta.
    script = os.path.join(os.path.dirname(__file__), "..", "scripts",
                          "grow_checkpoint.py")
    res = subprocess.run(
        [sys.executable, script, donor_path, "--to", "small",
         "--out", out], capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr
    assert "transplanted" in res.stdout
    grown = torch.load(out, map_location="cpu", weights_only=False)
    assert grown["grow"]["target_preset"] == "small"
    # the trainer's weights-only init must accept it strictly:
    tr = DiTTrainer(preset="small", seed=1, num_clips=4, batch_size=1,
                    ckpt_dir=None)
    tr.load_dit_weights(out)
    assert tr.init_from == out and tr.step == 0
