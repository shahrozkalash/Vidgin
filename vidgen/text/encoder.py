"""Part 5: text conditioning.

Track A uses TinyTextEncoder: a small learned word-embedding table over a fixed
caption vocabulary (Moving MNIST captions), producing [B, L, Dt] context
tokens + padding mask consumed by the DiT's cross-attention. The cross-attention
interface is the real one, so a frozen umT5-XXL encoder can replace this module
later — see FrozenEncoderStub. CFG null-embedding: the empty-string embedding
is precomputed once and cached; training drops captions to null with p=0.1.

Flagship upgrade (2026-10-21): TransformerTextEncoder is a full transformer
text encoder (word-hash tokenization, learned positions, pre-norm encoder
stack) selected per preset by VidgenConfig.text_layers via build_text_encoder.
The flagship presets (vidgen9b / vidgen14b) use it at Dt=1024 with 128-token
captions (~210M / ~311M params); tiny / small / vidgen480p keep the legacy
encoder, so every existing checkpoint loads unchanged.
"""

from __future__ import annotations

import zlib

import torch
import torch.nn as nn

VOCAB = [
    "<pad>", "<unk>", "digit", "moving", "left", "right", "up", "down",
    "up-left", "up-right", "down-left", "down-right", "and", "static",
    "with", "black", "background",
] + [str(d) for d in range(10)]


class TinyTextEncoder(nn.Module):
    """Learned word embeddings -> [B, L, Dt] + boolean padding mask.

    Device contract: every tensor returned by forward() and null_embedding()
    lives on the embedding weight's device, so GPU training never hits
    "Expected all tensors to be on the same device".
    """

    def __init__(self, vocab: list[str] | None = None, d_t: int = 256, max_len: int = 32):
        super().__init__()
        self.vocab = vocab or VOCAB
        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        self.d_t = d_t
        self.max_len = max_len
        self.embedding = nn.Embedding(len(self.vocab), d_t, padding_idx=0)
        # CFG null embedding: precomputed empty-string embedding, cached once.
        self.register_buffer("_null_ids", torch.zeros(1, 1, dtype=torch.long))

    def tokenize(self, caption: str) -> list[int]:
        ids = [self.word2id.get(w, self.word2id["<unk>"])
               for w in caption.lower().split()][: self.max_len]
        return ids

    def forward(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed captions -> ([B, L, Dt] embeds, [B, L] bool padding mask).

        Both tensors are allocated on the embedding weight's device: the
        mask must not stay on CPU when the module runs on CUDA.
        """
        device = self.embedding.weight.device
        b = len(captions)
        ids = torch.zeros(b, self.max_len, dtype=torch.long, device=device)
        mask = torch.zeros(b, self.max_len, dtype=torch.bool, device=device)
        for i, cap in enumerate(captions):
            toks = self.tokenize(cap)
            if toks:
                ids[i, : len(toks)] = torch.tensor(toks, device=device)
                mask[i, : len(toks)] = True
        return self.embedding(ids), mask

    def null_embedding(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached null (empty prompt) embedding for CFG: [1, 1, Dt], mask False."""
        emb = self.embedding(self._null_ids)  # [1,1,Dt]
        mask = torch.zeros(1, 1, dtype=torch.bool, device=emb.device)
        return emb, mask


class TransformerTextEncoder(nn.Module):
    """Full transformer text encoder for the flagship presets.

    Tokenization is word-hash: each whitespace-separated word is mapped by
    crc32 into one of `vocab_size` buckets (bucket 0 reserved for padding).
    crc32 is deterministic across processes and machines, so no vocab file
    ships with the model, and — unlike the legacy 27-word table — ANY word
    lands in a bucket of its own (hash collisions are possible but rare and
    shared across identical words only). On top sit learned positional
    embeddings and a pre-norm transformer encoder stack.

    Same interface as TinyTextEncoder: forward(captions) -> ([B, L, Dt]
    embeds, [B, L] bool padding mask) and null_embedding() -> ([1, 1, Dt],
    all-False mask), so the DiT, trainer, and sampler consume it unchanged.
    Every returned tensor lives on the embedding weight's device.
    """

    def __init__(self, d_t: int = 1024, max_len: int = 128, layers: int = 24,
                 heads: int = 16, ffn: int = 4096, vocab_size: int = 8192):
        super().__init__()
        self.d_t = d_t
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_t, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_t)
        layer = nn.TransformerEncoderLayer(
            d_model=d_t, nhead=heads, dim_feedforward=ffn, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm_final = nn.LayerNorm(d_t)
        # CFG null embedding: the empty-prompt token id row, as in the legacy
        # encoder — encoded once per call (cheap) so it always matches the
        # module's current weights and device.
        self.register_buffer("_null_ids", torch.zeros(1, 1, dtype=torch.long))

    def tokenize(self, caption: str) -> list[int]:
        return [1 + zlib.crc32(w.encode("utf-8")) % (self.vocab_size - 1)
                for w in caption.lower().split()[: self.max_len]]

    def _encode_ids(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.embedding(ids) + self.pos_embedding(pos).unsqueeze(0)
        pad = ~mask
        # A fully-padded row (empty caption) would softmax over an all-masked
        # row inside attention and yield NaNs; unmask its first slot for the
        # internal pass — that row's outputs are discarded downstream via the
        # real mask, and padded positions are zeroed on the way out.
        empty_rows = pad.all(dim=1)
        if bool(empty_rows.any()):
            pad = pad.clone()
            pad[empty_rows, 0] = False
        x = self.encoder(x, src_key_padding_mask=pad)
        x = self.norm_final(x)
        return x * mask.unsqueeze(-1).to(x.dtype)

    def forward(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.embedding.weight.device
        b = len(captions)
        ids = torch.zeros(b, self.max_len, dtype=torch.long, device=device)
        mask = torch.zeros(b, self.max_len, dtype=torch.bool, device=device)
        for i, cap in enumerate(captions):
            toks = self.tokenize(cap)
            if toks:
                ids[i, : len(toks)] = torch.tensor(toks, device=device)
                mask[i, : len(toks)] = True
        return self._encode_ids(ids, mask), mask

    def null_embedding(self) -> tuple[torch.Tensor, torch.Tensor]:
        """CFG null (empty prompt) context: [1, 1, Dt], mask all False."""
        ones = torch.ones(1, 1, dtype=torch.bool, device=self._null_ids.device)
        emb = self._encode_ids(self._null_ids, ones)
        mask = torch.zeros(1, 1, dtype=torch.bool, device=emb.device)
        return emb, mask


def build_text_encoder(cfg) -> nn.Module:
    """Config-gated factory used by the trainer and the sampler.

    cfg.text_layers == 0 (every preset and checkpoint from before the
    flagship text upgrade) returns the legacy TinyTextEncoder, so old
    checkpoints load unchanged; text_layers > 0 returns the
    TransformerTextEncoder at the config's width/depth.
    """
    if getattr(cfg, "text_layers", 0) > 0:
        return TransformerTextEncoder(
            d_t=cfg.text_dim, max_len=cfg.text_max_len,
            layers=cfg.text_layers, heads=cfg.text_heads,
            ffn=cfg.text_ffn or 4 * cfg.text_dim,
            vocab_size=cfg.text_vocab_size)
    return TinyTextEncoder(d_t=cfg.text_dim, max_len=cfg.text_max_len)


class FrozenEncoderStub:
    """Placeholder for the frozen umT5-XXL (4.7B) encoder (Part 5, GPU-only).

    Interface contract the real encoder must satisfy:
        encode(captions: list[str]) -> (embeds [B, L, Dt=4096], mask [B, L])
    with requires_grad=False on all parameters and a cached null embedding.
    As with TinyTextEncoder, every returned tensor must live on the module's
    device (never a CPU mask with CUDA embeddings).
    Instantiating this stub raises an informative error; it exists so the
    DiT cross-attention interface stays honest about the swap path.
    """

    EXPECTED_DIM = 4096
    EXPECTED_MAX_LEN = 512

    def __init__(self, *args, **kwargs):
        raise ImportError(
            "FrozenEncoderStub is a placeholder: the real umT5-XXL encoder needs "
            "transformers + >=24GB GPU VRAM (Part 5 GPU-only item). Track A trains "
            "with TinyTextEncoder instead; DiT consumes [B, L, Dt] + mask either way."
        )
