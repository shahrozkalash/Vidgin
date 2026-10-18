"""Part 5: text conditioning.

Track A uses TinyTextEncoder: a small learned word-embedding table over a fixed
caption vocabulary (Moving MNIST captions), producing [B, L, Dt] context
tokens + padding mask consumed by the DiT's cross-attention. The cross-attention
interface is the real one, so a frozen umT5-XXL encoder can replace this module
later — see FrozenEncoderStub. CFG null-embedding: the empty-string embedding
is precomputed once and cached; training drops captions to null with p=0.1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

VOCAB = [
    "<pad>", "<unk>", "digit", "moving", "left", "right", "up", "down",
    "up-left", "up-right", "down-left", "down-right", "and", "static",
    "with", "black", "background",
] + [str(d) for d in range(10)]


class TinyTextEncoder(nn.Module):
    """Learned word embeddings -> [B, L, Dt] + boolean padding mask."""

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
        b = len(captions)
        ids = torch.zeros(b, self.max_len, dtype=torch.long)
        mask = torch.zeros(b, self.max_len, dtype=torch.bool)
        for i, cap in enumerate(captions):
            toks = self.tokenize(cap)
            if toks:
                ids[i, : len(toks)] = torch.tensor(toks)
                mask[i, : len(toks)] = True
        return self.embedding(ids.to(self.embedding.weight.device)), mask

    def null_embedding(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached null (empty prompt) embedding for CFG: [1, 1, Dt], mask False."""
        emb = self.embedding(self._null_ids)  # [1,1,Dt]
        mask = torch.zeros(1, 1, dtype=torch.bool, device=emb.device)
        return emb, mask


class FrozenEncoderStub:
    """Placeholder for the frozen umT5-XXL (4.7B) encoder (Part 5, GPU-only).

    Interface contract the real encoder must satisfy:
        encode(captions: list[str]) -> (embeds [B, L, Dt=4096], mask [B, L])
    with requires_grad=False on all parameters and a cached null embedding.
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
