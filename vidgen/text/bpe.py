"""Subword (BPE) tokenizer for Vidgin's text encoders — from scratch.

Phase 4 of the master improvement programme (2026-10-22). The word-level
:class:`~vidgen.text.tokenizer.CaptionTokenizer` maps every unseen word
to a single ``<unk>`` id, so spelling variants and rare words are
indistinguishable to the text encoder. Byte-pair encoding fixes that
without any external dependency or pretrained model: words are split
into frequent character sequences learned from Vidgin's OWN captions,
so "walking", "walks" and the typo "walkking" share subwords instead of
collapsing.

Implementation notes:

* Pretokenization reuses the caption pipeline's ``normalize_tokens``
  (lowercase words, punctuation dropped) so BPE and word-level see the
  same word stream and their coverage numbers are comparable.
* Each word is represented as its characters plus an end-of-word
  marker ``</w>``; merges are learned by frequency, ties broken
  lexicographically, so training is a deterministic function of the
  caption multiset — same corpus, same tokenizer, on any machine.
* Special tokens keep the SAME ids as the word-level tokenizer
  (PAD 0, UNK 1, BOS 2, EOS 3) and the class mirrors its interface
  (``encode``/``decode``/``to_dict``/``from_dict``/``save``/``load``,
  ``len()``, ``.words``), so encoders, trainers and checkpoint code
  treat both tokenizers identically. A character never seen in training
  becomes ``<unk>`` in place and blocks merges across it.
* Honest limit: BPE gives the encoder more symbols to work with; it
  does not give it understanding. Whether it beats word-level for
  Vidgin is an empirical question — ``scripts/build_tokenizer.py
  --method bpe`` prints both coverage reports for the same corpus.
"""

from __future__ import annotations

import json
from collections import Counter

from vidgen.text.tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNK_ID,
    UNK_TOKEN,
    normalize_tokens,
)

BPE_FORMAT_TAG = "vidgin-bpe-tokenizer-v1"
END_OF_WORD = "</w>"


class BPETokenizer:
    """Byte-pair-encoding tokenizer trained on Vidgin's own captions."""

    def __init__(self, units: list[str], merges: list[tuple[str, str]]):
        units = list(units)
        if units[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError(
                "BPE units must start with the special tokens "
                f"{SPECIAL_TOKENS}, got {units[:4]}"
            )
        if len(set(units)) != len(units):
            raise ValueError("BPE vocabulary contains duplicate units")
        self.units = units
        self.unit2id = {u: i for i, u in enumerate(units)}
        self.merges = [tuple(m) for m in merges]
        self.merge_rank = {pair: i for i, pair in enumerate(self.merges)}

    # The word-level tokenizer exposes `.words`; encoders size their
    # embedding from it. For BPE the units play that role.
    @property
    def words(self) -> list[str]:
        return self.units

    def __len__(self) -> int:
        return len(self.units)

    def __eq__(self, other) -> bool:
        return (isinstance(other, BPETokenizer)
                and other.units == self.units
                and other.merges == self.merges)

    # ------------------------------------------------------------- training
    @classmethod
    def train(cls, captions, vocab_size: int = 2048,
              min_pair_freq: int = 2) -> "BPETokenizer":
        """Learn merges from caption text up to ``vocab_size`` units.

        ``vocab_size`` counts the special tokens. Deterministic: pair
        choice is by descending frequency with lexicographic tie-break.
        """
        if vocab_size <= len(SPECIAL_TOKENS) + 1:
            raise ValueError(
                f"vocab_size must exceed {len(SPECIAL_TOKENS)} specials + base chars")
        word_freq: Counter[str] = Counter()
        for caption in captions:
            word_freq.update(normalize_tokens(caption))
        # word -> symbol sequence (chars + end-of-word marker)
        seqs: dict[str, list[str]] = {
            w: list(w) + [END_OF_WORD] for w in word_freq
        }
        base_chars = sorted({c for w in word_freq for c in w})
        merges: list[tuple[str, str]] = []
        while len(SPECIAL_TOKENS) + len(base_chars) + len(merges) < vocab_size:
            pairs: Counter[tuple[str, str]] = Counter()
            for w, seq in seqs.items():
                f = word_freq[w]
                for a, b in zip(seq, seq[1:]):
                    pairs[(a, b)] += f
            if not pairs:
                break
            best, best_freq = None, -1
            for pair in sorted(pairs):  # lexicographic tie-break
                if pairs[pair] > best_freq:
                    best, best_freq = pair, pairs[pair]
            if best is None or best_freq < min_pair_freq:
                break
            merges.append(best)
            merged = best[0] + best[1]
            for w, seq in seqs.items():
                out: list[str] = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                seqs[w] = out
        # Vocabulary order: specials, base chars (sorted), merged units
        # in the order they were learned.
        units = list(SPECIAL_TOKENS) + base_chars + [a + b for a, b in merges]
        return cls(units, merges)

    # ------------------------------------------------------------- encoding
    def _encode_word(self, word: str) -> list[str]:
        known = set(self.units)
        seq = [c if c in known else UNK_TOKEN for c in word] + [END_OF_WORD]
        # Apply merges greedily by rank (standard BPE encoding).
        while len(seq) > 1:
            best_i, best_rank = -1, None
            for i in range(len(seq) - 1):
                rank = self.merge_rank.get((seq[i], seq[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_i, best_rank = i, rank
            if best_i < 0:
                break
            seq[best_i:best_i + 2] = [seq[best_i] + seq[best_i + 1]]
        return seq

    def tokenize(self, text: str) -> list[str]:
        units: list[str] = []
        for word in normalize_tokens(text):
            units.extend(self._encode_word(word))
        return units

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        """[BOS] + subword ids (unknown chars -> UNK) + [EOS]."""
        ids = [BOS_ID]
        ids.extend(self.unit2id.get(u, UNK_ID) for u in self.tokenize(text))
        ids.append(EOS_ID)
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids) -> str:
        from vidgen.text.tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN

        text = ""
        for i in ids:
            unit = self.units[int(i)] if 0 <= int(i) < len(self.units) else UNK_TOKEN
            if unit in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
                continue
            if unit == END_OF_WORD:
                text += " "
            else:
                text += unit.replace(END_OF_WORD, " ")
        return " ".join(text.split())

    def oov_rate(self, captions) -> float:
        """Fraction of caption WORDS containing an unseen character."""
        total = 0
        oov = 0
        known = set(self.units)
        for caption in captions:
            for word in normalize_tokens(caption):
                total += 1
                if any(c not in known for c in word):
                    oov += 1
        return (oov / total) if total else 0.0

    def unk_token_rate(self, captions) -> float:
        """Fraction of emitted subword tokens that are ``<unk>``."""
        total = 0
        unk = 0
        for caption in captions:
            for unit in self.tokenize(caption):
                total += 1
                if unit == UNK_TOKEN:
                    unk += 1
        return (unk / total) if total else 0.0

    # -------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        return {"format": BPE_FORMAT_TAG, "units": list(self.units),
                "merges": [list(m) for m in self.merges]}

    @classmethod
    def from_dict(cls, data: dict) -> "BPETokenizer":
        if not isinstance(data, dict) or data.get("format") != BPE_FORMAT_TAG:
            raise ValueError(
                f"not a {BPE_FORMAT_TAG} tokenizer payload: {str(data)[:80]!r}"
            )
        return cls(list(data["units"]), [tuple(m) for m in data["merges"]])

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            f.write("\n")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
