"""Caption-trained tokenizer for Vidgin's own text encoders.

Phase 2 of the targeted improvement programme (2026-10-22). Until now the
text encoders tokenized in two lossy ways:

* the legacy encoder used a fixed 27-word table — every real-world word
  outside it collapsed into the same ``<unk>`` id;
* the flagship transformer encoder hashed each word with crc32 into one
  of 8,192 buckets — different words can collide into the same bucket,
  and the mapping has no relationship to the project's own captions.

:class:`CaptionTokenizer` replaces both with a word-level vocabulary
*trained on Vidgin's own dataset captions* (built by
``scripts/build_tokenizer.py``): deterministic (same captions -> same
vocabulary), serializable to JSON, and stored inside every checkpoint
trained with it, so training, resume, and inference always tokenize
identically. Unknown words map to ``<unk>`` — a tokenizer gives the
encoder distinct symbols to learn from; the *meaning* still has to be
learned by the encoder during training, and words never seen in the
caption corpus cannot be understood. That limitation is by design and
documented, not hidden.

:func:`audit_hash_collisions` quantifies the old scheme: it groups words
that the crc32 word-hash would force to share one embedding bucket.
"""

from __future__ import annotations

import json
import re
import zlib
from collections import Counter

FORMAT_TAG = "vidgin-caption-tokenizer-v1"

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def normalize_tokens(text: str) -> list[str]:
    """Lowercase word tokens: letters/digits, keeping inner apostrophes.

    Punctuation is dropped (it carries little signal in short captions
    and would explode the vocabulary); numbers stay as their own tokens.
    """
    return _WORD_RE.findall(text.lower())


def hash_bucket(word: str, buckets: int = 8192) -> int:
    """The legacy flagship mapping: crc32 of the word into 1..buckets-1."""
    return 1 + zlib.crc32(word.encode("utf-8")) % (buckets - 1)


def audit_hash_collisions(words, buckets: int = 8192) -> dict[int, list[str]]:
    """Groups of DISTINCT words the word-hash forces into one bucket.

    Returns ``{bucket_id: [word, ...]}`` containing only buckets shared
    by two or more distinct words — the distinctions the old tokenizer
    could never represent.
    """
    by_bucket: dict[int, list[str]] = {}
    for word in sorted(set(words)):
        by_bucket.setdefault(hash_bucket(word, buckets), []).append(word)
    return {b: ws for b, ws in by_bucket.items() if len(ws) > 1}


class CaptionTokenizer:
    """Word-level tokenizer with a vocabulary learned from captions."""

    def __init__(self, words: list[str]):
        words = list(words)
        if words[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError(
                "tokenizer words must start with the special tokens "
                f"{SPECIAL_TOKENS}, got {words[:4]}"
            )
        if len(set(words)) != len(words):
            raise ValueError("tokenizer vocabulary contains duplicates")
        self.words = words
        self.word2id = {w: i for i, w in enumerate(words)}

    def __len__(self) -> int:
        return len(self.words)

    def __eq__(self, other) -> bool:
        return isinstance(other, CaptionTokenizer) and other.words == self.words

    # ------------------------------------------------------------- training
    @classmethod
    def train(cls, captions, max_vocab: int = 8192, min_freq: int = 1) -> "CaptionTokenizer":
        """Build a vocabulary from caption text.

        Words are ordered by descending frequency, ties broken
        alphabetically, so the vocabulary is a deterministic function of
        the caption multiset. ``max_vocab`` counts the special tokens.
        """
        if max_vocab <= len(SPECIAL_TOKENS):
            raise ValueError(f"max_vocab must exceed {len(SPECIAL_TOKENS)} specials")
        counts: Counter[str] = Counter()
        for caption in captions:
            counts.update(normalize_tokens(caption))
        kept = [w for w, c in counts.items() if c >= min_freq]
        kept.sort(key=lambda w: (-counts[w], w))
        kept = kept[: max_vocab - len(SPECIAL_TOKENS)]
        return cls(SPECIAL_TOKENS + kept)

    # ------------------------------------------------------------- encoding
    def tokenize(self, text: str) -> list[str]:
        return normalize_tokens(text)

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        """[BOS] + word ids (unknown -> UNK) + [EOS], optionally truncated."""
        ids = [BOS_ID]
        ids.extend(self.word2id.get(w, UNK_ID) for w in normalize_tokens(text))
        ids.append(EOS_ID)
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids) -> str:
        words = []
        for i in ids:
            token = self.words[int(i)] if 0 <= int(i) < len(self.words) else UNK_TOKEN
            if token in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
                continue
            words.append(token)
        return " ".join(words)

    def oov_rate(self, captions) -> float:
        """Fraction of caption tokens NOT in the vocabulary (0.0 = perfect)."""
        total = 0
        oov = 0
        for caption in captions:
            for token in normalize_tokens(caption):
                total += 1
                if token not in self.word2id:
                    oov += 1
        return (oov / total) if total else 0.0

    # -------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        return {"format": FORMAT_TAG, "words": list(self.words)}

    @classmethod
    def from_dict(cls, data: dict) -> "CaptionTokenizer":
        if not isinstance(data, dict) or data.get("format") != FORMAT_TAG:
            raise ValueError(
                f"not a {FORMAT_TAG} tokenizer payload: {str(data)[:80]!r}"
            )
        return cls(list(data["words"]))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            f.write("\n")

    @classmethod
    def load(cls, path: str) -> "CaptionTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def tokenizer_from_dict(data: dict):
    """Rebuild EITHER tokenizer kind from a serialized payload.

    Dispatches on the payload's ``format`` tag: word-level
    (:class:`CaptionTokenizer`) or subword
    (:class:`~vidgen.text.bpe.BPETokenizer`). Checkpoint code must use
    this instead of assuming the word-level class, so a checkpoint
    trained with BPE resumes and samples with BPE. Raises ValueError
    for unknown formats — an incompatible tokenizer is a hard error,
    never a silent fallback.
    """
    if not isinstance(data, dict):
        raise ValueError(f"tokenizer payload must be a dict, got {type(data)}")
    fmt = data.get("format")
    if fmt == FORMAT_TAG:
        return CaptionTokenizer.from_dict(data)
    from vidgen.text.bpe import BPE_FORMAT_TAG, BPETokenizer

    if fmt == BPE_FORMAT_TAG:
        return BPETokenizer.from_dict(data)
    raise ValueError(f"unknown tokenizer format: {fmt!r}")


def load_tokenizer(path: str):
    """Load a tokenizer JSON file of either kind (see tokenizer_from_dict)."""
    with open(path, "r", encoding="utf-8") as f:
        return tokenizer_from_dict(json.load(f))
