"""Word Level Tokeniser (WLT): a small byte-level BPE built with HuggingFace ``tokenizers``.

Byte-level pre-tokenisation makes encode/decode exactly lossless for any text (including
spaces), and no ``<unk>`` is ever needed. The WLT has no special tokens of its own;
``MTTTokenizer`` places the special ids in front of it.

``ByteWLT`` is a dependency-free stand-in with the same interface (one id per UTF-8
byte). It is used in tests and quick debugging, never in the real experiment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol, Sequence


class WordTokenizer(Protocol):
    """Interface MTTTokenizer needs from a word-level tokeniser."""

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class WLT:
    """Small BPE (default vocab 2000) trained on the project's own corpus."""

    def __init__(self, tokenizer) -> None:  # tokenizers.Tokenizer
        self._tok = tokenizer

    @classmethod
    def train(cls, corpus: Iterable[str], vocab_size: int = 2000) -> "WLT":
        """Train a byte-level BPE on ``corpus`` (an iterable of strings)."""
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[],
            show_progress=False,
        )
        tok.train_from_iterator(list(corpus), trainer=trainer)
        return cls(tok)

    @classmethod
    def load(cls, path: str | Path) -> "WLT":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        self._tok.save(str(path))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        if text == "":
            return []
        return list(self._tok.encode(text).ids)

    def decode(self, ids: Sequence[int]) -> str:
        if len(ids) == 0:
            return ""
        return self._tok.decode(list(ids), skip_special_tokens=False)


class ByteWLT:
    """One token per UTF-8 byte. Test/debug stand-in for ``WLT``."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")
