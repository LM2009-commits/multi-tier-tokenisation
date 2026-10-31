"""Multi Tier Tokenisation: WLT for normal text, CLT for marked spans.

Id layout of the combined vocabulary::

    0 .. 6                    special tokens  <pad> <bos> <eos> <unk> <sep> <CLT> </CLT>
    7 .. 7+W-1                WLT ids (offset by 7)
    7+W .. 7+W+C-1            CLT ids (offset by 7+W)

A CLT span of the text is emitted as ``<CLT> c1 c2 ... cn </CLT>``. Because the id ranges
of WLT and CLT never overlap, the markers are redundant for decoding, but they are what the
model sees, they make sequences self-describing, and ``decode(strict=True)`` checks them.
"""
from __future__ import annotations

from typing import Sequence

from .clt import CLT, to_display
from .wlt import WordTokenizer

PAD, BOS, EOS, UNK, SEP, CLT_OPEN, CLT_CLOSE = range(7)
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<sep>", "<CLT>", "</CLT>"]
NUM_SPECIAL = len(SPECIAL_TOKENS)

Span = tuple[int, int]  # half-open character offsets [start, end)

__all__ = [
    "PAD", "BOS", "EOS", "UNK", "SEP", "CLT_OPEN", "CLT_CLOSE", "SPECIAL_TOKENS",
    "NUM_SPECIAL", "Span", "MTTTokenizer", "normalise_spans", "to_display",
]


def normalise_spans(spans: Sequence[Span] | None, length: int) -> list[Span]:
    """Sort and validate spans: ``0 <= start < end <= length``, no overlaps.

    Adjacent spans (``end == next start``) are allowed.
    """
    out = sorted((int(s), int(e)) for s, e in (spans or ()))
    prev_end = 0
    for s, e in out:
        if not 0 <= s < e <= length:
            raise ValueError(f"invalid span {(s, e)} for text of length {length}")
        if s < prev_end:
            raise ValueError(f"overlapping spans near offset {s}")
        prev_end = e
    return out


class MTTTokenizer:
    """Combined encoder/decoder over a word-level tokeniser and a CLT."""

    def __init__(self, wlt: WordTokenizer, clt: CLT | None = None) -> None:
        self.wlt = wlt
        self.clt = clt or CLT()
        self.wlt_offset = NUM_SPECIAL
        self.clt_offset = NUM_SPECIAL + wlt.vocab_size
        self.vocab_size = self.clt_offset + self.clt.vocab_size

    # ------------------------------------------------------------------ encode
    def encode(self, text: str, spans: Sequence[Span] | None = None) -> list[int]:
        """Encode ``text``; character ranges in ``spans`` go through CLT, rest through WLT.

        No ``<bos>``/``<eos>`` are added here (the pipeline does that).
        """
        ids: list[int] = []
        pos = 0
        for start, end in normalise_spans(spans, len(text)):
            ids.extend(self._encode_wlt(text[pos:start]))
            ids.append(CLT_OPEN)
            ids.extend(i + self.clt_offset for i in self.clt.encode(text[start:end]))
            ids.append(CLT_CLOSE)
            pos = end
        ids.extend(self._encode_wlt(text[pos:]))
        return ids

    def _encode_wlt(self, text: str) -> list[int]:
        return [i + self.wlt_offset for i in self.wlt.encode(text)]

    # ------------------------------------------------------------------ decode
    def decode(self, ids: Sequence[int], strict: bool = True) -> str:
        """Reconstruct text from ids.

        ``<pad> <bos> <eos> <unk> <sep>`` are skipped. With ``strict=True`` malformed
        sequences (CLT ids outside ``<CLT>..</CLT>``, unbalanced markers, bad ids) raise
        ``ValueError``; with ``strict=False`` (used on model output) tokens are decoded by
        id range and markers are ignored, so decoding never raises.
        """
        segments: list[tuple[str, list[int]]] = []  # (kind, local ids)
        in_clt = False

        def add(kind: str, local_id: int) -> None:
            if segments and segments[-1][0] == kind:
                segments[-1][1].append(local_id)
            else:
                segments.append((kind, [local_id]))

        for i in ids:
            if i in (PAD, BOS, EOS, UNK, SEP):
                continue
            if i == CLT_OPEN:
                if strict and in_clt:
                    raise ValueError("nested <CLT>")
                in_clt = True
            elif i == CLT_CLOSE:
                if strict and not in_clt:
                    raise ValueError("</CLT> without <CLT>")
                in_clt = False
            elif self.wlt_offset <= i < self.clt_offset:
                if strict and in_clt:
                    raise ValueError("WLT id inside <CLT> span")
                add("wlt", i - self.wlt_offset)
            elif self.clt_offset <= i < self.vocab_size:
                if strict and not in_clt:
                    raise ValueError("CLT id outside <CLT> span")
                add("clt", i - self.clt_offset)
            elif strict:
                raise ValueError(f"invalid token id {i}")
        if strict and in_clt:
            raise ValueError("unterminated <CLT> span")

        parts = [
            self.wlt.decode(local) if kind == "wlt" else self.clt.decode(local, strict=strict)
            for kind, local in segments
        ]
        return "".join(parts)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def to_display(text: str) -> str:
        """Numeric display view of ``text`` (see ``clt.to_display``)."""
        return to_display(text)

    def id_to_token(self, i: int) -> str:
        """Readable name of an id (for debugging)."""
        if i < NUM_SPECIAL:
            return SPECIAL_TOKENS[i]
        if i < self.clt_offset:
            return f"w{i - self.wlt_offset}"
        return f"c:{self.clt.tokens[i - self.clt_offset]}"
