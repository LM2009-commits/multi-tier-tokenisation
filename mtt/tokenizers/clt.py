"""Character Level Tokeniser (CLT).

Every supported character is exactly ONE token id. Supported characters are
``a-z``, ``0-9``, ASCII punctuation and the space. A capital letter is encoded as the
two tokens ``<CAP>`` + lowercase letter, so the round trip is lossless.

The numeric "display view" (``a=1 ... z=26``) is for humans only; see ``to_display``.
"""
from __future__ import annotations

import string
from typing import Sequence

CAP = "<CAP>"
LETTERS = string.ascii_lowercase
SUPPORTED_CHARS = LETTERS + string.digits + string.punctuation + " "


class CLT:
    """Lossless character tokeniser with a ``<CAP>`` marker for capitals.

    Ids are local to the CLT (``0 .. vocab_size-1``); ``MTTTokenizer`` adds an offset.
    """

    def __init__(self) -> None:
        self.tokens: list[str] = [CAP, *SUPPORTED_CHARS]
        self._to_id: dict[str, int] = {tok: i for i, tok in enumerate(self.tokens)}
        self.cap_id: int = self._to_id[CAP]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def supports(self, text: str) -> bool:
        """Return True if every character of ``text`` can be encoded."""
        return all(ch in SUPPORTED_CHARS or ch in string.ascii_uppercase for ch in text)

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` to ids, one id per character (two for a capital letter)."""
        ids: list[int] = []
        for ch in text:
            if ch in string.ascii_uppercase:
                ids.append(self.cap_id)
                ids.append(self._to_id[ch.lower()])
            elif ch in SUPPORTED_CHARS:
                ids.append(self._to_id[ch])
            else:
                raise ValueError(f"CLT cannot encode character {ch!r}")
        return ids

    def decode(self, ids: Sequence[int], strict: bool = True) -> str:
        """Decode ids back to text.

        With ``strict=False`` malformed input (out-of-range ids, a dangling ``<CAP>``)
        is skipped instead of raising. This is what we use on model output.
        """
        out: list[str] = []
        pending_cap = False
        for i in ids:
            if not 0 <= i < self.vocab_size:
                if strict:
                    raise ValueError(f"invalid CLT id {i}")
                continue
            if i == self.cap_id:
                if pending_cap and strict:
                    raise ValueError("two consecutive <CAP> tokens")
                pending_cap = True
                continue
            ch = self.tokens[i]
            if pending_cap:
                if ch in LETTERS:
                    ch = ch.upper()
                elif strict:
                    raise ValueError(f"<CAP> must be followed by a letter, got {ch!r}")
                pending_cap = False
            out.append(ch)
        if pending_cap and strict:
            raise ValueError("dangling <CAP> at end of sequence")
        return "".join(out)


def _display_symbol(ch: str) -> str:
    if ch in LETTERS:
        return str(ord(ch) - 96)
    if ch in string.ascii_uppercase:
        return "^" + str(ord(ch.lower()) - 96)
    if ch in string.digits + string.punctuation:
        return f"[{ch}]"
    raise ValueError(f"cannot display character {ch!r}")


def to_display(text: str) -> str:
    """Human-readable numeric view: ``"how are you" -> "8.15.23, 1.18.5, 25.15.21"``.

    Letters become 1-26 joined by dots, words are joined by ", ". Capitals get a ``^``
    prefix (``H -> ^8``) and digits/punctuation are shown in brackets (``!`` -> ``[!]``).
    This view is for display only; the model always sees one token id per character.
    """
    if text == "":
        return ""
    words = text.split(" ")
    return ", ".join(".".join(_display_symbol(ch) for ch in w) for w in words)
