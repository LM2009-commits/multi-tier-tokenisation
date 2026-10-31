"""Routers decide which character spans of a prompt are sent through the CLT.

A router maps a prompt to a ``Route``: the CLT spans (character offsets of the *target
words*, never the whole prompt), the detected task name and whether the answer should be
CLT-encoded too. Any object with ``route(prompt) -> Route`` is a router (``Router``
protocol), so the rule-based version and the learned version are interchangeable.
"""
from __future__ import annotations

import re
import zlib
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

Span = tuple[int, int]


@dataclass(frozen=True)
class Route:
    """Routing decision for one prompt."""

    spans: tuple[Span, ...] = ()
    task: str | None = None
    answer_clt: bool = False  # should the answer be CLT-encoded?

    @property
    def needs_clt(self) -> bool:
        return len(self.spans) > 0


class Router(Protocol):
    """Interface for all routers (rule-based, learned, ...)."""

    def route(self, prompt: str) -> Route: ...


class NullRouter:
    """Never routes anything to CLT (equivalent to WLT-only)."""

    def route(self, prompt: str) -> Route:
        return Route()


# --------------------------------------------------------------------------- rules
_Q = r"[\"'\u201c\u201d\u2018\u2019]?"  # optional quote character
_W = r"(?!(?:the|a|an|my|this|that|these|those)\b)(?P<word>[a-z]+)"  # target word
_LETTER = rf"{_Q}(?P<letter>[a-z]){_Q}(?![a-z])(?:['\u2019]?s)?"  # r / 'r' / r's
_ORD = (
    r"(?P<ord>\d+(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|"
    r"ninth|tenth|last)"
)
_PREP_IN = r"(?:in|inside|within)"
_THE_WORD = r"(?:the\s+word\s+)?"


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_COUNT = (
    _rx(
        rf"\bhow\s+many\s+(?:times\s+(?:does|do|is|are)\s+)?(?:the\s+)?(?:letters?\s+)?{_LETTER}"
        rf"(?:\s+(?:are|is|appear|appears|occur|occurs|show\s+up))*(?:\s+there)?"
        rf"\s+{_PREP_IN}\s+{_THE_WORD}{_Q}{_W}"
    ),
    _rx(
        rf"\bcount\s+(?:the\s+)?(?:number\s+of\s+)?(?:times\s+)?(?:(?:the\s+)?letters?\s+)?"
        rf"{_LETTER}(?:\s+(?:appears?|occurs?))?\s+(?:{_PREP_IN}|of)\s+{_THE_WORD}{_Q}{_W}"
    ),
)
_SPELL = (_rx(rf"\bspell\b(?:\s+out\b)?(?:\s+the\b)?(?:\s+word\b)?\s*:?\s*{_Q}{_W}"),)
_REVERSE = (
    _rx(
        rf"\breverse\b(?:\s+the\b)?(?:\s+(?:word\b|letters\s+(?:of|in)\b|spelling\s+of\b))?"
        rf"\s*:?\s*{_Q}{_W}"
    ),
    _rx(
        rf"(?<![a-z]){_Q}{_W}{_Q}\s+(?:(?:spelled|spelt|written)\s+)?"
        rf"(?:backwards?|in\s+reverse)\b"
    ),
)
_NTH = (
    _rx(rf"\b{_ORD}\s+(?:letter|character)\s+(?:of|in)\s+{_THE_WORD}{_Q}{_W}"),
    _rx(
        rf"\b(?:letter|character)\s+(?:number\s+|#\s*)?(?P<ord>\d+)\s+(?:of|in)\s+"
        rf"{_THE_WORD}{_Q}{_W}"
    ),
)
_CIPHER_TRIG = r"(?:caesar|rot\s*-?\s*\d+|atbash|a1z26|cipher(?:text)?|encrypt|decrypt)"
_CIPHER = (
    _rx(
        rf"\b(?:shift|rotate)\s+(?:(?:each|every)\s+letter\s+(?:of|in)\s+|the\s+letters\s+(?:of|in)\s+)?"
        rf"{_Q}{_W}{_Q}\s+(?:forward\s+|back(?:ward)?s?\s+)?by\s+\d+"
    ),
    _rx(rf"\b(?:encrypt|decrypt)\s+{_THE_WORD}{_Q}{_W}"),
    _rx(
        rf"\b{_CIPHER_TRIG}\b[^\n]*?(?:\b(?:to|on|of|for|with|into)\b|:)\s*{_Q}{_W}{_Q}\s*[.?!]?\s*$"
    ),
)


@dataclass(frozen=True)
class _Rule:
    task: str
    patterns: tuple[re.Pattern[str], ...]
    answer_clt: bool


_RULES: tuple[_Rule, ...] = (
    _Rule("cipher", _CIPHER, True),
    _Rule("reverse", _REVERSE, True),
    _Rule("count", _COUNT, False),  # the answer is a number: stays WLT
    _Rule("nth", _NTH, True),
    _Rule("spell", _SPELL, True),
)


class RuleRouter:
    """Version 1 router: regex triggers that extract the target word(s) as spans.

    Args:
        route_letters: for letter counting, also route the single target letter
            (``r`` in ``r's``) so it is seen in the same alphabet as the word.
    """

    def __init__(self, route_letters: bool = True) -> None:
        self.route_letters = route_letters

    def route(self, prompt: str) -> Route:
        for rule in _RULES:
            for pattern in rule.patterns:
                match = pattern.search(prompt)
                if match is None:
                    continue
                spans: list[Span] = [match.span("word")]
                if self.route_letters and "letter" in pattern.groupindex:
                    spans.append(match.span("letter"))
                return Route(tuple(sorted(spans)), rule.task, rule.answer_clt)
        return Route()


# ------------------------------------------------------------------- learned router
class HashedLogReg:
    """Logistic regression on hashed character n-grams, trained with plain SGD (numpy)."""

    def __init__(self, n_features: int = 4096, ngram_range: tuple[int, int] = (2, 4)) -> None:
        self.n_features = n_features
        self.ngram_range = ngram_range
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

    def indices(self, text: str) -> np.ndarray:
        """Hashed (deterministic crc32) char n-gram feature indices of ``text``."""
        t = f"^{text.lower()}$"
        lo, hi = self.ngram_range
        idx = {
            zlib.crc32(t[i : i + n].encode("utf-8")) % self.n_features
            for n in range(lo, hi + 1)
            for i in range(len(t) - n + 1)
        }
        return np.fromiter(sorted(idx), dtype=np.int64, count=len(idx))

    def _logit(self, idx: np.ndarray) -> float:
        z = self.w[idx].sum() / np.sqrt(max(len(idx), 1)) + self.b
        return float(np.clip(z, -30.0, 30.0))

    def predict_proba(self, text: str) -> float:
        return 1.0 / (1.0 + np.exp(-self._logit(self.indices(text))))

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        epochs: int = 10,
        lr: float = 0.5,
        seed: int = 0,
    ) -> "HashedLogReg":
        feats = [self.indices(t) for t in texts]
        y = np.asarray(labels, dtype=np.float64)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            for i in rng.permutation(len(feats)):
                idx = feats[i]
                p = 1.0 / (1.0 + np.exp(-self._logit(idx)))
                g = p - y[i]
                self.w[idx] -= lr * g / np.sqrt(len(idx))
                self.b -= 0.1 * lr * g
        return self


_WORD_RE = re.compile(r"[A-Za-z]+")


class LearnedRouter:
    """Learned router: two tiny classifiers plus a frequency-based span extractor.

    * ``needs`` predicts whether the prompt needs CLT.
    * ``letter_answer`` predicts whether the answer is made of letters (CLT-encoded).
    * Spans are the alphabetic words that are *rare* in the training prompts (template
      words appear thousands of times, target words only a handful). Single letters are
      NOT routed (limitation: use ``RuleRouter(route_letters=False)`` for a like-for-like
      comparison).
    """

    def __init__(
        self,
        needs: HashedLogReg,
        letter_answer: HashedLogReg,
        template_words: frozenset[str],
        threshold: float = 0.5,
    ) -> None:
        self.needs = needs
        self.letter_answer = letter_answer
        self.template_words = template_words
        self.threshold = threshold

    @classmethod
    def fit(
        cls,
        prompts: Sequence[str],
        needs_labels: Sequence[int],
        letter_answer_labels: Sequence[int],
        min_df_frac: float = 0.004,
        seed: int = 0,
    ) -> "LearnedRouter":
        """Train on labelled prompts. Template words = words in >= ``min_df_frac`` of prompts."""
        df: Counter[str] = Counter()
        for p in prompts:
            df.update({m.group().lower() for m in _WORD_RE.finditer(p)})
        min_df = max(5, int(min_df_frac * len(prompts)))
        template = frozenset(w for w, c in df.items() if c >= min_df)
        needs = HashedLogReg().fit(prompts, needs_labels, seed=seed)
        letter = HashedLogReg().fit(prompts, letter_answer_labels, seed=seed + 1)
        return cls(needs, letter, template)

    def route(self, prompt: str) -> Route:
        if self.needs.predict_proba(prompt) < self.threshold:
            return Route()
        spans = tuple(
            m.span()
            for m in _WORD_RE.finditer(prompt)
            if len(m.group()) >= 2 and m.group().lower() not in self.template_words
        )
        if not spans:
            return Route()
        answer_clt = self.letter_answer.predict_proba(prompt) >= self.threshold
        return Route(spans, "learned", answer_clt)
