"""Turns dataset examples into token sequences for each tokenisation mode.

Sequence layout (all modes)::

    <bos> prompt_ids <sep> answer_ids <eos>

The loss is computed on ``answer_ids <eos>`` only. Modes:

* ``wlt_only`` : everything through WLT.
* ``clt_only`` : prompt and answer entirely through CLT (one ``<CLT>..</CLT>`` span each).
* ``mtt``      : a router picks the prompt spans for CLT; the answer is CLT-encoded when
                 the router says the answer is made of letters (spell/reverse/nth).
* ``selfesc``  : (stretch) the prompt stays WLT; the model must itself emit
                 ``<CLT> target word </CLT>`` before the answer ("zoom in"), then the
                 answer is encoded exactly as in ``mtt`` mode.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .data import Dataset, Example, LETTER_TASKS, TASKS, generate_dataset, router_training_set, wlt_corpus
from .router import LearnedRouter, NullRouter, Route, Router, RuleRouter
from .tokenizers.mtt import BOS, CLT_CLOSE, CLT_OPEN, EOS, SEP, MTTTokenizer
from .tokenizers.wlt import WLT

MODES = ("wlt_only", "clt_only", "mtt", "selfesc")
ROUTERS = ("rule", "rule_words", "learned")
NEEDS_CLT_TASKS = frozenset(LETTER_TASKS)
LETTER_ANSWER_TASKS = frozenset({"spell", "reverse", "nth"})


def build_tokenizer(dataset: Dataset, include_test_words: bool = False,
                    vocab_size: int = 2000) -> MTTTokenizer:
    """Train the WLT BPE on the dataset's training corpus and wrap it in an MTTTokenizer."""
    wlt = WLT.train(wlt_corpus(dataset, include_test_words), vocab_size=vocab_size)
    return MTTTokenizer(wlt)


def build_router(name: str, dataset: Dataset | None = None, seed: int = 0) -> Router:
    """Create a router by name: ``rule``, ``rule_words`` (no letter spans) or ``learned``."""
    if name == "rule":
        return RuleRouter(route_letters=True)
    if name == "rule_words":
        return RuleRouter(route_letters=False)
    if name == "learned":
        if dataset is None:
            raise ValueError("the learned router needs the dataset to train on")
        prompts, needs, letter_answer = router_training_set(dataset)
        return LearnedRouter.fit(prompts, needs, letter_answer, seed=seed)
    raise ValueError(f"unknown router {name!r}; choose from {ROUTERS}")


def _full(text: str) -> list[tuple[int, int]]:
    return [(0, len(text))] if text else []


def encode_example(ex: Example, mode: str, tok: MTTTokenizer, router: Router | None = None
                   ) -> tuple[list[int], list[int]]:
    """Return ``(prompt_ids, answer_ids)``; ``answer_ids`` already ends with ``<eos>``."""
    if mode == "wlt_only":
        return tok.encode(ex.prompt), tok.encode(ex.answer) + [EOS]
    if mode == "clt_only":
        return (tok.encode(ex.prompt, _full(ex.prompt)),
                tok.encode(ex.answer, _full(ex.answer)) + [EOS])
    if mode == "mtt":
        route: Route = (router or RuleRouter()).route(ex.prompt)
        answer_spans = _full(ex.answer) if route.answer_clt else []
        return tok.encode(ex.prompt, list(route.spans)), tok.encode(ex.answer, answer_spans) + [EOS]
    if mode == "selfesc":
        scratch: list[int] = []
        if ex.task in NEEDS_CLT_TASKS:
            word = ex.target_words[0]
            scratch = tok.encode(word, _full(word))
        answer_spans = _full(ex.answer) if ex.task in LETTER_ANSWER_TASKS else []
        return tok.encode(ex.prompt), scratch + tok.encode(ex.answer, answer_spans) + [EOS]
    raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")


def generation_prompt(prompt_ids: Sequence[int]) -> list[int]:
    """Model input at inference time: ``<bos> prompt <sep>``."""
    return [BOS, *prompt_ids, SEP]


def training_sequence(prompt_ids: Sequence[int], answer_ids: Sequence[int]) -> tuple[list[int], int]:
    """Full sequence and the index of the first answer token (start of the loss region)."""
    head = generation_prompt(prompt_ids)
    return head + list(answer_ids), len(head)


def extract_answer(generated: Sequence[int], mode: str, tok: MTTTokenizer) -> str:
    """Turn generated ids (after ``<sep>``) into the answer string, tolerating junk.

    For ``selfesc`` the scratch ``<CLT> word </CLT>`` is dropped first; if the model never
    closes it the answer is empty (counts as wrong).
    """
    ids = list(generated)
    if EOS in ids:
        ids = ids[: ids.index(EOS)]
    if mode == "selfesc" and ids and ids[0] == CLT_OPEN:
        if CLT_CLOSE not in ids:
            return ""
        ids = ids[ids.index(CLT_CLOSE) + 1:]
    return tok.decode(ids, strict=False)


def sequence_lengths(examples: Sequence[Example], mode: str, tok: MTTTokenizer,
                     router: Router | None = None) -> dict[str, dict[str, float]]:
    """Mean token counts per task: full sequence, prompt part and answer part.

    ``full`` counts ``<bos> prompt <sep> answer <eos>``.
    """
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for ex in examples:
        p, a = encode_example(ex, mode, tok, router)
        s = sums[ex.task]
        s[0] += len(p) + len(a) + 2
        s[1] += len(p)
        s[2] += len(a)
        s[3] += 1
    return {t: {"full": s[0] / s[3], "prompt": s[1] / s[3], "answer": s[2] / s[3]}
            for t, s in sums.items() if s[3]}


__all__ = [
    "MODES", "ROUTERS", "TASKS", "NEEDS_CLT_TASKS", "LETTER_ANSWER_TASKS", "NullRouter",
    "build_tokenizer", "build_router", "encode_example", "generation_prompt",
    "training_sequence", "extract_answer", "sequence_lengths", "generate_dataset",
]
