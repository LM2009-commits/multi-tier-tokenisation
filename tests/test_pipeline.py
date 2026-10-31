import pytest

from mtt.data import LETTER_TASKS
from mtt.pipeline import (MODES, encode_example, extract_answer, generation_prompt,
                          sequence_lengths, training_sequence)
from mtt.router import RuleRouter
from mtt.tokenizers.mtt import BOS, CLT_CLOSE, CLT_OPEN, EOS, SEP


def first(ds, task, split="test"):
    return next(e for e in getattr(ds, split) if e.task == task)


def test_wlt_only_has_no_clt(tok, small_dataset):
    p, a = encode_example(first(small_dataset, "spell"), "wlt_only", tok)
    assert CLT_OPEN not in p + a and a[-1] == EOS


def test_clt_only_is_all_clt(tok, small_dataset):
    ex = first(small_dataset, "spell")
    p, a = encode_example(ex, "clt_only", tok)
    assert p[0] == CLT_OPEN and p[-1] == CLT_CLOSE and a[0] == CLT_OPEN and a[-2] == CLT_CLOSE
    assert tok.decode(p) == ex.prompt and tok.decode(a) == ex.answer


@pytest.mark.parametrize("task", ["count", "spell", "reverse", "nth"])
def test_mtt_routes_targets_and_letter_answers(tok, small_dataset, task):
    ex = first(small_dataset, task)
    p, a = encode_example(ex, "mtt", tok, RuleRouter())
    assert CLT_OPEN in p and tok.decode(p) == ex.prompt and tok.decode(a) == ex.answer
    assert (CLT_OPEN in a) == (task != "count")


def test_mtt_leaves_category_alone(tok, small_dataset):
    ex = first(small_dataset, "category")
    p, a = encode_example(ex, "mtt", tok, RuleRouter())
    assert CLT_OPEN not in p + a
    assert encode_example(ex, "wlt_only", tok) == (p, a)


def test_selfesc_emits_scratch_then_answer(tok, small_dataset):
    ex = first(small_dataset, "reverse")
    p, a = encode_example(ex, "selfesc", tok)
    assert CLT_OPEN not in p and a[0] == CLT_OPEN
    word = ex.target_words[0]
    assert tok.decode(a[: a.index(CLT_CLOSE) + 1]) == word
    assert extract_answer(a, "selfesc", tok) == ex.answer
    cat = first(small_dataset, "category")
    pc, ac = encode_example(cat, "selfesc", tok)
    assert CLT_OPEN not in ac and extract_answer(ac, "selfesc", tok) == cat.answer


def test_extract_answer_is_robust(tok):
    assert extract_answer([], "mtt", tok) == ""
    assert extract_answer([CLT_OPEN], "selfesc", tok) == ""  # scratch never closed
    assert extract_answer([tok.wlt_offset + 5, EOS, tok.wlt_offset + 6], "wlt_only", tok) == tok.decode(
        [tok.wlt_offset + 5])


@pytest.mark.parametrize("mode", MODES)
def test_gold_answers_round_trip_through_extraction(tok, small_dataset, mode):
    router = RuleRouter()
    for ex in small_dataset.test:
        _, a = encode_example(ex, mode, tok, router)
        assert extract_answer(a, mode, tok) == ex.answer


def test_sequence_layout(tok, small_dataset):
    ex = first(small_dataset, "nth")
    p, a = encode_example(ex, "mtt", tok, RuleRouter())
    seq, start = training_sequence(p, a)
    assert seq[0] == BOS and seq[start - 1] == SEP and seq[start:] == a
    assert generation_prompt(p) == seq[:start]


def test_clt_costs_sequence_length(tok, small_dataset):
    lengths = {m: sequence_lengths(small_dataset.test, m, tok, RuleRouter()) for m in MODES}
    for task in LETTER_TASKS:
        assert lengths["clt_only"][task]["full"] > lengths["mtt"][task]["full"]
        assert lengths["mtt"][task]["full"] >= lengths["wlt_only"][task]["full"] - 40  # BPE vs bytes
