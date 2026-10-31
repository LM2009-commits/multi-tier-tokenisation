import pytest

from mtt.tokenizers.mtt import (BOS, CLT_CLOSE, CLT_OPEN, EOS, NUM_SPECIAL, MTTTokenizer,
                                normalise_spans, to_display)


def roundtrip(tok, text, spans=None):
    ids = tok.encode(text, spans)
    assert tok.decode(ids) == text
    return ids


def test_no_spans_is_pure_wlt(tok):
    ids = roundtrip(tok, "Hello, world!")
    assert CLT_OPEN not in ids and all(NUM_SPECIAL <= i < tok.clt_offset for i in ids)


def test_empty_string(tok):
    assert roundtrip(tok, "") == [] and roundtrip(tok, "", []) == []


def test_span_is_wrapped_in_markers(tok):
    text = "How many r's are in strawberry?"
    start = text.index("strawberry")
    ids = roundtrip(tok, text, [(start, start + 10)])
    assert ids.count(CLT_OPEN) == 1 and ids.count(CLT_CLOSE) == 1
    inner = ids[ids.index(CLT_OPEN) + 1 : ids.index(CLT_CLOSE)]
    assert len(inner) == 10 and all(i >= tok.clt_offset for i in inner)  # one id per char


def test_span_at_start_end_and_whole_text(tok):
    roundtrip(tok, "banana split", [(0, 6)])
    roundtrip(tok, "banana split", [(7, 12)])
    ids = roundtrip(tok, "banana", [(0, 6)])
    assert ids[0] == CLT_OPEN and ids[-1] == CLT_CLOSE


def test_adjacent_and_multiple_spans(tok):
    roundtrip(tok, "abcdef", [(0, 3), (3, 6)])
    roundtrip(tok, "a b c d e", [(0, 1), (4, 5), (8, 9)])
    roundtrip(tok, "ab", [(1, 2), (0, 1)])  # unsorted input is fine


def test_capitals_and_punctuation_in_spans(tok):
    roundtrip(tok, "Say: HeLLo, World! 42", [(5, 21)])
    roundtrip(tok, "!!!???", [(0, 3)])


def test_non_ascii_outside_span_ok_inside_span_raises(tok):
    roundtrip(tok, "caf\u00e9 au lait", [(6, 8)])
    with pytest.raises(ValueError):
        tok.encode("caf\u00e9", [(0, 4)])


@pytest.mark.parametrize("spans", [[(2, 2)], [(3, 2)], [(0, 99)], [(-1, 2)], [(0, 3), (2, 4)]])
def test_invalid_spans_rejected(tok, spans):
    with pytest.raises(ValueError):
        tok.encode("abcdef", spans)


def test_normalise_spans():
    assert normalise_spans([(3, 5), (0, 2)], 5) == [(0, 2), (3, 5)]
    assert normalise_spans(None, 0) == []


def test_decode_ignores_specials(tok):
    ids = [BOS] + tok.encode("hi", [(0, 2)]) + [EOS]
    assert tok.decode(ids) == "hi"


def test_strict_decode_detects_malformed(tok):
    clt_id = tok.clt_offset + 5
    for bad in ([clt_id], [CLT_OPEN, clt_id], [CLT_CLOSE], [CLT_OPEN, CLT_OPEN, CLT_CLOSE],
                [CLT_OPEN, NUM_SPECIAL, CLT_CLOSE], [tok.vocab_size + 3]):
        with pytest.raises(ValueError):
            tok.decode(bad)
        tok.decode(bad, strict=False)  # lenient decoding never raises


def test_id_ranges_disjoint(tok):
    assert tok.clt_offset == NUM_SPECIAL + tok.wlt.vocab_size
    assert tok.vocab_size == tok.clt_offset + tok.clt.vocab_size


def test_display_function():
    assert to_display("how are you") == "8.15.23, 1.18.5, 25.15.21"
    assert MTTTokenizer.to_display("how are you") == "8.15.23, 1.18.5, 25.15.21"
