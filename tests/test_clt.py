import string

import pytest

from mtt.tokenizers.clt import CAP, CLT, to_display


def test_one_id_per_character(clt):
    assert len(clt.encode("hello world 123!")) == len("hello world 123!")


def test_round_trip_all_supported_characters(clt):
    text = string.ascii_lowercase + string.digits + string.punctuation + " "
    assert clt.decode(clt.encode(text)) == text


def test_round_trip_capitals_and_mixed_case(clt):
    for text in ["Hello", "HELLO", "hELLo World", "A", "Zebra, Yak & Xylophone."]:
        assert clt.decode(clt.encode(text)) == text


def test_capital_uses_cap_marker(clt):
    ids = clt.encode("Ab")
    assert ids[0] == clt.cap_id and len(ids) == 3
    assert clt.tokens[ids[0]] == CAP


def test_empty_string(clt):
    assert clt.encode("") == [] and clt.decode([]) == ""


def test_unsupported_character_raises(clt):
    with pytest.raises(ValueError):
        clt.encode("caf\u00e9")
    assert not clt.supports("caf\u00e9") and clt.supports("Cafe 1!")


def test_strict_and_lenient_decode(clt):
    dangling = [clt.cap_id]
    with pytest.raises(ValueError):
        clt.decode(dangling)
    assert clt.decode(dangling, strict=False) == ""
    with pytest.raises(ValueError):
        clt.decode([9999])
    assert clt.decode([9999] + clt.encode("a"), strict=False) == "a"


def test_display_view_example():
    assert to_display("how are you") == "8.15.23, 1.18.5, 25.15.21"


def test_display_view_extras():
    assert to_display("") == ""
    assert to_display("a") == "1"
    assert to_display("Hi!") == "^8.9.[!]"
    assert to_display("z9") == "26.[9]"
