import pytest

pytest.importorskip("tokenizers")

from mtt.data import generate_dataset, wlt_corpus  # noqa: E402
from mtt.tokenizers.wlt import WLT  # noqa: E402


def test_bpe_trains_round_trips_and_saves(tmp_path):
    ds = generate_dataset(0, 200, 20)
    wlt = WLT.train(wlt_corpus(ds), vocab_size=2000)
    assert 256 < wlt.vocab_size <= 2000
    for text in ["", "How many r's are in strawberry?", "  spaces  ", "UPPER lower 123 !?", "caf\u00e9"]:
        assert wlt.decode(wlt.encode(text)) == text
    wlt.save(tmp_path / "wlt.json")
    again = WLT.load(tmp_path / "wlt.json")
    assert again.encode("Spell the word: banana") == wlt.encode("Spell the word: banana")


def test_common_words_are_single_tokens():
    ds = generate_dataset(0, 400, 20)
    wlt = WLT.train(wlt_corpus(ds), vocab_size=2000)
    word = ds.train_words[0]
    assert len(wlt.encode(" " + word)) < len(word)  # BPE compresses seen words
