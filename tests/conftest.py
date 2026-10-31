"""Shared fixtures. Tokeniser tests run with the dependency-free ByteWLT and, when the
HuggingFace ``tokenizers`` library is installed, also with the real BPE WLT."""
import pytest

from mtt.data import generate_dataset, wlt_corpus
from mtt.tokenizers.clt import CLT
from mtt.tokenizers.mtt import MTTTokenizer
from mtt.tokenizers.wlt import ByteWLT


@pytest.fixture(scope="session")
def small_dataset():
    return generate_dataset(seed=0, n_train=200, n_test=30)


@pytest.fixture
def clt():
    return CLT()


@pytest.fixture(params=["byte", "hf"])
def tok(request, small_dataset):
    if request.param == "byte":
        return MTTTokenizer(ByteWLT())
    pytest.importorskip("tokenizers")
    from mtt.tokenizers.wlt import WLT

    return MTTTokenizer(WLT.train(wlt_corpus(small_dataset), vocab_size=600))
