import pytest

from tokenizer import BPETokenizer


@pytest.fixture
def tokenizer() -> BPETokenizer:
    return BPETokenizer(max_vocab_size=384)


@pytest.fixture(scope="session")
def corpus() -> str:
    with open("tests/corpus.txt", "r") as f:
        content = f.read()

    return content
