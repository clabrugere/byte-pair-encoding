import pytest

from tokenizer import BPETokenizer
from tokenizer.bpe import merge_pair, most_frequent_pair, string_to_byte


@pytest.mark.parametrize(
    "input, output",
    [
        ("hello", [104, 101, 108, 108, 111]),
        ("world", [119, 111, 114, 108, 100]),
    ],
)
def test_string_to_byte(input: str, output: list[int]) -> None:
    assert string_to_byte(input, "utf-8") == output


@pytest.mark.parametrize(
    "input, expected",
    [
        (([1, 2, 3, 4, 3, 4]), ((3, 4), 2)),
        (([1, 2, 3, 4, 5, 6]), ((1, 2), 1)),
    ],
)
def test_most_frequent_pair(input: list[int], expected: tuple[tuple[int, int], int]) -> None:
    assert most_frequent_pair(input) == expected


@pytest.mark.parametrize(
    "input, pair, index, expected",
    [
        (([1, 2, 3, 4, 3, 4]), (3, 4), 5, [1, 2, 5, 5]),
        (([1, 2, 3, 4, 5, 6]), (1, 2), 7, [7, 3, 4, 5, 6]),
    ],
)
def test_merge_pair(input: list[int], pair: tuple[int, int], index: int, expected: list[int]) -> None:
    assert merge_pair(input, pair, index) == expected


@pytest.mark.parametrize(
    "tokenizer, corpus, input, add_special",
    [
        ("tokenizer", "corpus", "some regular sentence.", False),
        ("tokenizer", "corpus", "<BOS>now with special tokens<EOS>", True),
        ("tokenizer", "corpus", "😱 didn't appear in the corpus", False),
    ],
    indirect=["tokenizer", "corpus"],
)
def test_tokenizer(tokenizer: BPETokenizer, corpus: str, input: str, add_special: bool) -> None:
    if add_special:
        tokenizer.register_special_token("<BOS>")
        tokenizer.register_special_token("<EOS>")

    tokenizer.train(corpus)
    assert tokenizer.decode(tokenizer.encode(input)) == input
    assert tokenizer.next_id == len(tokenizer.id_to_token) + len(tokenizer.id_to_special)
