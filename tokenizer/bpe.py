import logging
import re
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def string_to_byte(input: str, encoding: str) -> list[int]:
    return list(map(int, input.encode(encoding)))


def most_frequent_pair(indices: list[int]) -> tuple[tuple[int, int], int]:
    counts = Counter(zip(indices, indices[1:]))
    pair, count = counts.most_common(1)[0]

    return pair, count


def merge_pair(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    merged = []
    i = 0

    while i < len(indices):
        if i < len(indices) - 1 and (indices[i], indices[i + 1]) == pair:
            merged.append(new_index)
            i += 2
        else:
            merged.append(indices[i])
            i += 1

    return merged


class BPETokenizer:
    def __init__(
        self,
        max_vocab_size: int,
    ):
        if max_vocab_size <= 256:
            logger.warning("max_vocab_size is smaller than 256 so no pair merges will be done.")

        self.max_vocab_size = max_vocab_size
        self.reset()

    def reset(self) -> None:
        # UTF-8 encoding represents characters with 1, 2, 3 or 4 consecutive bytes, which means that the base vocabulary
        # size is 256. It also guarantee that we can't have out of vocabulary tokens as long as the input string can be
        # encoded in UTF-8.
        self.pairs = {}
        self.id_to_token = {i: bytes([i]) for i in range(256)}
        self.next_id = 256
        self.special_to_id = {}
        self.id_to_special = {}

        self.register_special_token("<BOS>")
        self.register_special_token("<EOS>")

    def register_special_token(self, token: str) -> None:
        if token not in self.special_to_id:
            logger.info(f"Registering special token {token} with id {self.next_id}.")

            self.special_to_id[token] = self.next_id
            self.id_to_special[self.next_id] = token
            self.next_id += 1

    def train(self, input: str, stop_early: bool = False) -> None:
        indices = string_to_byte(input, "utf-8")
        num_iter = 0

        while self.vocab_size < self.max_vocab_size:
            num_iter += 1
            pair, count = most_frequent_pair(indices)

            if stop_early and count == 1:
                break

            indices = merge_pair(indices, pair, self.next_id)
            self.pairs[pair] = self.next_id
            self.id_to_token[self.next_id] = self.id_to_token[pair[0]] + self.id_to_token[pair[1]]
            self.next_id += 1

        logger.warning(f"Stopping compression after {num_iter} pair merges with vocab size of {self.vocab_size}.")

    def _encode_non_special(self, input: str) -> list[int]:
        indices = string_to_byte(input, "utf-8")
        i = 0

        while i < len(indices) - 1:
            pair = (indices[i], indices[i + 1])
            if pair in self.pairs:
                indices = merge_pair(indices, pair, self.pairs[pair])
            else:
                i += 1

        return indices

    def encode(self, input: str) -> list[int]:
        special_pattern = re.compile(f"({'|'.join(re.escape(t) for t in self.special_to_id.keys())})")
        splits = special_pattern.split(input)

        indices = []

        for split in splits:
            if split in self.special_to_id:
                indices.append(self.special_to_id[split])
            else:
                indices.extend(self._encode_non_special(split))

        return indices

    def decode(self, indices: list[int]) -> str:
        decoded = []

        for id in indices:
            if id in self.id_to_special:
                decoded.append(self.id_to_special[id].encode("utf-8"))
            else:
                decoded.append(self.id_to_token[id])

        decoded = b"".join(decoded).decode("utf-8")

        return decoded

    @property
    def vocab_size(self) -> int:
        return self.next_id
