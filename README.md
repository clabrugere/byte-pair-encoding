# Minimal byte-pair encoding for tokenization

## About <a name = "about"></a>

Simple implementation of a pure python BPE tokenizer following Andrej Karpathy's [video](https://www.youtube.com/watch?v=zduSFxRajkE) on the subject.

It varies a bit from the original implementation from the video:
- instances keep track of the next available integer index so there is no need to explicitly pass an integer when registering a special token.
- uses `Counter` object to count pair frequencies and return the most frequent pair for merging.
- `encode` implementation differs because it is clearer for me.
- add `stop_early` argument to the `train` method such that compression stops when no more pair appear more than once.
- special tokens are included in the vocabulary size because ultimately this value will determines the size of the embedding table in the language model.

## How it works

The algorithm for byte pair encoding is pretty straightforward: it is about representing strings as arrays of bytes and iteratively compressing this representation by merging the most common pairs of bytes and mapping them to a new id. As a result, an encoded string can be represented with a larger vocabulary but with a smaller list of tokens. Applied to language modeling, it allows to control the tradeoff between the size of the embedding table (proportional to the vocabulary size) and the amount of tokens that can be fit in a fixed context size (attention has a quadratic complexity with respect to the context size).

Here are the main steps for its training:
1. It builds an initial map `int -> byte` of the 256 elements corresponding to all the possible values in one byte. UTF-8 encodes string using contiguous sequences of 1, 2, 3 (4 bytes at most), representing code points. For instance "a" is represented as integer 97, "é" is represented by the two integers [195, 169], etc.
2. It initializes a map of signature `(int, int) -> int` to store the pairs merged together and their new indices.
3. It converts the input corpus into its bytes representation (ultimately represented as a list of integers).
4. Iteratively, it looks for the most common pair of consecutive tokens and merge it. The  new token resulting from the concatenation of the pair is affected a new integer id that is added to the vocabulary, increasing its size by one. The pair is added to the pair map with the corresponding id. It continues merging tokens until it has reached a given vocabulary size.

Now to encode an input string:
1. Converts the input to its byte representation.
2. Scans the encoded input looking for pairs present in the pair map in order to merge them using the corresponding indices. The result is a list of integers ready to be ingested by a language model. Its length is hopefully smaller than the raw bytes representation of the input thanks to the merged pairs.

Finally to decode a list of ids back to something readable, it simply uses the vocabulary to replace ids with tokens and the resulting list of bytes is finally decoded to get a string.

## Getting Started <a name = "getting_started"></a>

```python
from tokenizer import BPETokenizer


# load a corpus of text in a string
with open('../tests/corpus.txt', 'r') as f:
    corpus = f.read()

# train the tokenizer on it. By default, <BOS> and <EOS> special tokens are automatically registered
tokenizer = BPETokenizer(max_vocab_size=2048)
tokenizer.train(corpus)

# register additional special tokens. Its ID is automatically determined as the next available integer.
tokenizer.register_special_token("<EOP>")

# encode some string to get indices ready to be passed to a language model
encoded = tokenizer.encode("Some sentence you want to encode as a list of integers.")

# decode the output of a language model
encoded = [115, 101, 113, 117, 269, 99, 258, 294, 110, 380, 98, 268, 115]
decoded = tokenizer.decode(encoded)
```

## Potential improvements

1. The `train` expects a string as the input corpus. Obviously it's not very scalable to  large corpus at it would require to fit everything in memory. Instead, chunks of text could be streamed but that would require an extensive redesign.
2. When iterating to merge pairs, we always scan the entire corpus (with already merged pairs from the last iterations) to count the pairs frequencies and get the highest one. Surely there is a more efficient way to do that.