"""Level 1: a byte-level BPE tokenizer.

The vocabulary is `ranks`, a mapping from each token's bytes to its id. Every
single byte is a token, and every merge adds a longer byte string with the next
id. A lower id means the pair was merged earlier in training, which is also the
order in which pairs are merged when encoding. tiktoken uses the same
representation.

Typical usage example:

    gpt2 = BPETokenizer.from_tiktoken("gpt2")
    ids = gpt2.encode("Hello, world!")
    text = gpt2.decode(ids)
"""

from __future__ import annotations
import itertools

from ast import Tuple
from collections.abc import Mapping, Sequence, Set
from collections import Counter

import regex
import tiktoken

# GPT-2's pre-tokenization regex.
GPT2_SPLIT_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++"
    r"|\s++$|\s+(?!\S)|\s"
)


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replaces every occurrence of pair in ids with new_id.

    Scans left to right so overlapping occurrences merge greedily:
    [a,a,a] with pair (a,a) becomes [new_id, a]
    """
    i = 0
    result = []
    while i + 1 < len(ids):
        j = i + 1
        pair_to_eval = (ids[i], ids[j])
        if pair_to_eval == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1

    if i < len(ids):
        result.append(ids[i])

    return result


class BPETokenizer:
    """A byte-level BPE vocabulary that encodes text to ids and back.

    Attributes:
        ranks: Token bytes -> id. Contains every single byte.
        pattern: Regex that splits text into chunks before BPE, or None to
            treat the whole text as one chunk.
        special_tokens: Special-token string -> id, e.g.
            {"<|endoftext|>": 50256}. These ids are not in `ranks`.
    """

    def __init__(
        self,
        ranks: Mapping[bytes, int],
        pattern: str | None = GPT2_SPLIT_PATTERN,
        special_tokens: Mapping[str, int] | None = None,
    ):
        """Wraps an existing vocabulary.

        Args:
            ranks: Token bytes -> id. Must contain every single byte.
            pattern: Regex that splits text into chunks before BPE. Merges
                never cross chunks. None means the whole text is one chunk.
            special_tokens: Special-token string -> id. None means there are
                no special tokens.
        """
        self.ranks = ranks
        self.vocab = {}
        for token_bytes, token_id in self.ranks.items():
            self.vocab[token_id] = token_bytes
        self.pattern = pattern
        self.special_tokens = special_tokens

    @property
    def vocab_size(self) -> int:
        """The number of ordinary tokens, excluding special tokens."""
        return len(self.vocab)

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        pattern: str | None = GPT2_SPLIT_PATTERN,
    ) -> BPETokenizer:
        """Learns a vocabulary from text.

        Starts from the 256 single bytes, with ids 0..255 in byte order
        (bytes([i]) -> i). Splits the text into chunks with `pattern`, then
        repeatedly merges the most frequent adjacent pair within chunks. The
        merged bytes get the next id. Ties may be broken any way.

        Args:
            text: The training text.
            vocab_size: Target number of tokens, including the 256 bytes.
                Training stops earlier if no adjacent pairs remain.
            pattern: Split regex, as in `__init__`.

        Returns:
            A tokenizer with the learned ranks and no special tokens.
        """

        chunks = text
        if pattern:
            chunks = regex.findall(pattern, text)
        else:
            chunks = [text]

        ranks = {bytes([i]): i for i in range(256)}
        vocab = {}
        for token_bytes, token_id in ranks.items():
            vocab[token_id] = token_bytes

        encoded_chunks = [list(chunk.encode("utf-8")) for chunk in chunks]

        while len(vocab) < vocab_size:
            counter = Counter()
            for chunk in encoded_chunks:
                counter.update(zip(chunk, chunk[1:]))

            if not counter:
                break
            new_id = len(ranks)
            pair = counter.most_common(1)[0][0]
            a, b = pair
            ranks[vocab[a] + vocab[b]] = new_id
            vocab[new_id] = vocab[a] + vocab[b]
            encoded_chunks = [
                _merge(chunk, pair, new_id) for chunk in encoded_chunks
            ]

        return cls(
            ranks=ranks,
            pattern=pattern,
        )

    @classmethod
    def from_tiktoken(cls, name: str = "gpt2") -> BPETokenizer:
        """Loads an existing tiktoken vocabulary.

        Args:
            name: A tiktoken encoding name, e.g. "gpt2".

        Returns:
            A tokenizer with the encoding's ranks, split pattern and special
            tokens.
        """
        enc = tiktoken.get_encoding(name)
        return cls(
            ranks=enc._mergeable_ranks,
            pattern=enc._pat_str,
            special_tokens=enc._special_tokens,
        )

    def _encode_chunk(self, chunk_bytes: bytes) -> list[int]:
        """Encodes one chunk by applying merges, lowest rank first."""
        ids = [self.ranks[bytes([b])] for b in chunk_bytes]
        while True:
            candidates = []
            for a, b in itertools.pairwise(ids):
                joined = self.vocab[a] + self.vocab[b]
                if joined in self.ranks:
                    candidates.append((self.ranks[joined], (a, b)))
            if not candidates:
                break

            rank, pair = min(candidates)
            ids = _merge(ids, pair, rank)

        return ids

    def encode(
        self,
        text: str,
        allowed_special: Set[str] | None = None,
    ) -> list[int]:
        """Encodes text to token ids.

        Args:
            text: The text to encode.
            allowed_special: Special tokens to encode as their single id. All
                other text, including special-token strings not listed here,
                is encoded as ordinary text.

        Returns:
            The token ids.
        """
        allowed = allowed_special or set()
        special = {
            name: token_id
            for name, token_id in (self.special_tokens or {}).items()
            if name in allowed
        }
        if not special:
            return self._encode_ordinary(text)

        # Longest first, so a special token never loses to its own prefix.
        names = sorted(special, key=len, reverse=True)
        pattern = "(" + "|".join(regex.escape(name) for name in names) + ")"
        ids = []
        for piece in regex.split(pattern, text):
            if piece in special:
                ids.append(special[piece])
            elif piece:
                ids.extend(self._encode_ordinary(piece))
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        """Encodes text with no special tokens: split into chunks, BPE each."""
        chunks = regex.findall(self.pattern, text) if self.pattern else [text]
        ids = []
        for chunk in chunks:
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Decodes token ids to text.

        Args:
            ids: The token ids.

        Returns:
            Every token's bytes concatenated and decoded as UTF-8, with
            errors="replace".
        """
        result = []
        for token_id in ids:
            result.append(self.vocab[token_id])

        return b"".join(result).decode("utf-8", errors="replace")
