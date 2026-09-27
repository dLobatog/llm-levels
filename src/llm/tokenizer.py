"""Level 1 · Byte-level BPE tokenizer.

The vocabulary is `ranks: dict[bytes, int]`: each token's bytes mapped to its id. Single bytes are tokens, and
every merge adds a longer byte string with the next id. A lower id means the pair was merged earlier in training,
which is also the order in which pairs are merged when encoding. This is the representation tiktoken uses.
"""

from __future__ import annotations

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""


class BPETokenizer:
    def __init__(
        self,
        ranks: dict[bytes, int],
        pattern: str | None = GPT2_SPLIT_PATTERN,
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        """Wrap a vocabulary.

        ranks: token bytes -> id. Must contain every single byte.
        pattern: regex that splits text into chunks before BPE (merges never cross chunks). None = no split.
        special_tokens: e.g. {"<|endoftext|>": 50256}. Their ids are separate from `ranks`.
        """
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Number of ordinary tokens (len of ranks), excluding special tokens."""
        raise NotImplementedError

    @classmethod
    def train(cls, text: str, vocab_size: int, pattern: str | None = GPT2_SPLIT_PATTERN) -> BPETokenizer:
        """Learn a vocabulary of `vocab_size` tokens from `text`.

        Start from the 256 single bytes with ids 0..255 in byte order (bytes([i]) -> i). Split the text into chunks
        with `pattern`, then repeatedly merge the most frequent adjacent pair within chunks; the merged bytes get the
        next id. Stop at `vocab_size` tokens, or earlier if no adjacent pairs remain. Ties may be broken any way.
        """
        raise NotImplementedError

    @classmethod
    def from_tiktoken(cls, name: str = "gpt2") -> BPETokenizer:
        """Load an existing tiktoken vocabulary: its ranks, split pattern and special tokens."""
        raise NotImplementedError

    def encode(self, text: str, allowed_special: set[str] | None = None) -> list[int]:
        """Text -> ids.

        Special tokens listed in `allowed_special` become their single id. Anything else, including special-token
        strings that are not allowed, is encoded as ordinary text.
        """
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Ids -> text: concatenate each token's bytes and decode as UTF-8 with errors="replace"."""
        raise NotImplementedError
