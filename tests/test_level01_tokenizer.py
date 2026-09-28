"""Level 1 proof: byte-level BPE that trains, round-trips and matches GPT-2."""

import random

import pytest
import tiktoken

from llm import tokenizer

HAND_PICKED = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "I'm sure they'll say we've done it, but I don't think it's right.",
    "def add(a, b):\n    return a + b\n",
    "for (int i = 0; i < n; ++i) { total += x[i]; }",
    "12345 3.14159 1,000,000 2024-09-26 0.0001",
    "1 2 3 4 5",
    "   leading spaces",
    "trailing spaces   ",
    "tabs\tand\ttabs",
    "\n\nnewlines\n\n\n",
    "mixed  \n  whitespace \t\n",
    "héllo wörld, ça va? naïve café",
    "日本語のテキストを分割します。",
    "Привет, как дела?",
    "emoji 👋🏽 and family 👨‍👩‍👧 and flags 🇪🇸",
    "https://example.com/path?query=1&x=y#frag",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "UPPER lower MiXeD CamelCase snake_case kebab-case",
    "'s 't 're 've 'm 'll 'd",
    "",
    " ",
    "a",
]

ALPHABET = [
    *"abcdefghijklmnopqrstuvwxyz ABCXYZ0123456789",
    *".,;:!?'\"()[]{}-_=+*/\\\n\t",
    *"éñüßçøåæœ",
    *["日", "本", "語", "👋", "🏽", "—", "“", "”", "€", "中", "文"],
]


def random_strings(count: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [
        "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 60)))
        for _ in range(count)
    ]


RANDOM = random_strings(400)

TRAIN_TEXT = (
    "Once upon a time there was a little cat. The cat liked to sit on the "
    "mat. One day the cat saw a dog. The dog liked the mat too. The cat and "
    "the dog sat on the mat together. "
) * 20


@pytest.fixture(scope="module")
def trained():
    """One tokenizer trained on TRAIN_TEXT, shared by the tests that read it."""
    return tokenizer.BPETokenizer.train(TRAIN_TEXT, vocab_size=300)


# Training ─────────────────────────────────────────────────────────────────────


def test_bytes_only_vocabulary_encodes_to_utf8_bytes():
    bpe = tokenizer.BPETokenizer(
        {bytes([i]): i for i in range(256)}, pattern=None
    )
    text = "héllo 👋"
    assert bpe.encode(text) == list(text.encode("utf-8"))
    assert bpe.decode(list(text.encode("utf-8"))) == text


def test_first_merge_is_the_most_frequent_pair():
    bpe = tokenizer.BPETokenizer.train(
        "aaabdaaabac", vocab_size=257, pattern=None
    )
    assert bpe.vocab_size == 257
    assert bpe.decode([256]) == "aa", (
        "in 'aaabdaaabac' the pair 'aa' occurs most often, so token 256 "
        "should be b'aa'"
    )


def test_training_reaches_the_requested_vocab_size(trained):
    bpe = trained
    assert bpe.vocab_size == 300


def test_merges_never_cross_chunk_boundaries(trained):
    # With the GPT-2 split, " cat" and "." are different chunks, so no token
    # may contain both a letter and ".".
    bpe = trained
    for token_id in range(256, 300):
        piece = bpe.decode([token_id])
        has_letter = any(char.isalpha() for char in piece)
        assert not (has_letter and "." in piece), (
            f"token {token_id}={piece!r} crosses a chunk boundary"
        )


def test_training_compresses_the_text():
    bpe = tokenizer.BPETokenizer.train(TRAIN_TEXT, vocab_size=320)
    n_bytes = len(TRAIN_TEXT.encode("utf-8"))
    n_tokens = len(bpe.encode(TRAIN_TEXT))
    assert n_tokens < 0.5 * n_bytes, (
        f"{n_tokens} tokens for {n_bytes} bytes: expected at least 2x "
        "compression"
    )


@pytest.mark.parametrize("text", HAND_PICKED)
def test_trained_tokenizer_round_trips_unseen_text(trained, text):
    bpe = trained
    assert bpe.decode(bpe.encode(text)) == text


# GPT-2 compatibility ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def reference():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def gpt2():
    return tokenizer.BPETokenizer.from_tiktoken("gpt2")


def test_gpt2_vocab_size(gpt2):
    assert gpt2.vocab_size == 50257 - 1, (
        "GPT-2 has 50,256 ordinary tokens plus <|endoftext|>"
    )


def test_gpt2_split_pattern_is_the_real_one(reference):
    assert tokenizer.GPT2_SPLIT_PATTERN == reference._pat_str


@pytest.mark.parametrize("text", HAND_PICKED)
def test_encode_matches_tiktoken(gpt2, reference, text):
    assert gpt2.encode(text) == reference.encode_ordinary(text)


def test_encode_matches_tiktoken_on_400_random_strings(gpt2, reference):
    mismatches = []
    for text in RANDOM:
        ours, theirs = gpt2.encode(text), reference.encode_ordinary(text)
        if ours != theirs:
            mismatches.append(
                f"{text!r}\n  yours:    {ours}\n  tiktoken: {theirs}"
            )
    detail = "\n".join(mismatches[:3])
    assert not mismatches, (
        f"{len(mismatches)}/400 strings differ. First ones:\n{detail}"
    )


@pytest.mark.parametrize("text", HAND_PICKED)
def test_decode_inverts_tiktoken(gpt2, reference, text):
    assert gpt2.decode(reference.encode_ordinary(text)) == text


def test_allowed_special_token_is_a_single_id(gpt2, reference):
    text = "first document<|endoftext|>second document"
    allowed = {"<|endoftext|>"}
    expected = reference.encode(text, allowed_special=allowed)
    assert gpt2.encode(text, allowed_special=allowed) == expected
    assert 50256 in expected


def test_disallowed_special_token_is_ordinary_text(gpt2, reference):
    text = "first document<|endoftext|>second document"
    assert gpt2.encode(text) == reference.encode_ordinary(text)
