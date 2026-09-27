"""Level 1 proof: byte-level BPE that trains, round-trips, and matches GPT-2's tokenizer exactly."""

import random

import pytest
import tiktoken

from llm.tokenizer import GPT2_SPLIT_PATTERN, BPETokenizer

GPT2 = tiktoken.get_encoding("gpt2")

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

ALPHABET = list("abcdefghijklmnopqrstuvwxyz ABCXYZ0123456789.,;:!?'\"()[]{}-_=+*/\\\n\t") + list("éñüßçøåæœ") + [
    "日", "本", "語", "👋", "🏽", "—", "“", "”", "€", "中", "文",
]


def random_strings(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        length = rng.randint(1, 60)
        out.append("".join(rng.choice(ALPHABET) for _ in range(length)))
    return out


RANDOM = random_strings(400)

TRAIN_TEXT = (
    "Once upon a time there was a little cat. The cat liked to sit on the mat. "
    "One day the cat saw a dog. The dog liked the mat too. The cat and the dog sat on the mat together. "
) * 20


# ── Training ─────────────────────────────────────────────────────────────────


def test_bytes_only_vocabulary_encodes_to_utf8_bytes():
    tok = BPETokenizer({bytes([i]): i for i in range(256)}, pattern=None)
    text = "héllo 👋"
    assert tok.encode(text) == list(text.encode("utf-8"))
    assert tok.decode(list(text.encode("utf-8"))) == text


def test_first_merge_is_the_most_frequent_pair():
    tok = BPETokenizer.train("aaabdaaabac", vocab_size=257, pattern=None)
    assert tok.vocab_size == 257
    assert tok.decode([256]) == "aa", "in 'aaabdaaabac' the pair 'aa' occurs most often, so token 256 should be b'aa'"


def test_training_reaches_the_requested_vocab_size():
    tok = BPETokenizer.train(TRAIN_TEXT, vocab_size=300)
    assert tok.vocab_size == 300


def test_merges_never_cross_chunk_boundaries():
    # With the GPT-2 split, " cat" and "." are different chunks, so no token may contain both a letter and "."
    tok = BPETokenizer.train(TRAIN_TEXT, vocab_size=300)
    for i in range(256, 300):
        piece = tok.decode([i])
        assert not (any(c.isalpha() for c in piece) and "." in piece), f"token {i}={piece!r} crosses a chunk boundary"


def test_training_compresses_the_text():
    tok = BPETokenizer.train(TRAIN_TEXT, vocab_size=320)
    n_bytes = len(TRAIN_TEXT.encode("utf-8"))
    n_tokens = len(tok.encode(TRAIN_TEXT))
    assert n_tokens < 0.5 * n_bytes, f"{n_tokens} tokens for {n_bytes} bytes: expected at least 2x compression"


@pytest.mark.parametrize("text", HAND_PICKED)
def test_trained_tokenizer_round_trips_unseen_text(text):
    tok = BPETokenizer.train(TRAIN_TEXT, vocab_size=300)
    assert tok.decode(tok.encode(text)) == text


# ── GPT-2 compatibility ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def gpt2():
    return BPETokenizer.from_tiktoken("gpt2")


def test_gpt2_vocab_size(gpt2):
    assert gpt2.vocab_size == 50257 - 1, "GPT-2 has 50,256 ordinary tokens plus <|endoftext|>"


def test_gpt2_split_pattern_is_the_real_one(gpt2):
    assert GPT2_SPLIT_PATTERN == GPT2._pat_str


@pytest.mark.parametrize("text", HAND_PICKED)
def test_encode_matches_tiktoken(gpt2, text):
    assert gpt2.encode(text) == GPT2.encode_ordinary(text)


def test_encode_matches_tiktoken_on_400_random_strings(gpt2):
    mismatches = [(t, gpt2.encode(t), GPT2.encode_ordinary(t)) for t in RANDOM]
    mismatches = [m for m in mismatches if m[1] != m[2]]
    detail = "\n".join(f"{t!r}\n  yours:   {a}\n  tiktoken: {b}" for t, a, b in mismatches[:3])
    assert not mismatches, f"{len(mismatches)}/400 strings differ. First ones:\n{detail}"


@pytest.mark.parametrize("text", HAND_PICKED)
def test_decode_inverts_tiktoken(gpt2, text):
    assert gpt2.decode(GPT2.encode_ordinary(text)) == text


def test_allowed_special_token_is_a_single_id(gpt2):
    text = "first document<|endoftext|>second document"
    expected = GPT2.encode(text, allowed_special={"<|endoftext|>"})
    assert gpt2.encode(text, allowed_special={"<|endoftext|>"}) == expected
    assert 50256 in expected


def test_disallowed_special_token_is_ordinary_text(gpt2):
    text = "first document<|endoftext|>second document"
    assert gpt2.encode(text) == GPT2.encode_ordinary(text)
