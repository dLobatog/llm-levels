# Level 1 · Tokenizer

**Goal:** turn text into integers and back with byte-level BPE, and reproduce GPT-2's tokenizer exactly.

Everything downstream sees only these integers. A mistake here doesn't crash anything. It silently changes what the model is learning.

## Before you write code

Answer these first, out loud or in a note:

1. Why use **bytes** as the base alphabet rather than characters or whole words? What does each choice do to the vocabulary size and to unknown inputs?
2. BPE training repeatedly merges the most frequent adjacent pair. What do you need to count, and how do the counts change after a merge?
3. GPT-2 splits text with a regex *before* running BPE. What would the merges look like without that split?
4. When **encoding** new text, which pair do you merge first: the most frequent pair in *this* text, or something else? Why does the answer have to be deterministic?

## What to build

`src/llm/tokenizer.py`, class `BPETokenizer`. The stub has the exact contract. In short:

| Method | Does |
|---|---|
| `BPETokenizer(ranks, pattern, special_tokens)` | wraps a vocabulary: `ranks` maps each token's **bytes** to its id |
| `train(text, vocab_size, pattern)` | start from 256 single bytes (ids 0–255 in byte order), merge until the vocabulary has `vocab_size` tokens |
| `from_tiktoken("gpt2")` | load GPT-2's real vocabulary, split pattern and special tokens |
| `encode(text, allowed_special)` | text → ids; allowed special tokens map to their single id |
| `decode(ids)` | ids → text |
| `vocab_size` | number of ordinary tokens (excluding special tokens) |

The vocabulary maps bytes to ids, the same representation `tiktoken` uses. A lower id means "merged earlier", which is also the merge priority at encode time.

## Proof

```bash
uv run pytest tests/test_level01_tokenizer.py -q
uv run ruff check src/llm/tokenizer.py
```

The tests check:
- a known training example;
- vocabulary size;
- compression;
- round trips on text the tokenizer never saw;
- **exact equality with `tiktoken`'s GPT-2 ids** on several hundred strings: English, code, numbers, whitespace runs, contractions, accents, CJK and emoji;
- special-token handling.

Ruff must be clean too. The code follows the Google Python Style Guide (see the README).

## Hints (read only if stuck)

<details><summary>Loading GPT-2</summary>

`tiktoken.get_encoding("gpt2")` has private attributes `_mergeable_ranks` (bytes → id), `_special_tokens` and `_pat_str`. You can copy them. Look at which byte has id 0 in GPT-2. It isn't `\x00`.
</details>

<details><summary>The regex</summary>

Use the `regex` package, not `re`. The pattern needs `\p{L}`, `\p{N}` and possessive quantifiers (`++`), which `re` handles differently or not at all.
</details>

<details><summary>Encoding</summary>

Within one chunk: turn it into single-byte tokens, then repeatedly merge the adjacent pair whose *concatenated bytes* have the lowest id in the vocabulary. Stop when no adjacent pair is in the vocabulary.
</details>

## When the tests pass

Answer these without notes, then move on:

1. Why do LLMs struggle to count the r's in "strawberry"?
2. Doubling the vocabulary size: what gets cheaper, what gets more expensive, and where do the extra parameters live?
3. How does GPT-2 tokenize "12345" versus "1 2 3 4 5", and why does that matter for arithmetic?
4. You serve a model with a slightly different tokenizer from the one it was trained with. What breaks, and how would you notice?
5. The same sentence costs 2–3× more tokens in some languages. Why, and what does that cost in money and in context length?

## Stretch

- Profile your encoder against `tiktoken`, which is written in Rust. Where does the time go?
- Train a 4,096-token vocabulary on a few megabytes of TinyStories and read the first 50 merges. What did it learn first?
