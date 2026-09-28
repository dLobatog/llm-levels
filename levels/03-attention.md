# Level 3 · Attention

**Goal:** build causal self-attention, check it against PyTorch's, and give your bigram an attention layer. It then beats the best any bigram can do, because each prediction can use the whole context instead of one token.

Your level 2 code stays: `get_batch`, the loss, `generate` and `train` work unchanged on the new model.

## Before you write code

Answer these first, out loud or in a note:

1. Your bigram predicts from one token. If the model could look at every earlier token, how might a position decide *which* earlier tokens matter to it?
2. Warm-up: how would you compute, for every position *t* at once, the plain average of the vectors at positions 0…*t*, with a single matrix multiplication?
3. The scores are divided by √head_dim before the softmax. What happens to a softmax when its inputs get large, and why would that hurt training?
4. Why must the causal mask be applied before the softmax, not after it?
5. A weighted average doesn't care about order. So how could the model tell "the cat sat" from "sat the cat"?
6. Why split the width into several small heads instead of using one big head?

## What to build

`src/llm/attention.py`. The stubs have the exact contracts.

| Build | Does |
|---|---|
| `attention(q, k, v, causal)` | scores q·k/√d, mask the future, softmax, weighted sum of v |
| `CausalSelfAttention(d_model, n_heads)` | attributes `qkv` (d → 3d) and `proj` (d → d); split into heads, attend, merge |
| `AttentionLM(vocab_size, d_model, n_heads, context_length)` | token + position embeddings → attention → linear head; `forward` and `generate` as in your bigram |

Write the attention yourself: `torch.nn.functional.scaled_dot_product_attention` is the answer key in the tests, not a building block.

## Proof

```bash
uv run pytest tests/test_level03_attention.py -q
uv run ruff check src
```

One section at a time: add `-k attn`, `-k heads` or `-k lm`.

The tests check:
- **the function:** it matches PyTorch's attention with and without the mask; the first position sees only itself; equal keys give the running average of the values; and no position receives gradient from a later one;
- **the heads:** the output shape; for 1, 2, 4 and 8 heads, the same result as a reference built from *your* `qkv` and `proj` weights; no looking ahead; and a `ValueError` when the heads don't divide the width;
- **the model:** shapes; a `ValueError` for sequences longer than the context; later tokens never change earlier logits; **swapping two earlier tokens changes the prediction**; generation beyond the context length; and, on the level 2 paragraph, a loss below 1.5, where the best possible bigram gets 1.929.

About that last number: the model trains and is measured on the same 946 bytes, so part of what it learns is memorization. The test shows the model *uses context*. Whether it *generalizes* needs text it hasn't seen, which is level 5's job.

## See it

Once the tests pass, run [`labs/03_attention.py`](../labs/03_attention.py) a cell at a time: averaging the past three ways, the reshape dance on the numbers 0–31, your heads against a plain loop, your model's loss against the best possible bigram, heatmaps of where each of your heads looks, and what happens without the mask or without positions.

## Hints (read only if stuck)

<details><summary>The mask</summary>

`torch.ones(T, T, dtype=torch.bool).tril()` is True where a position may look. `scores.masked_fill(~mask, float("-inf"))` hides the rest: e^−∞ = 0 after the softmax.
</details>

<details><summary>Splitting into heads</summary>

`q.view(B, T, n_heads, head_dim).transpose(1, 2)` gives `(B, n_heads, T, head_dim)`, and your `attention` handles the extra dimension for free. Merging back is the reverse: `transpose(1, 2).reshape(B, T, d_model)`.
</details>

<details><summary>Positions</summary>

`nn.Embedding(context_length, d_model)` looked up with `torch.arange(T)` gives one learned vector per position. Add it to the token embeddings.
</details>

<details><summary>Keeping each token's own identity</summary>

After attention, a position's vector is an average of other positions' values. Adding the input back, `x + attention(x)`, lets it keep its own token too. It's optional here; GPT-2 does it everywhere (the "residual connection").
</details>

<details><summary>Generating past the context</summary>

The position table only has `context_length` rows, so feed the model `ids[:, -context_length:]`.
</details>

## When the tests pass

Answer these without notes, then move on:

1. What do queries, keys and values each do, in one sentence each?
2. How do the time and memory of attention grow with the sequence length *T*? Why does that make long contexts expensive?
3. Your model reaches a loss far below 1.929 on the paragraph. Why is that less impressive than it sounds, and what would you check?
4. What breaks if you remove the position embedding? Which test would catch it?
5. Four heads of width 8 vs one head of width 32: the same number of parameters. What can the four do that the one can't?

## Stretch

- Set `causal=False` inside your model and train again. The loss drops suspiciously low. Why is it cheating, and why would its generated text be bad anyway?
- Remove the position embedding, retrain, and compare the samples.
- Plot the attention weights of your trained model for one sentence as a heatmap. What does each head look at?
