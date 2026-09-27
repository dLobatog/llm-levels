# Level 2 · Data and the training loop

**Goal:** turn a stream of token ids into training examples, and train the simplest language model there is, a bigram, with a loop you write yourself. The loss should start at ln(vocab size) and then fall.

The bigram is deliberately dumb: it predicts the next token from the current one alone. What matters here is everything around it. From level 4 on, GPT-2 replaces the bigram, but the batches, the loss and the loop stay the same.

## Before you write code

Answer these first, out loud or in a note:

1. Take one window of 8 tokens. How many predictions does the model learn from, and how many tokens of the stream does the window use?
2. Before any training, what loss should the model have, exactly? Why? What does that tell you about how to initialize the model?
3. A bigram sees only the current token. On English text, what is the lowest loss any bigram could reach, and how could you compute it without training anything?
4. Why sample random windows instead of walking through the data from start to end?
5. What goes wrong if you forget to zero the gradients between steps?

## What to build

Three small files. The stubs have the exact contracts.

| File | Build | Does |
|---|---|---|
| `src/llm/data.py` | `get_batch(ids, batch_size, context_length, generator)` | random windows `x`, and targets `y` = `x` shifted by one |
| `src/llm/bigram.py` | `Bigram(vocab_size)` | a `vocab_size × vocab_size` table of next-token logits |
| | `forward(ids, targets=None)` | `(logits, loss)`, with loss the mean cross-entropy |
| | `generate(ids, max_new_tokens, generator)` | samples one token at a time |
| `src/llm/training.py` | `train(model, ids, steps, batch_size, context_length, lr, generator)` | the AdamW loop; returns every step's loss |

## Proof

```bash
uv run pytest tests/test_level02_training_loop.py -q
uv run ruff check src
```

The tests check:
- **batches:** shapes, the shift by one, windows that are real slices of the data, every valid start reachable and none running off the end, and reproducibility with a seed;
- **the model:** the untrained loss is ln(V) ± 0.05, the loss equals the mean negative log-likelihood, the prediction depends only on the current token, and generation keeps the prompt and is reproducible;
- **training:** on a repeating sequence the loss starts at ln(V) and falls below 0.05, and the trained model follows the cycle. On a paragraph of English it gets within 0.05 nats of **the best loss any bigram can reach**, computed by counting pairs.

## Hints (read only if stuck)

<details><summary>The table</summary>

`nn.Embedding(V, V)` is exactly a table of learnable rows looked up by id. Check what its default initialization does to the loss before training.
</details>

<details><summary>Cross-entropy</summary>

`torch.nn.functional.cross_entropy` wants logits of shape `(N, C)` and targets of shape `(N,)`. Your logits are `(B, T, V)`, so reshape both.
</details>

<details><summary>Batches</summary>

`torch.randint(low, high, size, generator=...)` excludes `high`. `ids[starts[:, None] + torch.arange(T)]` builds every window at once.
</details>

<details><summary>Sampling</summary>

Softmax the logits of the last position, then `torch.multinomial(probs, 1, generator=...)` draws one id per row. `@torch.no_grad()` keeps generation out of the autograd graph.
</details>

## When the tests pass

Answer these without notes, then move on:

1. Why is cross-entropy the loss for next-token prediction? What does a loss of ln(V) mean in words, and what would a loss of 0 mean?
2. A bigram has V² parameters. How many is that for GPT-2's vocabulary, and why is that a dead end?
3. On the repeating sequence your loss went to almost 0. Is that good? What would you need in order to know whether a model generalizes?
4. Why does PyTorch accumulate gradients instead of overwriting them?
5. Sampling vs always taking the most likely token (greedy): what does each produce from the trained bigram, and why?

## Stretch

- Train your level 1 tokenizer to about 512 tokens on a text you like, then train the bigram on those ids instead of bytes. Does the sample read better?
- Add a temperature to `generate` by dividing the logits by it before the softmax. What do 0.5 and 2 do?
- Time a training step on CPU and on MPS. For a model this small, which wins, and why?
