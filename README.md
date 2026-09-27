# llm-levels

Build a GPT from an empty file up to GRPO, one verified level at a time.

Every level ends with a **proof**: tests that pass only if the thing actually works. Two of them are the backbone of the whole project. Your GPT-2 must reproduce OpenAI's logits, and your Qwen3 must reproduce Hugging Face's. A level is unlocked when its tests pass *and* you can explain what you built without looking.

No inference engines, no `transformers` in the implementation. Plain PyTorch. Reference libraries (`tiktoken`, `transformers`) are answer keys in the tests. Outside tests they only supply data, such as GPT-2's vocabulary and pretrained weights.

## The ladder

| # | Level | You build | Proof |
|---|---|---|---|
| 0 | Setup | the repo, dependencies, this ladder | `python ladder.py` runs |
| 1 | Tokenizer | byte-level BPE: training, encoding, decoding | exact match with `tiktoken`'s GPT-2 ids on hundreds of strings |
| 2 | Data and the training loop | next-token windows, batching, a bigram model | initial loss ≈ ln(vocab), then it drops |
| 3 | Attention | one head, causal mask, multi-head via reshapes | matches `scaled_dot_product_attention`; output *t* has zero gradient from tokens after *t* |
| 4 | GPT-2 | LayerNorm, GELU, MLP, residuals, positions, weight tying, sampling | exactly 124M parameters; OpenAI's weights in *your* model give Hugging Face's logits |
| 5 | Pretraining | AdamW, warmup + cosine, clipping, eval, checkpoints | overfits one batch; a small GPT writes coherent TinyStories |
| 6 | Inference and modern blocks | KV cache, batched generation, RoPE, RMSNorm, SwiGLU, GQA | cached output = uncached output; Qwen3-0.6B in your code matches Hugging Face |
| 7 | Fine-tuning | chat template, answer-only loss masking, LoRA from scratch | held-out quality before/after; LoRA memory vs full fine-tuning |
| 8 | Preference optimization | DPO, derived from the KL-regularized objective | reward margins grow while the policy stays near the reference |
| 9 | RL with verifiable rewards | GRPO: group sampling, verifier rewards, clipped objective, KL | reward climbs on arithmetic, then GSM8K; experiments on length bias and reward hacking |
| 10 | Stretch | inference-time scaling, distillation, quantization, evaluation harness | measured, not assumed |

## How it works

- **Tests are the proof.** Each level has a brief in `levels/` and a test file in `tests/`. The brief says what to build and why. The tests say whether you did.
- **Core math by hand.** Attention, norms, losses and update rules get written without an AI assistant. Plumbing (data loading, logging, scripts) can be AI-assisted, but every line gets reviewed.
- **Questions before code.** Each brief opens with questions to answer before writing anything, and ends with follow-ups to answer out loud once the tests pass.
- **Google Python style.** Code follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). Ruff checks the mechanical parts (80 columns, import order, naming, Google docstrings, type annotations), configured in `pyproject.toml` with the guide's section numbers. The rest is on you: import modules rather than names (`import collections`, then `collections.Counter`), use descriptive names, and keep functions short.
- **Briefs appear one level at a time.** Later levels are written once the earlier ones are done, so they can build on what actually got built.

## Running

```bash
uv sync
uv run python ladder.py            # where am I?
uv run pytest tests/test_level01_tokenizer.py
uv run ruff check && uv run ruff format --check   # style
```

## Layout

```
levels/   one brief per level
src/llm/  your implementation (starts as stubs with exact contracts)
tests/    the proofs
ladder.py progress
```
