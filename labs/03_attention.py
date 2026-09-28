"""Lab 3: open up your attention and watch it work.

Run one cell at a time and answer each cell's question before running it.

  Neovim:   cursor in a cell, <leader>rc (runs it in the IPython pane)
  VS Code:  Shift+Enter (Interactive Window, plots appear inline)
  Terminal: uv run python labs/03_attention.py

Everything here calls your own code in src/llm/attention.py.
"""

# %% Setup
import math

import lovely_tensors
import matplotlib.pyplot as plt
import torch
from torch.nn import functional

from llm import attention
from llm import training

lovely_tensors.monkey_patch()  # tensors print as shape + statistics

TEXT = (
    "The lighthouse keeper wrote the same line in her log every night: wind "
    "from the west, sea calm, lamp lit. Nobody read the log. The ships that "
    "passed did not need to know about the wind; they only needed the lamp. "
    "One winter the lamp went out for a single hour, and she wrote that down "
    "too, in small careful letters, because a log that only tells the truth "
    "when the truth is boring is not a log at all. In spring a boy from the "
    "village asked her what she did all night. She said she kept a light on "
    "and wrote down the weather. He said that did not sound like much. She "
    "agreed, and then she showed him the chart of the rocks below the cliff, "
    "every one of them marked with the name of a ship that had found it "
    "before the lighthouse was built. After that he came back every evening "
    "to help her wind the clock that turned the lamp, and he learned to "
    "write the wind, the sea and the lamp in the log, in small careful "
    "letters, the same line every night."
)
BEST_BIGRAM = 1.929  # level 2: the best any bigram can do on TEXT
ids = torch.tensor(list(TEXT.encode("utf-8")))


def as_text(token_ids: torch.Tensor) -> str:
    """Decodes a 1-D tensor of byte ids, showing spaces as ␣."""
    raw = bytes(token_ids.tolist()).decode("utf-8", errors="replace")
    return raw.replace(" ", "␣")


# %% 1. Averaging the past, three ways
# Predict: will the loop, the triangle matrix and the -inf softmax agree?
x = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
T = x.shape[0]
by_loop = torch.stack([x[: t + 1].mean(dim=0) for t in range(T)])
triangle = torch.ones(T, T).tril()
by_matmul = (triangle / triangle.sum(dim=1, keepdim=True)) @ x
scores = torch.zeros(T, T).masked_fill(triangle == 0, float("-inf"))
by_softmax = functional.softmax(scores, dim=-1) @ x
print(by_loop.p)
print("matmul == loop:", torch.allclose(by_matmul, by_loop))
print("softmax == loop:", torch.allclose(by_softmax, by_loop))
print(functional.softmax(scores, dim=-1).p)

# %% 2. Your attention() with equal keys is that same average
# Predict: if every key is identical, what weights does each position get?
q = torch.randn(T, 2)
k = torch.ones(T, 2)
print(attention.attention(q, k, x, causal=True).p)

# %% 3. The reshape dance, with numbers you can follow
# Predict each print before running: where do 4..7 and 8..11 end up?
q = torch.arange(32).view(4, 8)  # 4 positions, width 8
print(q.p)
print(q.view(4, 2, 4).p)  # cut each row into 2 heads of 4: nothing moves
print(q.view(4, 2, 4).transpose(0, 1).p)  # one (4 × 4) matrix per head
print(q.view(2, 4, 4).p)  # the wrong way: positions 0 and 1 chopped up
back = q.view(4, 2, 4).transpose(0, 1).transpose(0, 1).reshape(4, 8)
print("round trip gives q back:", torch.equal(back, q))

# %% 4. Your heads == a plain loop over heads
# Predict: how big is the difference between the two?
torch.manual_seed(0)
layer = attention.CausalSelfAttention(8, 2)
x = torch.randn(1, 5, 8)
q, k, v = layer.qkv(x).split(8, dim=-1)
heads = []
for h in range(2):
    cols = slice(4 * h, 4 * (h + 1))
    heads.append(attention.attention(q[..., cols], k[..., cols], v[..., cols]))
by_loop = layer.proj(torch.cat(heads, dim=-1))
print("max difference:", (layer(x) - by_loop).abs().max().item())

# %% 5. Train your model: can it beat the best possible bigram?
generator = torch.Generator().manual_seed(0)
torch.manual_seed(0)
model = attention.AttentionLM(256, 32, 4, 32)
prompt = torch.tensor([list(b"The ")])
untrained = model.generate(prompt, 100, torch.Generator().manual_seed(1))
losses = training.train(model, ids, 1000, 32, 32, 1e-2, generator)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses, lw=1, label="your attention model")
ax.axhline(math.log(256), ls=":", c="grey", label="ln 256: knows nothing")
ax.axhline(BEST_BIGRAM, ls="--", c="green", label="best any bigram can do")
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.legend()
plt.show()
print("untrained:", as_text(untrained[0]))
trained = model.generate(prompt, 150, torch.Generator().manual_seed(1))
print("trained:  ", as_text(trained[0]))


# %% 6. Where does each of your heads look?
# Predict: will any head mostly look at the previous character?
def head_weights(
    lm: attention.AttentionLM, token_ids: torch.Tensor
) -> torch.Tensor:
    """Your trained model's attention weights, shape (heads, T, T)."""
    t = token_ids.shape[1]
    with torch.no_grad():
        x = lm.embeddings(token_ids) + lm.position_embeddings(torch.arange(t))
        q, k, _ = lm.attn.qkv(x).split(lm.d_model, dim=-1)
        h, hd = lm.n_heads, lm.d_model // lm.n_heads
        q = q.view(1, t, h, hd).transpose(1, 2)
        k = k.view(1, t, h, hd).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(hd)
        allowed = torch.ones(t, t, dtype=torch.bool).tril()
        scores = scores.masked_fill(~allowed, float("-inf"))
        return functional.softmax(scores, dim=-1)[0]


sentence = "the lamp in the log"
sentence_ids = torch.tensor([list(sentence.encode())])
weights = head_weights(model, sentence_ids)
labels = [c if c != " " else "␣" for c in sentence]
fig, axes = plt.subplots(1, model.n_heads, figsize=(4 * model.n_heads, 4.3))
for h, ax in enumerate(axes):
    ax.imshow(weights[h], cmap="magma", vmin=0, vmax=1)
    ax.set_title(f"head {h}")
    ax.set_xticks(range(len(labels)), labels, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("looks at")
axes[0].set_ylabel("position")
fig.tight_layout()
plt.show()
for h in range(model.n_heads):
    diagonal = weights[h].diagonal().mean().item()
    previous = weights[h].diagonal(-1).mean().item()
    print(f"head {h}: {diagonal:.2f} on itself, {previous:.2f} on the previous")


# %% 7. Break it on purpose: no mask, and no positions
# Predict: which one gets a suspiciously low loss, and why is it cheating?
def train_variant(no_mask: bool, no_positions: bool) -> tuple[float, str]:
    """Trains a fresh model with one ingredient removed."""
    original = attention.attention
    if no_mask:
        attention.attention = lambda q, k, v, causal=True: original(
            q, k, v, causal=False
        )
    try:
        torch.manual_seed(0)
        lm = attention.AttentionLM(256, 32, 4, 32)
        if no_positions:
            lm.position_embeddings.weight.data.zero_()
            lm.position_embeddings.weight.requires_grad_(False)
        run = training.train(lm, ids, 1000, 32, 32, 1e-2, generator)
        sample = lm.generate(prompt, 100, torch.Generator().manual_seed(1))
    finally:
        attention.attention = original
    return sum(run[-50:]) / 50, as_text(sample[0])


for name, no_mask, no_positions in [
    ("no mask", True, False),
    ("no positions", False, True),
]:
    final, sample = train_variant(no_mask, no_positions)
    print(f"{name:13} final loss {final:.3f}\n  {sample}\n")

# %% 8. Your turn
# Change d_model, n_heads or the context length in cell 5 and rerun cells
# 5-6. Does a single head (n_heads=1) look different from four?
