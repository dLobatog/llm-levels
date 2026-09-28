"""Lab 2: watch your bigram learn.

Run one cell at a time and read the output before moving on. Each cell starts
with a question: answer it before you run the cell.

  Neovim:   cursor in a cell, <leader>rc (runs it in the IPython pane)
  VS Code:  Shift+Enter (Interactive Window, plots appear inline)
  Terminal: uv run python labs/02_bigram.py (plots open in windows)

Everything here calls your own code: data.get_batch, bigram.Bigram and
training.train.
"""

# %% Setup: the text, as bytes
import collections
import math

import lovely_tensors
import matplotlib.pyplot as plt
import torch
from torch.nn import functional

from llm import bigram
from llm import data
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
ids = torch.tensor(list(TEXT.encode("utf-8")))
used = sorted(set(ids.tolist()))  # the byte values that actually appear
print(ids)
print(f"{len(used)} distinct bytes out of 256")


def as_text(token_ids: torch.Tensor) -> str:
    """Decodes a 1-D tensor of byte ids, showing spaces as ␣."""
    raw = bytes(token_ids.tolist()).decode("utf-8", errors="replace")
    return raw.replace(" ", "␣")


def probs_table(model: torch.nn.Module) -> torch.Tensor:
    """Softmaxed rows of the model's table, restricted to the used bytes."""
    table = next(model.parameters()).detach()
    rows = functional.softmax(table, dim=-1)
    return rows[used][:, used]


def draw_table(ax: plt.Axes, probs: torch.Tensor, title: str) -> None:
    """Draws a probability table as a heatmap with byte labels."""
    labels = [as_text(torch.tensor([b])) for b in used]
    ax.imshow(probs, cmap="magma", vmin=0, vmax=0.5)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(len(used)), labels, fontsize=5)
    ax.set_yticks(range(len(used)), labels, fontsize=5)
    ax.set_xlabel("next byte", fontsize=7)
    ax.set_ylabel("current byte", fontsize=7)


# %% 1. A batch, as text
# Predict: what shapes will x and y have? How does each y row relate to x?
generator = torch.Generator().manual_seed(0)
x, y = data.get_batch(ids, 4, 24, generator)
print("x:", x)
print("y:", y)
for row_x, row_y in zip(x, y, strict=True):
    print(f"x  {as_text(row_x)}")
    print(f"y   {as_text(row_y)}")

# %% 2. The untrained model
# Predict: what's in the table? What loss will it report, and why?
torch.manual_seed(0)
model = bigram.Bigram(256)
table = next(model.parameters())
print("table:", table)
logits, loss = model(x, y)
print("logits:", logits)
print(f"loss {loss.item():.4f}   ln(256) = {math.log(256):.4f}")

# %% 3. What the untrained model writes
# Predict: every byte is equally likely. What will the text look like?
prompt = torch.tensor([[ord("T")]])
out = model.generate(prompt, 80, torch.Generator().manual_seed(1))
print(as_text(out[0]))

# %% 4. Watch it learn: the table at 6 moments of training
# Predict: which cells of the table will light up first, and why those?
snapshots = [(0, probs_table(model))]
losses = []
for steps in (5, 10, 25, 60, 1400):  # snapshots at 5, 15, 40, 100, 1500
    losses += training.train(model, ids, steps, 32, 64, 0.1, generator)
    snapshots.append((len(losses), probs_table(model)))

fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
for ax, (step, probs) in zip(axes.flat, snapshots, strict=True):
    draw_table(ax, probs, f"after {step} steps")
fig.suptitle("each row: where the model thinks that byte goes next")
fig.tight_layout()
plt.show()

# %% 5. The loss curve
# Predict: where does it start, and can it ever go below the dashed line?
pairs = collections.Counter(
    zip(ids[:-1].tolist(), ids[1:].tolist(), strict=True)
)
firsts = collections.Counter(ids[:-1].tolist())
n = len(ids) - 1
best = -sum(c / n * math.log(c / firsts[a]) for (a, _), c in pairs.items())

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses, lw=1)
ax.axhline(math.log(256), ls=":", c="grey", label="ln 256: knows nothing")
ax.axhline(
    best, ls="--", c="green", label=f"best any bigram can do: {best:.3f}"
)
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.legend()
plt.show()
with torch.no_grad():
    _, full = model(ids[None, :-1], ids[None, 1:])
print(f"your model on the whole paragraph: {full.item():.3f}")

# %% 6. Training rediscovers counting
# Predict: after "h", how often does the model expect "e"? Compare the count.
h = used.index(ord("h"))
after_h = collections.Counter(
    b for a, b in zip(TEXT, TEXT[1:], strict=False) if a == "h"
)
counted = torch.tensor(
    [after_h.get(chr(b), 0) for b in used], dtype=torch.float
)
counted /= counted.sum()
learned = probs_table(model)[h]
top = torch.argsort(learned, descending=True)[:8]
labels = [as_text(torch.tensor([used[i]])) for i in top]
fig, ax = plt.subplots(figsize=(8, 3.5))
positions = torch.arange(len(top))
ax.bar(positions - 0.2, counted[top], width=0.4, label="counted")
ax.bar(positions + 0.2, learned[top], width=0.4, label="learned")
ax.set_xticks(positions, labels)
ax.set_title('what follows "h"')
ax.legend()
plt.show()

# %% 7. What the trained model writes: sampling vs greedy
# Predict: greedy always takes the most likely next byte. What goes wrong?
out = model.generate(prompt, 120, torch.Generator().manual_seed(1))
print("sampled:", as_text(out[0]))
greedy = prompt
with torch.no_grad():
    for _ in range(120):
        logits, _ = model(greedy[:, -1:])
        greedy = torch.cat([greedy, logits[:, -1].argmax(-1, keepdim=True)], 1)
print("greedy: ", as_text(greedy[0]))

# %% 8. Your turn
# Paste a paragraph of your own into TEXT (try Spanish), rerun from the top,
# and look at the table. Which letters have the most confident rows, and why?
