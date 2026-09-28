"""Level 2: the training loop.

Typical usage example:

    losses = train(model, ids, steps=1000, batch_size=32, context_length=64,
                   lr=1e-2)
"""

import torch
from torch import nn

from llm import data


def train(
    model: nn.Module,
    ids: torch.Tensor,
    steps: int,
    batch_size: int,
    context_length: int,
    lr: float,
    generator: torch.Generator | None = None,
) -> list[float]:
    """Trains `model` on random windows of `ids` with AdamW.

    Every step samples a batch with `data.get_batch`, computes the loss,
    backpropagates it and updates the parameters.

    Args:
        model: A model whose `forward(ids, targets)` returns
            `(logits, loss)`.
        ids: A 1-D tensor of token ids: the training stream.
        steps: Number of optimizer steps.
        batch_size: Windows per step.
        context_length: Tokens per window.
        lr: AdamW learning rate.
        generator: Random generator for the batches, for reproducible runs.

    Returns:
        The loss at every step, as Python floats, in order.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        x, y = data.get_batch(ids, batch_size, context_length, generator)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses
