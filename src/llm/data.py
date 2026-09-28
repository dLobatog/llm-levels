"""Level 2: training batches cut from a stream of token ids.

Typical usage example:

    x, y = get_batch(ids, batch_size=32, context_length=64)
"""

import torch


def get_batch(
    ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples random training windows from a stream of token ids.

    Each row of `x` is `context_length` consecutive tokens starting at a
    uniformly random position, and the same row of `y` holds the token that
    follows each of them, so `y` is `x` shifted one step to the left. Every
    start position whose window and targets fit inside `ids` can be drawn.

    Args:
        ids: A 1-D tensor of token ids: the whole training stream.
        batch_size: Number of windows (rows).
        context_length: Tokens per window (columns).
        generator: Random generator for the start positions, for
            reproducible batches. None uses torch's global generator.

    Returns:
        `(x, y)`, two long tensors of shape `(batch_size, context_length)`.

    Raises:
        ValueError: If `ids` is too short to hold one window and its targets.
    """
    if len(ids) <= context_length:
        raise ValueError

    start = torch.randint(
        0, len(ids) - context_length, (batch_size,), generator=generator
    )
    x_list, y_list = [], []
    for s in start.tolist():
        x_list.append(ids[s : s + context_length])
        y_list.append(ids[s + 1 : s + 1 + context_length])

    return torch.stack(x_list), torch.stack(y_list)
