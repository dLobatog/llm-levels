"""Level 3: causal self-attention, and a language model that uses it.

Typical usage example:

    out = attention(q, k, v, causal=True)
    layer = CausalSelfAttention(d_model=64, n_heads=4)
    model = AttentionLM(vocab_size=256, d_model=64, n_heads=4,
                        context_length=64)
"""

import torch
from torch import nn


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Scaled dot-product attention.

    Every position scores every position by q · k / sqrt(head_dim), turns
    its scores into weights with a softmax, and returns the weighted average
    of the values. With `causal`, a position only sees itself and earlier
    positions. Write it yourself: PyTorch's scaled_dot_product_attention is
    the answer key in the tests.

    Args:
        q: Queries, shape `(..., T, head_dim)`.
        k: Keys, same shape as `q`.
        v: Values, same shape as `q`.
        causal: Whether to hide later positions from earlier ones.

    Returns:
        Tensor of shape `(..., T, head_dim)`.
    """
    raise NotImplementedError


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention over a sequence of vectors.

    Attributes:
        qkv: Linear layer `d_model → 3 · d_model` that produces the queries,
            keys and values, concatenated in that order.
        proj: Linear layer `d_model → d_model` applied to the merged heads.
    """

    def __init__(self, d_model: int, n_heads: int):
        """Creates the two linear layers.

        Args:
            d_model: Width of each input and output vector.
            n_heads: Number of heads. Each head works on
                `d_model // n_heads` of the width.

        Raises:
            ValueError: If `n_heads` does not divide `d_model`.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mixes each position with the earlier ones.

        Splits `qkv(x)` into queries, keys and values, splits each into
        `n_heads` heads of shape `(B, n_heads, T, head_dim)`, runs causal
        attention per head, merges the heads back into `(B, T, d_model)` and
        applies `proj`.

        Args:
            x: Tensor of shape `(B, T, d_model)`.

        Returns:
            Tensor of shape `(B, T, d_model)`.
        """
        raise NotImplementedError


class AttentionLM(nn.Module):
    """Your bigram with a middle: embeddings, one attention layer, a head.

    ids → token embedding + position embedding → CausalSelfAttention →
    linear layer to `vocab_size` scores. `forward` and `generate` follow the
    same contract as `bigram.Bigram`, so `training.train` works unchanged.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        context_length: int,
    ):
        """Creates the layers.

        Args:
            vocab_size: Number of distinct token ids.
            d_model: Width of the vectors inside the model.
            n_heads: Attention heads.
            context_length: Longest sequence the model accepts; one position
                embedding per position.
        """
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Scores the next token at every position, using all earlier ones.

        Args:
            ids: Long tensor of shape `(B, T)` with `T <= context_length`.
            targets: Optional long tensor of the same shape.

        Returns:
            `(logits, loss)`, as in `bigram.Bigram.forward`.

        Raises:
            ValueError: If `T` is longer than `context_length`.
        """
        raise NotImplementedError

    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Samples one token at a time, as in `bigram.Bigram.generate`.

        Only the last `context_length` tokens are fed to the model, so the
        text can grow longer than the context.

        Args:
            ids: Long tensor of shape `(B, T)`: the prompts.
            max_new_tokens: How many tokens to add to each row.
            generator: Random generator for sampling.

        Returns:
            Long tensor of shape `(B, T + max_new_tokens)` that starts with
            `ids`. No gradients are tracked.
        """
        raise NotImplementedError
