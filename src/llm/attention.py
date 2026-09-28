"""Level 3: causal self-attention, and a language model that uses it.

Typical usage example:

    out = attention(q, k, v, causal=True)
    layer = CausalSelfAttention(d_model=64, n_heads=4)
    model = AttentionLM(vocab_size=256, d_model=64, n_heads=4,
                        context_length=64)
"""

import torch
from torch import math
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
    head_dim = q.shape[-1]

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)  #  (.., T, T)
    if causal:
        t = scores.shape[-1]
        allowed = torch.ones(t, t, dtype=torch.bool).tril()
        scores = scores.masked_fill(~allowed, float("-inf"))
    weights = torch.softmax(scores, dim=-1)  #  (.., T, T)
    out = weights @ v  # (.. T, head_dim)

    return out


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
        if d_model % n_heads != 0:
            raise ValueError
        self.d_model = d_model
        self.n_heads = n_heads
        self.qkv = nn.Linear(in_features=d_model, out_features=3 * d_model)
        self.proj = nn.Linear(in_features=d_model, out_features=d_model)

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
        qkv = self.qkv(x)  # (B, T, D)
        D = self.d_model
        B, T, _ = qkv.shape
        H = self.n_heads
        q, k, v = qkv.split(self.d_model, dim=-1)  # (B, T, D) (x3)
        q = q.view(B, T, H, D // H)  # (B, T, H, D)
        k = k.view(B, T, H, D // H)  # (B, T, H, D)
        v = v.view(B, T, H, D // H)  # (B, T, H, D)
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)  # (B, H, T, D)
        v = v.transpose(1, 2)  # (B, H, T, D)

        out = attention(q, k, v, causal=True)  # (B, H, T , D)
        out = out.transpose(1, 2)  # B, T, H, D
        out = out.reshape((B, T, D))
        out = self.proj(out)

        return out


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
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.context_length = context_length
        self.position_embeddings = nn.Embedding(
            num_embeddings=context_length, embedding_dim=d_model
        )
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads)
        self.head = nn.Linear(
            out_features=vocab_size,
            in_features=d_model,
        )
        self.loss = nn.CrossEntropyLoss()

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
        B, T = ids.shape
        if T > self.context_length:
            raise ValueError("Sequence length larger than context length")
        positions = torch.arange(T, device=ids.device)  # 0, 1, …, T-1
        out = self.embeddings(ids)
        out = out + self.position_embeddings(positions)  # (B,T,d) + (T,d)
        out = self.attn(out)
        logits = self.head(out)
        loss = None
        if targets is not None:
            reshaped_logits = logits.reshape(-1, self.vocab_size)
            targets = targets.reshape(-1)
            loss = self.loss(reshaped_logits, targets)

        return (logits, loss)

    @torch.no_grad()
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
        for _ in range(max_new_tokens):
            logits, _ = self(ids[:, -self.context_length :])
            last = logits[:, -1, :]  # (B, V) for every row
            probs = torch.softmax(input=last, dim=-1)  # (B, T)
            next_id = torch.multinomial(
                input=probs, num_samples=1, generator=generator
            )  # (B, 1)
            ids = torch.cat((ids, next_id), dim=1)

        return ids
