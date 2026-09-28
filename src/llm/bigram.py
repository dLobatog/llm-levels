"""Level 2: the bigram language model, a table of next-token scores.

Typical usage example:

    model = Bigram(vocab_size=256)
    logits, loss = model(x, y)
    text_ids = model.generate(prompt, max_new_tokens=100)
"""

import torch
from torch import nn


class Bigram(nn.Module):
    """Predicts the next token from the current token alone.

    The whole model is a `vocab_size` × `vocab_size` table: row `a` holds the
    logits (unnormalized log-probabilities) of the token that follows `a`.
    """

    def __init__(self, vocab_size: int):
        """Creates the table.

        Args:
            vocab_size: Number of distinct token ids.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=vocab_size
        )
        self.loss = nn.CrossEntropyLoss()
        nn.init.zeros_(self.embeddings.weight)

    def forward(
        self,
        ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Scores the next token at every position.

        Args:
            ids: Long tensor of shape `(batch, time)`.
            targets: Optional long tensor of the same shape: the true next
                token at every position.

        Returns:
            `(logits, loss)`. `logits` has shape `(batch, time, vocab_size)`.
            `loss` is the mean cross-entropy over all positions, or None when
            no targets are given.
        """
        logits = self.embeddings(ids)  # (B, T, D)
        loss = None
        if targets is not None:
            reshaped_logits = logits.reshape(-1, self.vocab_size)
            loss = self.loss(reshaped_logits, targets.reshape(-1))

        return logits, loss

    @torch.no_grad
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Extends each row by sampling one token at a time.

        Args:
            ids: Long tensor of shape `(batch, time)`: the prompts.
            max_new_tokens: How many tokens to add to each row.
            generator: Random generator for sampling, for reproducible text.
                None uses torch's global generator.

        Returns:
            Long tensor of shape `(batch, time + max_new_tokens)` that starts
            with `ids`. No gradients are tracked.
        """
        for _ in range(max_new_tokens):
            logits, loss = self.forward(ids=ids)  # (B, T, V)
            last = logits[:, -1, :]  # (B, V) for every row
            probs = torch.softmax(input=last, dim=-1)  # (B, T)
            next_id = torch.multinomial(
                input=probs, num_samples=1, generator=generator
            )  # (B, 1)
            ids = torch.cat((ids, next_id), dim=1)

        return ids
