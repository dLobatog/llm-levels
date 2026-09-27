"""Level 2 proof: batches, a bigram model, and a training loop that learns."""

import collections
import math

import pytest
import torch
from torch.nn import functional

from llm import bigram
from llm import data
from llm import training

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


def text_ids():
    return torch.tensor(list(TEXT.encode("utf-8")))


def cycle_ids(vocab_size=10, repeats=100):
    return torch.arange(vocab_size).repeat(repeats)


def seeded(seed):
    return torch.Generator().manual_seed(seed)


def scramble(model, seed=0):
    """Gives the model random parameters, so tests don't depend on its init."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_()


# Batches ─────────────────────────────────────────────────────────────────────


def test_batch_shapes_and_dtype():
    x, y = data.get_batch(torch.arange(100), 4, 8, seeded(0))
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert x.dtype == torch.long and y.dtype == torch.long


def test_batch_targets_are_inputs_shifted_by_one():
    # With data 0, 1, 2, ..., every window is a run of consecutive numbers
    # and every target is its input plus one.
    x, y = data.get_batch(torch.arange(1000), 16, 12, seeded(0))
    steps = x[:, 1:] - x[:, :-1]
    assert torch.equal(steps, torch.ones(16, 11, dtype=torch.long))
    assert torch.equal(y, x + 1)


def test_batch_windows_come_from_the_data():
    ids = torch.randint(0, 50, (300,), generator=seeded(1))
    x, y = data.get_batch(ids, 8, 10, seeded(2))
    windows = ids.unfold(0, 11, 1)  # every contiguous run of 11 tokens
    for row_x, row_y in zip(x, y, strict=True):
        pair = torch.cat([row_x, row_y[-1:]])
        assert (windows == pair).all(dim=1).any(), (
            "each row of x (plus the last target) must be a contiguous slice"
        )


def test_batch_every_start_is_used_and_none_runs_off_the_end():
    # 20 tokens and windows of 8: starts 0..11 are valid, since the last
    # target sits one past the window.
    ids = torch.arange(20)
    starts = set()
    for seed in range(300):
        x, y = data.get_batch(ids, 8, 8, seeded(seed))
        assert int(y.max()) <= 19
        starts.update(x[:, 0].tolist())
    assert starts == set(range(12)), f"starts used: {sorted(starts)}"


def test_batch_exactly_one_window_fits():
    x, y = data.get_batch(torch.arange(9), 3, 8, seeded(0))
    assert torch.equal(x, torch.arange(8).repeat(3, 1))
    assert torch.equal(y, torch.arange(1, 9).repeat(3, 1))


def test_same_generator_same_batch():
    ids = torch.arange(500)
    a = data.get_batch(ids, 4, 16, seeded(7))
    b = data.get_batch(ids, 4, 16, seeded(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_batch_too_little_data_raises():
    with pytest.raises(ValueError):
        data.get_batch(torch.arange(8), 2, 8, seeded(0))


# The bigram model ────────────────────────────────────────────────────────────


def test_model_logits_shape_and_no_loss_without_targets():
    model = bigram.Bigram(30)
    logits, loss = model(torch.randint(0, 30, (2, 5), generator=seeded(0)))
    assert logits.shape == (2, 5, 30)
    assert loss is None


def test_model_initial_loss_is_ln_vocab():
    torch.manual_seed(0)
    model = bigram.Bigram(256)
    ids = torch.randint(0, 256, (16, 32), generator=seeded(0))
    targets = torch.randint(0, 256, (16, 32), generator=seeded(1))
    _, loss = model(ids, targets)
    assert abs(loss.item() - math.log(256)) < 0.05, (
        f"untrained loss {loss.item():.3f}, expected about ln(256) = "
        f"{math.log(256):.3f}. What should a model that knows nothing predict?"
    )


def test_model_loss_is_mean_negative_log_likelihood():
    model = bigram.Bigram(40)
    scramble(model)
    ids = torch.randint(0, 40, (3, 7), generator=seeded(0))
    targets = torch.randint(0, 40, (3, 7), generator=seeded(1))
    logits, loss = model(ids, targets)
    log_probs = functional.log_softmax(logits, dim=-1)
    expected = -log_probs.gather(-1, targets[..., None]).mean()
    torch.testing.assert_close(loss, expected)


def test_model_prediction_depends_only_on_the_current_token():
    model = bigram.Bigram(20)
    scramble(model)
    logits, _ = model(torch.tensor([[5, 7, 5, 9, 5]]))
    torch.testing.assert_close(logits[0, 0], logits[0, 2])
    torch.testing.assert_close(logits[0, 0], logits[0, 4])
    a, _ = model(torch.tensor([[1, 2, 3, 4]]))
    b, _ = model(torch.tensor([[9, 9, 9, 4]]))
    torch.testing.assert_close(a[0, -1], b[0, -1])


def test_model_generate_keeps_the_prompt_and_adds_tokens():
    model = bigram.Bigram(25)
    scramble(model)
    prompt = torch.randint(0, 25, (3, 4), generator=seeded(0))
    out = model.generate(prompt, 10, seeded(1))
    assert out.shape == (3, 14)
    assert torch.equal(out[:, :4], prompt)
    assert int(out.min()) >= 0 and int(out.max()) < 25
    assert not out.requires_grad


def test_model_generate_is_reproducible():
    model = bigram.Bigram(25)
    scramble(model)
    prompt = torch.zeros(2, 1, dtype=torch.long)
    a = model.generate(prompt, 30, seeded(3))
    b = model.generate(prompt, 30, seeded(3))
    assert torch.equal(a, b)


# The training loop ───────────────────────────────────────────────────────────


def test_learn_returns_one_loss_per_step():
    model = bigram.Bigram(10)
    losses = training.train(model, cycle_ids(), 25, 4, 8, 0.1, seeded(0))
    assert len(losses) == 25
    assert all(isinstance(loss, float) for loss in losses)


def test_learn_loss_starts_at_ln_vocab_and_drops():
    # 0, 1, ..., 9, 0, 1, ... : every next token is fully determined.
    torch.manual_seed(0)
    model = bigram.Bigram(10)
    losses = training.train(model, cycle_ids(), 300, 16, 16, 0.1, seeded(0))
    assert abs(losses[0] - math.log(10)) < 0.05, (
        f"first loss {losses[0]:.3f}, expected about ln(10) = 2.303"
    )
    final = sum(losses[-20:]) / 20
    assert final < 0.05, f"loss only got down to {final:.3f}"


def test_learn_is_reproducible():
    runs = []
    for _ in range(2):
        torch.manual_seed(0)
        model = bigram.Bigram(10)
        losses = training.train(model, cycle_ids(), 20, 4, 8, 0.1, seeded(5))
        runs.append(losses)
    assert runs[0] == runs[1]


def test_learn_then_generate_follows_the_cycle():
    torch.manual_seed(0)
    model = bigram.Bigram(10)
    training.train(model, cycle_ids(), 400, 16, 16, 0.1, seeded(0))
    out = model.generate(torch.tensor([[3]]), 40, seeded(1))[0].tolist()
    follows = sum((b - a) % 10 == 1 for a, b in zip(out, out[1:], strict=False))
    assert follows >= 36, f"only {follows}/40 steps followed the cycle: {out}"


def test_learn_reaches_the_best_possible_bigram_loss():
    # The best any bigram can do on this text is the entropy of the next byte
    # given the current one, computed by counting pairs.
    ids = text_ids()
    firsts_list, seconds_list = ids[:-1].tolist(), ids[1:].tolist()
    pairs = collections.Counter(zip(firsts_list, seconds_list, strict=True))
    firsts = collections.Counter(firsts_list)
    n = len(ids) - 1
    best = -sum(c / n * math.log(c / firsts[a]) for (a, _), c in pairs.items())

    torch.manual_seed(0)
    model = bigram.Bigram(256)
    training.train(model, ids, 1500, 32, 64, 0.1, seeded(0))
    with torch.no_grad():
        _, loss = model(ids[None, :-1], ids[None, 1:])
    assert loss.item() - best < 0.05, (
        f"best possible bigram loss on this text: {best:.3f} nats; "
        f"yours after training: {loss.item():.3f}"
    )
