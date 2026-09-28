"""Level 3 proof: causal attention that matches PyTorch, and uses context."""

import math

import pytest
import torch
from torch.nn import functional

from llm import attention
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
BEST_BIGRAM_LOSS = 1.929  # level 2: the best any bigram can do on TEXT


def seeded(seed):
    return torch.Generator().manual_seed(seed)


def qkv(*shape, seed=0):
    g = seeded(seed)
    return tuple(torch.randn(*shape, generator=g) for _ in range(3))


# The attention function ──────────────────────────────────────────────────────


def test_attn_matches_pytorch_causal():
    q, k, v = qkv(2, 3, 7, 8)
    expected = functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    got = attention.attention(q, k, v, causal=True)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_attn_matches_pytorch_without_mask():
    q, k, v = qkv(2, 3, 7, 8)
    expected = functional.scaled_dot_product_attention(q, k, v)
    got = attention.attention(q, k, v, causal=False)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_attn_first_position_only_sees_itself():
    q, k, v = qkv(1, 5, 4)
    out = attention.attention(q, k, v, causal=True)
    torch.testing.assert_close(out[:, 0], v[:, 0])


def test_attn_equal_keys_average_the_past():
    # If every key is the same, no earlier token stands out: position t gets
    # the plain average of values 0..t.
    q, _, v = qkv(1, 6, 4)
    k = torch.ones(1, 6, 4)
    out = attention.attention(q, k, v, causal=True)
    running_mean = v.cumsum(dim=1) / torch.arange(1, 7)[None, :, None]
    torch.testing.assert_close(out, running_mean)


def test_attn_future_gets_zero_gradient():
    q, k, v = qkv(1, 6, 4)
    k.requires_grad_(True)
    v.requires_grad_(True)
    out = attention.attention(q, k, v, causal=True)
    out[0, 2].sum().backward()  # position 2 may only depend on 0, 1, 2
    assert torch.all(k.grad[0, 3:] == 0) and torch.all(v.grad[0, 3:] == 0)
    assert torch.any(v.grad[0, :3] != 0)


# Multi-head causal self-attention ────────────────────────────────────────────


def reference_heads(module, x, n_heads):
    """What CausalSelfAttention must compute, using the module's weights."""
    b, t, d = x.shape
    q, k, v = module.qkv(x).split(d, dim=-1)
    q, k, v = (
        z.view(b, t, n_heads, d // n_heads).transpose(1, 2) for z in (q, k, v)
    )
    y = functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    return module.proj(y.transpose(1, 2).reshape(b, t, d))


def test_heads_output_shape():
    module = attention.CausalSelfAttention(32, 4)
    assert module(torch.randn(2, 9, 32)).shape == (2, 9, 32)


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_heads_match_the_reference(n_heads):
    torch.manual_seed(0)
    module = attention.CausalSelfAttention(32, n_heads)
    x = torch.randn(2, 9, 32, generator=seeded(1))
    torch.testing.assert_close(
        module(x), reference_heads(module, x, n_heads), atol=1e-5, rtol=1e-5
    )


def test_heads_never_look_ahead():
    torch.manual_seed(0)
    module = attention.CausalSelfAttention(16, 4)
    x = torch.randn(1, 8, 16, requires_grad=True)
    module(x)[0, 3].sum().backward()
    assert torch.all(x.grad[0, 4:] == 0), "position 3 used a later position"


def test_heads_must_divide_the_model_width():
    with pytest.raises(ValueError):
        attention.CausalSelfAttention(30, 4)


# A language model with an attention head ─────────────────────────────────────


def text_ids():
    return torch.tensor(list(TEXT.encode("utf-8")))


def test_lm_logits_shape_and_no_loss_without_targets():
    model = attention.AttentionLM(256, 32, 4, 16)
    logits, loss = model(torch.randint(0, 256, (3, 10), generator=seeded(0)))
    assert logits.shape == (3, 10, 256)
    assert loss is None


def test_lm_rejects_sequences_longer_than_its_context():
    model = attention.AttentionLM(256, 32, 4, 16)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 17, dtype=torch.long))


def test_lm_logits_ignore_the_future():
    torch.manual_seed(0)
    model = attention.AttentionLM(50, 16, 2, 12)
    a = torch.randint(0, 50, (1, 12), generator=seeded(0))
    b = a.clone()
    b[0, 6:] = (b[0, 6:] + 1) % 50  # change everything after position 5
    torch.testing.assert_close(model(a)[0][0, :6], model(b)[0][0, :6])


def test_lm_word_order_matters():
    # Swap two earlier tokens and keep the last one: attention alone would not
    # notice (a weighted average doesn't care about order). Positions must.
    torch.manual_seed(0)
    model = attention.AttentionLM(50, 16, 2, 12)
    a = torch.tensor([[3, 7, 11, 19, 5]])
    b = torch.tensor([[7, 3, 11, 19, 5]])
    assert not torch.allclose(model(a)[0][0, -1], model(b)[0][0, -1]), (
        "swapping earlier tokens changed nothing: is position used?"
    )


def test_lm_generate_runs_past_its_context():
    torch.manual_seed(0)
    model = attention.AttentionLM(40, 16, 2, 8)
    prompt = torch.zeros(2, 3, dtype=torch.long)
    out = model.generate(prompt, 30, seeded(1))
    assert out.shape == (2, 33)
    assert torch.equal(out[:, :3], prompt)
    assert not out.requires_grad


def test_lm_beats_the_best_possible_bigram():
    # Same paragraph as level 2. No bigram can get below 1.929 on it; a model
    # that sees the context can.
    ids = text_ids()
    torch.manual_seed(0)
    model = attention.AttentionLM(256, 32, 4, 32)
    training.train(model, ids, 1000, 32, 32, 1e-2, seeded(0))
    g = seeded(123)
    with torch.no_grad():
        losses = [
            model(*data.get_batch(ids, 32, 32, g))[1].item() for _ in range(20)
        ]
    loss = sum(losses) / len(losses)
    assert loss < 1.5, (
        f"loss {loss:.3f}; the best possible bigram gets {BEST_BIGRAM_LOSS}"
    )
    assert math.isfinite(loss)
