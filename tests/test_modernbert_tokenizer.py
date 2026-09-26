import pytest
import torch

from model import KronosTokenizer
from modernbert_finance.tokenizer import FrozenKronosTokenizer


def make_frozen_tokenizer():
    torch.manual_seed(17)
    tokenizer = KronosTokenizer(
        6, 32, 4, 64, 2, 2, 0.2, 0.2, 0.2, 4, 4, 0.25, 1.0, 1.0, 1.0, 4
    )
    return FrozenKronosTokenizer(tokenizer)


def test_tokenizer_returns_frozen_two_codebooks_and_is_deterministic():
    adapter = make_frozen_tokenizer()
    history = torch.randn(2, 120, 6)
    first = adapter(history)
    second = adapter(history)

    assert tuple(first[0].shape) == (2, 120)
    assert tuple(first[1].shape) == (2, 120)
    assert first[0].dtype == torch.long
    assert first[1].dtype == torch.long
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert all(not parameter.requires_grad for parameter in adapter.parameters())


def test_parent_train_cannot_enable_tokenizer_dropout():
    adapter = make_frozen_tokenizer()
    adapter.train()
    assert adapter.training
    assert not adapter.tokenizer.training


def test_invalid_windows_and_non_finite_values_are_rejected():
    adapter = make_frozen_tokenizer()
    with pytest.raises(ValueError, match="120, 6"):
        adapter(torch.randn(1, 119, 6))
    with pytest.raises(ValueError, match="NaN"):
        invalid = torch.zeros(1, 120, 6)
        invalid[0, 0, 0] = float("nan")
        adapter(invalid)
    with pytest.raises(TypeError, match="floating"):
        adapter(torch.zeros(1, 120, 6, dtype=torch.long))


def test_new_financial_embeddings_receive_gradients_after_tokenization():
    adapter = make_frozen_tokenizer()
    s1, s2 = adapter(torch.randn(2, 120, 6))
    emb_s1 = torch.nn.Embedding(adapter.s1_vocab_size, 8)
    emb_s2 = torch.nn.Embedding(adapter.s2_vocab_size, 8)
    loss = (emb_s1(s1) + emb_s2(s2)).square().mean()
    loss.backward()

    assert emb_s1.weight.grad is not None
    assert emb_s2.weight.grad is not None
    assert all(parameter.grad is None for parameter in adapter.parameters())
