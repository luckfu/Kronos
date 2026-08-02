import numpy as np
import pandas as pd
import torch

from finetune.prepare_a_share import add_size_buckets
from model import Kronos


def build_tiny_model(use_size_percentile=True):
    return Kronos(
        s1_bits=2,
        s2_bits=2,
        n_layers=2,
        d_model=8,
        n_heads=2,
        ff_dim=16,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.0,
        learn_te=True,
        num_size_buckets=10,
        context_layer=1,
        use_size_percentile=use_size_percentile,
        size_mlp_hidden_dim=4,
    )


def test_size_bucket_preparation_preserves_continuous_rank():
    frame = pd.DataFrame({
        'date': pd.to_datetime(['2026-01-02'] * 20),
        'market_cap': np.arange(1, 21, dtype=float),
    })

    result = add_size_buckets(frame, bucket_count=10)

    assert result['size_percentile'].is_monotonic_increasing
    assert result['size_percentile'].between(0.0, 1.0).all()
    assert result['size_bucket'].between(0, 9).all()
    assert result.iloc[-1]['size_percentile'] == 1.0
    assert result.iloc[-1]['size_bucket'] == 9


def test_continuous_condition_starts_as_noop_and_is_trainable():
    model = build_tiny_model()
    hidden = torch.zeros(2, 3, model.d_model)
    buckets = torch.tensor([2, 2])
    percentiles = torch.tensor([0.2, 0.8])

    initial = model._add_context(
        hidden, size_bucket=buckets, size_percentile=percentiles
    )
    assert torch.equal(initial[0], initial[1])

    with torch.no_grad():
        model.size_mlp[0].weight.fill_(1.0)
        model.size_mlp[0].bias.zero_()
        model.size_mlp[-1].weight.fill_(0.1)
        model.size_mlp[-1].bias.zero_()

    conditioned = model._add_context(
        hidden, size_bucket=buckets, size_percentile=percentiles
    )
    assert not torch.equal(conditioned[0], conditioned[1])


def test_discrete_model_rejects_unconfigured_continuous_condition():
    model = build_tiny_model(use_size_percentile=False)
    hidden = torch.zeros(1, 2, model.d_model)

    try:
        model._add_context(hidden, size_percentile=torch.tensor([0.5]))
    except ValueError as exc:
        assert 'use_size_percentile is false' in str(exc)
    else:
        raise AssertionError('Expected an explicit configuration error')
