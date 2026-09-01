from types import SimpleNamespace
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from finetune.dataset import QlibDataset, build_causal_size_path, fixed_validation_limits
from model import Kronos


def tiny_model():
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
        context_layer=1,
        use_size_path=True,
        size_path_input_dim=4,
        size_mlp_hidden_dim=4,
    )


def test_dynamic_size_path_starts_as_exact_noop_and_receives_gradient():
    model = tiny_model()
    hidden = torch.randn(2, 5, model.d_model)
    path = torch.rand(2, 5, 4)

    conditioned = model._add_context(hidden, size_path=path)
    assert torch.equal(conditioned, hidden)

    conditioned.sum().backward()
    assert model.size_path_mlp[-1].weight.grad.abs().sum().item() > 0

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.size_path_mlp[-1].weight.fill_(0.1)
    model._add_context(hidden, size_path=path).sum().backward()
    assert model.size_path_mlp[0].weight.grad.abs().sum().item() > 0


def test_dynamic_size_path_rejects_misaligned_sequence():
    model = tiny_model()
    with pytest.raises(ValueError, match="size_path must have shape"):
        model._add_context(
            torch.zeros(2, 5, model.d_model), size_path=torch.zeros(2, 4, 4)
        )


def test_causal_path_carries_only_last_visible_value_into_forecast():
    path = build_causal_size_path([np.nan, 0.2, 0.3, np.nan, 0.45], 3)

    assert path.shape == (8, 4)
    assert path[:, 0].tolist() == pytest.approx([
        0.5, 0.2, 0.3, 0.3, 0.45, 0.45, 0.45, 0.45
    ])
    assert path[:, 1].tolist() == pytest.approx([
        0.0, -0.3, 0.1, 0.0, 0.15, 0.0, 0.0, 0.0
    ])
    assert path[:, 2].tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert path[:, 3].tolist() == [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def test_bootstrap_validation_cap_and_full_only_behavior():
    bootstrap = SimpleNamespace(
        validation_full_only=False,
        validation_quick_samples=2_000,
        validation_large_samples=2_000,
    )
    full = SimpleNamespace(
        validation_full_only=True,
        validation_quick_samples=2_000,
        validation_large_samples=2_000,
    )

    assert fixed_validation_limits(bootstrap, 24_000, 123_982) == (2_000, 2_000)
    assert fixed_validation_limits(full, 24_000, 123_982) == (123_982, 123_982)


def test_dataset_returns_120_observed_and_10_causal_size_tokens(tmp_path, monkeypatch):
    dates = pd.bdate_range("2025-01-01", periods=131)
    values = np.arange(131, dtype=np.float64) + 10.0
    percentile = np.linspace(0.1, 0.9, 131)
    frame = pd.DataFrame({
        "open": values,
        "high": values + 1,
        "low": values - 1,
        "close": values + 0.25,
        "volume": values * 100,
        "amount": values * 1000,
        "size_percentile": percentile,
    }, index=pd.DatetimeIndex(dates, name="datetime"))
    panel_path = tmp_path / "panel.pkl"
    with panel_path.open("wb") as handle:
        pickle.dump({"sh.600001": frame}, handle)

    monkeypatch.setenv("KRONOS_TRAIN_DATA_PATHS", str(panel_path))
    monkeypatch.setenv("KRONOS_TRAIN_SAMPLES_PER_SEGMENT", "1")
    monkeypatch.setenv("KRONOS_LOOKBACK_WINDOW", "120")
    monkeypatch.setenv("KRONOS_PREDICT_WINDOW", "10")
    monkeypatch.setenv("KRONOS_USE_SECTOR_FEATURES", "0")
    monkeypatch.setenv("KRONOS_USE_SIZE_FEATURES", "0")
    monkeypatch.setenv("KRONOS_USE_SIZE_PERCENTILE", "0")
    monkeypatch.setenv("KRONOS_USE_SIZE_PATH", "1")
    monkeypatch.setenv("KRONOS_METADATA_PATH", "")

    sample = QlibDataset("train")[0]
    size_path = sample[5].numpy()

    assert len(sample) == 6
    assert size_path.shape == (130, 4)
    assert size_path[:120, 3].tolist() == [1.0] * 120
    assert size_path[120:, 3].tolist() == [0.0] * 10
    assert size_path[120:, 0].tolist() == pytest.approx([percentile[119]] * 10)
    assert percentile[120] != size_path[120, 0]
