"""Unit tests for Kairos Phase A/B diagnostic helpers (CPU, synthetic)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from modernbert_finance.ablations.label_time_diagnostics import (
    BINARY_HEADS,
    cross_prior_transfer,
    document_cutoff_contract,
    head_correlations,
    positive_rate_tables,
)
from modernbert_finance.ablations.structure_ablations import run_structure_ablations


def _fake_targets(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-07-01", periods=n, freq="D")
    data = {
        "symbol": ["sh.600000"] * n,
        "start_index": np.arange(n, dtype=np.int32),
        "asof_date": dates.strftime("%Y-%m-%d"),
        "mfe10": rng.normal(0.02, 0.05, size=n).astype(np.float32),
        "mae10": rng.normal(-0.02, 0.04, size=n).astype(np.float32),
    }
    for head in BINARY_HEADS:
        data[head] = rng.integers(0, 2, size=n, dtype=np.uint8)
    return pd.DataFrame(data)


def test_cutoff_contract_documents_10d_horizon() -> None:
    contract = document_cutoff_contract()
    assert contract["lookback"] == 120
    assert contract["horizon"] == 10
    assert contract["asof_position_in_window"] == 119


def test_positive_rates_and_cross_prior() -> None:
    train = _fake_targets(500, seed=1)
    val = _fake_targets(300, seed=2)
    # Force val into 2025H2 / 2026H1 buckets
    val.loc[:149, "asof_date"] = "2025-08-15"
    val.loc[150:, "asof_date"] = "2026-03-01"
    tables = positive_rate_tables(train, val)
    assert "train" in tables["macro_prior_log_loss"]
    assert "val_2025H2" in tables["macro_prior_log_loss"]
    cross = cross_prior_transfer(train, val)
    assert np.isfinite(cross["macro_val_log_loss_with_train_prior"])


def test_head_correlations_shape() -> None:
    frame = _fake_targets(200)
    corr = head_correlations(frame)
    assert len(corr["matrix"]) == 8
    assert corr["n_rows_used"] == 200


def test_structure_ablations_runs() -> None:
    rng = np.random.default_rng(3)
    n, d = 300, 16
    x = rng.normal(size=(n, d))
    targets = {h: (x[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(np.int64) for h in BINARY_HEADS}
    # Decorrelate a bit
    targets["down_005"] = 1 - targets["up_005"]
    targets["mfe10"] = rng.normal(scale=0.02, size=n)
    targets["mae10"] = rng.normal(scale=0.02, size=n)
    report = run_structure_ablations(x, targets, seed=4)
    assert report["n_samples"] == n
    assert "dilution_summary" in report
    assert "up_005" in report["per_head_independent"]
