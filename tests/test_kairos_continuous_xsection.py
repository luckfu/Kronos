"""Unit tests for continuous / cross-section Kairos ablations (synthetic)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from modernbert_finance.ablations.continuous_xsection_ablations import (
    BINARY_HEADS,
    _eval_regression,
    _fit_eval_logistic,
    run_continuous_and_xsection,
)
from modernbert_finance.ablations.label_time_diagnostics import BINARY_HEADS as BH


def test_eval_regression_beats_noise_on_signal() -> None:
    rng = np.random.default_rng(0)
    n, d = 400, 8
    x = rng.normal(size=(n, d))
    y = x[:, 0] * 0.5 + rng.normal(scale=0.1, size=n)
    report = _eval_regression(x, y, seed=1)
    assert not report["skipped"]
    assert report["ridge_r2"] > 0.5
    assert report["beats_naive_mae"]


def test_binary_heads_alias() -> None:
    assert list(BINARY_HEADS) == list(BH)


def test_logistic_macro_runs() -> None:
    rng = np.random.default_rng(2)
    n, d = 300, 6
    x = rng.normal(size=(n, d))
    y = (x[:, 0] > 0).astype(np.int64)
    r = _fit_eval_logistic(x, y, seed=3)
    assert not r["skipped"]
    assert r["delta_model_minus_prior"] < 0


def test_build_enriched_matrix_filters_targets_to_signal_window() -> None:
    """Features and targets must share the same asof signal window (Phase U)."""
    from modernbert_finance.ablations.continuous_xsection_ablations import (
        build_enriched_matrix,
    )
    from modernbert_finance.build_targets import WINDOW, target_row

    rng = np.random.default_rng(7)
    n_days = 350
    dates = pd.bdate_range("2023-01-01", periods=n_days)

    def _frame() -> pd.DataFrame:
        base = 50 + np.cumsum(rng.normal(0, 0.5, size=n_days))
        return pd.DataFrame(
            {
                "open": base,
                "high": base + 0.8,
                "low": base - 0.8,
                "close": base,
                "volume": rng.integers(1e4, 1e5, size=n_days).astype(float),
                "amount": rng.integers(1e5, 1e6, size=n_days).astype(float),
                "sector": "IND",
            },
            index=dates,
        )

    panel = {"aa.0001": _frame(), "bb.0002": _frame()}
    rows = []
    for symbol in sorted(panel):
        frame = panel[symbol]
        for start in range(len(frame) - WINDOW + 1):
            rows.append(target_row(symbol, frame, start))
    targets = pd.DataFrame(rows)
    packed = build_enriched_matrix(
        panel, targets, signal_start="2023-07-01", signal_end="2023-12-31"
    )
    assert packed["n_samples"] == len(packed["targets"]["mfe10"])
    assert packed["n_samples"] < len(targets)
    asof = pd.to_datetime(packed["meta"]["asof_date"])
    assert asof.min() >= pd.Timestamp("2023-07-01")
    assert asof.max() <= pd.Timestamp("2023-12-31")
