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
