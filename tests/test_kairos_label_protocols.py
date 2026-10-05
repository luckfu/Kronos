"""Unit tests for Phase E alternative label protocols (synthetic)."""

from __future__ import annotations

import numpy as np

from modernbert_finance.ablations.label_protocol_ablations import (
    _eval_regression_residualized,
    _first_touch_class_clean,
    _fit_eval_logistic_mask,
)


def test_clean_ft_same_day_both_is_neither() -> None:
    # Day1: both sides hit → neither
    high = np.array([1.10, 1.20])
    low = np.array([0.90, 0.80])
    assert _first_touch_class_clean(high, low, 1.0, up_th=0.05, dn_th=0.05) == 2


def test_clean_ft_up_first() -> None:
    high = np.array([1.06, 1.20])
    low = np.array([0.99, 0.80])
    assert _first_touch_class_clean(high, low, 1.0, up_th=0.05, dn_th=0.05) == 0


def test_clean_ft_dn_first() -> None:
    high = np.array([1.01, 1.20])
    low = np.array([0.94, 0.80])
    assert _first_touch_class_clean(high, low, 1.0, up_th=0.05, dn_th=0.05) == 1


def test_residualized_regression_recovers_signal() -> None:
    rng = np.random.default_rng(0)
    n, d = 500, 6
    x = rng.normal(size=(n, d))
    vol = np.abs(rng.normal(size=n)) + 0.01
    # y driven by x0 after removing vol component
    y = 0.4 * x[:, 0] + 2.0 * vol + rng.normal(scale=0.05, size=n)
    report = _eval_regression_residualized(x, y, vol, seed=1)
    assert not report["skipped"]
    assert report["ridge_r2"] > 0.4
    assert report["beats_naive_mae"]


def test_logistic_mask_decisive() -> None:
    rng = np.random.default_rng(4)
    n, d = 400, 5
    x = rng.normal(size=(n, d))
    y = (x[:, 0] > 0).astype(np.int64)
    mask = rng.random(n) > 0.3
    r = _fit_eval_logistic_mask(x, y, mask, seed=5)
    assert not r["skipped"]
    assert r["delta_model_minus_prior"] < 0
