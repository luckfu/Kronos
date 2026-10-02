"""Unit tests for Phase G2 MFE buy-worth-it binary labels (synthetic)."""

from __future__ import annotations

import numpy as np

from modernbert_finance.ablations.buy_profit_mfe_ablations import (
    MFE10_DEF,
    PROFIT_THRESHOLD,
)


def test_profit_threshold_default() -> None:
    assert PROFIT_THRESHOLD == 0.10


def test_mfe10_definition_string() -> None:
    assert MFE10_DEF == "max(high[T+1:T+10]) / close[T] - 1"


def test_mfe_buy_worth_binary_definition() -> None:
    """y = 1{mfe10 >= 0.10}: path MFE, not close-to-close."""
    mfe = np.array([-0.05, 0.0, 0.099, 0.10, 0.25], dtype=np.float64)
    y = (mfe >= PROFIT_THRESHOLD).astype(np.int64)
    np.testing.assert_array_equal(y, [0, 0, 0, 1, 1])
    assert float(y.mean()) == 0.4


def test_mfe_vs_close_to_close_semantics() -> None:
    """Path can touch +10% even if close[T+10] is flat/down."""
    # high-touch MFE = 0.12, but close-to-close = 0.02 → G2 pos, G neg
    mfe10 = 0.12
    fwd_ret_10 = 0.02
    assert (mfe10 >= PROFIT_THRESHOLD) == 1
    assert (fwd_ret_10 >= PROFIT_THRESHOLD) == 0


def test_mfe10_daily_cs_percentile_top_quintile() -> None:
    from modernbert_finance.ablations.buy_profit_mfe_ablations import (
        CS_TOP_PERCENTILE,
        mfe10_daily_cs_percentile,
    )

    # Two days, 5 names each → rank pct in {0.2,0.4,0.6,0.8,1.0}; top quintile = pct>=0.80
    mfe = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.02, 0.04, 0.06, 0.08, 0.50])
    asof = ["2024-01-02"] * 5 + ["2024-01-03"] * 5
    pct = mfe10_daily_cs_percentile(mfe, asof)
    y = (pct >= CS_TOP_PERCENTILE).astype(np.int64)
    assert CS_TOP_PERCENTILE == 0.80
    np.testing.assert_allclose(pct[:5], [0.2, 0.4, 0.6, 0.8, 1.0])
    np.testing.assert_array_equal(y[:5], [0, 0, 0, 1, 1])
    assert int(y.sum()) == 4  # two days × top two ranks (>=0.80)


def test_soft_absolute_threshold_constant() -> None:
    from modernbert_finance.ablations.buy_profit_mfe_ablations import (
        SOFT_ABSOLUTE_THRESHOLD,
    )

    assert SOFT_ABSOLUTE_THRESHOLD == 0.08
