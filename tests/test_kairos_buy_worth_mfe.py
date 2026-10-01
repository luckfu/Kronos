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
