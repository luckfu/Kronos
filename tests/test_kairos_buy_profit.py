"""Unit tests for Phase G buy-profit binary labels (synthetic)."""

from __future__ import annotations

import numpy as np

from modernbert_finance.ablations.buy_profit_ablations import PROFIT_THRESHOLD


def test_profit_threshold_default() -> None:
    assert PROFIT_THRESHOLD == 0.10


def test_buy_profit_binary_definition() -> None:
    fwd = np.array([-0.05, 0.0, 0.099, 0.10, 0.25], dtype=np.float64)
    y = (fwd >= PROFIT_THRESHOLD).astype(np.int64)
    np.testing.assert_array_equal(y, [0, 0, 0, 1, 1])
    assert float(y.mean()) == 0.4
