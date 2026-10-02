"""Unit tests for Phase AA cost-aware top-K backtest helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from modernbert_finance.ablations.phase_aa_kronos_cost_aware_topk_backtest import (
    LIMIT_UP,
    path_from_entry,
    select_top_k,
    simulate_trade,
)


def _row(**kwargs):
    base = {
        "asof_date": "2026-08-11",
        "symbol": "sh.600000",
        "predicted_return_10d": 0.05,
        "return_10d": 0.08,
        "actual_return_d1": 0.01,
    }
    for i in range(2, 11):
        base[f"actual_return_d{i}"] = 0.01 * i
    base.update(kwargs)
    return pd.Series(base)


def test_path_from_entry_relative_to_d1():
    row = _row(actual_return_d1=0.0, actual_return_d5=0.10, actual_return_d10=0.08)
    path = dict(path_from_entry(row))
    assert abs(path[5] - 0.10) < 1e-12
    assert abs(path[10] - 0.08) < 1e-12


def test_limit_up_skips_entry():
    row = _row(actual_return_d1=LIMIT_UP + 0.001)
    assert simulate_trade(row, 0.10, 0.05, 0.003) is None


def test_take_profit_exit_and_costs():
    # entry d1=0; d3 hits +10%
    row = _row(
        actual_return_d1=0.0,
        actual_return_d2=0.04,
        actual_return_d3=0.11,
        actual_return_d10=0.20,
    )
    t = simulate_trade(row, 0.10, 0.05, 0.003)
    assert t is not None
    assert t["exit_reason"] == "take_profit"
    assert t["exit_horizon_signal"] == 3
    assert abs(t["gross_return"] - 0.11) < 1e-12
    assert abs(t["net_return"] - (0.11 - 0.006)) < 1e-12


def test_select_top_k_orders_by_score():
    day = pd.DataFrame(
        {
            "symbol": ["a", "b", "c", "d"],
            "predicted_return_10d": [0.1, -0.2, 0.3, 0.0],
        }
    )
    top = select_top_k(day, 2)
    assert list(top["symbol"]) == ["c", "a"]
