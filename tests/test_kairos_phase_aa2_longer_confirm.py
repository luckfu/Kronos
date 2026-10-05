"""Unit tests for Phase AA2 longer sealed confirm helpers."""
from __future__ import annotations

from modernbert_finance.ablations.phase_aa2_kronos_longer_sealed_confirm import (
    A_PRIORI_RULES,
    FOCUS_RULE,
    build_z2_splits,
    stride_dates,
)
import pandas as pd


def test_focus_rule_is_aa_winner_and_frozen_set():
    assert FOCUS_RULE == "topk50_tp10_nostop"
    names = [r[0] for r in A_PRIORI_RULES]
    assert names[0] == FOCUS_RULE
    assert "topk50_hold_d10" in names
    assert "topk20_tp10_nostop" in names
    # nostop / no SL on focus
    focus = A_PRIORI_RULES[0]
    assert focus[2] == 0.10 and focus[3] is None


def test_stride_dates():
    d = [f"d{i}" for i in range(10)]
    assert stride_dates(d, 3) == ["d0", "d3", "d6", "d9"]


def test_build_z2_splits_late_half_and_folds():
    dates = [f"2025-{m:02d}-{d:02d}" for m in range(7, 13) for d in range(1, 11)]
    # 6*10 = 60 synthetic dates
    frame = pd.DataFrame({"asof_date": dates * 2, "symbol": ["a", "b"] * len(dates)})
    # unique dates from frame
    frame = pd.DataFrame({"asof_date": dates})
    splits = build_z2_splits(frame)
    assert len(splits["early_dates"]) + len(splits["late_dates"]) == len(dates)
    assert splits["late_dates"][0] == dates[len(dates) // 2]
    assert all(len(f["dates"]) >= 20 for f in splits["folds"]) or len(dates) < 20
