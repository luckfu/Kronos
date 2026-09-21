import numpy as np
import pandas as pd

from finetune.neutralized_ls_backtest import (
    book_backtest,
    overlay_backtest,
    pick_with_hysteresis,
    residualize_column,
)


def _style_frame():
    rows = []
    for day, date in enumerate(("2026-08-11", "2026-08-12", "2026-08-13")):
        for index in range(20):
            sector = "银行" if index < 10 else "电子"
            size = index % 10
            predicted = 0.02 * size + (0.01 if sector == "银行" else 0.0)
            actual_d10 = predicted
            step = (1.0 + actual_d10) ** 0.1 - 1.0
            row = {
                "symbol": f"{index:03d}.SZ",
                "asof_date": date,
                "sector": sector,
                "size_decile": size,
                "predicted_return_d10": predicted,
            }
            cum = 0.0
            for horizon in range(1, 11):
                cum = (1.0 + cum) * (1.0 + step) - 1.0
                row[f"actual_return_d{horizon}"] = cum
            rows.append(row)
    return pd.DataFrame(rows)


def _alpha_frame():
    rows = []
    for date in ("2026-08-11", "2026-08-12", "2026-08-13"):
        for index in range(20):
            sector = "银行" if index < 10 else "电子"
            size = index % 10
            residual = 0.05 if index >= 18 else (-0.05 if index < 2 else 0.0)
            predicted = 0.01 * size + residual
            actual_d10 = residual
            step = (1.0 + actual_d10) ** 0.1 - 1.0 if actual_d10 > -1 else 0.0
            row = {
                "symbol": f"{index:03d}.SZ",
                "asof_date": date,
                "sector": sector,
                "size_decile": size,
                "predicted_return_d10": predicted,
            }
            cum = 0.0
            for horizon in range(1, 11):
                cum = (1.0 + cum) * (1.0 + step) - 1.0
                row[f"actual_return_d{horizon}"] = cum
            rows.append(row)
    return pd.DataFrame(rows)


def test_residualization_kills_pure_style_scores():
    frame = _style_frame().reset_index(drop=True)
    residual = residualize_column(frame, "predicted_return_d10")
    assert float(np.max(np.abs(residual))) < 1e-10


def test_style_only_overlay_is_flat_after_costs_near_zero_gross():
    result = overlay_backtest(_style_frame(), cost_per_side=0.0)
    assert abs(result["mean_overlapping_10d_ls"]) < 1e-8
    assert abs(result["total_return"]) < 1e-8


def test_residual_alpha_long_short_is_positive_after_costs():
    result = overlay_backtest(_alpha_frame(), cost_per_side=0.003)
    assert result["mean_overlapping_10d_ls"] > 0.05
    assert result["total_return"] > 0
    assert result["max_drawdown"] <= 0
    assert result["nav_days"] == 3 + 10 - 1
    assert result["mean_two_way_turnover"] is not None


def test_turnover_cost_reduces_nav():
    gross = overlay_backtest(_alpha_frame(), cost_per_side=0.0)
    net = overlay_backtest(_alpha_frame(), cost_per_side=0.003)
    assert net["total_return"] < gross["total_return"]


def test_five_day_book_does_not_rebalance_inside_the_hold():
    frame = _alpha_frame()
    five = book_backtest(frame, cost_per_side=0.0, hold=5, stride=5, enter=0.10)
    interior = [row["two_way_turnover"] for row in five["nav_curve"][1:]]
    assert five["nav_curve"][0]["two_way_turnover"] > 0
    assert max(interior) < 1e-12
    assert five["total_return"] > 0


def test_top5_selects_fewer_names_than_top10():
    frame = _alpha_frame()
    wide = overlay_backtest(frame, cost_per_side=0.0, top_fraction=0.10)
    tight = overlay_backtest(frame, cost_per_side=0.0, top_fraction=0.05)
    assert tight["layers"][0]["n_long"] < wide["layers"][0]["n_long"]


def test_hysteresis_keeps_name_inside_wider_band():
    rows = pd.DataFrame({
        "symbol": [f"{i:02d}" for i in range(20)],
        "residual_predicted_d10": list(range(20)),
    })
    prev_long = {"18", "19"}
    prev_short = {"00", "01"}
    # Rank 17 is 3rd highest: outside top 10% (2 names) but inside top 15% (3 names).
    longs, shorts = pick_with_hysteresis(
        rows, prev_long | {"17"}, prev_short, enter=0.10, exit_fraction=0.15,
    )
    assert "17" in longs
    assert "19" in longs
    longs_exit, _ = pick_with_hysteresis(
        rows, {"16", "19"}, prev_short, enter=0.10, exit_fraction=0.15,
    )
    assert "16" not in longs_exit
    assert "19" in longs_exit
