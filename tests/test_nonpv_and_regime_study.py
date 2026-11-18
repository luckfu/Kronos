"""Unit tests for point-in-time non-PV feature alignment and gated slot backtest."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune/oos_eval/base_clean_sealed_oos"))
sys.path.insert(0, str(ROOT / "tests"))

import few_names_backtest as fnb  # noqa: E402
import nonpv_and_regime_study as nrs  # noqa: E402
from test_signal_usage_backtest import FIRST_SIGNAL, LAST_SIGNAL, N_DAYS, N_SYMBOLS, make_market, scores_from  # noqa: E402


def test_gated_slots_respects_regime_capacity(tmp_path):
    market = make_market(tmp_path)
    rng = np.random.default_rng(7)
    noise = rng.normal(size=(N_DAYS, N_SYMBOLS))
    scores = scores_from(market, lambda day: noise[day])
    rebalance = list(range(FIRST_SIGNAL, LAST_SIGNAL + 1, 5))
    last_day = LAST_SIGNAL + 1
    k = 6

    gate = np.full(N_DAYS, 2, dtype=int)
    gate[rebalance[1]] = 1
    gate[rebalance[2]] = 0

    book = nrs.run_gated_slots(market, nrs.gated_plan(market, scores, k, gate, True), rebalance, k, last_day)
    daily = book["daily"]
    assert daily.loc[market.dates[rebalance[0] + 1], "names"] == k
    assert daily.loc[market.dates[rebalance[1] + 1], "names"] <= k // 2
    assert daily.loc[market.dates[rebalance[2] + 1], "names"] == 0
    assert daily.loc[market.dates[rebalance[2] + 1], "exposure"] == 0.0
    assert book["cash_rebalances"] == 1


def test_ungated_matches_few_names_backtest(tmp_path):
    market = make_market(tmp_path)
    rng = np.random.default_rng(11)
    noise = rng.normal(size=(N_DAYS, N_SYMBOLS))
    scores = scores_from(market, lambda day: noise[day])
    rebalance = list(range(FIRST_SIGNAL, LAST_SIGNAL + 1, 5))
    last_day = LAST_SIGNAL + 1
    k = 4

    gate = np.zeros(N_DAYS, dtype=int)
    ours = nrs.run_gated_slots(market, nrs.gated_plan(market, scores, k, gate, False), rebalance, k, last_day)["daily"]["ret"]
    theirs = fnb.run_slots(market, fnb.c2_plan(market, scores, k, True), rebalance, k, last_day)["daily"]["ret"]
    np.testing.assert_allclose(ours.to_numpy(), theirs.to_numpy(), atol=1e-12)


def test_yjbb_point_in_time_no_future_leak(tmp_path):
    raw_rows = []
    dates = pd.bdate_range("2026-01-01", "2026-09-15").strftime("%Y-%m-%d").tolist()
    for i, d in enumerate(dates):
        raw_rows.append({
            "symbol": "sh.600000", "date": d, "open": 10.0, "high": 10.2,
            "low": 9.8, "close": 10.0 + 0.01 * i, "volume": 1e6, "amount": 1e8, "market_cap": 1e11,
        })
    raw_path = tmp_path / "raw.csv"
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)

    npv = tmp_path / "non_pv"
    npv.mkdir()
    for q in ["20250930", "20251231", "20260331", "20260630", "20260930"]:
        ann = {"20250930": "2025-10-30", "20251231": "2026-04-20", "20260331": "2026-04-28",
               "20260630": "2026-08-20", "20260930": "2026-10-25"}[q]
        val = float(q[:4]) + float(q[4:6])
        pd.DataFrame([{
            "股票代码": "600000", "最新公告日期": ann, "每股收益": val, "营业总收入-营业总收入": val * 1e9,
            "营业总收入-同比增长": val, "营业总收入-季度环比增长": 1.0, "净利润-净利润": val * 1e8,
            "净利润-同比增长": val, "净利润-季度环比增长": 1.0, "每股净资产": 5.0, "净资产收益率": 10.0,
            "每股经营现金流量": 1.0, "销售毛利率": 30.0,
        }]).to_parquet(npv / f"yjbb_{q}.parquet", index=False)
    for q in ["20251231", "20260331", "20260630", "20260930"]:
        pd.DataFrame(columns=["股票代码", "预测指标", "业绩变动幅度", "预告类型", "公告日期"]).to_parquet(
            npv / f"yjyg_{q}.parquet", index=False
        )
    pd.DataFrame(columns=["代码", "上榜日", "龙虎榜净买额"]).to_parquet(npv / "lhb_20251201_20261025.parquet", index=False)
    pd.DataFrame({"date": dates, "close": np.linspace(3000, 3200, len(dates))}).to_parquet(
        npv / "idx_sh000300.parquet", index=False
    )

    panel, _ = nrs.build_panel(raw_path, npv)
    before = panel[panel["asof_date"] == "2026-08-19"].iloc[0]
    after = panel[panel["asof_date"] == "2026-08-20"].iloc[0]
    assert before["yb_period"] == "20260331"
    assert after["yb_period"] == "20260630"
