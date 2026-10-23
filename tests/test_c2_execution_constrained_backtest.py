"""Unit tests for the C2 step-8b execution-constrained book.

Synthetic prices only. Nothing here is a section 5.2 OOS number.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from finetune.small_clean_oos.execution_constrained import (
    PRED_ABS_MIN,
    ROUND_TRIP_COST,
    limit_ratio,
    run_execution_backtest,
    slippage_rate,
)
from finetune.small_clean_oos.run_c2_execution_backtest import (
    bars_with_unadjusted_flags,
    main,
    synthetic_smoke_inputs,
)

ROOT = Path(__file__).resolve().parents[1]


def _dates(start="2024-01-02", periods=40):
    return pd.bdate_range(start, periods=periods).strftime("%Y-%m-%d").tolist()


def _bars(symbols, dates, price=10.0, amount=500_000_000.0, overrides=None,
          volume_overrides=None, is_st=None, flags=None):
    overrides = overrides or {}
    volume_overrides = volume_overrides or {}
    rows = []
    for symbol in symbols:
        for date in dates:
            if (symbol, date) in overrides:
                open_, high, low, close = overrides[(symbol, date)]
            else:
                open_ = high = low = close = price
            row = {
                "symbol": symbol,
                "date": date,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "amount": amount,
                "volume": volume_overrides.get((symbol, date), 1_000_000.0),
            }
            if is_st and symbol in is_st:
                row["is_st"] = True
            if flags is not None:
                row["halted"] = False
                row["one_word_limit_up"] = False
                row["one_word_limit_down"] = False
                row.update(flags.get((symbol, date), {}))
            rows.append(row)
    return pd.DataFrame(rows)


def _preds(specs, dates=None, signal_index=20):
    if dates is None:
        dates = _dates()
    frame = pd.DataFrame(specs)
    asof = dates[signal_index]
    if "asof_date" not in frame.columns:
        frame["asof_date"] = asof
    for horizon in range(1, 11):
        column = f"label_date_d{horizon}"
        if column not in frame.columns:
            frame[column] = dates[signal_index + horizon]
    return frame


def _one(result, symbol):
    matches = [trade for trade in result["trades"] if trade["symbol"] == symbol]
    assert len(matches) == 1, result["rejects"]
    return matches[0]


def test_slippage_tiers_are_fixed_and_match_the_liquidity_floor():
    assert slippage_rate(500_000_000) == 0.0005
    assert slippage_rate(499_999_999) == 0.0010
    assert slippage_rate(200_000_000) == 0.0010
    assert slippage_rate(100_000_000) == 0.0015
    assert slippage_rate(50_000_000) == 0.0020
    assert slippage_rate(49_999_999) is None
    assert limit_ratio("sh.600000") == 0.10
    assert limit_ratio("sz.300750") == 0.20
    assert limit_ratio("sh.688001") == 0.20
    assert limit_ratio("bj.830001") == 0.30
    assert limit_ratio("sh.600000", is_st=True) == 0.05


def test_flat_trade_charges_30bps_round_trip_once_not_per_side():
    dates = _dates()
    signal = dates[20]
    preds = _preds([{"symbol": "sh.600000", "predicted_return_d10": PRED_ABS_MIN}])
    bars = _bars(["sh.600000"], dates, amount=500_000_000.0)
    result = run_execution_backtest(preds, bars)
    trade = _one(result, "sh.600000")
    slip = 0.0005
    gross = (10.0 * (1.0 - slip)) / (10.0 * (1.0 + slip)) - 1.0
    assert trade["entry_date"] == dates[21]
    assert trade["exit_date"] == dates[30]
    assert trade["exit_session"] == "planned_close"
    assert trade["weight"] == 0.1
    assert abs(trade["gross_return"] - gross) < 1e-12
    assert abs(trade["net_return"] - (gross - ROUND_TRIP_COST)) < 1e-12
    assert abs((trade["gross_return"] - trade["net_return"]) - 0.003) < 1e-12
    assert abs(trade["net_return"] - (gross - 0.006)) > 1e-4
    assert abs(result["book"]["sum_daily_net_return"] - trade["weight"] * trade["net_return"]) < 1e-12
    assert result["book"]["max_drawdown"] <= 0
    assert result["publishable_as_section_5_2"] is False


def test_threshold_is_not_a_full_sample_topk():
    dates = _dates()
    specs = [{"symbol": f"sh.6000{i:02d}", "predicted_return_d10": 0.04} for i in range(10)]
    specs.append({"symbol": "sh.600099", "predicted_return_d10": 0.05})
    result = run_execution_backtest(_preds(specs), _bars([row["symbol"] for row in specs], dates))
    assert [trade["symbol"] for trade in result["trades"]] == ["sh.600099"]
    assert result["trades"][0]["weight"] == 0.1
    assert result["counts"]["threshold_rows"] == 1


def test_two_sleeves_keep_one_tenth_capital_each():
    dates = _dates()
    # Label dates have to be attached per signal, not from a single index.
    rows = []
    for asof_index in (20, 21):
        row = {"symbol": "sh.600000", "asof_date": dates[asof_index], "predicted_return_d10": 0.06}
        for horizon in range(1, 11):
            row[f"label_date_d{horizon}"] = dates[asof_index + horizon]
        rows.append(row)
    result = run_execution_backtest(pd.DataFrame(rows), _bars(["sh.600000"], dates))
    assert len(result["trades"]) == 2
    assert {trade["weight"] for trade in result["trades"]} == {0.1}
    overlapped = [
        row for row in result["nav_curve"]
        if dates[22] <= row["date"] <= dates[30]
    ]
    assert overlapped
    assert all(abs(row["gross_exposure"] - 0.2) < 1e-12 for row in overlapped)


def test_one_word_limit_up_cancels_the_buy_and_does_not_retry():
    dates = _dates()
    entry = dates[21]
    bars = _bars(
        ["sh.600000"],
        dates,
        overrides={("sh.600000", entry): (11.0, 11.0, 11.0, 11.0)},
    )
    # The following session is a normal print. The cancelled buy must not slide onto it.
    result = run_execution_backtest(
        _preds([{"symbol": "sh.600000", "predicted_return_d10": 0.07}]),
        bars,
    )
    assert result["trades"] == []
    assert result["counts"]["rejects"]["skipped_one_word_limit_up"] == 1


def test_one_word_limit_down_can_be_bought_but_not_sold_short():
    dates = _dates()
    entry = dates[21]
    bars = _bars(
        ["sh.600000"],
        dates,
        overrides={("sh.600000", entry): (9.0, 9.0, 9.0, 9.0)},
    )
    long = run_execution_backtest(
        _preds([{"symbol": "sh.600000", "predicted_return_d10": 0.07}]),
        bars,
    )
    short = run_execution_backtest(
        _preds([{"symbol": "sh.600000", "predicted_return_d10": -0.07}]),
        bars,
    )
    assert _one(long, "sh.600000")["entry_open"] == 9.0
    assert short["trades"] == []
    assert short["counts"]["rejects"]["skipped_one_word_limit_down"] == 1


def test_chinext_ten_percent_one_word_is_not_a_limit_and_twenty_percent_is():
    dates = _dates()
    entry = dates[21]
    symbol = "sz.300750"
    not_limit = run_execution_backtest(
        _preds([{"symbol": symbol, "predicted_return_d10": 0.08}]),
        _bars([symbol], dates, overrides={(symbol, entry): (11.0, 11.0, 11.0, 11.0)}),
    )
    blocked = run_execution_backtest(
        _preds([{"symbol": symbol, "predicted_return_d10": 0.08}]),
        _bars([symbol], dates, overrides={(symbol, entry): (12.0, 12.0, 12.0, 12.0)}),
    )
    assert _one(not_limit, symbol)["entry_open"] == 11.0
    assert blocked["counts"]["rejects"]["skipped_one_word_limit_up"] == 1


def test_st_uses_the_five_percent_limit():
    dates = _dates()
    entry = dates[21]
    symbol = "sh.600000"
    blocked = run_execution_backtest(
        _preds([{"symbol": symbol, "predicted_return_d10": 0.08}]),
        _bars([symbol], dates, overrides={(symbol, entry): (10.5, 10.5, 10.5, 10.5)}, is_st={symbol}),
    )
    opened = run_execution_backtest(
        _preds([{"symbol": symbol, "predicted_return_d10": 0.08}]),
        _bars([symbol], dates, overrides={(symbol, entry): (11.0, 11.0, 11.0, 11.0)}, is_st={symbol}),
    )
    assert blocked["counts"]["rejects"]["skipped_one_word_limit_up"] == 1
    assert _one(opened, symbol)["entry_open"] == 11.0


def test_halt_defers_entry_to_the_resume_open():
    dates = _dates()
    halted_day = dates[21]
    resume = dates[22]
    bars = _bars(
        ["sh.600000"],
        dates,
        overrides={( "sh.600000", resume): (12.0, 12.2, 11.8, 12.0)},
        volume_overrides={("sh.600000", halted_day): 0.0},
    )
    result = run_execution_backtest(
        _preds([{"symbol": "sh.600000", "predicted_return_d10": 0.08}]),
        bars,
    )
    trade = _one(result, "sh.600000")
    assert trade["entry_date"] == resume
    assert trade["entry_open"] == 12.0
    assert trade["entry_defer_sessions"] == 1
    assert result["counts"]["entry_deferrals"] == 1


def test_one_word_limit_down_defers_the_long_exit_to_the_next_open():
    dates = _dates()
    planned = dates[30]
    nxt = dates[31]
    bars = _bars(
        ["sh.600000"],
        dates,
        overrides={
            ("sh.600000", planned): (9.0, 9.0, 9.0, 9.0),
            ("sh.600000", nxt): (8.0, 8.4, 7.8, 8.2),
        },
    )
    trade = _one(run_execution_backtest(
        _preds([{"symbol": "sh.600000", "predicted_return_d10": 0.08}]),
        bars,
    ), "sh.600000")
    assert trade["exit_date"] == nxt
    assert trade["exit_price"] == 8.0
    assert trade["exit_session"] == "deferred_open"
    assert trade["exit_defer_sessions"] >= 1
    slip = trade["slippage_per_side"]
    gross = (8.0 * (1.0 - slip)) / (10.0 * (1.0 + slip)) - 1.0
    assert abs(trade["gross_return"] - gross) < 1e-12
    assert abs(trade["net_return"] - (gross - ROUND_TRIP_COST)) < 1e-12


def test_one_word_limit_up_defers_the_short_cover():
    dates = _dates()
    planned = dates[30]
    nxt = dates[31]
    bars = _bars(
        ["sh.600000"],
        dates,
        overrides={
            ("sh.600000", planned): (11.0, 11.0, 11.0, 11.0),
            ("sh.600000", nxt): (12.0, 12.4, 11.8, 12.1),
        },
    )
    trade = _one(run_execution_backtest(
        _preds([{"symbol": "sh.600000", "predicted_return_d10": -0.08}]),
        bars,
    ), "sh.600000")
    assert trade["exit_date"] == nxt
    assert trade["exit_price"] == 12.0
    assert trade["exit_session"] == "deferred_open"


def test_explicit_flags_override_the_price_geometry():
    dates = _dates()
    entry = dates[21]
    forced = _bars(
        ["sh.600000"],
        dates,
        flags={("sh.600000", entry): {"one_word_limit_up": True}},
    )
    waived = _bars(
        ["sh.600000"],
        dates,
        overrides={("sh.600000", entry): (11.0, 11.0, 11.0, 11.0)},
        flags={("sh.600000", entry): {"one_word_limit_up": False, "one_word_limit_down": False}},
    )
    blocked = run_execution_backtest(
        _preds([{"symbol": "sh.600000", "predicted_return_d10": 0.08}]),
        forced,
    )
    filled = run_execution_backtest(
        _preds([{"symbol": "sh.600000", "predicted_return_d10": 0.08}]),
        waived,
    )
    assert blocked["limit_flag_mode"] == "explicit"
    assert blocked["counts"]["rejects"]["skipped_one_word_limit_up"] == 1
    assert _one(filled, "sh.600000")["entry_open"] == 11.0


def test_liquidity_uses_only_bars_on_or_before_the_signal():
    dates = _dates()
    signal = dates[20]
    symbol = "sh.600000"
    base = _bars([symbol], dates, amount=80_000_000.0)
    richer_future = base.copy()
    richer_future.loc[richer_future["date"] > signal, "amount"] = 1.0
    left = run_execution_backtest(_preds([{"symbol": symbol, "predicted_return_d10": 0.08}]), base)
    right = run_execution_backtest(_preds([{"symbol": symbol, "predicted_return_d10": 0.08}]), richer_future)
    assert _one(left, symbol)["median_amount"] == _one(right, symbol)["median_amount"]
    assert _one(left, symbol)["slippage_per_side"] == 0.0020

    poor = _bars([symbol], dates, amount=10_000_000.0)
    poor.loc[poor["date"] > signal, "amount"] = 1_000_000_000.0
    refused = run_execution_backtest(_preds([{"symbol": symbol, "predicted_return_d10": 0.2}]), poor)
    assert refused["trades"] == []
    assert refused["counts"]["rejects"]["illiquid"] == 1


def test_short_history_is_not_tradable():
    dates = _dates(periods=30)
    signal = dates[10]
    bars = _bars(["sh.600000"], dates[5:])  # dates[5].. ; signal dates[10] has only 6 bars
    row = {"symbol": "sh.600000", "asof_date": signal, "predicted_return_d10": 0.08}
    for horizon in range(1, 11):
        row[f"label_date_d{horizon}"] = dates[10 + horizon]
    result = run_execution_backtest(pd.DataFrame([row]), bars)
    assert result["trades"] == []
    assert result["counts"]["rejects"]["illiquid"] == 1


def test_halt_gap_rows_are_dropped_before_orders():
    dates = _dates()
    signal_index = 20
    symbols = ["sh.600001", "sh.600002", "sh.600003", "sh.600099"]
    rows = []
    for symbol in symbols:
        row = {"symbol": symbol, "asof_date": dates[signal_index], "predicted_return_d10": 0.09}
        for horizon in range(1, 11):
            if symbol == "sh.600099":
                row[f"label_date_d{horizon}"] = "2025-06-01"
            else:
                row[f"label_date_d{horizon}"] = dates[signal_index + horizon]
        rows.append(row)
    result = run_execution_backtest(pd.DataFrame(rows), _bars(symbols, dates))
    assert "sh.600099" not in {trade["symbol"] for trade in result["trades"]}
    assert result["counts"]["filled"] == 3
    assert result["counts"]["rejects"]["halt_gap"] == 1
    assert result["halt_gap_filter"] == "applied"


def test_actual_returns_do_not_change_the_fill():
    dates = _dates()
    bars = _bars(["sh.600000"], dates)
    base = _preds([{"symbol": "sh.600000", "predicted_return_d10": 0.08, "actual_return_d10": 0.5}])
    other = base.copy()
    other["actual_return_d10"] = -0.4
    left = run_execution_backtest(base, bars)
    right = run_execution_backtest(other, bars)
    assert _one(left, "sh.600000")["net_return"] == _one(right, "sh.600000")["net_return"]
    assert left["book"]["nav_end"] == right["book"]["nav_end"]


def test_non_c2_predictions_are_refused():
    dates = _dates()
    preds = _preds([{"symbol": "sh.600000", "predicted_return_d10": 0.08, "model": "bigbatch_best"}])
    try:
        run_execution_backtest(preds, _bars(["sh.600000"], dates))
    except ValueError as exc:
        assert "C2" in str(exc)
    else:
        raise AssertionError("e03b predictions should be refused")


def test_missing_amount_does_not_invent_a_liquidity_pass():
    dates = _dates()
    bars = _bars(["sh.600000"], dates).drop(columns=["amount"])
    result = run_execution_backtest(
        _preds([{"symbol": "sh.600000", "predicted_return_d10": 0.08}]),
        bars,
    )
    assert result["trades"] == []
    assert "amount_column_missing" in result["publication_blockers"]


def test_target_window_gate_opens_only_with_explicit_flags_and_mature_bars():
    dates = pd.bdate_range("2026-09-01", "2026-11-20").strftime("%Y-%m-%d").tolist()
    signal = "2026-10-23"
    assert signal in dates
    index = dates.index(signal)
    row = {"symbol": "sh.600000", "asof_date": signal, "predicted_return_d10": 0.08, "model": "c2_best"}
    for horizon in range(1, 11):
        row[f"label_date_d{horizon}"] = dates[index + horizon]
    bars = _bars(["sh.600000"], dates, flags={})
    result = run_execution_backtest(pd.DataFrame([row]), bars)
    assert result["planned_exit_max"] == "2026-11-06"
    assert result["publication_blockers"] == []
    assert result["publishable_as_section_5_2"] is True
    assert result["role"] == "section_5_2"

    inferred = run_execution_backtest(
        pd.DataFrame([row]),
        _bars(["sh.600000"], dates),
    )
    assert inferred["publishable_as_section_5_2"] is False
    assert "limit_flags_not_explicit_or_unadjusted" in inferred["publication_blockers"]


def test_consumed_sealed_window_and_old_panel_cannot_publish():
    dates = pd.bdate_range("2026-08-03", "2026-10-30").strftime("%Y-%m-%d").tolist()
    sealed = "2026-09-22"
    index = dates.index(sealed)
    row = {"symbol": "sh.600000", "asof_date": sealed, "predicted_return_d10": 0.08}
    for horizon in range(1, 11):
        row[f"label_date_d{horizon}"] = dates[index + horizon]
    result = run_execution_backtest(pd.DataFrame([row]), _bars(["sh.600000"], dates, flags={}))
    assert "consumed_sealed_window_2026-09-18_2026-09-30" in result["publication_blockers"]
    assert "signal_window_is_not_2026-10-09_2026-10-23" in result["publication_blockers"]
    assert result["publishable_as_section_5_2"] is False

    old = "2026-08-11"
    old_index = dates.index(old)
    old_row = {"symbol": "sh.600000", "asof_date": old, "predicted_return_d10": 0.08}
    for horizon in range(1, 11):
        old_row[f"label_date_d{horizon}"] = dates[old_index + horizon]
    old_result = run_execution_backtest(pd.DataFrame([old_row]), _bars(["sh.600000"], dates, flags={}))
    assert "signal_window_is_not_2026-10-09_2026-10-23" in old_result["publication_blockers"]
    assert old_result["metrics_are_section_5_2"] is False


def test_unadjusted_file_supplies_the_one_word_flag():
    dates = _dates()
    entry = dates[21]
    adjusted = _bars(["sh.600000"], dates)  # flat, would otherwise fill
    raw = _bars(
        ["sh.600000"],
        dates,
        overrides={("sh.600000", entry): (11.0, 11.0, 11.0, 11.0)},
    )
    merged, unmatched = bars_with_unadjusted_flags(adjusted, raw)
    assert unmatched == 0
    result = run_execution_backtest(
        _preds([{"symbol": "sh.600000", "predicted_return_d10": 0.08}]),
        merged,
        limit_flag_mode_override="unadjusted_file",
    )
    assert result["limit_flag_mode"] == "unadjusted_file"
    assert result["counts"]["rejects"]["skipped_one_word_limit_up"] == 1
    # The price path itself stayed at the adjusted print.
    assert float(merged.loc[merged["date"] == entry, "open"].iloc[0]) == 10.0


def test_driver_smoke_is_not_a_section_5_2_result(tmp_path):
    predictions, bars = synthetic_smoke_inputs()
    result = run_execution_backtest(predictions, bars, synthetic=True)
    assert result["publishable_as_section_5_2"] is False
    assert result["role"] == "interface_smoke"
    assert result["counts"]["filled"] == 1
    assert result["counts"]["rejects"]["skipped_one_word_limit_up"] == 1

    output = tmp_path / "c2_execution_smoke.json"
    completed = subprocess.run(
        [sys.executable, "-m", "finetune.small_clean_oos.run_c2_execution_backtest",
         "--smoke", "--output", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text())
    assert payload["publishable_as_section_5_2"] is False
    assert payload["role"] == "interface_smoke"
    assert "synthetic_or_interface_smoke" in payload["publication_blockers"]

    refused = subprocess.run(
        [sys.executable, "-m", "finetune.small_clean_oos.run_c2_execution_backtest",
         "--smoke", "--output", str(tmp_path / "c2_execution_constrained_8b.json")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode == 2


def test_driver_missing_inputs_write_a_blocker_without_metrics(tmp_path):
    output = tmp_path / "blocker.json"
    status = main(["--predictions", str(tmp_path / "missing.csv"), "--bars", str(tmp_path / "missing_bars.csv"),
                   "--output", str(output)])
    assert status == 0
    payload = json.loads(output.read_text())
    assert payload["metrics"] is None
    assert payload["publishable_as_section_5_2"] is False
    assert payload["reason"] == "input_path_missing"
