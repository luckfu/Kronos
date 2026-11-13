"""The vectorised take-profit simulation must match a plain per-trade loop (synthetic market, CPU)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune/oos_eval/base_clean_sealed_oos"))
sys.path.insert(0, str(ROOT / "tests"))

import daily_pick_backtest as dp  # noqa: E402
from test_signal_usage_backtest import FIRST_SIGNAL, N_SYMBOLS, make_market  # noqa: E402


def reference(market, signal, symbol, take_profit, time_stop):
    entry, last = signal + 1, len(market.dates) - 1
    basis = market.open[entry, symbol] * (1.0 + market.side_cost[entry, symbol])
    for day in range(entry + 1, last + 1):
        keep = 1.0 - market.side_cost[day, symbol]
        if not market.traded[day, symbol] or market.one_down[day, symbol]:
            continue
        if take_profit is not None:
            target = basis * (1.0 + take_profit) / keep
            if market.high[day, symbol] >= target:
                return max(market.open[day, symbol], target) * keep / basis - 1.0, dp.TP_HIT, day - entry
        if time_stop is not None and day - entry >= time_stop:
            return market.close[day, symbol] * keep / basis - 1.0, dp.TIME_STOP, day - entry
    return market.close[last, symbol] * (1.0 - market.side_cost[last, symbol]) / basis - 1.0, dp.OPEN, last - entry


def test_simulation_matches_reference_loop(tmp_path):
    market = make_market(tmp_path)
    symbols = np.arange(N_SYMBOLS)
    seen = set()
    for take_profit, time_stop in dp.EXITS.values():
        trades = dp.simulate(market, FIRST_SIGNAL, symbols, take_profit, time_stop)
        assert len(trades) == N_SYMBOLS
        for row in trades.itertuples():
            net, status, hold = reference(market, FIRST_SIGNAL, row.symbol, take_profit, time_stop)
            assert abs(row.net - net) < 1e-12 and row.status == status and row.hold == hold
            seen.add(status)
            if status == dp.TP_HIT:
                assert row.net >= take_profit - 1e-12 and row.hold >= 1          # T+1, and at least the target
    assert seen == {dp.TP_HIT, dp.TIME_STOP, dp.OPEN}


def test_unbuyable_names_are_dropped_and_selections_nest(tmp_path):
    market = make_market(tmp_path)
    market.one_up[FIRST_SIGNAL + 1, 3] = True
    trades = dp.simulate(market, FIRST_SIGNAL, np.arange(N_SYMBOLS), 0.05, None)
    assert 3 not in set(trades["symbol"]) and len(trades) == N_SYMBOLS - 1
    scores = np.full(market.traded.shape, np.nan)
    rng = np.random.default_rng(0)
    for day in (FIRST_SIGNAL, FIRST_SIGNAL + 1, FIRST_SIGNAL + 2):
        scores[day] = rng.normal(size=N_SYMBOLS)
    scores[FIRST_SIGNAL:FIRST_SIGNAL + 3, 7] = 99.0                             # always ranked first
    import few_names_lowvol_backtest as lv
    picks = dp.selections(market, scores, lv.trailing_volatility(market), [FIRST_SIGNAL, FIRST_SIGNAL + 1, FIRST_SIGNAL + 2])
    last = picks[FIRST_SIGNAL + 2]
    assert list(last["top1"]) == [7] and set(last["top3"]) <= set(last["top10"]) <= set(last["pool"])
    assert 7 in last["persist2"] and 7 in last["persist3"] and "persist3" not in picks[FIRST_SIGNAL + 1]
    assert set(last["persist3"]) <= set(last["persist2"]) <= set(last["top10"])
