"""Engine checks for the signal-usage backtest on a small synthetic market (CPU, no data files)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune/oos_eval/base_clean_sealed_oos"))

import signal_usage_backtest as sub  # noqa: E402

N_SYMBOLS, N_DAYS, FIRST_SIGNAL, LAST_SIGNAL = 40, 60, 25, 45
SIDE_COST = 0.0005 + sub.FEE_PER_SIDE      # top liquidity tier + fee


def make_bars(seed=0, mutate=None):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-05", periods=N_DAYS).strftime("%Y-%m-%d")
    rows = []
    for s in range(N_SYMBOLS):
        close = 20.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, N_DAYS))
        open_ = np.concatenate(([20.0], close[:-1])) * (1.0 + rng.normal(0.0, 0.005, N_DAYS))
        for d in range(N_DAYS):
            rows.append({"symbol": f"sh.{600000 + s}", "date": dates[d], "open": open_[d], "close": close[d],
                         "high": max(open_[d], close[d]) * 1.01, "low": min(open_[d], close[d]) * 0.99,
                         "volume": 1e6, "amount": 1e9})
    bars = pd.DataFrame(rows)
    return mutate(bars, dates) if mutate else bars


def make_market(tmp_path, **kwargs):
    path = tmp_path / "bars.csv"
    make_bars(**kwargs).to_csv(path, index=False)
    return sub.Market(path)


def scores_from(market, fn):
    matrix = np.full(market.traded.shape, np.nan)
    for day in range(FIRST_SIGNAL, LAST_SIGNAL + 1):
        matrix[day] = fn(day)
    return matrix


def run(market, matrix, book, **kwargs):
    return sub.run_book(market, matrix, list(range(FIRST_SIGNAL, LAST_SIGNAL + 1)), sub.BOOKS[book], **kwargs)


def test_universe_is_equal_weight_buy_and_hold(tmp_path):
    market = make_market(tmp_path)
    matrix = scores_from(market, lambda day: np.zeros(N_SYMBOLS))
    first, last = FIRST_SIGNAL + 1, LAST_SIGNAL + 1
    expected = float((market.close[last] / market.open[first]).mean())
    gross = run(market, matrix, "U", charge_costs=False)["daily"]
    net = run(market, matrix, "U")["daily"]
    assert len(gross) == last - first + 1
    assert gross["equity"].iloc[-1] == pytest.approx(expected, rel=1e-12)
    # One entry cost on the whole book, no later trading.
    assert net["equity"].iloc[-1] == pytest.approx(expected * (1.0 - SIDE_COST), rel=1e-12)
    assert net["one_way_turnover"].iloc[1:].abs().max() < 1e-12


def test_perfect_foresight_wins_and_buffer_cuts_turnover(tmp_path):
    market = make_market(tmp_path)
    future = lambda day: market.close[min(day + 10, N_DAYS - 1)] / market.close[day] - 1.0   # noqa: E731
    matrix = scores_from(market, future)
    universe = run(market, matrix, "U")["daily"]
    top, buffered = run(market, matrix, "T10")["daily"], run(market, matrix, "T10b30")["daily"]
    assert top["equity"].iloc[-1] > universe["equity"].iloc[-1]
    assert buffered["one_way_turnover"].iloc[1:].mean() < top["one_way_turnover"].iloc[1:].mean()
    # Exclusion book never holds a name ranked in the bottom 20% of the latest signal.
    result = run(market, matrix, "E20")
    pct = sub.pool_percentiles(matrix[LAST_SIGNAL], market.liquid[LAST_SIGNAL])
    assert not ((result["final_value"] > 0) & (pct <= 0.20)).any()


def test_no_lookahead_in_prices_or_scores(tmp_path):
    rng = np.random.default_rng(1)
    noise = rng.normal(size=(N_DAYS, N_SYMBOLS))
    base_market = make_market(tmp_path)
    base = run(base_market, scores_from(base_market, lambda day: noise[day]), "T10b30")["daily"]
    cut = 12                                   # rows of the NAV that must not change

    def shock_prices(bars, dates):             # change every bar after the cut
        late = bars["date"] > dates[FIRST_SIGNAL + cut]
        bars.loc[late, ["open", "high", "low", "close"]] *= 1.37
        return bars

    (tmp_path / "b").mkdir()
    shocked_market = make_market(tmp_path / "b", mutate=shock_prices)
    shocked = run(shocked_market, scores_from(shocked_market, lambda day: noise[day]), "T10b30")["daily"]
    np.testing.assert_allclose(shocked["ret"].iloc[:cut], base["ret"].iloc[:cut], rtol=0, atol=1e-12)

    later_scores = scores_from(base_market, lambda day: noise[day] if day < FIRST_SIGNAL + cut else -noise[day])
    changed = run(base_market, later_scores, "T10b30")["daily"]
    np.testing.assert_allclose(changed["ret"].iloc[:cut], base["ret"].iloc[:cut], rtol=0, atol=1e-12)
    assert not np.allclose(changed["ret"].iloc[cut + 1:], base["ret"].iloc[cut + 1:])


def test_one_word_limits_block_entries_and_exits(tmp_path):
    target = "sh.600003"

    def limit_up_on_entry(bars, dates):        # flat at +10% on the first execution day
        day, prev = dates[FIRST_SIGNAL + 1], dates[FIRST_SIGNAL]
        prev_close = float(bars.loc[(bars.symbol == target) & (bars.date == prev), "close"].iloc[0])
        price = np.floor(prev_close * 1.10 * 100 + 0.5) / 100
        bars.loc[(bars.symbol == target) & (bars.date == day), ["open", "high", "low", "close"]] = price
        return bars

    market = make_market(tmp_path, mutate=limit_up_on_entry)
    column = market.symbol_index[target]
    assert market.one_up[FIRST_SIGNAL + 1, column]

    def scores(day):                           # top 4 of 40 = the T10 book; the target always ranks first
        row = np.zeros(N_SYMBOLS)
        row[column] = 1.0
        others = (0, 1, 2) if day == FIRST_SIGNAL else (0, 1, 5)   # symbol 2 drops out on the second day
        row[list(others)] = (0.9, 0.8, 0.7)
        return row

    matrix = scores_from(market, scores)
    one_day = sub.run_book(market, matrix, [FIRST_SIGNAL], sub.BOOKS["T10"])
    assert one_day["final_value"][column] == 0.0             # could not be bought at the limit
    assert (one_day["final_value"] > 0).sum() == 3
    # The sale of symbol 2 frees cash, so the retried entry fills on the next session.
    two_days = sub.run_book(market, matrix, [FIRST_SIGNAL, FIRST_SIGNAL + 1], sub.BOOKS["T10"])
    assert two_days["final_value"][column] > 0.0
    assert two_days["final_value"][2] == 0.0


def test_sealed_window_is_refused(tmp_path):
    frame = pd.DataFrame({"symbol": ["sh.600000"], "asof_date": ["2026-10-26"], "predicted_return_d10": [0.01]})
    (tmp_path / "shards").mkdir()
    frame.to_csv(tmp_path / "shards" / "x_2026-10-26.csv.gz", index=False, compression="gzip")
    with pytest.raises(SystemExit, match="sealed"):
        sub.load_scores(f"shards:{tmp_path / 'shards'}")
