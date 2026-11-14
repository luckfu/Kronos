"""Engine checks for the few-name backtest on a small synthetic market (CPU, no data files)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune/oos_eval/base_clean_sealed_oos"))
sys.path.insert(0, str(ROOT / "tests"))

import few_names_backtest as fnb  # noqa: E402
import signal_usage_backtest as sub  # noqa: E402
from test_signal_usage_backtest import FIRST_SIGNAL, LAST_SIGNAL, N_DAYS, N_SYMBOLS, make_market, scores_from  # noqa: E402

REBALANCE = list(range(FIRST_SIGNAL, LAST_SIGNAL + 1, 3))
LAST_DAY = LAST_SIGNAL + 1


def noise_scores(market, seed=3):
    noise = np.random.default_rng(seed).normal(size=(N_DAYS, N_SYMBOLS))
    return scores_from(market, lambda day: noise[day])


def test_universe_matches_buy_and_hold_of_the_other_engine(tmp_path):
    market = make_market(tmp_path)
    scores = noise_scores(market)
    ours = fnb.universe(market, scores, FIRST_SIGNAL, LAST_DAY)
    theirs = sub.run_book(market, scores, list(range(FIRST_SIGNAL, LAST_SIGNAL + 1)), sub.BOOKS["U"])["daily"]
    np.testing.assert_allclose(ours.to_numpy(), theirs["ret"].to_numpy(), rtol=0, atol=1e-12)


def test_slots_hold_exactly_the_top_k_without_buffer(tmp_path):
    market = make_market(tmp_path)
    scores = noise_scores(market)
    k = 5
    book = fnb.run_slots(market, fnb.c2_plan(market, scores, k, buffer=False), REBALANCE, k, LAST_DAY)
    assert book["daily"]["names"].max() == k
    order, _ = fnb.ranked_pool(market, scores, REBALANCE[-1])
    assert set(np.flatnonzero(book["final_value"] > 0)) == set(order[:k])
    assert book["daily"]["one_way_turnover"][book["daily"]["one_way_turnover"] > 0].count() == len(REBALANCE)


def test_buffer_trades_less_and_only_sells_below_the_band(tmp_path):
    market = make_market(tmp_path)
    scores = noise_scores(market)
    k = 5
    plain = fnb.run_slots(market, fnb.c2_plan(market, scores, k, buffer=False), REBALANCE, k, LAST_DAY)
    buffered = fnb.run_slots(market, fnb.c2_plan(market, scores, k, buffer=True), REBALANCE, k, LAST_DAY)
    assert sum(buffered["orders"][1:]) < sum(plain["orders"][1:])
    _, pct = fnb.ranked_pool(market, scores, REBALANCE[-1])
    held = np.flatnonzero(buffered["final_value"] > 0)
    assert len(held) == k and (pct[held] > 1.0 - fnb.BUFFER_KEEP_FRACTION).all()


def test_perfect_foresight_beats_universe_and_costs_reduce_returns(tmp_path):
    market = make_market(tmp_path)
    future = scores_from(market, lambda day: market.close[min(day + 3, N_DAYS - 1)] / market.close[day] - 1.0)
    k = 5
    net = fnb.run_slots(market, fnb.c2_plan(market, future, k, False), REBALANCE, k, LAST_DAY)["daily"]["ret"]
    gross = fnb.run_slots(market, fnb.c2_plan(market, future, k, False), REBALANCE, k, LAST_DAY,
                          charge_costs=False)["daily"]["ret"]
    assert fnb.total(gross) > fnb.total(net) > fnb.total(fnb.universe(market, future, FIRST_SIGNAL, LAST_DAY))


def test_no_lookahead(tmp_path):
    market = make_market(tmp_path)
    scores = noise_scores(market)
    k, cut = 5, 9
    base = fnb.run_slots(market, fnb.c2_plan(market, scores, k, True), REBALANCE, k, LAST_DAY)["daily"]["ret"]

    def shock(bars, dates):
        bars.loc[bars["date"] > dates[FIRST_SIGNAL + cut], ["open", "high", "low", "close"]] *= 1.37
        return bars

    (tmp_path / "b").mkdir()
    shocked_market = make_market(tmp_path / "b", mutate=shock)
    shocked = fnb.run_slots(shocked_market, fnb.c2_plan(shocked_market, noise_scores(shocked_market), k, True),
                            REBALANCE, k, LAST_DAY)["daily"]["ret"]
    np.testing.assert_allclose(shocked.iloc[:cut], base.iloc[:cut], rtol=0, atol=1e-12)
    later = scores.copy()
    later[FIRST_SIGNAL + cut:] *= -1.0
    changed = fnb.run_slots(market, fnb.c2_plan(market, later, k, True), REBALANCE, k, LAST_DAY)["daily"]["ret"]
    np.testing.assert_allclose(changed.iloc[:cut], base.iloc[:cut], rtol=0, atol=1e-12)
    assert not np.allclose(changed.iloc[cut + 1:], base.iloc[cut + 1:])


def test_random_books_center_on_the_universe(tmp_path):
    market = make_market(tmp_path)
    scores = noise_scores(market)
    rng = np.random.default_rng(0)
    bench = fnb.total(fnb.universe(market, scores, FIRST_SIGNAL, LAST_DAY, charge_costs=False))
    excess = [fnb.total(fnb.run_slots(market, fnb.random_plan(market, scores, rng), REBALANCE, 5, LAST_DAY,
                                      charge_costs=False)["daily"]["ret"]) - bench for _ in range(300)]
    assert abs(np.mean(excess)) < 3.0 * np.std(excess, ddof=1) / np.sqrt(len(excess))
    assert np.std(excess) > 0.01           # five names really are noisy
