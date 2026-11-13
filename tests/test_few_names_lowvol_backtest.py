"""Checks for the volatility-gated few-name books on a small synthetic market (CPU, no data files)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune/oos_eval/base_clean_sealed_oos"))
sys.path.insert(0, str(ROOT / "tests"))

import few_names_backtest as fnb  # noqa: E402
import few_names_lowvol_backtest as lv  # noqa: E402
from test_signal_usage_backtest import FIRST_SIGNAL, LAST_SIGNAL, N_DAYS, N_SYMBOLS, make_market, scores_from  # noqa: E402

REBALANCE = list(range(FIRST_SIGNAL, LAST_SIGNAL + 1, 3))
LAST_DAY = LAST_SIGNAL + 1
K = 5


def setup(tmp_path, seed=3):
    market = make_market(tmp_path)
    noise = np.random.default_rng(seed).normal(size=(N_DAYS, N_SYMBOLS))
    return market, scores_from(market, lambda day: noise[day]), lv.trailing_volatility(market)


def run(market, scores, vol, variant, buffer=True, **kwargs):
    return fnb.run_slots(market, lv.make_plan(market, scores, vol, variant, K, buffer), REBALANCE, K, LAST_DAY, **kwargs)


def test_volatility_uses_only_past_closes(tmp_path):
    market, _, vol = setup(tmp_path)
    day = FIRST_SIGNAL
    expected = np.std(market.close[day - 19:day + 1] / market.close[day - 20:day] - 1.0, axis=0, ddof=1)
    np.testing.assert_allclose(vol[day], expected, rtol=1e-10)
    assert np.isnan(vol[19]).all() and np.isfinite(vol[20]).all()


def test_no_gate_is_the_previous_engine(tmp_path):
    market, scores, vol = setup(tmp_path)
    for buffer in (False, True):
        ours = run(market, scores, vol, "NONE", buffer)["daily"]["ret"]
        theirs = fnb.run_slots(market, fnb.c2_plan(market, scores, K, buffer), REBALANCE, K, LAST_DAY)["daily"]["ret"]
        np.testing.assert_array_equal(ours.to_numpy(), theirs.to_numpy())


def test_low_volatility_gate_limits_entries(tmp_path):
    market, scores, vol = setup(tmp_path)
    for variant, quantile in (("LV50", 0.50), ("LV70", 0.70)):
        for signal in REBALANCE:
            order, _ = lv.make_plan(market, scores, vol, variant, K, False)(signal)
            cutoff = np.quantile(vol[signal], quantile)
            assert (vol[signal][order] <= cutoff).all()
            assert (np.diff(scores[signal][order]) <= 0).all()          # best-first inside the gate
            assert len(order) == int((vol[signal] <= cutoff).sum())
    # Without a buffer the book holds exactly the gated top K.
    book = run(market, scores, vol, "LV50", buffer=False)
    order, _ = lv.make_plan(market, scores, vol, "LV50", K, False)(REBALANCE[-1])
    assert set(np.flatnonzero(book["final_value"] > 0)) == set(order[:K])


def test_stratified_takes_one_per_volatility_bucket(tmp_path):
    market, scores, vol = setup(tmp_path)
    signal = REBALANCE[2]
    order, _ = lv.make_plan(market, scores, vol, "STRAT", K, False)(signal)
    bucket = (np.argsort(np.argsort(vol[signal], kind="stable"), kind="stable") * lv.STRATA) // N_SYMBOLS
    assert sorted(bucket[order[:lv.STRATA]]) == list(range(lv.STRATA))            # one name from each bucket
    assert sorted(bucket[order[lv.STRATA:2 * lv.STRATA]]) == list(range(lv.STRATA))
    for b in range(lv.STRATA):                                                      # and it is the bucket's best
        inside = np.flatnonzero(bucket == b)
        assert inside[np.argmax(scores[signal][inside])] in order[:lv.STRATA]


def test_gate_applies_to_entries_only(tmp_path):
    market, scores, vol = setup(tmp_path)
    first, second = REBALANCE[0], REBALANCE[1]
    order, _ = lv.make_plan(market, scores, vol, "LV50", K, True)(first)
    held = order[0]
    scores = scores.copy()
    scores[second, held] = scores[second].max() + 1.0       # still ranked first ...
    vol = vol.copy()
    vol[second, held] = vol[second].max() * 10.0            # ... but now far outside the gate
    candidates, keep = lv.make_plan(market, scores, vol, "LV50", K, True)(second)
    assert held not in candidates and keep[held]
    book = fnb.run_slots(market, lv.make_plan(market, scores, vol, "LV50", K, True), [first, second], K, second + 1)
    assert book["final_value"][held] > 0


def test_no_lookahead(tmp_path):
    market, scores, vol = setup(tmp_path)
    cut = 9
    base = {v: run(market, scores, vol, v)["daily"]["ret"] for v in lv.VARIANTS}

    def shock(bars, dates):
        bars.loc[bars["date"] > dates[FIRST_SIGNAL + cut], ["open", "high", "low", "close"]] *= 1.37
        return bars

    (tmp_path / "b").mkdir()
    shocked = make_market(tmp_path / "b", mutate=shock)
    shocked_vol = lv.trailing_volatility(shocked)
    for variant in lv.VARIANTS:
        after = run(shocked, scores, shocked_vol, variant)["daily"]["ret"]
        np.testing.assert_allclose(after.iloc[:cut], base[variant].iloc[:cut], rtol=0, atol=1e-12)
