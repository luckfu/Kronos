"""Turnover-aware long-only backtest of two ways to *use* the C2 rank signal (CPU only).

Question 1 (exclusion): hold the tradable pool but stay out of C2's bottom names.
Question 2 (buffer): long C2's top decile, selling only after a name drops well
below the entry threshold.

Rules, books, datasets and the pass criteria are frozen in
``finetune/reports/20261112_signal_usage_backtest_prereg_cn.md``; nothing here is
tuned on the results.

Engine (shared by every book):

- Decide at the close of signal day T, trade at the next session's open.
- Value accounting: continuing positions are never rebalanced; a new position
  gets ``equity / target_count`` (scaled down when cash is short); sale proceeds
  stay in cash until a later buy needs them.
- Entry pool on T: traded on T, has a prediction, trailing 20-bar median amount
  >= 50M CNY.  Liquidity gates entries only.
- Percentile: rank of ``predicted_return_d10`` inside the pool (1 = best).
- Not buyable on a halt or one-word limit-up; not sellable on a halt or one-word
  limit-down; a held name without a prediction on T is kept.
- Costs per side: amount-tier slippage (5-20 bp) + 0.15% fee.
- NAV runs from the first execution day to the session after the last signal day.

Signal dates after SEAL_AFTER are refused: they belong to the acceptance window.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from finetune.small_clean_oos import execution_constrained as ec  # noqa: E402
from rank_book_backtest import infer_flags  # noqa: E402

SEAL_AFTER = "2026-10-23"
FEE_PER_SIDE = ec.ROUND_TRIP_COST / 2.0
FALLBACK_SLIPPAGE = ec.SLIPPAGE_TIERS_CNY[-1][1]
BOOTSTRAP = {"block": 10, "replicates": 2000, "seed": 20261112}

# name: (kind, enter_above, exit_at_or_below, first_day_enter_above, min_hold_sessions)
BOOKS = {
    "U": ("universe", 0.0, None, 0.0, 0),
    "E10": ("rank", 0.10, 0.10, 0.10, 0),
    "E20": ("rank", 0.20, 0.20, 0.20, 0),
    "E20b": ("rank", 0.40, 0.20, 0.20, 0),
    "T10": ("rank", 0.90, 0.90, 0.90, 0),
    "T10b30": ("rank", 0.90, 0.70, 0.90, 0),
    "T10b50": ("rank", 0.90, 0.50, 0.90, 0),
    "T10h10": ("rank", 0.90, 0.90, 0.90, 10),
}
PRIMARY_PAIRS = {"exclusion": ("E20b", "U"), "buffer": ("T10b30", "T10"), "long_book_vs_universe": ("T10b30", "U")}
EXTRA_PAIRS = [("E10", "U"), ("E20", "U"), ("T10", "U"), ("T10b50", "U"), ("T10h10", "U"),
               ("T10b50", "T10"), ("T10h10", "T10")]


def load_scores(spec: str) -> pd.DataFrame:
    """``shards:<dir>`` (evaluator output) or ``production:<dir>`` (Oracle archive copy)."""
    kind, path = spec.split(":", 1)
    if kind == "shards":
        frames = [pd.read_csv(p, usecols=["symbol", "asof_date", "predicted_return_d10"], dtype={"symbol": str})
                  for p in sorted(glob.glob(str(Path(path) / "*.csv.gz")))]
    elif kind == "production":
        frames = []
        for ranking in sorted(Path(path).glob("*/full_market/ranking.csv")):
            day = ranking.parts[-3]
            summary = json.loads((ranking.parent / "summary.json").read_text())
            # Read nothing from the acceptance window, and only production-parity (16-path) days.
            if day > SEAL_AFTER or summary["model"].get("sample_count") != 16:
                continue
            frame = pd.read_csv(ranking, usecols=["asof", "code", "predicted_return_d10"], dtype={"code": str})
            frames.append(frame.rename(columns={"asof": "asof_date", "code": "symbol"}))
    else:
        raise SystemExit(f"unknown score source {kind}")
    if not frames:
        raise SystemExit(f"no predictions under {path}")
    scores = pd.concat(frames, ignore_index=True)
    scores["asof_date"] = scores["asof_date"].astype(str)
    if scores["asof_date"].max() > SEAL_AFTER:
        raise SystemExit(f"Refusing: signal dates after {SEAL_AFTER} are the sealed acceptance window")
    if scores.duplicated(["symbol", "asof_date"]).any() or not np.isfinite(scores["predicted_return_d10"]).all():
        raise SystemExit("duplicate or non-finite predictions")
    return scores


class Market:
    """Date x symbol arrays of everything the engine needs from the bar panel."""

    def __init__(self, raw_path: Path):
        bars = infer_flags(ec.normalize_bars(pd.read_csv(raw_path, dtype={"symbol": str})))
        bars["median_amount"] = bars.groupby("symbol")["amount"].transform(
            lambda s: s.rolling(ec.LIQUIDITY_LOOKBACK, min_periods=ec.LIQUIDITY_LOOKBACK).median())
        self.dates = sorted(bars["date"].unique())
        self.symbols = sorted(bars["symbol"].unique())
        self.date_index = {d: i for i, d in enumerate(self.dates)}
        self.symbol_index = {s: i for i, s in enumerate(self.symbols)}

        def wide(column, fill=np.nan, dtype=np.float64):
            out = np.full((len(self.dates), len(self.symbols)), fill, dtype=dtype)
            rows = bars["date"].map(self.date_index).to_numpy()
            cols = bars["symbol"].map(self.symbol_index).to_numpy()
            out[rows, cols] = bars[column].to_numpy()
            return out

        close, open_ = wide("close"), wide("open")
        self.traded = np.isfinite(close)
        self.one_up = wide("one_word_limit_up", False, bool)
        self.one_down = wide("one_word_limit_down", False, bool)
        # Valuation prices: an untraded day is carried at the last close.
        self.close = pd.DataFrame(close).ffill().to_numpy()
        previous_close = np.vstack([np.full((1, close.shape[1]), np.nan), self.close[:-1]])
        self.open = np.where(self.traded, open_, previous_close)
        median = wide("median_amount")
        self.liquid = self.traded & (median >= ec.MIN_TRAILING_MEDIAN_AMOUNT)
        slip = np.full(median.shape, np.nan)
        for floor, rate in reversed(ec.SLIPPAGE_TIERS_CNY):   # ascending floors, so higher tiers overwrite
            slip[median >= floor] = rate
        self.side_cost = pd.DataFrame(slip).ffill().fillna(FALLBACK_SLIPPAGE).to_numpy() + FEE_PER_SIDE

    def score_matrix(self, scores: pd.DataFrame) -> np.ndarray:
        known = scores["symbol"].isin(self.symbol_index) & scores["asof_date"].isin(self.date_index)
        scores = scores[known]
        out = np.full(self.traded.shape, np.nan)
        out[scores["asof_date"].map(self.date_index).to_numpy(),
            scores["symbol"].map(self.symbol_index).to_numpy()] = scores["predicted_return_d10"].to_numpy()
        return out


def pool_percentiles(score_row: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Percentile of every scored name against the entry pool's score distribution."""
    reference = np.sort(score_row[pool])
    out = np.full(score_row.shape, np.nan)
    scored = np.isfinite(score_row)
    out[scored] = np.searchsorted(reference, score_row[scored], side="right") / max(1, len(reference))
    return out


def run_book(market: Market, scores: np.ndarray, signal_days: list[int], rule, charge_costs: bool = True) -> dict:
    kind, enter_above, exit_at, first_enter_above, min_hold = rule
    n_symbols = len(market.symbols)
    value = np.zeros(n_symbols)
    entered = np.full(n_symbols, -1)
    cash, previous_equity = 1.0, 1.0
    signal_set = set(signal_days)
    first_exec, last_exec = signal_days[0] + 1, signal_days[-1] + 1
    rows = []
    for day in range(first_exec, last_exec + 1):
        held = value > 0
        if held.any():   # overnight move: previous close -> today's open
            value[held] *= market.open[day, held] / market.close[day - 1, held]
        bought = sold = costs = 0.0
        if day - 1 in signal_set:
            signal = day - 1
            pool = market.liquid[signal] & np.isfinite(scores[signal])
            pct = pool_percentiles(scores[signal], pool)
            threshold = first_enter_above if day == first_exec else enter_above
            wants_in = pool & (pct > threshold) if kind == "rank" else pool
            wants_out = np.zeros(n_symbols, dtype=bool)
            if exit_at is not None:   # NaN percentile (no prediction today) compares False: keep
                wants_out = held & (pct <= exit_at) & (day - entered >= min_hold)
            sells = wants_out & market.traded[day] & ~market.one_down[day]
            if sells.any():
                sold = float(value[sells].sum())
                sell_cost = float((value[sells] * market.side_cost[day, sells]).sum()) if charge_costs else 0.0
                costs += sell_cost
                cash += sold - sell_cost
                value[sells] = 0.0
                entered[sells] = -1
            held = value > 0
            buys = wants_in & ~held & market.traded[day] & ~market.one_up[day]
            n_buys = int(buys.sum())
            equity = cash + float(value.sum())
            if n_buys and cash > 1e-9 * equity:   # a fully invested book has nothing to buy with
                target_count = int(held.sum()) + n_buys
                each = min(equity / target_count, cash / n_buys)
                rate = market.side_cost[day, buys] if charge_costs else 0.0
                value[buys] = each * (1.0 - rate)
                entered[buys] = day
                bought = each * n_buys
                costs += float(np.sum(each * rate))
                cash -= bought
        held = value > 0
        if held.any():   # intraday move: today's open -> today's close
            value[held] *= market.close[day, held] / market.open[day, held]
        equity = cash + float(value.sum())
        rows.append({"date": market.dates[day], "ret": equity / previous_equity - 1.0, "equity": equity,
                     "names": int(held.sum()), "exposure": 1.0 - cash / equity,
                     "one_way_turnover": (bought + sold) / 2.0 / equity, "cost": costs / previous_equity})
        previous_equity = equity
    return {"daily": pd.DataFrame(rows), "final_value": value, "final_cash": cash}


def block_ci(values: np.ndarray) -> list[float]:
    n, block = len(values), min(BOOTSTRAP["block"], len(values))
    rng = np.random.default_rng(BOOTSTRAP["seed"])
    draws = np.empty(BOOTSTRAP["replicates"])
    for r in range(BOOTSTRAP["replicates"]):
        idx: list[int] = []
        while len(idx) < n:
            start = int(rng.integers(0, n - block + 1))
            idx.extend(range(start, start + block))
        draws[r] = values[idx[:n]].mean()
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize(net: pd.DataFrame, gross: pd.DataFrame) -> dict:
    nav = (1.0 + net["ret"]).cumprod()
    std = float(net["ret"].std(ddof=1))
    return {
        "total_net_return": float(nav.iloc[-1] - 1.0),
        "total_gross_return": float((1.0 + gross["ret"]).prod() - 1.0),
        "cost_drag": float((1.0 + gross["ret"]).prod() - nav.iloc[-1]),
        "sharpe_annualized": float(net["ret"].mean() / std * np.sqrt(ec.TRADING_DAYS_PER_YEAR)) if std else None,
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()),
        "mean_names": float(net["names"].mean()),
        "mean_exposure": float(net["exposure"].mean()),
        # Day 1 builds the book from cash; turnover after that is what the rule itself costs.
        "mean_daily_one_way_turnover_after_day1": float(net["one_way_turnover"].iloc[1:].mean()),
        "nav_days": int(len(net)),
    }


def pair(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    diff = (left["ret"] - right["ret"]).to_numpy()
    ci = block_ci(diff)
    verdict = "better" if ci[0] > 0 else "worse" if ci[1] < 0 else "inconclusive"
    return {"mean_daily_net_diff": float(diff.mean()), "ci95": ci, "verdict": verdict,
            "total_return_diff": float((1 + left["ret"]).prod() - (1 + right["ret"]).prod())}


def run_dataset(market: Market, spec: str) -> dict:
    scores = load_scores(spec)
    matrix = market.score_matrix(scores)
    signal_days = sorted(market.date_index[d] for d in scores["asof_date"].unique() if d in market.date_index)
    if signal_days[-1] + 1 >= len(market.dates):
        raise SystemExit("bar panel ends before the last execution day")
    daily = {name: run_book(market, matrix, signal_days, rule)["daily"] for name, rule in BOOKS.items()}
    gross = {name: run_book(market, matrix, signal_days, rule, charge_costs=False)["daily"]
             for name, rule in BOOKS.items()}
    pool_sizes = [int((market.liquid[d] & np.isfinite(matrix[d])).sum()) for d in signal_days]
    return {
        "source": spec,
        "signal_days": len(signal_days),
        "signal_start": market.dates[signal_days[0]], "signal_end": market.dates[signal_days[-1]],
        "nav_start": market.dates[signal_days[0] + 1], "nav_end": market.dates[signal_days[-1] + 1],
        "consecutive_signal_days": bool(np.all(np.diff(signal_days) == 1)),
        "predictions": int(len(scores)), "mean_pool_size": float(np.mean(pool_sizes)),
        "books": {name: summarize(daily[name], gross[name]) for name in BOOKS},
        "primary": {key: pair(daily[a], daily[b]) | {"left": a, "right": b}
                    for key, (a, b) in PRIMARY_PAIRS.items()},
        "sensitivity": {f"{a}-{b}": pair(daily[a], daily[b]) for a, b in EXTRA_PAIRS},
        "daily_net_returns": {name: {"dates": frame["date"].tolist(), "ret": frame["ret"].round(8).tolist()}
                              for name, frame in daily.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--dataset", action="append", required=True, help="name=shards:<dir> | name=production:<dir>")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    market = Market(Path(args.raw))
    report = {
        "rules": {"books": {k: dict(zip(("kind", "enter_above", "exit_at_or_below", "first_day_enter_above",
                                         "min_hold_sessions"), v)) for k, v in BOOKS.items()},
                  "fee_per_side": FEE_PER_SIDE, "slippage_tiers": [list(t) for t in ec.SLIPPAGE_TIERS_CNY],
                  "min_trailing_median_amount": ec.MIN_TRAILING_MEDIAN_AMOUNT,
                  "decision": "close_of_signal_day", "execution": "next_session_open",
                  "limit_flags": "inferred_from_front_adjusted_bars", "bootstrap": BOOTSTRAP,
                  "sealed_after": SEAL_AFTER, "primary_dataset": args.dataset[0].split("=", 1)[0]},
        "datasets": {},
    }
    for item in args.dataset:
        name, spec = item.split("=", 1)
        report["datasets"][name] = run_dataset(market, spec)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    for name, data in report["datasets"].items():
        print(f"\n== {name}: {data['signal_days']} signal days {data['signal_start']}..{data['signal_end']}, "
              f"pool {data['mean_pool_size']:.0f}/day, NAV {data['nav_start']}..{data['nav_end']}")
        print(f"{'book':8s} {'net':>8s} {'gross':>8s} {'cost':>7s} {'maxDD':>8s} {'names':>7s} {'expo':>6s} {'turn/d':>7s}")
        for book, s in data["books"].items():
            print(f"{book:8s} {s['total_net_return']:+8.2%} {s['total_gross_return']:+8.2%} {s['cost_drag']:7.2%} "
                  f"{s['max_drawdown']:8.2%} {s['mean_names']:7.0f} {s['mean_exposure']:6.1%} "
                  f"{s['mean_daily_one_way_turnover_after_day1']:7.2%}")
        for key, p in data["primary"].items():
            print(f"  PRIMARY {key:22s} {p['left']}-{p['right']}: {p['mean_daily_net_diff']*1e4:+.2f} bp/day "
                  f"[{p['ci95'][0]*1e4:+.2f}, {p['ci95'][1]*1e4:+.2f}] total {p['total_return_diff']:+.2%} -> {p['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
