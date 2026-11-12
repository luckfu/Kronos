"""Few-name, low-frequency long-only backtest of the C2 rank signal (CPU only).

The live book is traded by hand, so it must hold few names and trade rarely.
This script tests K-slot books (K = 5/10/20) rebalanced every 1/2/3 signal
dates, against buy-and-hold of the whole pool and against random K-name books.

Rules, datasets, the primary book and the pass criteria are frozen in
``finetune/reports/20261112_few_names_backtest_prereg_cn.md``.

Engine: decide at the close of a rebalance signal day, trade at the next open,
no trading in between.  A held name is sold when it leaves the top K (no
buffer) or the top 30% of the pool (buffer); free slots are filled best-first
with names that can be bought that morning; each new position gets equity / K.
Market data, eligibility, costs and the sealed-window guard come from
``signal_usage_backtest``.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import signal_usage_backtest as sub

KS = (5, 10, 20)
STEPS = (1, 2, 3)                      # in signal dates of the dataset
BUFFER_KEEP_FRACTION = 0.30
PRIMARY = {"k": 10, "step": 3, "buffer": True}
RANDOM_BOOKS = 500
RANDOM_SEED = 20261112
LABEL_BUCKETS = (5, 10, 20, 50, 100)


def ranked_pool(market: sub.Market, scores: np.ndarray, signal: int) -> tuple[np.ndarray, np.ndarray]:
    """Pool members best-first, and every scored name's percentile inside the pool."""
    pool = market.liquid[signal] & np.isfinite(scores[signal])
    members = np.flatnonzero(pool)
    order = members[np.argsort(-scores[signal][members], kind="stable")]
    return order, sub.pool_percentiles(scores[signal], pool)


def c2_plan(market: sub.Market, scores: np.ndarray, k: int, buffer: bool):
    def plan(signal: int):
        order, pct = ranked_pool(market, scores, signal)
        keep = ~np.isfinite(scores[signal])              # no prediction today: no reason to sell
        if buffer:
            keep |= pct > 1.0 - BUFFER_KEEP_FRACTION
        else:
            keep[order[:k]] = True
        return order, keep
    return plan


def random_plan(market: sub.Market, scores: np.ndarray, rng: np.random.Generator):
    def plan(signal: int):
        pool = np.flatnonzero(market.liquid[signal] & np.isfinite(scores[signal]))
        return rng.permutation(pool), np.zeros(scores.shape[1], dtype=bool)   # replace everything
    return plan


def run_slots(market: sub.Market, plan, rebalance_days: list[int], k: int, last_day: int,
              charge_costs: bool = True) -> dict:
    n_symbols = len(market.symbols)
    value = np.zeros(n_symbols)
    cash, previous_equity = 1.0, 1.0
    rebalance = set(rebalance_days)
    rows, orders, blocked_sells = [], [], 0
    for day in range(rebalance_days[0] + 1, last_day + 1):
        held = value > 0
        if held.any():
            value[held] *= market.open[day, held] / market.close[day - 1, held]
        traded_value = cost = 0.0
        if day - 1 in rebalance:
            order, keep = plan(day - 1)
            leaving = held & ~keep
            sells = leaving & market.traded[day] & ~market.one_down[day]
            blocked_sells += int((leaving & ~sells).sum())
            if sells.any():
                proceeds = float(value[sells].sum())
                fee = float((value[sells] * market.side_cost[day, sells]).sum()) if charge_costs else 0.0
                cash += proceeds - fee
                cost += fee
                traded_value += proceeds
                value[sells] = 0.0
            held = value > 0
            free = k - int(held.sum())
            buyable = order[~held[order] & market.traded[day, order] & ~market.one_up[day, order]]
            buys = buyable[:max(free, 0)]
            equity = cash + float(value.sum())
            if len(buys) and cash > 1e-9 * equity:
                each = min(equity / k, cash / len(buys))
                rate = market.side_cost[day, buys] if charge_costs else np.zeros(len(buys))
                value[buys] = each * (1.0 - rate)
                cash -= each * len(buys)
                cost += float((each * rate).sum())
                traded_value += each * len(buys)
            orders.append(int(sells.sum()) + len(buys))
        held = value > 0
        if held.any():
            value[held] *= market.close[day, held] / market.open[day, held]
        equity = cash + float(value.sum())
        rows.append((market.dates[day], equity / previous_equity - 1.0, int(held.sum()),
                     traded_value / 2.0 / equity, cost))
        previous_equity = equity
    daily = pd.DataFrame(rows, columns=["date", "ret", "names", "one_way_turnover", "cost"]).set_index("date")
    return {"daily": daily, "orders": orders, "blocked_sells": blocked_sells, "final_value": value}


def universe(market: sub.Market, scores: np.ndarray, first_signal: int, last_day: int,
             charge_costs: bool = True) -> pd.Series:
    """Benchmark: buy every pool name once at the first trade day and never trade again."""
    order = ranked_pool(market, scores, first_signal)[0]
    hold_all = np.ones(scores.shape[1], dtype=bool)
    book = run_slots(market, lambda signal: (order, hold_all), [first_signal], len(order), last_day, charge_costs)
    return book["daily"]["ret"]


def total(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def evaluate(market: sub.Market, scores: np.ndarray, signal_days: list[int], with_random: bool,
             steps: tuple[int, ...] = STEPS, last_day: int | None = None) -> dict:
    """Every K x step x buffer book.  ``last_day`` defaults to the session after the last signal day."""
    last_day = signal_days[-1] + 1 if last_day is None else last_day
    rng = np.random.default_rng(RANDOM_SEED)
    out = {}
    for step in steps:
        phases = [signal_days[phase::step] for phase in range(step)]
        bench = [universe(market, scores, days[0], last_day) for days in phases]
        bench_gross = [total(universe(market, scores, days[0], last_day, charge_costs=False)) for days in phases]
        for k in KS:
            random_gross = None
            if with_random:
                random_gross = np.array([[total(run_slots(market, random_plan(market, scores, rng), days, k, last_day,
                                                          charge_costs=False)["daily"]["ret"]) - bench_gross[p]
                                          for p, days in enumerate(phases)] for _ in range(RANDOM_BOOKS)])
            for buffer in (False, True):
                per_phase, diffs = [], []
                for p, days in enumerate(phases):
                    net = run_slots(market, c2_plan(market, scores, k, buffer), days, k, last_day)
                    gross = run_slots(market, c2_plan(market, scores, k, buffer), days, k, last_day,
                                      charge_costs=False)
                    nav = (1.0 + net["daily"]["ret"]).cumprod()
                    diffs.append(net["daily"]["ret"] - bench[p])
                    per_phase.append({
                        "first_trade": net["daily"].index[0], "rebalances": len(days),
                        "net": total(net["daily"]["ret"]), "gross": total(gross["daily"]["ret"]),
                        "universe_net": total(bench[p]),
                        "net_excess": total(net["daily"]["ret"]) - total(bench[p]),
                        "gross_excess": total(gross["daily"]["ret"]) - bench_gross[p],
                        "max_drawdown": float((nav / nav.cummax() - 1.0).min()),
                        "orders_per_rebalance_after_first": float(np.mean(net["orders"][1:])) if len(net["orders"]) > 1 else None,
                        "mean_names": float(net["daily"]["names"].mean()),
                        "blocked_sells": net["blocked_sells"],
                    })
                mean_diff = pd.concat(diffs, axis=1).mean(axis=1).to_numpy()   # phases present on each day
                later_orders = [x["orders_per_rebalance_after_first"] for x in per_phase
                                if x["orders_per_rebalance_after_first"] is not None]
                ci = sub.block_ci(mean_diff)
                entry = {
                    "k": k, "step_signal_dates": step, "buffer": buffer, "phases": per_phase,
                    "mean_net": float(np.mean([x["net"] for x in per_phase])),
                    "mean_net_excess": float(np.mean([x["net_excess"] for x in per_phase])),
                    "mean_gross_excess": float(np.mean([x["gross_excess"] for x in per_phase])),
                    "net_excess_range": [min(x["net_excess"] for x in per_phase), max(x["net_excess"] for x in per_phase)],
                    "mean_max_drawdown": float(np.mean([x["max_drawdown"] for x in per_phase])),
                    # A start day with a single rebalance has no "after the first" orders.
                    "mean_orders_per_rebalance": float(np.mean(later_orders)) if later_orders else None,
                    "daily_net_diff_vs_universe": {"mean": float(mean_diff.mean()), "ci95": ci,
                                                   "verdict": "better" if ci[0] > 0 else "worse" if ci[1] < 0 else "inconclusive"},
                }
                if random_gross is not None and not buffer:
                    reference = random_gross.mean(axis=1)
                    entry["random_books_gross_excess"] = {
                        "books": RANDOM_BOOKS, "mean": float(reference.mean()), "std": float(reference.std(ddof=1)),
                        "q025": float(np.quantile(reference, 0.025)), "q975": float(np.quantile(reference, 0.975)),
                        "single_phase_q025": float(np.quantile(random_gross[:, 0], 0.025)),
                        "single_phase_q975": float(np.quantile(random_gross[:, 0], 0.975)),
                        "c2_percentile": float((reference < entry["mean_gross_excess"]).mean()),
                    }
                out[f"k{k}_step{step}_{'buffer' if buffer else 'nobuffer'}"] = entry
    return out


def label_profile(market: sub.Market, shard_dir: str, block: int, dates: set[str] | None = None) -> dict:
    """D10 label excess (vs the pool mean, no costs) of the top-N and bottom-N names by score."""
    columns = ["symbol", "asof_date", "predicted_return_d10", "actual_return_d10"]
    frame = pd.concat([pd.read_csv(p, usecols=columns, dtype={"symbol": str})
                       for p in sorted(glob.glob(str(Path(shard_dir) / "*.csv.gz")))], ignore_index=True)
    frame["asof_date"] = frame["asof_date"].astype(str)
    if frame["asof_date"].max() > sub.SEAL_AFTER:
        raise SystemExit("sealed acceptance window")
    if dates is not None:
        frame = frame[frame["asof_date"].isin(dates)]
    liquid = market.liquid[frame["asof_date"].map(market.date_index).to_numpy(),
                           frame["symbol"].map(market.symbol_index).to_numpy()]
    frame = frame[liquid]
    per_day = []
    for _, day in frame.groupby("asof_date", sort=True):
        ordered = day.sort_values(["predicted_return_d10", "symbol"], ascending=[False, True])
        excess = (ordered["actual_return_d10"] - day["actual_return_d10"].mean()).to_numpy()
        decile = max(1, int(np.ceil(0.10 * len(excess))))
        row = {f"top{n}": excess[:n].mean() for n in LABEL_BUCKETS} | {"top_decile": excess[:decile].mean()}
        row |= {f"bottom{n}": excess[-n:].mean() for n in LABEL_BUCKETS} | {"bottom_decile": excess[-decile:].mean()}
        per_day.append(row)
    table = pd.DataFrame(per_day)
    original = dict(sub.BOOTSTRAP)
    sub.BOOTSTRAP["block"] = block
    try:
        result = {col: {"mean": float(table[col].mean()), "ci95": sub.block_ci(table[col].to_numpy()),
                        "positive_days": float((table[col] > 0).mean())} for col in table.columns}
    finally:
        sub.BOOTSTRAP.update(original)
    return {"source": shard_dir, "signal_days": int(len(table)), "pool_rows": int(len(frame)),
            "bootstrap_block": block, "buckets": result}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--p16", required=True, help="16-path shards on sparse signal dates (primary)")
    parser.add_argument("--single", required=True, help="single-path shards; restricted to the --p16 dates")
    parser.add_argument("--p16-dense", required=True, help="16-path shards on consecutive dates (label profile only)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    market = sub.Market(Path(args.raw))
    p16 = sub.load_scores(f"shards:{args.p16}")
    dates = set(p16["asof_date"])
    single = sub.load_scores(f"shards:{args.single}")
    single = single[single["asof_date"].isin(dates)]
    signal_days = sorted(market.date_index[d] for d in dates)
    report = {
        "constraint": "hand-traded book: few names, few orders",
        "rules": {"ks": KS, "steps_in_signal_dates": STEPS, "signal_date_spacing_sessions": int(np.diff(signal_days).min()),
                  "buffer_keep_fraction": BUFFER_KEEP_FRACTION, "primary": PRIMARY, "random_books": RANDOM_BOOKS,
                  "random_seed": RANDOM_SEED, "bootstrap": sub.BOOTSTRAP, "sealed_after": sub.SEAL_AFTER,
                  "fee_per_side": sub.FEE_PER_SIDE},
        "window": {"signal_days": len(signal_days), "signal_start": market.dates[signal_days[0]],
                   "signal_end": market.dates[signal_days[-1]], "nav_end": market.dates[signal_days[-1] + 1]},
        "P16": evaluate(market, market.score_matrix(p16), signal_days, with_random=True),
        "S1": evaluate(market, market.score_matrix(single), signal_days, with_random=False),
        "label_profile": {
            "p16_sparse": label_profile(market, args.p16, block=4),
            "p16_dense": label_profile(market, args.p16_dense, block=10),
            "single_65d": label_profile(market, args.single, block=10),
            "single_on_p16_dates": label_profile(market, args.single, block=4, dates=dates),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    spacing = report["rules"]["signal_date_spacing_sessions"]
    for name in ("P16", "S1"):
        print(f"\n== {name}  (net excess vs buy-and-hold of the pool; mean over phases)")
        print(f"{'book':22s} {'net':>8s} {'excess':>8s} {'range over phases':>20s} {'gross ex':>9s} {'maxDD':>8s} {'orders':>7s}  verdict")
        for key, e in report[name].items():
            flag = " <- PRIMARY" if all(e[x] == PRIMARY[y] for x, y in (("k", "k"), ("step_signal_dates", "step"), ("buffer", "buffer"))) else ""
            lo, hi = e["net_excess_range"]
            print(f"K{e['k']:<3d}every {e['step_signal_dates'] * spacing}d {'buf' if e['buffer'] else '   '}    "
                  f"{e['mean_net']:+8.2%} {e['mean_net_excess']:+8.2%} {f'[{lo:+.1%}, {hi:+.1%}]':>20s} "
                  f"{e['mean_gross_excess']:+9.2%} {e['mean_max_drawdown']:8.1%} {e['mean_orders_per_rebalance']:7.1f}  "
                  f"{e['daily_net_diff_vs_universe']['verdict']}{flag}")
            if "random_books_gross_excess" in e:
                r = e["random_books_gross_excess"]
                print(f"{'':22s} random {RANDOM_BOOKS} books gross excess: 95% band [{r['q025']:+.1%}, {r['q975']:+.1%}] "
                      f"(one start day [{r['single_phase_q025']:+.1%}, {r['single_phase_q975']:+.1%}]); C2 at {r['c2_percentile']:.1%}")
    for name, profile in report["label_profile"].items():
        print(f"\n== label profile {name}: {profile['signal_days']} days")
        for bucket, s in profile["buckets"].items():
            print(f"   {bucket:14s} {s['mean']:+7.2%} [{s['ci95'][0]:+.2%}, {s['ci95'][1]:+.2%}]  positive days {s['positive_days']:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
