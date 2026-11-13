"""Few-name books with a volatility gate on entries (CPU only).

Earlier rounds showed that C2's very top names are high-volatility and follow
the market: they beat the pool when it rises and lose to it when it falls.
This script tests whether keeping high-volatility names out of the entry pool
removes that dependence without throwing the stock selection away.

Selections (``VARIANTS``): no gate, low-volatility half, drop the top 30% by
volatility, and equal picks from five volatility buckets.  The gate applies to
entries only; exits still follow the C2 rank with the usual 30% buffer.

Rules, datasets, the primary book and the pass criteria are frozen in
``finetune/reports/20261112_few_names_lowvol_backtest_prereg_cn.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import few_names_backtest as fnb
import signal_usage_backtest as sub

VOL_WINDOW = 20
VARIANTS = ("NONE", "LV50", "LV70", "STRAT")
KEEP_QUANTILE = {"LV50": 0.50, "LV70": 0.70}
STRATA = 5
PRIMARY = {"variant": "LV50", "k": 10}
RANDOM_BOOKS = 200
RANDOM_SEED = 20261112
LABEL_TOP = 10
LABEL_HORIZON = 10


def trailing_volatility(market: sub.Market) -> np.ndarray:
    """Std of the last 20 close-to-close returns, known at each day's close."""
    returns = market.close[1:] / market.close[:-1] - 1.0
    rolling = pd.DataFrame(returns).rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std().to_numpy()
    return np.vstack([np.full((1, rolling.shape[1]), np.nan), rolling])


def make_plan(market: sub.Market, scores: np.ndarray, vol: np.ndarray, variant: str, k: int, buffer: bool):
    """Plan for ``fnb.run_slots``: (candidates best-first, names a holder keeps)."""
    if variant == "NONE":
        return fnb.c2_plan(market, scores, k, buffer)
    keep_above = 1.0 - fnb.BUFFER_KEEP_FRACTION

    def plan(signal: int):
        score = scores[signal]
        pool = market.liquid[signal] & np.isfinite(score) & np.isfinite(vol[signal])
        keep = ~np.isfinite(score)                       # no prediction today: no reason to sell
        if variant == "STRAT":
            members = np.flatnonzero(pool)
            bucket = (pd.Series(vol[signal][members]).rank(method="first").to_numpy() - 1) * STRATA // len(members)
            within = np.empty(len(members))
            for b in range(STRATA):
                inside = bucket == b
                within[inside] = pd.Series(score[members[inside]]).rank(method="first").to_numpy() / inside.sum()
            order = members[np.lexsort((-score[members], -within))]     # best of each bucket first
            if buffer:
                keep[members[within > keep_above]] = True
                # A held name outside today's pool has no bucket: judge it against the whole pool.
                keep |= ~pool & (sub.pool_percentiles(score, pool) > keep_above)
        else:
            entry = pool & (vol[signal] <= np.quantile(vol[signal][pool], KEEP_QUANTILE[variant]))
            members = np.flatnonzero(entry)
            order = members[np.argsort(-score[members], kind="stable")]
            if buffer:
                keep |= sub.pool_percentiles(score, entry) > keep_above
        if not buffer:
            keep[order[:k]] = True
        return order, keep
    return plan


def book_runs(market, scores, vol, phases, last_day, variant, k, buffer, charge_costs=True) -> list[dict]:
    """One ``fnb.run_slots`` result per start day."""
    return [fnb.run_slots(market, make_plan(market, scores, vol, variant, k, buffer), days, k, last_day,
                          charge_costs) for days in phases]


def verdict(ci: list[float]) -> str:
    return "better" if ci[0] > 0 else "worse" if ci[1] < 0 else "inconclusive"


def paired(left: list[pd.Series], right: list[pd.Series]) -> dict:
    diff = pd.concat([a - b for a, b in zip(left, right)], axis=1).mean(axis=1).to_numpy()
    ci = sub.block_ci(diff)
    return {"mean_daily_diff": float(diff.mean()), "ci95": ci, "verdict": verdict(ci),
            "mean_total_diff": float(np.mean([fnb.total(a) - fnb.total(b) for a, b in zip(left, right)]))}


def evaluate(market, scores, vol, signal_days, step, last_day, ks, with_random: bool) -> dict:
    phases = [signal_days[phase::step] for phase in range(step)]
    bench = [fnb.universe(market, scores, days[0], last_day) for days in phases]
    bench_gross = [fnb.total(fnb.universe(market, scores, days[0], last_day, charge_costs=False)) for days in phases]
    rng = np.random.default_rng(RANDOM_SEED)
    signal_rows = np.array(signal_days)
    out, daily = {}, {}
    for k in ks:
        for variant in VARIANTS:
            runs = book_runs(market, scores, vol, phases, last_day, variant, k, buffer=True)
            net = [run["daily"]["ret"] for run in runs]
            daily[(variant, k)] = net
            excess = [fnb.total(r) - fnb.total(b) for r, b in zip(net, bench)]
            drawdowns = [float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()) for r in net]
            later_orders = [float(np.mean(run["orders"][1:])) for run in runs if len(run["orders"]) > 1]
            no_buffer_gross = [fnb.total(run["daily"]["ret"]) for run in
                               book_runs(market, scores, vol, phases, last_day, variant, k, False, charge_costs=False)]
            entry = {
                "variant": variant, "k": k,
                "mean_net": float(np.mean([fnb.total(r) for r in net])),
                "mean_net_excess": float(np.mean(excess)), "net_excess_range": [min(excess), max(excess)],
                "start_days_beating_pool": int(sum(x > 0 for x in excess)), "start_days": len(phases),
                "mean_max_drawdown": float(np.mean(drawdowns)),
                "mean_orders_per_rebalance": float(np.mean(later_orders)) if later_orders else None,
                "vs_universe": paired(net, bench),
                "no_buffer_gross_excess": float(np.mean([g - b for g, b in zip(no_buffer_gross, bench_gross)])),
            }
            if with_random:
                sims = np.empty(RANDOM_BOOKS)
                for i in range(RANDOM_BOOKS):
                    noise = np.full(scores.shape, np.nan)
                    block = rng.normal(size=(len(signal_rows), scores.shape[1]))
                    noise[signal_rows] = np.where(np.isfinite(scores[signal_rows]), block, np.nan)
                    gross = [fnb.total(run["daily"]["ret"]) for run in
                             book_runs(market, noise, vol, phases, last_day, variant, k, False, charge_costs=False)]
                    sims[i] = np.mean([g - b for g, b in zip(gross, bench_gross)])
                entry["random_same_rule_gross_excess"] = {
                    "books": RANDOM_BOOKS, "mean": float(sims.mean()), "q025": float(np.quantile(sims, 0.025)),
                    "q975": float(np.quantile(sims, 0.975)),
                    "c2_percentile": float((sims < entry["no_buffer_gross_excess"]).mean())}
            out[f"{variant}_k{k}"] = entry
        for variant in VARIANTS[1:]:
            out[f"{variant}_k{k}"]["vs_NONE"] = paired(daily[(variant, k)], daily[("NONE", k)])
    return out


def label_regime(market, scores, vol, signal_days, block: int) -> dict:
    """Top-10 picks' 10-day label excess over the whole pool, split by what the pool did (no costs)."""
    rows = []
    for signal in signal_days:
        if signal + LABEL_HORIZON >= len(market.dates):
            continue
        pool = fnb.ranked_pool(market, scores, signal)[0]
        forward = market.close[signal + LABEL_HORIZON] / market.close[signal] - 1.0
        row = {"pool": float(forward[pool].mean())}
        for variant in VARIANTS:
            picks = make_plan(market, scores, vol, variant, LABEL_TOP, False)(signal)[0][:LABEL_TOP]
            row[variant] = float(forward[picks].mean() - row["pool"])
            row[f"{variant}_vol_pct"] = float(np.mean([(vol[signal][pool] <= vol[signal][p]).mean() for p in picks]))
        rows.append(row)
    table = pd.DataFrame(rows)
    up, down = table[table["pool"] > 0], table[table["pool"] <= 0]
    original = dict(sub.BOOTSTRAP)
    sub.BOOTSTRAP["block"] = block
    try:
        result = {"signal_days": int(len(table)), "up_days": int(len(up)), "down_days": int(len(down))}
        for variant in VARIANTS:
            entry = {
                "all": float(table[variant].mean()), "all_ci95": sub.block_ci(table[variant].to_numpy()),
                "up": float(up[variant].mean()) if len(up) else None,
                "down": float(down[variant].mean()) if len(down) else None,
                "corr_with_pool_return": float(np.corrcoef(table["pool"], table[variant])[0, 1]),
                "picks_volatility_percentile": float(table[f"{variant}_vol_pct"].mean()),
            }
            if variant != "NONE":
                diff = (table[variant] - table["NONE"]).to_numpy()
                entry["minus_NONE"] = {"mean": float(diff.mean()), "ci95": sub.block_ci(diff)}
            result[variant] = entry
    finally:
        sub.BOOTSTRAP.update(original)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--full", required=True, help="sparse 16-path shards (both regimes)")
    parser.add_argument("--down", required=True, help="production archive copy (falling market)")
    parser.add_argument("--eval", required=True, help="dense 16-path shards (check for --down)")
    parser.add_argument("--ks", default="5,10,20")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    market = sub.Market(Path(args.raw))
    vol = trailing_volatility(market)
    ks = tuple(int(x) for x in args.ks.split(","))
    datasets = {
        "FULL": (f"shards:{args.full}", 3, "next_session", True, 4),
        "DOWN": (f"production:{args.down}", 10, "last_bar", True, 10),
        "EVAL": (f"shards:{args.eval}", 10, "last_bar", False, 10),
    }
    report = {"constraint": "hand-traded book: few names, few orders",
              "rules": {"variants": VARIANTS, "volatility_window": VOL_WINDOW, "keep_quantile": KEEP_QUANTILE,
                        "strata": STRATA, "ks": ks, "buffer_keep_fraction": fnb.BUFFER_KEEP_FRACTION,
                        "primary": PRIMARY, "random_books": RANDOM_BOOKS, "random_seed": RANDOM_SEED,
                        "bootstrap": sub.BOOTSTRAP, "sealed_after": sub.SEAL_AFTER},
              "datasets": {}}
    for name, (spec, step, end, with_random, block) in datasets.items():
        scores_frame = sub.load_scores(spec)
        signal_days = sorted(market.date_index[d] for d in scores_frame["asof_date"].unique())
        last_day = signal_days[-1] + 1 if end == "next_session" else len(market.dates) - 1
        scores = market.score_matrix(scores_frame)
        spacing = int(np.diff(signal_days).min())
        books = evaluate(market, scores, vol, signal_days, step, last_day, ks, with_random)
        regime = label_regime(market, scores, vol, signal_days, block)
        report["datasets"][name] = {
            "source": spec, "signal_days": len(signal_days), "signal_start": market.dates[signal_days[0]],
            "signal_end": market.dates[signal_days[-1]], "nav_end": market.dates[last_day],
            "rebalance_every_sessions": step * spacing, "books": books, "label_regime_top10": regime}
        print(f"\n== {name}: {len(signal_days)} signal days {market.dates[signal_days[0]]}..{market.dates[signal_days[-1]]}, "
              f"rebalance every {step * spacing} sessions, NAV to {market.dates[last_day]}")
        print(f"{'book':10s} {'net':>8s} {'excess':>8s} {'range over start days':>22s} {'beat':>6s} {'maxDD':>7s} "
              f"{'orders':>7s} {'vs pool':>13s} {'vs NONE (total, verdict)':>26s}  random same rule (gross)")
        for key, e in books.items():
            lo, hi = e["net_excess_range"]
            versus = e.get("vs_NONE")
            versus_text = "" if versus is None else f"{versus['mean_total_diff']:+.2%} {versus['verdict']}"
            rnd = e.get("random_same_rule_gross_excess")
            rnd_text = "" if rnd is None else (f"C2 {e['no_buffer_gross_excess']:+.1%} vs [{rnd['q025']:+.1%}, "
                                               f"{rnd['q975']:+.1%}] -> {rnd['c2_percentile']:.1%}")
            flag = " <- PRIMARY" if (e["variant"], e["k"]) == tuple(PRIMARY.values()) else ""
            orders = "-" if e["mean_orders_per_rebalance"] is None else f"{e['mean_orders_per_rebalance']:.1f}"
            print(f"{key:10s} {e['mean_net']:+8.2%} {e['mean_net_excess']:+8.2%} {f'[{lo:+.1%}, {hi:+.1%}]':>22s} "
                  f"{e['start_days_beating_pool']}/{e['start_days']:<3d} {e['mean_max_drawdown']:7.1%} {orders:>7s} "
                  f"{e['vs_universe']['verdict']:>13s} {versus_text:>26s}  {rnd_text}{flag}")
        print(f"  label view, top-{LABEL_TOP} picks, 10-day excess over the pool "
              f"({regime['up_days']} up-market days, {regime['down_days']} down-market days):")
        for variant in VARIANTS:
            r = regime[variant]
            up = "n/a" if r["up"] is None else f"{r['up']:+.2%}"
            down = "n/a" if r["down"] is None else f"{r['down']:+.2%}"
            print(f"    {variant:6s} all {r['all']:+.2%} | market up {up} | market down {down} | "
                  f"corr with market {r['corr_with_pool_return']:+.2f} | picks' volatility percentile "
                  f"{r['picks_volatility_percentile']:.0%}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
