"""Few-name books rebalanced every 5 or 10 sessions, on datasets with a prediction every session.

Companion to ``few_names_backtest.py`` (same engine, pool, costs and rules); the
sparse 16-path run there could only test 3/6/9-session spacing.  Books keep
their last positions until the final bar of the price file.

Frozen in ``finetune/reports/20261112_few_names_daily_backtest_prereg_cn.md``.

    python3 finetune/oos_eval/base_clean_sealed_oos/few_names_daily_backtest.py \\
        --raw scratch/oos65/combined_raw.csv \\
        --dataset PROD=production:scratch/prod_c2 \\
        --dataset EVAL=shards:scratch/c2_roll/eval/c2 \\
        --out finetune/reports/20261112_few_names_daily_backtest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import few_names_backtest as fnb
import signal_usage_backtest as sub

STEPS = (5, 10)
PRIMARY = {"k": 10, "step": 10, "buffer": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--dataset", action="append", required=True, help="name=shards:<dir> | name=production:<dir>")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    market = sub.Market(Path(args.raw))
    last_day = len(market.dates) - 1
    report = {"constraint": "hand-traded book: few names, few orders",
              "rules": {"ks": fnb.KS, "steps_in_sessions": STEPS, "primary": PRIMARY,
                        "buffer_keep_fraction": fnb.BUFFER_KEEP_FRACTION, "random_books": fnb.RANDOM_BOOKS,
                        "random_seed": fnb.RANDOM_SEED, "bootstrap": sub.BOOTSTRAP, "sealed_after": sub.SEAL_AFTER,
                        "nav_end": market.dates[last_day], "primary_dataset": args.dataset[0].split("=", 1)[0]},
              "datasets": {}}
    for item in args.dataset:
        name, spec = item.split("=", 1)
        scores = sub.load_scores(spec)
        signal_days = sorted(market.date_index[d] for d in scores["asof_date"].unique())
        if not np.all(np.diff(signal_days) == 1):
            raise SystemExit(f"{name}: signal dates are not consecutive sessions")
        books = fnb.evaluate(market, market.score_matrix(scores), signal_days, with_random=True,
                             steps=STEPS, last_day=last_day)
        report["datasets"][name] = {"source": spec, "signal_days": len(signal_days),
                                    "signal_start": market.dates[signal_days[0]],
                                    "signal_end": market.dates[signal_days[-1]], "books": books}
        print(f"\n== {name}: {len(signal_days)} signal days {market.dates[signal_days[0]]}..{market.dates[signal_days[-1]]}, "
              f"NAV to {market.dates[last_day]}  (net excess vs buy-and-hold of the pool; mean over start days)")
        print(f"{'book':20s} {'net':>8s} {'excess':>8s} {'range over start days':>22s} {'beat pool':>10s} {'gross ex':>9s} "
              f"{'maxDD':>7s} {'orders':>7s}  verdict")
        for e in books.values():
            lo, hi = e["net_excess_range"]
            wins = sum(x["net_excess"] > 0 for x in e["phases"])
            orders = "-" if e["mean_orders_per_rebalance"] is None else f"{e['mean_orders_per_rebalance']:.1f}"
            flag = " <- PRIMARY" if (e["k"], e["step_signal_dates"], e["buffer"]) == tuple(PRIMARY.values()) else ""
            print(f"K{e['k']:<3d}every {e['step_signal_dates']:>2d}d {'buf' if e['buffer'] else '   '}   "
                  f"{e['mean_net']:+8.2%} {e['mean_net_excess']:+8.2%} {f'[{lo:+.1%}, {hi:+.1%}]':>22s} "
                  f"{f'{wins}/{len(e["phases"])}':>10s} {e['mean_gross_excess']:+9.2%} {e['mean_max_drawdown']:7.1%} "
                  f"{orders:>7s}  {e['daily_net_diff_vs_universe']['verdict']}{flag}")
            if "random_books_gross_excess" in e:
                r = e["random_books_gross_excess"]
                print(f"{'':20s} random books gross excess 95% band [{r['q025']:+.1%}, {r['q975']:+.1%}]; "
                      f"C2 at {r['c2_percentile']:.1%}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
