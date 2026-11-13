"""Per-trade study of the hand-traded practice: buy one top-ranked name every day, sell at a profit target.

Each (signal day, name) is one trade with the same stake: buy at the next open,
sell at a net profit target from the following session on (T+1), optionally with
a 10-session time stop; a trade still open on the last bar is marked there.
Because a human picks the name, each "top N" selection reports the average over
its names (the expectation of a random pick), next to the whole pool under the
same exit rule.

Frozen in ``finetune/reports/20261112_daily_pick_take_profit_prereg_cn.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import few_names_backtest as fnb
import few_names_lowvol_backtest as lv
import signal_usage_backtest as sub

TP_HIT, TIME_STOP, OPEN = 0, 1, 2
EXITS = {"hold10": (None, 10)}
for _tp in (0.02, 0.05, 0.10):
    EXITS[f"tp{int(_tp * 100)}"] = (_tp, None)
    EXITS[f"tp{int(_tp * 100)}_stop10"] = (_tp, 10)
PRIMARY = {"selection": "top10", "exit": "tp5"}
PERSIST_TOP = 10
ONE_A_DAY = ("top1", "p3_rank12", "p3_best1")
MIN_DAYS_FOR_CI = 2 * sub.BOOTSTRAP["block"]


def simulate(market: sub.Market, signal: int, symbols: np.ndarray, take_profit, time_stop) -> pd.DataFrame:
    """One trade per symbol bought at the open after ``signal``; net return after both sides' costs."""
    entry, last = signal + 1, len(market.dates) - 1
    symbols = symbols[market.traded[entry, symbols] & ~market.one_up[entry, symbols]]
    basis = market.open[entry, symbols] * (1.0 + market.side_cost[entry, symbols])
    net = np.full(len(symbols), np.nan)
    status = np.full(len(symbols), OPEN)
    hold = np.full(len(symbols), last - entry)
    alive = np.ones(len(symbols), dtype=bool)
    for day in range(entry + 1, last + 1):               # T+1: no sale on the entry day
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        sym = symbols[idx]
        keep_rate = 1.0 - market.side_cost[day, sym]
        sellable = market.traded[day, sym] & ~market.one_down[day, sym]
        done = np.zeros(len(idx), dtype=bool)
        if take_profit is not None:
            target = basis[idx] * (1.0 + take_profit) / keep_rate
            hit = sellable & (market.high[day, sym] >= target)
            price = np.maximum(market.open[day, sym], target)          # a gap up sells at the open
            net[idx[hit]] = price[hit] * keep_rate[hit] / basis[idx[hit]] - 1.0
            status[idx[hit]] = TP_HIT
            done |= hit
        if time_stop is not None and day - entry >= time_stop:
            due = sellable & ~done
            net[idx[due]] = market.close[day, sym[due]] * keep_rate[due] / basis[idx[due]] - 1.0
            status[idx[due]] = TIME_STOP
            done |= due
        hold[idx[done]] = day - entry
        alive[idx[done]] = False
    idx = np.flatnonzero(alive)                           # still open: mark at the last close
    net[idx] = market.close[last, symbols[idx]] * (1.0 - market.side_cost[last, symbols[idx]]) / basis[idx] - 1.0
    return pd.DataFrame({"symbol": symbols, "net": net, "status": status, "hold": hold, "entry": entry})


def selections(market, scores, vol, signal_days: list[int]) -> dict[int, dict[str, np.ndarray]]:
    """Names of every selection on every signal day (indices into the market's symbol axis)."""
    order = {d: fnb.ranked_pool(market, scores, d)[0] for d in signal_days}
    out = {}
    for d in signal_days:
        day = {"pool": order[d], "top1": order[d][:1], "top2": order[d][:2], "top3": order[d][:3],
               "top10": order[d][:10],
               "top20": order[d][:20],
               "lowvol_top10": lv.make_plan(market, scores, vol, "LV50", 10, False)(d)[0][:10]}
        for run in (2, 3):                                # in the top 10 on `run` consecutive sessions
            previous = [d - i for i in range(1, run)]
            if all(p in order for p in previous):
                keep = set(order[d][:PERSIST_TOP]).intersection(*(set(order[p][:PERSIST_TOP]) for p in previous))
                day[f"persist{run}"] = np.array([s for s in order[d][:PERSIST_TOP] if s in keep], dtype=int)
        if "persist3" in day:                             # the user's own rule and two readings of it
            day["p3_rank12"] = np.array([s for s in order[d][:2] if s in set(day["persist3"])], dtype=int)
            day["p3_best2"] = day["persist3"][:2]
            day["p3_best1"] = day["persist3"][:1]
        out[d] = day
    return out


def summarize(trades: pd.DataFrame) -> dict:
    open_ = trades["status"] == OPEN
    return {
        "trades": int(len(trades)), "win_rate": float((trades["net"] > 0).mean()),
        "mean_net": float(trades["net"].mean()), "median_net": float(trades["net"].median()),
        "take_profit_share": float((trades["status"] == TP_HIT).mean()),
        "open_share": float(open_.mean()),
        "open_mean_net": float(trades.loc[open_, "net"].mean()) if open_.any() else None,
        "mean_hold_sessions": float(trades["hold"].mean()), "worst": float(trades["net"].min()),
        "p10": float(trades["net"].quantile(0.10)), "p90": float(trades["net"].quantile(0.90)),
    }


def evaluate(market, scores, vol, signal_days: list[int]) -> dict:
    picks = selections(market, scores, vol, signal_days)
    names = sorted({name for day in picks.values() for name in day})
    out = {}
    for exit_name, (take_profit, time_stop) in EXITS.items():
        pooled = {name: [] for name in names}
        daily = {name: {} for name in names}
        for d in signal_days:
            pool = simulate(market, d, picks[d]["pool"], take_profit, time_stop).set_index("symbol")
            for name in names:
                chosen = picks[d].get(name)
                if chosen is None or not len(chosen):
                    continue
                trades = pool.loc[pool.index.intersection(chosen)]
                if len(trades):
                    pooled[name].append(trades)
                    daily[name][d] = float(trades["net"].mean())
        result = {}
        for name in names:
            if not pooled[name]:
                continue
            trades = pd.concat(pooled[name])
            entry = summarize(trades) | {"signal_days_with_a_pick": len(daily[name])}
            if name != "pool":
                days = sorted(daily[name])
                diff = np.array([daily[name][d] - daily["pool"][d] for d in days])
                # A block bootstrap needs several blocks; with fewer days the interval collapses
                # to a point and would fake a verdict.
                if len(days) < MIN_DAYS_FOR_CI:
                    entry["vs_pool_same_exit"] = {"mean_diff": float(diff.mean()), "ci95": None,
                                                  "verdict": "too_few_days"}
                else:
                    ci = sub.block_ci(diff)
                    entry["vs_pool_same_exit"] = {
                        "mean_diff": float(diff.mean()), "ci95": ci,
                        "verdict": "better" if ci[0] > 0 else "worse" if ci[1] < 0 else "inconclusive"}
            if name in ONE_A_DAY:                         # the literal "buy every day" path
                span = np.zeros(len(market.dates) + 1)
                for row in trades.itertuples():
                    span[row.entry] += 1
                    if row.status != OPEN:
                        span[row.entry + row.hold + 1] -= 1
                entry["one_a_day"] = {"sum_net_in_stakes": float(trades["net"].sum()),
                                      "max_open_positions": int(np.cumsum(span).max()),
                                      "open_at_end": int((trades["status"] == OPEN).sum())}
            result[name] = entry
        out[exit_name] = result
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--dataset", action="append", required=True, help="name=shards:<dir> | name=production:<dir>")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    market = sub.Market(Path(args.raw))
    vol = lv.trailing_volatility(market)
    report = {"practice": "buy one top-ranked name every day, sell at a profit target",
              "rules": {"exits": {k: {"take_profit": v[0], "time_stop_sessions": v[1]} for k, v in EXITS.items()},
                        "primary": PRIMARY, "persist_top": PERSIST_TOP, "bootstrap": sub.BOOTSTRAP,
                        "sealed_after": sub.SEAL_AFTER, "last_bar": market.dates[-1]},
              "datasets": {}}
    for item in args.dataset:
        name, spec = item.split("=", 1)
        frame = sub.load_scores(spec)
        signal_days = sorted(market.date_index[d] for d in frame["asof_date"].unique())
        if not np.all(np.diff(signal_days) == 1):
            raise SystemExit(f"{name}: signal dates are not consecutive sessions")
        result = evaluate(market, market.score_matrix(frame), vol, signal_days)
        report["datasets"][name] = {"source": spec, "signal_days": len(signal_days),
                                    "signal_start": market.dates[signal_days[0]],
                                    "signal_end": market.dates[signal_days[-1]], "exits": result}
        print(f"\n== {name}: {len(signal_days)} signal days {market.dates[signal_days[0]]}..{market.dates[signal_days[-1]]}")
        for exit_name, rows in result.items():
            print(f"-- exit {exit_name}")
            print(f"   {'selection':13s} {'trades':>7s} {'win':>6s} {'mean':>7s} {'median':>7s} {'tp hit':>7s} {'open':>6s} "
                  f"{'open avg':>9s} {'hold':>5s} {'worst':>7s}  vs pool, same exit")
            for sel, e in rows.items():
                versus = e.get("vs_pool_same_exit")
                if versus is None:
                    text = ""
                elif versus["ci95"] is None:
                    text = f"{versus['mean_diff']:+.2%} (too few days for an interval)"
                else:
                    text = (f"{versus['mean_diff']:+.2%} [{versus['ci95'][0]:+.2%}, {versus['ci95'][1]:+.2%}] "
                            f"{versus['verdict']}")
                open_avg = "-" if e["open_mean_net"] is None else f"{e['open_mean_net']:+.1%}"
                flag = " <- PRIMARY" if (sel, exit_name) == tuple(PRIMARY.values()) else ""
                print(f"   {sel:13s} {e['trades']:7d} {e['win_rate']:6.0%} {e['mean_net']:+7.2%} {e['median_net']:+7.2%} "
                      f"{e['take_profit_share']:7.0%} {e['open_share']:6.0%} {open_avg:>9s} {e['mean_hold_sessions']:5.1f} "
                      f"{e['worst']:+7.1%}  {text}{flag}")
            for sel in ONE_A_DAY:
                if sel in rows:
                    one = rows[sel]["one_a_day"]
                    print(f"   buying {sel} every day: total {one['sum_net_in_stakes']:+.2f} stakes, at most "
                          f"{one['max_open_positions']} positions open at once, {one['open_at_end']} still open at the end")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
