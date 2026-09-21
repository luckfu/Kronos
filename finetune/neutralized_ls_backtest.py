"""Industry+size neutralized long/short backtest on frozen C2 18-day predictions.

Score neutralization is the same cross-sectional OLS used in the confirmatory
kernel: intercept + size_decile + industry dummies. Holdings are equal-weight
top/bottom 10% of residual predicted 10-day return. P&L uses raw realized
path returns, not residualized actuals.

The tradable book is a 10-layer overlay: each signal date opens a 10-session
layer with equal capital among live layers. Adjacent sealed signal dates in
this window are consecutive A-share sessions, so layer age maps to the next
session. Cost is 30 bps per side on traded notional (双边千分之三).

Does not decode, train, or retune T / top_p / sample_count.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


HOLD = 10
TOP_FRACTION = 0.10
COST_PER_SIDE = 0.003
TRADING_DAYS_PER_YEAR = 252
SCORE_COL = "predicted_return_d10"
RESIDUAL_SCORE_COL = "residual_predicted_d10"


def residualize_column(frame, column):
    """Cross-section OLS residual on intercept + size_decile + industry dummies."""
    result = np.full(len(frame), np.nan, dtype=np.float64)
    for _, rows in frame.groupby("asof_date", sort=True):
        loc = rows.index.to_numpy()
        y = rows[column].to_numpy(dtype=np.float64)
        size = rows["size_decile"].to_numpy(dtype=np.float64)
        dummies = pd.get_dummies(rows["sector"], drop_first=True).to_numpy(dtype=np.float64)
        design = np.column_stack([np.ones(len(rows)), size, dummies])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        result[loc] = y - design @ coef
    return result


def incremental_return(row, horizon):
    current = float(row[f"actual_return_d{horizon}"])
    if horizon == 1:
        return current
    previous = float(row[f"actual_return_d{horizon - 1}"])
    return (1.0 + current) / (1.0 + previous) - 1.0


def equal_side_weights(long_symbols, short_symbols):
    weights = {}
    if long_symbols:
        weight = 1.0 / len(long_symbols)
        for symbol in long_symbols:
            weights[symbol] = weights.get(symbol, 0.0) + weight
    if short_symbols:
        weight = -1.0 / len(short_symbols)
        for symbol in short_symbols:
            weights[symbol] = weights.get(symbol, 0.0) + weight
    return weights


def pick_with_hysteresis(rows, prev_long, prev_short, enter=TOP_FRACTION, exit_fraction=None,
                         score_col=RESIDUAL_SCORE_COL):
    """Keep names until they leave the wider band; enter only at the inner band."""
    band = enter if exit_fraction is None else exit_fraction
    count_enter = max(1, int(math.ceil(len(rows) * enter)))
    count_exit = max(count_enter, int(math.ceil(len(rows) * band)))
    ranked = rows.sort_values(score_col, kind="mergesort")
    symbols = ranked["symbol"].tolist()
    long_enter = set(symbols[-count_enter:])
    short_enter = set(symbols[:count_enter])
    long_keep = set(symbols[-count_exit:])
    short_keep = set(symbols[:count_exit])
    new_long = set(long_enter) | (set(prev_long) & long_keep)
    new_short = set(short_enter) | (set(prev_short) & short_keep)
    overlap = new_long & new_short
    new_long -= overlap
    new_short -= overlap
    return new_long, new_short


def pick_long_short(rows, score_col=RESIDUAL_SCORE_COL, top_fraction=TOP_FRACTION):
    count = max(1, int(math.ceil(len(rows) * top_fraction)))
    ranked = rows.sort_values(score_col, kind="mergesort")
    short = ranked.head(count)
    long = ranked.tail(count)
    long_w = 1.0 / len(long)
    short_w = -1.0 / len(short)
    weights = {}
    for symbol in long["symbol"]:
        weights[symbol] = weights.get(symbol, 0.0) + long_w
    for symbol in short["symbol"]:
        weights[symbol] = weights.get(symbol, 0.0) + short_w
    return weights, long, short


def layer_daily_return(long_rows, short_rows, horizon):
    col = f"actual_return_d{horizon}"
    prev_col = f"actual_return_d{horizon - 1}" if horizon > 1 else None
    if prev_col is None:
        long_ret = float(long_rows[col].mean())
        short_ret = float(short_rows[col].mean())
    else:
        long_ret = float(
            ((1.0 + long_rows[col]) / (1.0 + long_rows[prev_col]) - 1.0).mean()
        )
        short_ret = float(
            ((1.0 + short_rows[col]) / (1.0 + short_rows[prev_col]) - 1.0).mean()
        )
    return long_ret - short_ret


def overlay_backtest(frame, cost_per_side=COST_PER_SIDE, hold=HOLD, top_fraction=TOP_FRACTION, stride=1):
    work = frame.reset_index(drop=True).copy()
    work["asof_date"] = pd.to_datetime(work["asof_date"]).dt.strftime("%Y-%m-%d")
    work[RESIDUAL_SCORE_COL] = residualize_column(work, SCORE_COL)
    signal_dates = sorted(work["asof_date"].unique())
    layers = []
    for index, date in enumerate(signal_dates):
        if index % stride != 0:
            continue
        rows = work[work["asof_date"] == date]
        residual_std = float(rows[RESIDUAL_SCORE_COL].std(ddof=0))
        if not np.isfinite(residual_std) or residual_std < 1e-12:
            layers.append({
                "asof_date": date,
                "index": index,
                "weights": {},
                "n_long": 0,
                "n_short": 0,
                "daily": [0.0] * hold,
                "holding_10d": 0.0,
            })
            continue
        weights, long_rows, short_rows = pick_long_short(rows, top_fraction=top_fraction)
        daily = [layer_daily_return(long_rows, short_rows, horizon) for horizon in range(1, hold + 1)]
        layers.append({
            "asof_date": date,
            "index": index,
            "weights": weights,
            "n_long": int(len(long_rows)),
            "n_short": int(len(short_rows)),
            "daily": daily,
            "holding_10d": float(
                long_rows["actual_return_d10"].mean() - short_rows["actual_return_d10"].mean()
            ) if "actual_return_d10" in long_rows else float("nan"),
        })

    n_signals = len(signal_dates)
    n_days = n_signals + hold - 1
    scale = stride / hold
    nav = 1.0
    peak = 1.0
    max_drawdown = 0.0
    curve = []
    prev_weights = {}
    turnovers = []
    for day in range(1, n_days + 1):
        active = [layer for layer in layers if 1 <= day - layer["index"] <= hold]
        if not active:
            continue
        weights = {}
        gross = 0.0
        for layer in active:
            horizon = day - layer["index"]
            gross += scale * layer["daily"][horizon - 1]
            for symbol, weight in layer["weights"].items():
                weights[symbol] = weights.get(symbol, 0.0) + scale * weight
        traded = 0.0
        names = set(prev_weights) | set(weights)
        for symbol in names:
            traded += abs(weights.get(symbol, 0.0) - prev_weights.get(symbol, 0.0))
        two_way_turnover = 0.5 * traded
        cost = cost_per_side * traded
        net = gross - cost
        nav *= 1.0 + net
        peak = max(peak, nav)
        drawdown = nav / peak - 1.0
        max_drawdown = min(max_drawdown, drawdown)
        asof = signal_dates[day - 1] if day <= n_signals else f"hold+{day - n_signals}"
        curve.append({
            "day_index": day,
            "asof_or_hold": asof,
            "n_layers": len(active),
            "gross_exposure": len(active) * scale,
            "gross_return": gross,
            "two_way_turnover": two_way_turnover,
            "cost": cost,
            "net_return": net,
            "nav": nav,
            "drawdown": drawdown,
        })
        turnovers.append(two_way_turnover)
        prev_weights = weights

    metrics = nav_metrics(curve, nav, max_drawdown)
    overlapping_10d = [layer["holding_10d"] for layer in layers]
    n_positive = int(sum(value > 0 for value in overlapping_10d if np.isfinite(value)))
    return {
        "kind": "overlay",
        "cost_per_side": cost_per_side,
        "hold_sessions": hold,
        "rebalance_stride": stride,
        "top_fraction": top_fraction,
        "signal_dates": n_signals,
        "mean_two_way_turnover": float(np.mean(turnovers)) if turnovers else None,
        "mean_overlapping_10d_ls": float(np.nanmean(overlapping_10d)) if overlapping_10d else None,
        "positive_overlapping_10d_ls_ratio": n_positive / len(layers) if layers else None,
        "nav_curve": curve,
        "layers": [
            {
                "asof_date": layer["asof_date"],
                "n_long": layer["n_long"],
                "n_short": layer["n_short"],
                "holding_10d": layer["holding_10d"],
            }
            for layer in layers
        ],
        **metrics,
    }


def nav_metrics(curve, nav, max_drawdown):
    returns = np.array([row["net_return"] for row in curve], dtype=np.float64)
    n = len(returns)
    mean = float(returns.mean()) if n else float("nan")
    std = float(returns.std(ddof=1)) if n > 1 else float("nan")
    total = float(nav - 1.0)
    annualized = (nav ** (TRADING_DAYS_PER_YEAR / n) - 1.0) if n and nav > 0 else float("nan")
    sharpe = mean / std * math.sqrt(TRADING_DAYS_PER_YEAR) if std else float("nan")
    calmar = annualized / abs(max_drawdown) if max_drawdown < 0 else float("nan")
    return {
        "nav_days": n,
        "total_return": total,
        "annualized_return": annualized,
        "daily_mean": mean,
        "daily_std": std,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }


def book_backtest(
    frame,
    cost_per_side=COST_PER_SIDE,
    hold=5,
    stride=5,
    enter=TOP_FRACTION,
    exit_fraction=None,
):
    """Single-book (or non-overlapping) rebalance with optional hysteresis.

    Holdings are frozen between rebalance dates. Each book earns the next
    `hold` session increments of the names selected on that signal date.
    """
    work = frame.reset_index(drop=True).copy()
    work["asof_date"] = pd.to_datetime(work["asof_date"]).dt.strftime("%Y-%m-%d")
    work[RESIDUAL_SCORE_COL] = residualize_column(work, SCORE_COL)
    signal_dates = sorted(work["asof_date"].unique())
    by_date = {date: group.reset_index(drop=True) for date, group in work.groupby("asof_date", sort=True)}
    books = []
    prev_long, prev_short = set(), set()
    for index, date in enumerate(signal_dates):
        if index % stride != 0:
            continue
        rows = by_date[date]
        residual_std = float(rows[RESIDUAL_SCORE_COL].std(ddof=0))
        if not np.isfinite(residual_std) or residual_std < 1e-12:
            prev_long, prev_short = set(), set()
            books.append({
                "asof_date": date,
                "index": index,
                "weights": {},
                "n_long": 0,
                "n_short": 0,
                "rows": rows,
            })
            continue
        longs, shorts = pick_with_hysteresis(
            rows, prev_long, prev_short, enter=enter, exit_fraction=exit_fraction,
        )
        prev_long, prev_short = longs, shorts
        by_symbol = rows.set_index("symbol")
        books.append({
            "asof_date": date,
            "index": index,
            "weights": equal_side_weights(longs, shorts),
            "n_long": len(longs),
            "n_short": len(shorts),
            "by_symbol": by_symbol,
        })

    n_signals = len(signal_dates)
    n_days = n_signals + hold - 1
    nav = 1.0
    peak = 1.0
    max_drawdown = 0.0
    curve = []
    prev_weights = {}
    turnovers = []
    for day in range(1, n_days + 1):
        active = [book for book in books if 1 <= day - book["index"] <= hold]
        if not active:
            continue
        scale = 1.0 / len(active)
        weights = {}
        gross = 0.0
        for book in active:
            horizon = day - book["index"]
            book_return = 0.0
            for symbol, weight in book["weights"].items():
                if symbol not in book["by_symbol"].index:
                    continue
                book_return += weight * incremental_return(book["by_symbol"].loc[symbol], horizon)
                weights[symbol] = weights.get(symbol, 0.0) + scale * weight
            gross += scale * book_return
        traded = 0.0
        names = set(prev_weights) | set(weights)
        for symbol in names:
            traded += abs(weights.get(symbol, 0.0) - prev_weights.get(symbol, 0.0))
        two_way_turnover = 0.5 * traded
        cost = cost_per_side * traded
        net = gross - cost
        nav *= 1.0 + net
        peak = max(peak, nav)
        drawdown = nav / peak - 1.0
        max_drawdown = min(max_drawdown, drawdown)
        asof = signal_dates[day - 1] if day <= n_signals else f"hold+{day - n_signals}"
        curve.append({
            "day_index": day,
            "asof_or_hold": asof,
            "n_layers": len(active),
            "gross_exposure": float(sum(abs(w) for w in weights.values()) / 2.0) if weights else 0.0,
            "gross_return": gross,
            "two_way_turnover": two_way_turnover,
            "cost": cost,
            "net_return": net,
            "nav": nav,
            "drawdown": drawdown,
        })
        turnovers.append(two_way_turnover)
        prev_weights = weights

    metrics = nav_metrics(curve, nav, max_drawdown)
    return {
        "kind": "book",
        "cost_per_side": cost_per_side,
        "hold_sessions": hold,
        "rebalance_stride": stride,
        "top_fraction": enter,
        "exit_fraction": enter if exit_fraction is None else exit_fraction,
        "hysteresis": exit_fraction is not None and exit_fraction > enter,
        "signal_dates": n_signals,
        "mean_two_way_turnover": float(np.mean(turnovers)) if turnovers else None,
        "mean_n_long": float(np.mean([book["n_long"] for book in books])) if books else None,
        "mean_n_short": float(np.mean([book["n_short"] for book in books])) if books else None,
        "nav_curve": curve,
        **metrics,
    }


RULES = [
    {
        "name": "overlay_1d_tb10",
        "kind": "overlay",
        "hold": 10,
        "stride": 1,
        "enter": 0.10,
        "note": "frozen baseline: 10-layer daily overlay, top/bottom 10%",
    },
    {
        "name": "overlay_1d_tb5",
        "kind": "overlay",
        "hold": 10,
        "stride": 1,
        "enter": 0.05,
        "note": "same overlay, concentrated top/bottom 5%",
    },
    {
        "name": "overlay_5d_tb10",
        "kind": "overlay",
        "hold": 10,
        "stride": 5,
        "enter": 0.10,
        "note": "new 10-day sleeve every 5 sessions (2 layers)",
    },
    {
        "name": "book_5d_tb10",
        "kind": "book",
        "hold": 5,
        "stride": 5,
        "enter": 0.10,
        "note": "full book rebalanced every 5 sessions, top/bottom 10%",
    },
    {
        "name": "book_5d_tb5",
        "kind": "book",
        "hold": 5,
        "stride": 5,
        "enter": 0.05,
        "note": "5-day rebalance, concentrated top/bottom 5%",
    },
    {
        "name": "book_1d_hyst_10_15",
        "kind": "book",
        "hold": 1,
        "stride": 1,
        "enter": 0.10,
        "exit": 0.15,
        "note": "daily mark, sell only after leaving top/bottom 15%",
    },
    {
        "name": "book_5d_hyst_10_15",
        "kind": "book",
        "hold": 5,
        "stride": 5,
        "enter": 0.10,
        "exit": 0.15,
        "note": "5-day rebalance plus 10/15 hysteresis",
    },
    {
        "name": "book_5d_tb5_hyst_5_8",
        "kind": "book",
        "hold": 5,
        "stride": 5,
        "enter": 0.05,
        "exit": 0.08,
        "note": "5-day rebalance, 5% enter, stay until 8%",
    },
]


def run_rule(frame, rule, cost_per_side=COST_PER_SIDE):
    if rule["kind"] == "overlay":
        result = overlay_backtest(
            frame,
            cost_per_side=cost_per_side,
            hold=rule["hold"],
            top_fraction=rule["enter"],
            stride=rule["stride"],
        )
        gross = overlay_backtest(
            frame, cost_per_side=0.0, hold=rule["hold"],
            top_fraction=rule["enter"], stride=rule["stride"],
        )
    else:
        result = book_backtest(
            frame,
            cost_per_side=cost_per_side,
            hold=rule["hold"],
            stride=rule["stride"],
            enter=rule["enter"],
            exit_fraction=rule.get("exit"),
        )
        gross = book_backtest(
            frame, cost_per_side=0.0, hold=rule["hold"], stride=rule["stride"],
            enter=rule["enter"], exit_fraction=rule.get("exit"),
        )
    result["name"] = rule["name"]
    result["note"] = rule["note"]
    result["gross_zero_cost"] = {
        "total_return": gross["total_return"],
        "annualized_return": gross["annualized_return"],
        "sharpe": gross["sharpe"],
        "max_drawdown": gross["max_drawdown"],
        "calmar": gross["calmar"],
    }
    return result


def compact_rule(result):
    return {
        "name": result["name"],
        "note": result["note"],
        "kind": result["kind"],
        "hold_sessions": result["hold_sessions"],
        "rebalance_stride": result["rebalance_stride"],
        "top_fraction": result["top_fraction"],
        "exit_fraction": result.get("exit_fraction"),
        "hysteresis": result.get("hysteresis"),
        "nav_days": result["nav_days"],
        "mean_two_way_turnover": result["mean_two_way_turnover"],
        "mean_n_long": result.get("mean_n_long"),
        "total_return": result["total_return"],
        "annualized_return": result["annualized_return"],
        "sharpe": result["sharpe"],
        "max_drawdown": result["max_drawdown"],
        "calmar": result["calmar"],
        "gross_sharpe": result["gross_zero_cost"]["sharpe"],
        "gross_total_return": result["gross_zero_cost"]["total_return"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cost-per-side", type=float, default=COST_PER_SIDE)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    path = Path(args.predictions)
    frame = pd.read_csv(path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.sweep:
        rows = []
        payload = {
            "predictions": str(path),
            "cost_per_side": args.cost_per_side,
            "training_performed": False,
            "weights_touched": False,
            "purpose": "postprocess_trading_rules_on_frozen_c2_18d_scores",
            "rules": [],
        }
        curve_dir = output.with_suffix("").parent / (output.stem + "_curves")
        curve_dir.mkdir(parents=True, exist_ok=True)
        for rule in RULES:
            result = run_rule(frame, rule, cost_per_side=args.cost_per_side)
            compact = compact_rule(result)
            rows.append(compact)
            payload["rules"].append(compact)
            pd.DataFrame(result["nav_curve"]).to_csv(curve_dir / f"{rule['name']}.csv", index=False)
        table = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
        table.to_csv(output.with_suffix(".csv"), index=False)
        payload["best_net_sharpe"] = table.iloc[0].to_dict() if len(table) else None
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(table.to_string(index=False))
        return
    result = overlay_backtest(frame, cost_per_side=args.cost_per_side)
    gross = overlay_backtest(frame, cost_per_side=0.0)
    curve = pd.DataFrame(result["nav_curve"])
    curve.to_csv(output.with_suffix(".csv"), index=False)
    serializable = {
        key: value for key, value in result.items() if key != "nav_curve"
    }
    serializable["nav_end"] = result["nav_curve"][-1]["nav"] if result["nav_curve"] else None
    serializable["gross_zero_cost"] = {
        "total_return": gross["total_return"],
        "annualized_return": gross["annualized_return"],
        "sharpe": gross["sharpe"],
        "max_drawdown": gross["max_drawdown"],
        "calmar": gross["calmar"],
        "nav_end": gross["nav_curve"][-1]["nav"] if gross["nav_curve"] else None,
    }
    output.write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "predictions": str(path),
        "output": str(output),
        "total_return": result["total_return"],
        "annualized_return": result["annualized_return"],
        "sharpe": result["sharpe"],
        "max_drawdown": result["max_drawdown"],
        "calmar": result["calmar"],
        "mean_two_way_turnover": result["mean_two_way_turnover"],
        "mean_overlapping_10d_ls": result["mean_overlapping_10d_ls"],
        "gross_zero_cost_total_return": serializable["gross_zero_cost"]["total_return"],
        "gross_zero_cost_sharpe": serializable["gross_zero_cost"]["sharpe"],
        "nav_days": result["nav_days"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
