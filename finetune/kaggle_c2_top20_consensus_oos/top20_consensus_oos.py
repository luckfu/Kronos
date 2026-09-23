"""Exploratory Top20 persistence rules on the sealed C2 18-day OOS predictions."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"))
OUTPUT = Path(os.environ.get("KAGGLE_OUTPUT_ROOT", "/kaggle/working/top20_consensus_oos"))
TOP_N = 20
HOLD = 10
COST_PER_SIDE = 0.003
EXPECTED_ROWS = 92751
EXPECTED_DATES = 18


def find_predictions():
    matches = list(INPUT.glob("**/predictions_prod_t065_p80_n5.csv.gz"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one C2 prediction file, found {matches}")
    return matches[0]


def load_frame():
    path = find_predictions()
    frame = pd.read_csv(path)
    if len(frame) != EXPECTED_ROWS or frame["asof_date"].nunique() != EXPECTED_DATES:
        raise RuntimeError(f"Unexpected OOS shape: {len(frame)} rows / {frame.asof_date.nunique()} dates")
    frame["asof_date"] = pd.to_datetime(frame["asof_date"]).dt.strftime("%Y-%m-%d")
    frame = frame.sort_values(["asof_date", "predicted_return_d10", "symbol"],
                              ascending=[True, False, True], kind="mergesort")
    frame["rank"] = frame.groupby("asof_date", sort=False).cumcount() + 1
    return frame, path


def incremental_return(row, horizon):
    current = float(row[f"actual_return_d{horizon}"])
    if horizon == 1:
        return current
    previous = float(row[f"actual_return_d{horizon - 1}"])
    return (1.0 + current) / (1.0 + previous) - 1.0


def selected_dates(frame, consecutive):
    dates = sorted(frame.asof_date.unique())
    top = {
        date: set(frame.loc[(frame.asof_date == date) & (frame["rank"] <= TOP_N), "symbol"])
        for date in dates
    }
    if consecutive == 1:
        return {date: top[date] for date in dates}
    result = {}
    for index, date in enumerate(dates):
        if index + 1 < consecutive:
            result[date] = set()
            continue
        names = set(top[date])
        for prior in dates[index - consecutive + 1:index]:
            names &= top[prior]
        result[date] = names
    return result


def event_metrics(frame, selected):
    by_date = {date: group.set_index("symbol") for date, group in frame.groupby("asof_date")}
    rows = []
    for date in sorted(selected):
        symbols = sorted(selected[date])
        if not symbols:
            continue
        group = by_date[date].loc[symbols]
        row = {"asof_date": date, "n_selected": len(symbols)}
        for horizon in (1, 3, 5, 10):
            values = group[f"actual_return_d{horizon}"].astype(float)
            row[f"mean_return_d{horizon}"] = float(values.mean())
            row[f"win_rate_d{horizon}"] = float((values > 0).mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table, {}
    summary = {
        "confirmation_dates": int(len(table)),
        "mean_selected": float(table.n_selected.mean()),
        "median_selected": float(table.n_selected.median()),
    }
    for horizon in (1, 3, 5, 10):
        summary[f"mean_return_d{horizon}"] = float(table[f"mean_return_d{horizon}"].mean())
        summary[f"win_rate_d{horizon}"] = float(table[f"win_rate_d{horizon}"].mean())
    return table, summary


def portfolio(frame, selected):
    dates = sorted(frame.asof_date.unique())
    by_date = {date: group.set_index("symbol") for date, group in frame.groupby("asof_date")}
    layers = []
    for index, date in enumerate(dates):
        names = sorted(selected[date])
        if not names:
            continue
        group = by_date[date].loc[names]
        daily = [float(np.mean([incremental_return(group.loc[symbol], h) for symbol in names]))
                 for h in range(1, HOLD + 1)]
        layers.append({"index": index, "asof_date": date, "symbols": names, "daily": daily})

    if not layers:
        return {"nav_days": 0, "total_return": 0.0, "net_curve": [], "layers": []}
    n_days = len(dates) + HOLD - 1
    nav = 1.0
    peak = 1.0
    max_dd = 0.0
    previous = {}
    curve = []
    turnovers = []
    for day in range(n_days):
        active = [layer for layer in layers if 0 <= day - layer["index"] < HOLD]
        if not active:
            continue
        scale = 1.0 / len(active)
        gross = sum(scale * layer["daily"][day - layer["index"]] for layer in active)
        weights = {}
        for layer in active:
            for symbol in layer["symbols"]:
                weights[symbol] = weights.get(symbol, 0.0) + scale / len(layer["symbols"])
        turnover = 0.5 * sum(abs(weights.get(s, 0.0) - previous.get(s, 0.0))
                             for s in set(weights) | set(previous))
        net = gross - COST_PER_SIDE * 2.0 * turnover
        nav *= 1.0 + net
        peak = max(peak, nav)
        max_dd = min(max_dd, nav / peak - 1.0)
        curve.append({
            "day_index": day,
            "asof_or_hold": dates[day] if day < len(dates) else f"hold+{day - len(dates) + 1}",
            "active_layers": len(active),
            "gross_return": gross,
            "two_way_turnover": turnover,
            "cost": COST_PER_SIDE * 2.0 * turnover,
            "net_return": net,
            "nav": nav,
            "drawdown": nav / peak - 1.0,
        })
        turnovers.append(turnover)
        previous = weights
    returns = np.array([row["net_return"] for row in curve], dtype=float)
    gross_returns = np.array([row["gross_return"] for row in curve], dtype=float)
    std = float(returns.std(ddof=1)) if len(returns) > 1 else None
    gross_std = float(gross_returns.std(ddof=1)) if len(gross_returns) > 1 else None
    return {
        "nav_days": len(curve),
        "total_return": float(nav - 1.0),
        "gross_total_return": float(np.prod(1.0 + gross_returns) - 1.0),
        "mean_two_way_turnover": float(np.mean(turnovers)) if turnovers else None,
        "sharpe": None if not std or not math.isfinite(std) else float(returns.mean() / std * math.sqrt(252)),
        "gross_sharpe": None if not gross_std or not math.isfinite(gross_std) else float(gross_returns.mean() / gross_std * math.sqrt(252)),
        "max_drawdown": float(max_dd),
        "net_curve": curve,
        "layers": [{"asof_date": layer["asof_date"], "n_selected": len(layer["symbols"])}
                   for layer in layers],
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, source = load_frame()
    strategies = {}
    for consecutive in (1, 2, 3):
        name = "daily_top20" if consecutive == 1 else f"consecutive_{consecutive}d_top20"
        selected = selected_dates(frame, consecutive)
        events, event_summary = event_metrics(frame, selected)
        portfolio_summary = portfolio(frame, selected)
        events.to_csv(OUTPUT / f"{name}_events.csv", index=False)
        strategies[name] = {
            "consecutive_days": consecutive,
            "event_summary": event_summary,
            "portfolio": portfolio_summary,
        }
    result = {
        "status": "complete",
        "purpose": "exploratory_rule_comparison_on_reused_18d_c2_oos",
        "source": str(source),
        "signal_dates": EXPECTED_DATES,
        "samples": EXPECTED_ROWS,
        "top_n": TOP_N,
        "hold_sessions": HOLD,
        "cost_per_side": COST_PER_SIDE,
        "execution": "signal confirmed after asof close; returns start at next session d1",
        "strategies": strategies,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        name: {
            "events": value["event_summary"],
            "total_return": value["portfolio"]["total_return"],
            "gross_total_return": value["portfolio"]["gross_total_return"],
            "sharpe": value["portfolio"]["sharpe"],
            "max_drawdown": value["portfolio"]["max_drawdown"],
        }
        for name, value in strategies.items()
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
