"""Backtest early-exit trading rules from cached 10-day forecast paths."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = range(1, 11)


def simulate(
    rows, side, take_profit=None, stop_loss=None, top_fraction=0.2, top_count=None
):
    rows = rows.sort_values(["asof_date", "predicted_return_d10"])
    selected = []
    for _, day in rows.groupby("asof_date", sort=True):
        count = int(top_count) if top_count is not None else max(1, int(len(day) * top_fraction))
        count = min(count, len(day))
        selected.append(day.tail(count) if side == "long" else day.head(count))
    selected = pd.concat(selected, ignore_index=True)
    outcomes = []
    for row in selected.itertuples(index=False):
        exit_day = 10
        exit_return = float(getattr(row, "actual_return_d10"))
        reason = "horizon"
        if take_profit is not None and stop_loss is not None:
            for day in HORIZONS:
                value = float(getattr(row, f"actual_return_d{day}"))
                if side == "long":
                    if value >= take_profit:
                        exit_day, exit_return, reason = day, value, "take_profit"
                        break
                    if value <= -stop_loss:
                        exit_day, exit_return, reason = day, value, "stop_loss"
                        break
                else:
                    if value <= -take_profit:
                        exit_day, exit_return, reason = day, -value, "take_profit"
                        break
                    if value >= stop_loss:
                        exit_day, exit_return, reason = day, -value, "stop_loss"
                        break
        if side == "short" and reason == "horizon":
            exit_return = -exit_return
        outcomes.append({
            "model": row.model,
            "asof_date": row.asof_date,
            "symbol": row.symbol,
            "side": side,
            "exit_day": exit_day,
            "exit_return": exit_return,
            "exit_reason": reason,
        })
    trades = pd.DataFrame(outcomes)
    daily = trades.groupby("asof_date")["exit_return"].mean().sort_index()
    equity = (1.0 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return trades, {
        "trades": int(len(trades)),
        "signal_dates": int(daily.size),
        "mean_trade_return": float(trades.exit_return.mean()),
        "median_trade_return": float(trades.exit_return.median()),
        "win_rate": float((trades.exit_return > 0).mean()),
        "mean_daily_return": float(daily.mean()),
        "compounded_return": float(equity.iloc[-1] - 1.0),
        "max_daily_loss": float(daily.min()),
        "max_drawdown": float(drawdown.min()),
        "mean_exit_day": float(trades.exit_day.mean()),
        "exit_reason_counts": trades.exit_reason.value_counts().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", action="append", nargs=3,
        metavar=("LABEL", "SOURCE_MODEL", "CSV"), required=True,
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    all_trades = []
    for label, source_model, csv_path in args.input:
        frame = pd.read_csv(csv_path)
        frame = frame.loc[frame["model"] == source_model].copy()
        if frame.empty:
            raise ValueError(f"No rows for source model {source_model!r} in {csv_path}")
        if frame["identity"].duplicated().any():
            raise ValueError(f"Duplicate identities after filtering {source_model!r}")
        frame["model"] = label
        model_results = {}
        rules = {
            "hold_d10": (None, None),
            "tp3_sl2": (0.03, 0.02),
            "tp5_sl3": (0.05, 0.03),
        }
        for rule, (take_profit, stop_loss) in rules.items():
            for side in ("long", "short"):
                trades, metrics = simulate(frame, side, take_profit, stop_loss)
                key = f"{side}_{rule}"
                model_results[key] = metrics
                trades["rule"] = rule
                all_trades.append(trades)
        summaries[label] = model_results
    trades = pd.concat(all_trades, ignore_index=True)
    trades.to_csv(output / "august_path_trades.csv.gz", index=False, compression="gzip")
    (output / "august_path_summary.json").write_text(
        json.dumps({
            "evaluation_role": "cached_august_10d_path_trading_backtest",
            "input_models": [label for label, _, _ in args.input],
            "selection": "top or bottom 20 percent per signal date by predicted_return_d10",
            "execution": "close-to-close; first observed close barrier hit exits, otherwise day 10",
            "barriers": {"tp3_sl2": {"take_profit": 0.03, "stop_loss": 0.02}, "tp5_sl3": {"take_profit": 0.05, "stop_loss": 0.03}},
            "costs_included": False,
            "metrics": summaries,
        }, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
