"""Evaluate concentrated Top-5 August portfolios from cached forecasts."""

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest_august_path import simulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", nargs=3, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cost-bps-per-side", type=float, default=20.0)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    all_trades = []
    rules = {"hold_d10": (None, None), "tp3_sl2": (0.03, 0.02), "tp5_sl3": (0.05, 0.03)}
    round_trip_cost = 2 * args.cost_bps_per_side / 10000
    for label, source_model, path in args.input:
        frame = pd.read_csv(path)
        frame = frame.loc[frame.model == source_model].copy()
        if frame.empty or frame.identity.duplicated().any():
            raise ValueError(f"Invalid filtered input for {source_model}")
        result[label] = {}
        for rule, (tp, sl) in rules.items():
            trades, gross = simulate(frame, "long", tp, sl, top_count=5)
            trades["net_return"] = trades.exit_return - round_trip_cost
            daily = trades.groupby("asof_date").net_return.mean().sort_index()
            equity = (1 + daily).cumprod()
            drawdown = equity / equity.cummax() - 1
            result[label][rule] = {
                **gross,
                "mean_trade_return_net": float(trades.net_return.mean()),
                "win_rate_net": float((trades.net_return > 0).mean()),
                "compounded_return_net": float(equity.iloc[-1] - 1),
                "max_drawdown_net": float(drawdown.min()),
            }
            trades["model"] = label
            trades["rule"] = rule
            all_trades.append(trades)
    pd.concat(all_trades, ignore_index=True).to_csv(
        output / "august_top5_trades.csv", index=False
    )
    summary = {
        "selection": "Top 5 per signal date by predicted_return_d10",
        "cost_bps_per_side": args.cost_bps_per_side,
        "overlapping_vintages": True,
        "metrics": result,
    }
    (output / "august_top5_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
