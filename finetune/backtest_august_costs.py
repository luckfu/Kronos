"""Cost and concentration sensitivity for the cached August path backtest."""

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest_august_path import simulate


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
    result = {}
    trades_all = []
    rules = {"hold_d10": (None, None), "tp3_sl2": (0.03, 0.02), "tp5_sl3": (0.05, 0.03)}
    for label, source_model, path in args.input:
        frame = pd.read_csv(path)
        frame = frame.loc[frame["model"] == source_model].copy()
        if frame.empty:
            raise ValueError(f"No rows for source model {source_model!r} in {path}")
        if frame["identity"].duplicated().any():
            raise ValueError(f"Duplicate identities after filtering {source_model!r}")
        result[label] = {}
        for fraction in (0.1, 0.2):
            for rule, (tp, sl) in rules.items():
                for side in ("long", "short"):
                    trades, _ = simulate(frame, side, tp, sl, top_fraction=fraction)
                    key = f"{side}_{rule}_top{int(fraction * 100)}"
                    result[label][key] = {}
                    for cost_bps in (0, 10, 20, 40, 60):
                        net = trades.exit_return - 2 * cost_bps / 10000
                        daily = net.groupby(trades.asof_date).mean().sort_index()
                        equity = (1 + daily).cumprod()
                        dd = equity / equity.cummax() - 1
                        result[label][key][str(cost_bps)] = {
                            "cost_bps_per_side": cost_bps,
                            "mean_trade_return_net": float(net.mean()),
                            "win_rate_net": float((net > 0).mean()),
                            "compounded_return_net": float(equity.iloc[-1] - 1),
                            "max_drawdown_net": float(dd.min()),
                        }
                    trades["label"] = label; trades["rule"] = rule; trades["top_fraction"] = fraction
                    trades_all.append(trades)
    pd.concat(trades_all, ignore_index=True).to_csv(output / "august_cost_sensitivity_trades.csv.gz", index=False, compression="gzip")
    (output / "august_cost_sensitivity_summary.json").write_text(json.dumps({
        "evaluation_role": "cost_and_concentration_sensitivity",
        "cost_definition": "round_trip = 2 * cost_bps_per_side; applied to every completed trade",
        "overlapping_vintages": True,
        "metrics": result,
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
