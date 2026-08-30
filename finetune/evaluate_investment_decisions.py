"""Turn saved multi-horizon forecasts into date-clustered investment metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = (1, 3, 5, 10)


def max_drawdown(cumulative_returns):
    equity = 1.0 + np.asarray(cumulative_returns, dtype=float)
    return float(np.min(equity / np.maximum.accumulate(equity) - 1.0))


def portfolio_path(rows, horizon):
    return np.array(
        [rows[f"actual_return_d{day}"].mean() for day in range(1, horizon + 1)],
        dtype=float,
    )


def evaluate_horizon(rows, horizon, top_fraction):
    score = f"predicted_return_d{horizon}"
    actual = f"actual_return_d{horizon}"
    per_date = []
    for asof_date, date_rows in rows.groupby("asof_date"):
        ranked = date_rows.sort_values(score, ascending=False)
        count = max(1, int(np.ceil(len(ranked) * top_fraction)))
        top = ranked.head(count)
        bottom = ranked.tail(count)
        market_path = portfolio_path(ranked, horizon)
        top_path = portfolio_path(top, horizon)
        bottom_path = portfolio_path(bottom, horizon)
        per_date.append({
            "asof_date": asof_date,
            "horizon_day": horizon,
            "rank_ic": float(date_rows[score].corr(date_rows[actual], method="spearman")),
            "top_count": count,
            "market_return": float(market_path[-1]),
            "top_return": float(top_path[-1]),
            "bottom_return": float(bottom_path[-1]),
            "top_excess_return": float(top_path[-1] - market_path[-1]),
            "top_bottom_spread": float(top_path[-1] - bottom_path[-1]),
            "top_win_rate": float((top[actual] > 0).mean()),
            "top_max_drawdown": max_drawdown(top_path),
            "bottom_max_drawdown": max_drawdown(bottom_path),
        })
    frame = pd.DataFrame(per_date)
    return frame, {
        "dates": int(len(frame)),
        "mean_rank_ic": float(frame["rank_ic"].mean()),
        "rank_ic_positive_rate": float((frame["rank_ic"] > 0).mean()),
        "mean_market_return": float(frame["market_return"].mean()),
        "mean_top_return": float(frame["top_return"].mean()),
        "mean_bottom_return": float(frame["bottom_return"].mean()),
        "mean_top_excess_return": float(frame["top_excess_return"].mean()),
        "mean_top_bottom_spread": float(frame["top_bottom_spread"].mean()),
        "mean_top_win_rate": float(frame["top_win_rate"].mean()),
        "mean_top_max_drawdown": float(frame["top_max_drawdown"].mean()),
        "worst_top_max_drawdown": float(frame["top_max_drawdown"].min()),
    }


def run(predictions_path, output_dir, top_fraction):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(predictions_path)
    required = {f"{kind}_return_d{day}" for kind in ("predicted", "actual") for day in range(1, 11)}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise RuntimeError(f"Predictions lack path columns: {missing}")
    summaries = {}
    date_frames = []
    for (set_name, model), selected in rows.groupby(["set", "model"]):
        model_summary = {}
        for horizon in HORIZONS:
            frame, metrics = evaluate_horizon(selected, horizon, top_fraction)
            frame["set"] = set_name
            frame["model"] = model
            date_frames.append(frame)
            model_summary[f"day_{horizon}"] = metrics
        summaries.setdefault(set_name, {})[model] = model_summary
    per_date = pd.concat(date_frames, ignore_index=True)
    per_date.to_csv(output / "per_date_investment_metrics.csv.gz", index=False, compression="gzip")
    summary = {
        "schema_version": 1,
        "task": "evaluation_only_investment_decision_metrics",
        "source_predictions": str(predictions_path),
        "decision_rule": {
            "selection": f"top {top_fraction:.0%} by predicted cumulative return at each holding horizon",
            "weighting": "equal weight within each asof_date",
            "rebalancing": "separate static portfolios per asof_date and holding horizon",
            "costs": "excluded",
        },
        "metrics": summaries,
        "limitations": [
            "Only six future signal dates are available.",
            "Per-date demeaning does not change cross-sectional ranks, so it is not separately reported here.",
            "Forecast sampling uncertainty was not retained in the source prediction file.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if not 0 < args.top_fraction < 0.5:
        raise SystemExit("--top-fraction must be between 0 and 0.5")
    run(args.predictions, args.output_dir, args.top_fraction)


if __name__ == "__main__":
    main()
