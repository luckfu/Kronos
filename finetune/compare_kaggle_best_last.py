"""Compare the final Kaggle V3 best and last checkpoints on symbol holdouts."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune.evaluate_unseen_a_share import (
    FEATURE_COLUMNS,
    LOOKBACK,
    PRED_LEN,
    paired_bootstrap,
    run_model,
    summarize,
)
from model import KronosTokenizer


def load_holdout(path, manifest_path):
    holdout = pickle.load(open(path, "rb"))
    manifest = pd.read_csv(manifest_path)
    names = manifest.set_index("symbol")["name"].to_dict()
    panel = {}
    for symbol, frame in holdout.items():
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index)
        frame.index.name = "date"
        frame["name"] = names.get(symbol, symbol)
        panel[symbol] = frame.sort_index()
    calendar = pd.DatetimeIndex(sorted(set().union(*(set(frame.index) for frame in panel.values()))))
    return panel, calendar


def build_periods(panel, calendar, start, end, period_count, lookback=LOOKBACK):
    dates = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    valid_positions = np.arange(0, len(dates) - PRED_LEN)
    if len(valid_positions) == 0:
        raise RuntimeError(f"No valid signal dates in {start}..{end}")
    positions = np.linspace(valid_positions[0], valid_positions[-1], period_count).round().astype(int)
    periods = []
    for period_index, position in enumerate(positions):
        signal_date = pd.Timestamp(dates[position])
        future_dates = pd.DatetimeIndex(dates[position + 1:position + PRED_LEN + 1])
        records = []
        for symbol, frame in panel.items():
            if signal_date not in frame.index or not set(future_dates).issubset(frame.index):
                continue
            context = frame.loc[:signal_date].tail(lookback)
            if len(context) != lookback or context.index[-1] != signal_date:
                continue
            latest = context.iloc[-1]
            if any(float(latest[col]) <= 0 for col in ("close", "volume", "amount")):
                continue
            entry_open = float(frame.loc[future_dates[0], "open"])
            exit_close = float(frame.loc[future_dates[-1], "close"])
            average_close = float(frame.loc[future_dates, "close"].mean())
            records.append({
                "symbol": symbol,
                "name": str(latest["name"]),
                "context": context,
                "future_dates": future_dates,
                "size_bucket": int(np.clip(round(float(latest["size_bucket"])), 0, 9)),
                "size_percentile": float(np.clip(float(latest["size_percentile"]), 0, 1)),
                "last_close": float(latest["close"]),
                "actual_close_return": average_close / float(latest["close"]) - 1,
                "realized_return": exit_close / entry_open - 1,
            })
        if len(records) < 100:
            raise RuntimeError(f"{signal_date:%Y-%m-%d}: only {len(records)} eligible holdout stocks")
        periods.append({
            "period": period_index,
            "signal_date": signal_date,
            "entry_date": future_dates[0],
            "exit_date": future_dates[-1],
            "records": records,
        })
    return periods


def build_signal_periods(
    panel, calendar, start, end, period_count, lookback=LOOKBACK
):
    """Build periods whose signal dates, rather than labels, are date-bounded."""
    signal_dates = calendar[
        (calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))
    ]
    calendar_positions = calendar.get_indexer(signal_dates)
    valid_positions = calendar_positions[
        calendar_positions + PRED_LEN < len(calendar)
    ]
    if len(valid_positions) == 0:
        raise RuntimeError(f"No valid signal dates in {start}..{end}")
    selected = np.linspace(0, len(valid_positions) - 1, period_count)
    selected = np.unique(selected.round().astype(int))
    periods = []
    for period_index, selected_position in enumerate(selected):
        position = int(valid_positions[selected_position])
        signal_date = pd.Timestamp(calendar[position])
        future_dates = pd.DatetimeIndex(
            calendar[position + 1:position + PRED_LEN + 1]
        )
        records = []
        for symbol, frame in panel.items():
            if signal_date not in frame.index or not set(future_dates).issubset(frame.index):
                continue
            context = frame.loc[:signal_date].tail(lookback)
            if len(context) != lookback or context.index[-1] != signal_date:
                continue
            latest = context.iloc[-1]
            if any(float(latest[col]) <= 0 for col in ("close", "volume", "amount")):
                continue
            entry_open = float(frame.loc[future_dates[0], "open"])
            exit_close = float(frame.loc[future_dates[-1], "close"])
            average_close = float(frame.loc[future_dates, "close"].mean())
            records.append({
                "symbol": symbol,
                "name": str(latest["name"]),
                "context": context,
                "future_dates": future_dates,
                "size_bucket": int(np.clip(round(float(latest["size_bucket"])), 0, 9)),
                "size_percentile": float(np.clip(float(latest["size_percentile"]), 0, 1)),
                "last_close": float(latest["close"]),
                "actual_close_return": average_close / float(latest["close"]) - 1,
                "realized_return": exit_close / entry_open - 1,
            })
        if len(records) < 100:
            raise RuntimeError(f"{signal_date:%Y-%m-%d}: only {len(records)} eligible holdout stocks")
        periods.append({
            "period": period_index,
            "signal_date": signal_date,
            "entry_date": future_dates[0],
            "exit_date": future_dates[-1],
            "records": records,
        })
    return periods


def periods_with_lookback(periods, lookback):
    trimmed = []
    for period in periods:
        records = []
        for record in period["records"]:
            context = record["context"].tail(lookback)
            if len(context) != lookback:
                raise ValueError(
                    f"Record {record['symbol']} has {len(context)} rows; "
                    f"{lookback} required"
                )
            records.append({**record, "context": context})
        trimmed.append({**period, "records": records})
    return trimmed


def compare_window(best_path, last_path, tokenizer, panel, calendar, label, start, end, args, device, index):
    shared_lookback = max(args.best_lookback, args.last_lookback)
    if args.signal_start:
        periods = build_signal_periods(
            panel, calendar, start, end, args.period_count, shared_lookback
        )
    else:
        periods = build_periods(
            panel, calendar, start, end, args.period_count, shared_lookback
        )
    print(
        f"Window {label}: periods={len(periods)}, "
        f"first={periods[0]['signal_date']:%Y-%m-%d}, "
        f"last={periods[-1]['signal_date']:%Y-%m-%d}, "
        f"records/period={len(periods[0]['records'])}"
    )
    seed = args.seed + index * 1000
    best_periods = periods_with_lookback(periods, args.best_lookback)
    last_periods = periods_with_lookback(periods, args.last_lookback)
    best = run_model("best", best_path, tokenizer, best_periods, device, args.batch_size, args.sample_count, seed)
    latest = run_model("latest", last_path, tokenizer, last_periods, device, args.batch_size, args.sample_count, seed)
    for frame in (best, latest):
        frame["window"] = label
        frame["period_local"] = frame["period"]
        frame["period"] = frame["period"] + index * 1000
    predictions = pd.concat([best, latest], ignore_index=True)
    summaries = {model: summarize(frame)[0] for model, frame in predictions.groupby("model")}
    correlations, spreads, top8_overlap = [], {"best": [], "latest": []}, []
    for _, frame in predictions.groupby("period"):
        for model, model_frame in frame.groupby("model"):
            model_frame = model_frame.sort_values("score", ascending=False)
            n = max(1, len(model_frame) // 5)
            spreads[model].append(float(
                model_frame.head(n)["actual_close_return"].mean()
                - model_frame.tail(n)["actual_close_return"].mean()
            ))
        by_symbol = frame.pivot_table(index="symbol", columns="model", values="score").dropna()
        correlations.append(float(by_symbol["best"].corr(by_symbol["latest"], method="spearman")))
        best_top = set(frame[frame["model"] == "best"].nlargest(8, "score")["symbol"])
        latest_top = set(frame[frame["model"] == "latest"].nlargest(8, "score")["symbol"])
        top8_overlap.append(len(best_top & latest_top) / 8.0)
    return predictions, {
        "summary": summaries,
        "paired_bootstrap": {
            "rank_ic": paired_bootstrap(predictions, "rank_ic", draws=5000, seed=args.seed + index, higher_is_better=True),
            "direction_accuracy": paired_bootstrap(predictions, "direction_accuracy", draws=5000, seed=args.seed + index, higher_is_better=True),
            "return_mae": paired_bootstrap(predictions, "return_mae", draws=5000, seed=args.seed + index, higher_is_better=False),
        },
        "mean_top_bottom_actual_return_spread": {
            model: float(np.mean(values)) for model, values in spreads.items()
        },
        "mean_best_latest_score_rank_correlation": float(np.mean(correlations)),
        "mean_top8_overlap": float(np.mean(top8_overlap)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", default="./data/a_share_v3/processed_datasets/symbol_holdout_data.pkl")
    parser.add_argument("--manifest", default="./data/a_share_v3/universe_manifest.csv")
    parser.add_argument("--best-model", required=True)
    parser.add_argument("--last-model", required=True)
    parser.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    parser.add_argument("--output-dir", default="./outputs/backtest_results/kaggle_v3_best_vs_last_holdout")
    parser.add_argument("--period-count", type=int, default=24)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--best-lookback", type=int, default=LOOKBACK)
    parser.add_argument("--last-lookback", type=int, default=LOOKBACK)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--window", choices=("2024", "2025", "2026"), default=None)
    parser.add_argument("--signal-start", default="")
    parser.add_argument("--signal-end", default="")
    parser.add_argument("--signal-label", default="time_holdout")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    panel, calendar = load_holdout(args.holdout, args.manifest)
    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    print(f"holdout_symbols={len(panel)}, calendar_days={len(calendar)}, Device={device}")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer).to(device).eval()
    windows = [
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026", "2026-01-01", "2026-07-31"),
    ]
    if args.signal_start or args.signal_end:
        if not args.signal_start or not args.signal_end:
            parser.error('--signal-start and --signal-end must be provided together')
        windows = [(args.signal_label, args.signal_start, args.signal_end)]
    elif args.window:
        windows = [window for window in windows if window[0] == args.window]
    predictions, results = [], {}
    for index, window in enumerate(windows):
        frame, result = compare_window(
            args.best_model, args.last_model, tokenizer, panel, calendar,
            *window, args, device, index
        )
        predictions.append(frame)
        results[window[0]] = result
        print(json.dumps(result, indent=2))
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    summary = {
        "configuration": {
            "holdout_symbols": len(panel),
            "holdout_source": args.holdout,
            "windows": [{"label": label, "start": start, "end": end} for label, start, end in windows],
            "best_lookback": args.best_lookback,
            "last_lookback": args.last_lookback,
            "forecast_days": PRED_LEN,
            "signal_aggregation": "mean predicted close across forecast horizon",
            "sample_count": args.sample_count,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": str(device),
            "best_model": args.best_model,
            "last_model": args.last_model,
        },
        "windows": results,
        "overall": {model: summarize(frame)[0] for model, frame in predictions.groupby("model")},
    }
    with open(output / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print("RESULT_PATH", output)
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
