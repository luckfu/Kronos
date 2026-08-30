"""Evaluate 10-session long/short path events on a temporally isolated set."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model.kronos import auto_regressive_inference
from finetune.evaluate_v1_beta_checkpoints import (
    FEATURES, LOOKBACK, PREDICT, WindowStore, batches, find_evaluation_root,
    load_model, load_samples, stack_batch,
)


def select_by_date(records, per_date, seed):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["asof_date"]].append(record)
    selected = []
    for index, date in enumerate(sorted(grouped)):
        rows = grouped[date]
        rng = np.random.default_rng(seed + index)
        count = min(per_date, len(rows))
        selected.extend(rows[int(i)] for i in rng.choice(len(rows), count, replace=False))
    return selected


def first_hit(values, level, upward):
    indices = np.flatnonzero(values >= level if upward else values <= level)
    return int(indices[0]) + 1 if len(indices) else None


def wins_before_stop(target_path, stop_path):
    """Return whether a target occurs strictly before the stop on each path.

    Daily OHLC bars cannot order an intraday target and stop hit.  A same-day
    tie is therefore conservatively not a win.
    """
    result = np.zeros(len(target_path), dtype=bool)
    for index, (target_day, stop_day) in enumerate(zip(target_path, stop_path)):
        result[index] = target_day is not None and (
            stop_day is None or target_day < stop_day
        )
    return result


def actual_events(item, long_target, short_target, stop):
    # The next-session open is known only after the signal and is the assumed entry.
    scale = item["std"] + 1e-5
    raw = item["x"][LOOKBACK:LOOKBACK + PREDICT] * scale + item["mean"]
    entry = float(raw[0, FEATURES.index("open")])
    highs = raw[:, FEATURES.index("high")] / entry - 1
    lows = raw[:, FEATURES.index("low")] / entry - 1
    long_hit = first_hit(highs, long_target, True)
    short_hit = first_hit(lows, -short_target, False)
    long_stop = first_hit(lows, -stop, False)
    short_stop = first_hit(highs, stop, True)
    return {
        "entry_open": entry,
        "actual_long_hit_day": long_hit,
        "actual_short_hit_day": short_hit,
        "actual_long_win": long_hit is not None and (long_stop is None or long_hit < long_stop),
        "actual_short_win": short_hit is not None and (short_stop is None or short_hit < short_stop),
        "actual_max_high_return": float(highs.max()),
        "actual_min_low_return": float(lows.min()),
    }


def infer(model, tokenizer, records, store, device, samples, temperature, top_p, long_target, short_target, stop, seed):
    rows = []
    close = FEATURES.index("close")
    high = FEATURES.index("high")
    low = FEATURES.index("low")
    grouped = defaultdict(list)
    for record in records:
        grouped[record["asof_date"]].append(record)
    for date_index, (date, date_records) in enumerate(sorted(grouped.items())):
        torch.manual_seed(seed + date_index)
        np.random.seed(seed + date_index)
        for items in batches(date_records, store, 32):
            batch = stack_batch(items, device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                paths = auto_regressive_inference(
                    tokenizer, model, batch["x"][:, :LOOKBACK], batch["stamp"][:, :LOOKBACK],
                    batch["stamp"][:, LOOKBACK:LOOKBACK + PREDICT], max_context=512,
                    pred_len=PREDICT, clip=5, T=temperature, top_k=0, top_p=top_p,
                    sample_count=samples, verbose=False, sector_id=batch["sector"],
                    size_percentile=batch["percentile"], return_samples=True,
                )[:, :, -PREDICT:, :]
            for index, item in enumerate(items):
                scale = item["std"] + 1e-5
                predicted = paths[index] * scale + item["mean"]
                # Use each generated path's next-session open as its internally coherent entry.
                entry = predicted[:, 0, FEATURES.index("open")]
                highs = predicted[:, :, high] / entry[:, None] - 1
                lows = predicted[:, :, low] / entry[:, None] - 1
                long_target_days = [first_hit(path, long_target, True) for path in highs]
                short_target_days = [first_hit(path, -short_target, False) for path in lows]
                long_stop_days = [first_hit(path, -stop, False) for path in lows]
                short_stop_days = [first_hit(path, stop, True) for path in highs]
                long_wins = wins_before_stop(long_target_days, long_stop_days)
                short_wins = wins_before_stop(short_target_days, short_stop_days)
                actual = actual_events(item, long_target, short_target, stop)
                rows.append({
                    "asof_date": item["asof_date"], "symbol": item["symbol"],
                    "pred_long_probability": float(np.mean([day is not None for day in long_target_days])),
                    "pred_short_probability": float(np.mean([day is not None for day in short_target_days])),
                    "pred_long_win_probability": float(long_wins.mean()),
                    "pred_short_win_probability": float(short_wins.mean()),
                    "pred_median_close_return": float(np.median(predicted[:, -1, close] / entry - 1)),
                    **actual,
                })
        print(f"event inference {date_index + 1}/{len(grouped)} {date} ({len(date_records)} symbols)", flush=True)
    return pd.DataFrame(rows)


def event_metrics(frame, side):
    probability = f"pred_{side}_win_probability"
    actual = f"actual_{side}_win"
    values = []
    for _, rows in frame.groupby("asof_date"):
        n = max(1, len(rows) // 5)
        cutoff = float(rows[probability].nlargest(n).iloc[-1])
        above = rows[rows[probability] > cutoff]
        tied = rows[rows[probability] == cutoff]
        tied_slots = n - len(above)
        # Generated probabilities are discrete (for example, increments of 1/50).
        # Fractionally include the cutoff tie so metrics do not depend on row order.
        selected_actual = (
            float(above[actual].sum())
            + tied_slots * float(tied[actual].mean())
        ) / n
        selected_probability = (
            float(above[probability].sum()) + tied_slots * cutoff
        ) / n
        values.append({
            "top20_precision": selected_actual,
            "base_rate": float(rows[actual].mean()),
            "top20_probability": selected_probability,
            "top20_cutoff_probability": cutoff,
            "top20_cutoff_tie_count": int(len(tied)),
            "rank_ic": float(rows[probability].corr(rows[actual].astype(float), method="spearman")),
        })
    result = pd.DataFrame(values)
    return {key: float(result[key].mean()) for key in result.columns} | {"per_date": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-date", type=int, default=500)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--long-target", type=float, default=0.05)
    parser.add_argument("--short-target", type=float, default=0.05)
    parser.add_argument("--stop", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    root, manifest = find_evaluation_root(Path(args.input_root))
    groups = load_samples(root / manifest["artifacts"]["samples_file"])
    panel = __import__("pickle").loads((root / manifest["artifacts"]["panel_file"]).read_bytes())
    store = WindowStore(panel, manifest["model_contract"]["sector_labels"])
    records = select_by_date(groups["future_all"], args.per_date, args.seed)
    device = torch.device("cuda:0")
    tokenizer = __import__("model").KronosTokenizer.from_pretrained(args.tokenizer).to(device).eval()
    model = load_model(Path(args.model), device)
    frame = infer(model, tokenizer, records, store, device, args.samples, args.temperature,
                  args.top_p, args.long_target, args.short_target, args.stop, args.seed)
    frame.to_csv(output / "event_predictions.csv.gz", index=False, compression="gzip")
    summary = {"task": "future_path_event_evaluation", "training_performed": False,
               "temporal_isolation_verified": True, "samples": len(frame),
               "dates": sorted(frame["asof_date"].unique().tolist()),
               "entry_rule": "next_session_open", "long_event": f"high reaches +{args.long_target:.0%} before low reaches -{args.stop:.0%}",
               "short_event": f"low reaches -{args.short_target:.0%} before high reaches +{args.stop:.0%}",
               "generation": {"samples": args.samples, "temperature": args.temperature, "top_p": args.top_p},
               "model_sha256": hashlib.sha256((Path(args.model) / "model.safetensors").read_bytes()).hexdigest(),
               "long": event_metrics(frame, "long"), "short": event_metrics(frame, "short")}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
