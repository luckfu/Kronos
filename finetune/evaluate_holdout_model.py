"""Evaluate one checkpoint on a fixed A-share symbol holdout."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from finetune.compare_kaggle_best_last import (
    build_signal_periods,
    load_holdout,
)
from finetune.evaluate_unseen_a_share import run_model, summarize
from model import KronosTokenizer


def top_bottom_spread(predictions):
    values = []
    for _, frame in predictions.groupby("period"):
        ranked = frame.sort_values("score", ascending=False)
        count = max(1, len(ranked) // 5)
        values.append(
            float(
                ranked.head(count)["actual_close_return"].mean()
                - ranked.tail(count)["actual_close_return"].mean()
            )
        )
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lookback", type=int, required=True)
    parser.add_argument("--signal-start", required=True)
    parser.add_argument("--signal-end", required=True)
    parser.add_argument("--period-count", type=int, default=16)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    panel, calendar = load_holdout(args.holdout, args.manifest)
    periods = build_signal_periods(
        panel,
        calendar,
        args.signal_start,
        args.signal_end,
        args.period_count,
        args.lookback,
    )
    device = torch.device(
        "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(
        f"Device={device}, periods={len(periods)}, "
        f"first={periods[0]['signal_date']:%Y-%m-%d}, "
        f"last={periods[-1]['signal_date']:%Y-%m-%d}"
    )
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer).to(device).eval()
    predictions = run_model(
        args.label,
        args.model,
        tokenizer,
        periods,
        device,
        args.batch_size,
        args.sample_count,
        args.seed,
    )
    predictions.to_csv(output / "predictions.csv", index=False)
    result = {
        "configuration": {
            "model": args.model,
            "label": args.label,
            "lookback": args.lookback,
            "signal_start": args.signal_start,
            "signal_end": args.signal_end,
            "period_count": args.period_count,
            "sample_count": args.sample_count,
            "seed": args.seed,
            "device": str(device),
        },
        "summary": summarize(predictions)[0],
        "mean_top_bottom_actual_return_spread": top_bottom_spread(predictions),
    }
    with open(output / "summary.json", "w") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
