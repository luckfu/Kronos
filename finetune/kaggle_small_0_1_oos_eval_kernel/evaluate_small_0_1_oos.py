"""Evaluate Kronos-small Stage 2 best/last on the sealed time-OOS package."""

import argparse
import gc
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_v1_beta_checkpoints import WindowStore, evaluate_predictions, load_samples
from model import Kronos, KronosTokenizer


def summarize(frame):
    horizons = range(1, 11)
    rows = []
    for h in horizons:
        pred = frame[f"predicted_return_d{h}"]
        actual = frame[f"actual_return_d{h}"]
        rows.append({
            "horizon_day": h,
            "samples": int(len(frame)),
            "direction_accuracy": float((np.sign(pred) == np.sign(actual)).mean()),
            "rank_ic": None if pred.corr(actual, method="spearman") != pred.corr(actual, method="spearman") else float(pred.corr(actual, method="spearman")),
            "pearson_ic": None if pred.corr(actual) != pred.corr(actual) else float(pred.corr(actual)),
        })
    by_date = []
    for date, group in frame.groupby("asof_date", sort=True):
        pred = group["predicted_return_d10"]
        actual = group["actual_return_d10"]
        rank_ic = pred.corr(actual, method="spearman")
        by_date.append({
            "asof_date": date,
            "samples": int(len(group)),
            "direction_accuracy": float((np.sign(pred) == np.sign(actual)).mean()),
            "rank_ic": None if pd.isna(rank_ic) else float(rank_ic),
        })
    ic = pd.Series([x["rank_ic"] for x in by_date], dtype=float).dropna()
    return {
        "samples": int(len(frame)),
        "signal_dates": int(frame["asof_date"].nunique()),
        "overall_direction_accuracy_d10": float((np.sign(frame.predicted_return_d10) == np.sign(frame.actual_return_d10)).mean()),
        "mean_rank_ic_d10": float(ic.mean()) if len(ic) else None,
        "rank_ic_positive_rate_d10": float((ic > 0).mean()) if len(ic) else None,
        "by_horizon": rows,
        "by_signal_date_d10": by_date,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--best-model", required=True)
    parser.add_argument("--last-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--max-per-date", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    root = Path(args.evaluation_root)
    manifest = json.loads((root / "evaluation_manifest.json").read_text())
    if not manifest["temporal_isolation"].get("strictly_after_parent_latest_training_target"):
        raise RuntimeError("Evaluation package is not temporally isolated")
    with (root / manifest["artifacts"]["panel_file"]).open("rb") as handle:
        panel = pickle.load(handle)
    all_records = load_samples(root / manifest["artifacts"]["samples_file"])["future_all"]
    isolation = manifest["temporal_isolation"]
    start = isolation["future_signal_start"]
    end = isolation["future_signal_end"]
    records = [r for r in all_records if start <= r["asof_date"] <= end]
    if not records:
        raise RuntimeError("No records in the manifest-defined future signal range")
    if args.max_per_date:
        rng = np.random.default_rng(args.seed)
        sampled = []
        for date in sorted({r["asof_date"] for r in records}):
            candidates = [r for r in records if r["asof_date"] == date]
            take = min(args.max_per_date, len(candidates))
            sampled.extend(candidates[int(i)] for i in sorted(rng.choice(len(candidates), take, replace=False)))
        records = sampled
    store = WindowStore(panel, manifest["model_contract"]["sector_labels"])
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer).to(device).eval()
    models = {"best_segment_530": Path(args.best_model), "last_segment_534": Path(args.last_model)}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shard_dir = output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for label, model_path in models.items():
        pending_dates = [
            date for date in sorted({r["asof_date"] for r in records})
            if not (shard_dir / f"{label}_{date}.csv.gz").is_file()
        ]
        if not pending_dates:
            print(f"{label}: all date shards already exist", flush=True)
            continue
        model = Kronos.from_pretrained(
            model_path,
            num_sectors=86,
            num_size_buckets=0,
            context_layer=6,
            use_size_percentile=True,
            size_mlp_hidden_dim=64,
        ).to(device).eval()
        for date in pending_dates:
            date_records = [r for r in records if r["asof_date"] == date]
            frame = evaluate_predictions(
                label, model, tokenizer, date_records, store, device,
                args.batch_size, args.sample_count, args.seed, True,
            )
            shard_path = shard_dir / f"{label}_{date}.csv.gz"
            frame.to_csv(shard_path, index=False, compression="gzip")
            print(f"saved {shard_path} ({len(frame):,} rows)", flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frames = [pd.read_csv(path) for path in sorted(shard_dir.glob("*.csv.gz"))]
    if not frames:
        raise RuntimeError("No completed prediction shards")
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    summary = {
        "evaluation_name": "kronos_small_0_1_stage2_time_oos",
        "training_performed": False,
        "device": str(device),
        "manifest": manifest["name"],
        "temporal_isolation": manifest["temporal_isolation"],
        "models": {label: str(path) for label, path in models.items()},
        "sample_count": len(records),
        "metrics": {label: summarize(group) for label, group in predictions.groupby("model")},
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
