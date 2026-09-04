"""Forecast direction accuracy on the matured August evaluation package."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_v1_beta_checkpoints import (
    WindowStore,
    batches,
    evaluate_predictions,
    find_evaluation_root,
    load_model,
    load_samples,
    stack_batch,
)
from model import KronosTokenizer


LABELS = ("short", "neutral", "long")


def install_numpy_pickle_compat():
    """Allow NumPy 1.x to read arrays pickled by NumPy 2.x."""
    if not hasattr(np, "_core"):
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        import numpy.core.numeric as numpy_numeric

        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.multiarray", numpy_multiarray)
        sys.modules.setdefault("numpy._core.numeric", numpy_numeric)
        setattr(np, "_core", numpy_core)


def metrics(frame):
    frame = frame.copy()
    frame["predicted_direction"] = np.where(
        frame.predicted_return_10d < -0.01,
        "short",
        np.where(frame.predicted_return_10d > 0.01, "long", "neutral"),
    )
    confusion = {
        actual: {
            predicted: int(((frame.direction == actual) & (frame.predicted_direction == predicted)).sum())
            for predicted in LABELS
        }
        for actual in LABELS
    }
    precision = {}
    recall = {}
    for label in LABELS:
        tp = confusion[label][label]
        predicted_total = sum(confusion[actual][label] for actual in LABELS)
        actual_total = sum(confusion[label].values())
        precision[label] = tp / predicted_total if predicted_total else None
        recall[label] = tp / actual_total if actual_total else None
    actual_counts = frame["direction"].value_counts().reindex(LABELS, fill_value=0)
    predicted_counts = (
        frame["predicted_direction"].value_counts().reindex(LABELS, fill_value=0)
    )
    by_date = []
    for date, rows in frame.groupby("asof_date", sort=True):
        by_date.append({
            "asof_date": date,
            "samples": int(len(rows)),
            "direction_accuracy": float((rows.predicted_direction == rows.direction).mean()),
            "balanced_accuracy": float(np.mean([
                float((rows.loc[rows.direction == label, "predicted_direction"] == label).mean())
                if (rows.direction == label).any() else 0.0
                for label in LABELS
            ])),
        })
    return {
        "samples": int(len(frame)),
        "signal_dates": int(frame.asof_date.nunique()),
        "direction_accuracy": float((frame.predicted_direction == frame.direction).mean()),
        "balanced_accuracy": float(np.mean([v for v in recall.values() if v is not None])),
        "actual_distribution": {
            label: {
                "count": int(actual_counts[label]),
                "fraction": float(actual_counts[label] / len(frame)),
            }
            for label in LABELS
        },
        "predicted_distribution": {
            label: {
                "count": int(predicted_counts[label]),
                "fraction": float(predicted_counts[label] / len(frame)),
            }
            for label in LABELS
        },
        "precision": precision,
        "recall": recall,
        "confusion_matrix_actual_rows": confusion,
        "by_signal_date": by_date,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--candidate-label", default="beta_v2_0_last")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Direction evaluation requires CUDA")
    root, manifest = find_evaluation_root(Path(args.evaluation_root))
    install_numpy_pickle_compat()
    with (root / manifest["artifacts"]["panel_file"]).open("rb") as handle:
        import pickle
        panel = pickle.load(handle)
    groups = load_samples(root / manifest["artifacts"]["samples_file"])
    records = groups["future_all"]
    store = WindowStore(panel, manifest["model_contract"]["sector_labels"])
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer).to(device).eval()
    model_paths = {
        "beta_v1_1_best818": Path(args.baseline_model),
        args.candidate_label: Path(args.candidate_model),
    }
    frames = []
    for label, path in model_paths.items():
        model = load_model(path, device)
        frame = evaluate_predictions(
            label, model, tokenizer, records, store, device,
            args.batch_size, args.sample_count, args.seed, True,
        )
        frames.append(frame)
        del model
        torch.cuda.empty_cache()
    all_rows = pd.concat(frames, ignore_index=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(output / "august_direction_predictions.csv.gz", index=False, compression="gzip")
    summary = {
        "schema_version": 1,
        "evaluation_role": "matured_august_time_oos_direction_accuracy",
        "sample_count": len(records),
        "signal_start": min(r["asof_date"] for r in records),
        "signal_end": max(r["asof_date"] for r in records),
        "target_start": min(r["target_date"] for r in records),
        "target_end": max(r["target_date"] for r in records),
        "direction_definition": "short return_10d < -1%; neutral -1% <= return_10d <= 1%; long return_10d > 1%",
        "forecast_definition": "predicted_close_d10 / last_close - 1",
        "models": {label: {"path": str(path)} for label, path in model_paths.items()},
        "metrics": {label: metrics(frame) for label, frame in zip(model_paths, frames)},
    }
    base = frames[0].copy(); cand = frames[1].copy()
    merged = base[["identity", "asof_date", "direction"]].copy()
    merged["baseline_predicted_direction"] = np.where(base.predicted_return_10d < -0.01, "short", np.where(base.predicted_return_10d > 0.01, "long", "neutral"))
    merged["candidate_predicted_direction"] = np.where(cand.predicted_return_10d < -0.01, "short", np.where(cand.predicted_return_10d > 0.01, "long", "neutral"))
    summary["paired"] = {
        "candidate_minus_baseline_accuracy": float((merged.candidate_predicted_direction == merged.direction).mean() - (merged.baseline_predicted_direction == merged.direction).mean()),
        "candidate_wins": int(((merged.candidate_predicted_direction == merged.direction) & (merged.baseline_predicted_direction != merged.direction)).sum()),
        "baseline_wins": int(((merged.baseline_predicted_direction == merged.direction) & (merged.candidate_predicted_direction != merged.direction)).sum()),
        "ties": int(((merged.candidate_predicted_direction == merged.direction) == (merged.baseline_predicted_direction == merged.direction)).sum()),
    }
    (output / "direction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
