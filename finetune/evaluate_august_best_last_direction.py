"""Evaluate Beta v2.1 Best and Last on the signed August matured OOS package."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_v1_beta_checkpoints import (
    WindowStore,
    evaluate_predictions,
    load_model,
    load_samples,
)
from model import KronosTokenizer


LABELS = ("short", "neutral", "long")


def metrics(frame):
    frame = frame.copy()
    frame["predicted_direction"] = np.where(
        frame.predicted_return_10d < -0.01, "short",
        np.where(frame.predicted_return_10d > 0.01, "long", "neutral"),
    )
    confusion = {
        actual: {predicted: int(((frame.direction == actual) &
                                 (frame.predicted_direction == predicted)).sum())
                 for predicted in LABELS}
        for actual in LABELS
    }
    precision, recall = {}, {}
    for label in LABELS:
        tp = confusion[label][label]
        precision[label] = tp / sum(confusion[a][label] for a in LABELS) if sum(confusion[a][label] for a in LABELS) else None
        recall[label] = tp / sum(confusion[label].values()) if sum(confusion[label].values()) else None
    per_date = []
    for date, rows in frame.groupby("asof_date", sort=True):
        ranked = rows.sort_values("predicted_return_10d", ascending=False)
        tail = max(1, len(ranked) // 5)
        ic = rows["predicted_return_10d"].corr(rows["return_10d"], method="spearman")
        per_date.append({
            "asof_date": date,
            "samples": int(len(rows)),
            "direction_accuracy": float((rows.predicted_direction == rows.direction).mean()),
            "rank_ic": None if pd.isna(ic) else float(ic),
            "return_mae": float(np.abs(rows.predicted_return_10d - rows.return_10d).mean()),
            "top_bottom_spread": float(ranked.head(tail).return_10d.mean() - ranked.tail(tail).return_10d.mean()),
        })
    date_df = pd.DataFrame(per_date)
    ic = date_df.rank_ic.dropna()
    actual_counts = frame.direction.value_counts().reindex(LABELS, fill_value=0)
    predicted_counts = frame.predicted_direction.value_counts().reindex(LABELS, fill_value=0)
    return {
        "samples": int(len(frame)), "signal_dates": int(frame.asof_date.nunique()),
        "direction_accuracy": float((frame.predicted_direction == frame.direction).mean()),
        "balanced_accuracy": float(np.mean(list(recall.values()))),
        "precision": precision, "recall": recall,
        "actual_distribution": {x: {"count": int(actual_counts[x]), "fraction": float(actual_counts[x] / len(frame))} for x in LABELS},
        "predicted_distribution": {x: {"count": int(predicted_counts[x]), "fraction": float(predicted_counts[x] / len(frame))} for x in LABELS},
        "confusion_matrix_actual_rows": confusion,
        "mean_rank_ic": float(ic.mean()) if len(ic) else None,
        "rank_ic_std": float(ic.std(ddof=1)) if len(ic) > 1 else None,
        "rank_icir": float(ic.mean() / ic.std(ddof=1)) if len(ic) > 1 and ic.std(ddof=1) else None,
        "rank_ic_positive_rate": float((ic > 0).mean()) if len(ic) else None,
        "return_mae": float(np.abs(frame.predicted_return_10d - frame.return_10d).mean()),
        "mean_top_bottom_spread": float(date_df.top_bottom_spread.mean()),
        "by_signal_date": per_date,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation-root", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--best-model", required=True)
    p.add_argument("--last-model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")
    root = Path(args.evaluation_root)
    manifest = json.loads((root / "evaluation_manifest.json").read_text())
    if manifest.get("name") != "kronos_beta_v2_august_2026_full_matured_time_oos":
        raise RuntimeError(f"Unexpected manifest: {manifest.get('name')}")
    import pickle
    with (root / manifest["artifacts"]["panel_file"]).open("rb") as f:
        panel = pickle.load(f)
    records = load_samples(root / manifest["artifacts"]["samples_file"])["future_all"]
    store = WindowStore(panel, manifest["model_contract"]["sector_labels"])
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer).to(device).eval()
    models = [("best_segment_475", Path(args.best_model)), ("last_segment_673", Path(args.last_model))]
    frames = []
    for label, path in models:
        model = load_model(path, device)
        frames.append(evaluate_predictions(label, model, tokenizer, records, store, device, args.batch_size, 1, 20260902, True))
        del model
        torch.cuda.empty_cache()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(output / "august_best_last_direction_predictions.csv.gz", index=False, compression="gzip")
    result = {"manifest": manifest["name"], "samples": len(records), "signal_start": min(r["asof_date"] for r in records), "signal_end": max(r["asof_date"] for r in records), "target_end": max(r["target_date"] for r in records), "models": {label: {"path": str(path), "metrics": metrics(frame)} for (label, path), frame in zip(models, frames)}}
    (output / "august_best_last_direction_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
