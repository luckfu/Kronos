"""Compare Last1058, Best@185, and Last@528 without updating weights."""

import argparse
import gc
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from finetune.evaluate_v1_beta_checkpoints import (
    WindowStore,
    evaluate_predictions,
    evaluate_token_sets,
    load_model,
    load_samples,
    prediction_summary,
    sample_identity,
    sha256_file,
    token_summary,
)
from model import KronosTokenizer


FUTURE_NATURAL_SAMPLES = 3000


def load_evaluation(root):
    root = Path(root)
    manifest_path = root / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("name") != "kronos_v1_beta_checkpoint_evaluation_20260826":
        raise RuntimeError(f"Unexpected evaluation manifest: {manifest.get('name')}")
    if not manifest["temporal_isolation"].get("strictly_after_parent_latest_training_target"):
        raise RuntimeError("Evaluation set is not temporally isolated")
    for file_key, hash_key in (("panel_file", "panel_sha256"), ("samples_file", "samples_sha256")):
        path = root / manifest["artifacts"][file_key]
        if sha256_file(path) != manifest["artifacts"][hash_key]:
            raise RuntimeError(f"Evaluation artifact SHA mismatch: {path}")
    return manifest


def date_balanced_sample(records, count, seed):
    by_date = {}
    for record in records:
        by_date.setdefault(record["asof_date"], []).append(record)
    dates = sorted(by_date)
    base, remainder = divmod(count, len(dates))
    extra = set(np.random.default_rng(seed).choice(dates, size=remainder, replace=False)) if remainder else set()
    selected = []
    for offset, date in enumerate(dates):
        quota = base + int(date in extra)
        candidates = by_date[date]
        if quota > len(candidates):
            raise RuntimeError(f"Future quota exceeds candidates on {date}")
        indexes = np.random.default_rng(seed + offset).choice(len(candidates), quota, replace=False)
        selected.extend(candidates[int(index)] for index in sorted(indexes))
    return selected


def paired_token_deltas(token_rows, reference="last1058"):
    pivot = token_rows.pivot_table(
        index=["set", "identity", "asof_date"], columns="model", values="objective_loss"
    ).dropna().reset_index()
    results = {}
    for challenger in sorted(set(pivot.columns) - {"set", "identity", "asof_date", reference}):
        by_set = {}
        for set_name, rows in pivot.groupby("set"):
            difference = rows[challenger] - rows[reference]
            date_means = difference.groupby(rows["asof_date"]).mean()
            by_set[set_name] = {
                "definition": f"{challenger} minus {reference}; negative favors challenger",
                "mean_sample_difference": float(difference.mean()),
                "mean_date_difference": float(date_means.mean()),
                "date_count": int(len(date_means)),
            }
        results[challenger] = by_set
    return results


def max_drawdown(returns):
    equity = np.concatenate(([1.0], 1.0 + np.asarray(returns, dtype=float)))
    return float(np.min(equity / np.maximum.accumulate(equity) - 1.0))


def path_summary(frame):
    horizons = range(1, 11)
    per_horizon = []
    for horizon in horizons:
        predicted = frame[f"predicted_return_d{horizon}"]
        actual = frame[f"actual_return_d{horizon}"]
        per_horizon.append({
            "horizon_day": horizon,
            "rank_ic": float(predicted.corr(actual, method="spearman")),
            "return_mae": float(np.abs(predicted - actual).mean()),
            "direction_accuracy": float((np.sign(predicted) == np.sign(actual)).mean()),
        })
    predicted_path = frame[[f"predicted_return_d{horizon}" for horizon in horizons]].to_numpy()
    actual_path = frame[[f"actual_return_d{horizon}" for horizon in horizons]].to_numpy()
    predicted_mdd = np.apply_along_axis(max_drawdown, 1, predicted_path)
    actual_mdd = np.apply_along_axis(max_drawdown, 1, actual_path)
    ranked = frame.sort_values("predicted_return_d10", ascending=False)
    tail = max(1, len(ranked) // 5)
    portfolio_paths = {}
    for name, selected in (("top_quintile", ranked.head(tail)), ("bottom_quintile", ranked.tail(tail))):
        returns = np.array([selected[f"actual_return_d{horizon}"].mean() for horizon in horizons])
        portfolio_paths[name] = {
            "daily_mean_return": returns.tolist(),
            "terminal_return": float(returns[-1]),
            "max_drawdown": max_drawdown(returns),
        }
    return {
        "path_return_mae": float(np.abs(predicted_path - actual_path).mean()),
        "path_max_drawdown_mae": float(np.abs(predicted_mdd - actual_mdd).mean()),
        "per_horizon": per_horizon,
        "selected_portfolio_paths": portfolio_paths,
    }


def run_evaluation(evaluation_root, output_dir, tokenizer_path, last1058, best, last,
                   batch_size=64, forecast_batch_size=64, sample_count=3, seed=20260827):
    evaluation_root = Path(evaluation_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_evaluation(evaluation_root)
    model_paths = {"last1058": Path(last1058), "best185": Path(best), "last528": Path(last)}
    for label, path in model_paths.items():
        if not (path / "model.safetensors").is_file() or not (path / "config.json").is_file():
            raise RuntimeError(f"Incomplete {label} model directory: {path}")
    hashes = {label: sha256_file(path / "model.safetensors") for label, path in model_paths.items()}
    with (evaluation_root / manifest["artifacts"]["panel_file"]).open("rb") as handle:
        panel = pickle.load(handle)
    groups = load_samples(evaluation_root / manifest["artifacts"]["samples_file"])
    future_natural = date_balanced_sample(groups["future_all"], FUTURE_NATURAL_SAMPLES, seed)
    groups["future_all"] = list({sample_identity(row): row for row in future_natural + groups["future_balanced"]}.values())
    store = WindowStore(panel, manifest["model_contract"]["sector_labels"])
    balanced_ids = {sample_identity(record) for record in groups["future_balanced"]}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Formal checkpoint evaluation requires CUDA")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path).to(device).eval()
    print(json.dumps({"task": "evaluation_only", "device": str(device), "models": {k: str(v) for k, v in model_paths.items()}, "model_hashes": hashes, "sample_sets": {k: len(v) for k, v in groups.items()}}, ensure_ascii=False, indent=2), flush=True)
    token_frames, prediction_frames = [], []
    forecast_sets = {"future_natural": future_natural, "future_balanced": groups["future_balanced"]}
    for label, path in model_paths.items():
        print(f"Loading {label}: {path}", flush=True)
        model = load_model(path, device)
        token_frames.append(evaluate_token_sets(label, model, tokenizer, groups, store, device, batch_size, True))
        for set_name, records in forecast_sets.items():
            frame = evaluate_predictions(label, model, tokenizer, records, store, device, forecast_batch_size, sample_count, seed, True)
            frame["set"] = set_name
            prediction_frames.append(frame)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    token_rows = pd.concat(token_frames, ignore_index=True)
    prediction_rows = pd.concat(prediction_frames, ignore_index=True)
    token_rows.to_csv(output / "token_losses.csv.gz", index=False, compression="gzip")
    prediction_rows.to_csv(output / "future_predictions.csv.gz", index=False, compression="gzip")
    summary = {
        "schema_version": 1,
        "evaluation_name": "kronos_v1_beta_a800_natural_528_comparison_20260827",
        "training_performed": False,
        "configuration": {"seed": seed, "device": str(device), "future_natural_samples": len(future_natural), "future_balanced_samples": len(groups["future_balanced"]), "forecast_sample_count": sample_count, "model_hashes": hashes},
        "models": {key: str(value) for key, value in model_paths.items()},
        "token_metrics": {label: token_summary(rows, balanced_ids) for label, rows in token_rows.groupby("model")},
        "token_deltas_vs_last1058": paired_token_deltas(token_rows),
        "future_forecast_metrics": {set_name: {label: prediction_summary(rows.drop(columns="set")) for label, rows in set_rows.groupby("model")} for set_name, set_rows in prediction_rows.groupby("set")},
        "future_path_metrics": {set_name: {label: path_summary(rows.drop(columns="set")) for label, rows in set_rows.groupby("model")} for set_name, set_rows in prediction_rows.groupby("set")},
        "interpretation": {"primary_sets": ["future_natural", "future_balanced"], "primary_reason": "Future targets are strictly later than the training target cutoff.", "limitation": "The future sample has six signal dates; results are comparative evidence, not a new sealed final test after model selection."},
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    shutil.copy2(evaluation_root / "evaluation_manifest.json", output)
    (output / "experiment_manifest.json").write_text(json.dumps({"task": "evaluation_only", "training_performed": False, "models": summary["models"], "model_hashes": hashes, "output_files": ["summary.json", "evaluation_manifest.json", "token_losses.csv.gz", "future_predictions.csv.gz"]}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--last1058", required=True)
    parser.add_argument("--best", required=True)
    parser.add_argument("--last", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--forecast-batch-size", type=int, default=64)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    options = vars(args)
    options["tokenizer_path"] = options.pop("tokenizer")
    run_evaluation(**options)


if __name__ == "__main__":
    main()
