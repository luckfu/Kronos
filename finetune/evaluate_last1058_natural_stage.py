"""Evaluation-only comparison of Last1058 and the natural-validation stage."""

import gc
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from finetune.evaluate_v1_beta_checkpoints import (
    EXPECTED_BEST_SEGMENT,
    EXPECTED_EVALUATION_NAME,
    EXPECTED_LAST_SEGMENT,
    WindowStore,
    evaluate_predictions,
    evaluate_token_sets,
    find_evaluation_root,
    load_model,
    load_samples,
    prediction_summary,
    sample_identity,
    sha256_file,
    token_summary,
)
from model import KronosTokenizer


NATURAL_STAGE_ID = "v1_beta_last1058_natural_validation_v1"
FUTURE_NATURAL_SAMPLES = 3000


def find_roots(input_root):
    parent = None
    natural = None
    for manifest_path in Path(input_root).glob("**/experiment_manifest.json"):
        root = manifest_path.parent
        try:
            manifest = json.loads(manifest_path.read_text())
            progress = json.loads((root / "progress.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not (root / "checkpoints/last_model/model.safetensors").is_file():
            continue
        if manifest.get("stage_id") == NATURAL_STAGE_ID:
            if int(progress.get("current_segment", -1)) != 150:
                raise RuntimeError(f"Natural stage did not finish 150 segments: {progress}")
            metric_path = root / "checkpoints/best_model/best_metric.json"
            if not metric_path.is_file():
                raise RuntimeError("Natural stage Best metric is missing")
            natural = (root, json.loads(metric_path.read_text()), progress)
            continue
        metric_path = root / "checkpoints/best_model/best_metric.json"
        if not metric_path.is_file():
            continue
        metric = json.loads(metric_path.read_text())
        if (
            int(metric.get("segment", -1)) == EXPECTED_BEST_SEGMENT
            and int(progress.get("current_segment", -1)) == EXPECTED_LAST_SEGMENT
        ):
            parent = (root, metric, progress)
    if parent is None or natural is None:
        raise RuntimeError(
            "Expected exactly one Last1058 parent and one completed natural stage; "
            f"parent={parent is not None}, natural={natural is not None}"
        )
    return parent, natural


def date_balanced_sample(records, count, seed):
    by_date = {}
    for record in records:
        by_date.setdefault(record["asof_date"], []).append(record)
    dates = sorted(by_date)
    base, remainder = divmod(count, len(dates))
    rng = np.random.default_rng(seed)
    extra_dates = set(rng.choice(dates, size=remainder, replace=False)) if remainder else set()
    selected = []
    for offset, date in enumerate(dates):
        records_for_date = by_date[date]
        quota = base + int(date in extra_dates)
        if quota > len(records_for_date):
            raise RuntimeError(f"Future quota exceeds candidates on {date}")
        positions = np.random.default_rng(seed + offset).choice(
            len(records_for_date), quota, replace=False
        )
        selected.extend(records_for_date[int(index)] for index in sorted(positions))
    return selected


def pairwise_token_deltas(token_rows):
    objective = token_rows.pivot_table(
        index=["set", "identity", "asof_date"], columns="model", values="objective_loss"
    ).dropna().reset_index()
    comparisons = {}
    for challenger in ("natural_best21", "natural_last150"):
        result = {}
        for set_name, rows in objective.groupby("set"):
            difference = rows[challenger] - rows["last1058"]
            date_means = difference.groupby(rows["asof_date"]).mean()
            result[set_name] = {
                "definition": f"{challenger} minus last1058; negative favors challenger",
                "mean_sample_difference": float(difference.mean()),
                "mean_date_difference": float(date_means.mean()),
                "date_count": int(len(date_means)),
            }
        comparisons[challenger] = result
    return comparisons


def forecast_set_summary(predictions):
    return {
        set_name: {
            model: prediction_summary(rows.drop(columns="set"))
            for model, rows in set_rows.groupby("model")
        }
        for set_name, set_rows in predictions.groupby("set")
    }


def run_evaluation(
    input_root, output_dir, tokenizer_path="NeoQuasar/Kronos-Tokenizer-base",
    batch_size=64, forecast_batch_size=64, sample_count=3, seed=20260827,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evaluation_root, evaluation_manifest = find_evaluation_root(input_root)
    (parent_root, parent_best, parent_progress), (natural_root, natural_best, natural_progress) = find_roots(input_root)
    if int(natural_best.get("segment", -1)) != 21:
        raise RuntimeError(f"Expected natural stage Best@21, got {natural_best}")

    with (evaluation_root / evaluation_manifest["artifacts"]["panel_file"]).open("rb") as handle:
        panel = pickle.load(handle)
    groups = load_samples(evaluation_root / evaluation_manifest["artifacts"]["samples_file"])
    future_natural = date_balanced_sample(
        groups["future_all"], FUTURE_NATURAL_SAMPLES, seed
    )
    # Include the balanced future sample in token evaluation so its subset metrics
    # are exact, while retaining a natural-proportion sample for forecasting.
    future_union = {
        sample_identity(record): record
        for record in future_natural + groups["future_balanced"]
    }
    groups["future_all"] = list(future_union.values())
    store = WindowStore(panel, evaluation_manifest["model_contract"]["sector_labels"])
    future_balanced_ids = {sample_identity(record) for record in groups["future_balanced"]}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Formal stage evaluation requires CUDA")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path).to(device).eval()
    model_paths = {
        "last1058": parent_root / "checkpoints/last_model",
        "natural_best21": natural_root / "checkpoints/best_model",
        "natural_last150": natural_root / "checkpoints/last_model",
    }
    hashes = {label: sha256_file(path / "model.safetensors") for label, path in model_paths.items()}
    print(json.dumps({
        "task": "evaluation_only", "device": str(device), "sample_sets": {name: len(rows) for name, rows in groups.items()},
        "parent": {"best": parent_best, "progress": parent_progress},
        "natural_stage": {"best": natural_best, "progress": natural_progress},
        "model_hashes": hashes,
    }, ensure_ascii=False, indent=2), flush=True)

    token_frames = []
    prediction_frames = []
    forecast_sets = {
        "future_natural": future_natural,
        "future_balanced": groups["future_balanced"],
    }
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
        "evaluation_name": "kronos_v1_beta_last1058_natural_stage_comparison_20260827",
        "training_performed": False,
        "configuration": {
            "future_natural_samples": FUTURE_NATURAL_SAMPLES,
            "future_balanced_samples": len(groups["future_balanced"]),
            "forecast_sample_count": sample_count, "seed": seed, "device": str(device), "model_hashes": hashes,
        },
        "lineage": {"parent_best": parent_best, "parent_progress": parent_progress, "natural_best": natural_best, "natural_progress": natural_progress},
        "token_metrics": {label: token_summary(rows, future_balanced_ids) for label, rows in token_rows.groupby("model")},
        "token_deltas_vs_last1058": pairwise_token_deltas(token_rows),
        "future_forecast_metrics": forecast_set_summary(prediction_rows),
        "interpretation": {
            "primary_sets": ["future_natural", "future_balanced"],
            "primary_reason": "Both use 2026-08 samples strictly after the original parent training targets.",
            "historical_sets": ["historical_natural", "historical_balanced"],
            "limitation": "Only six future signal dates are available; use outcomes as directional evidence, not deployment proof.",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    shutil.copy2(evaluation_root / "evaluation_manifest.json", output)
    (output / "experiment_manifest.json").write_text(json.dumps({
        "task": "evaluation_only", "training_performed": False, "models": {key: str(value) for key, value in model_paths.items()},
        "output_files": ["summary.json", "evaluation_manifest.json", "token_losses.csv.gz", "future_predictions.csv.gz"],
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary
