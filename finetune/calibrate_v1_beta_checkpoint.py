"""Calibrate Best467 decoding on a fixed historical validation subset."""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model.kronos import auto_regressive_inference
from finetune.evaluate_v1_beta_checkpoints import (
    FEATURES,
    LOOKBACK,
    PREDICT,
    WindowStore,
    batches,
    find_evaluation_root,
    find_parent_root,
    load_model,
    load_samples,
    sample_identity,
    stack_batch,
)


def choose_records(records, date_count=8, per_direction=20, seed=20260827):
    by_date = defaultdict(list)
    for record in records:
        by_date[record["asof_date"]].append(record)
    dates = []
    for date in sorted(by_date):
        counts = defaultdict(int)
        for record in by_date[date]:
            counts[record["direction"]] += 1
        if all(counts[name] >= per_direction for name in ("short", "neutral", "long")):
            dates.append(date)
    if len(dates) < date_count:
        raise RuntimeError(
            f"Only {len(dates)} historical dates have all direction coverage; "
            f"need {date_count}"
        )
    date_positions = np.linspace(0, len(dates) - 1, date_count).round().astype(int)
    selected_dates = [dates[int(position)] for position in date_positions]
    selected = []
    for date_index, date in enumerate(selected_dates):
        groups = defaultdict(list)
        for record in by_date[date]:
            groups[record["direction"]].append(record)
        for direction_index, direction in enumerate(("short", "neutral", "long")):
            rng = np.random.default_rng(seed + date_index * 101 + direction_index)
            positions = rng.choice(len(groups[direction]), per_direction, replace=False)
            selected.extend(groups[direction][int(position)] for position in positions)
    return sorted(selected, key=sample_identity), selected_dates


def per_sample_cross_entropy(logits, targets):
    return F.cross_entropy(logits.transpose(1, 2), targets, reduction="none").mean(1)


def teacher_forcing_metrics(model, tokenizer, records, store, device, batch_size):
    sums = defaultdict(float)
    count = 0
    model.eval()
    with torch.no_grad():
        for items in batches(records, store, batch_size):
            batch = stack_batch(items, device)
            encoded = tokenizer.encode(batch["x"], half=True)
            token_in = [encoded[0][:, :-1], encoded[1][:, :-1]]
            token_out = [encoded[0][:, 1:], encoded[1][:, 1:]]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(
                    token_in[0], token_in[1], batch["stamp"][:, :-1],
                    sector_id=batch["sector"],
                    size_percentile=batch["percentile"],
                    use_teacher_forcing=True,
                    s1_targets=token_out[0],
                )
            history = slice(0, LOOKBACK - 1)
            forecast = slice(LOOKBACK - 1, LOOKBACK - 1 + PREDICT)

            def loss(selected):
                return (
                    per_sample_cross_entropy(logits[0][:, selected], token_out[0][:, selected])
                    + per_sample_cross_entropy(logits[1][:, selected], token_out[1][:, selected])
                ) / 2

            history_loss = loss(history)
            forecast_loss = loss(forecast)
            full_loss = loss(slice(None))
            for key, value in (
                ("objective_loss", forecast_loss + 0.02 * history_loss),
                ("forecast_loss", forecast_loss),
                ("history_loss", history_loss),
                ("full_loss", full_loss),
            ):
                sums[key] += float(value.sum().item())
            count += len(items)
    return {key: value / count for key, value in sums.items()} | {"samples": count}


def direction(values):
    values = np.asarray(values)
    return np.where(values < -0.01, "short", np.where(values > 0.01, "long", "neutral"))


def balanced_accuracy(actual, predicted):
    recalls = []
    for label in ("short", "neutral", "long"):
        mask = actual == label
        recalls.append(float(np.mean(predicted[mask] == label)) if mask.any() else 0.0)
    return float(np.mean(recalls))


def score_predictions(frame, score_column):
    rows = frame.copy()
    scores = rows[score_column].to_numpy(dtype=float)
    actual = rows["direction"].to_numpy()
    predicted = direction(scores)
    ranked = rows.sort_values(score_column, ascending=False)
    tail = max(1, len(ranked) // 5)
    spread = float(
        ranked.head(tail)["return_10d"].mean()
        - ranked.tail(tail)["return_10d"].mean()
    )
    per_date = []
    for date, group in rows.groupby("asof_date", sort=True):
        date_scores = group[score_column].to_numpy(dtype=float)
        date_actual = group["return_10d"].to_numpy(dtype=float)
        date_ranked = group.sort_values(score_column, ascending=False)
        date_tail = max(1, len(date_ranked) // 5)
        per_date.append({
            "asof_date": str(date),
            "rank_ic": float(pd.Series(date_scores).corr(pd.Series(date_actual), method="spearman")),
            "direction_accuracy": float(np.mean(direction(date_scores) == group["direction"].to_numpy())),
            "top_bottom_spread": float(
                date_ranked.head(date_tail)["return_10d"].mean()
                - date_ranked.tail(date_tail)["return_10d"].mean()
            ),
        })
    return {
        "score": score_column,
        "samples": len(rows),
        "dates": int(rows["asof_date"].nunique()),
        "predicted_mean": float(np.mean(scores)),
        "predicted_std": float(np.std(scores)),
        "predicted_direction_fractions": {
            key: float(value)
            for key, value in pd.Series(predicted).value_counts(normalize=True).to_dict().items()
        },
        "direction_accuracy": float(np.mean(predicted == actual)),
        "balanced_accuracy": balanced_accuracy(actual, predicted),
        "return_mae": float(np.mean(np.abs(scores - rows["return_10d"].to_numpy(dtype=float)))),
        "rank_ic": float(rows[score_column].corr(rows["return_10d"], method="spearman")),
        "top_bottom_spread": spread,
        "per_date": per_date,
    }


def forecast_grid(model, tokenizer, records, store, device, configs, batch_size, seed):
    all_results = {}
    by_date = defaultdict(list)
    for record in records:
        by_date[record["asof_date"]].append(record)
    for config_index, config in enumerate(configs):
        rows = []
        for date_index, (asof_date, date_records) in enumerate(sorted(by_date.items())):
            torch.manual_seed(seed + config_index * 1000 + date_index)
            np.random.seed(seed + config_index * 1000 + date_index)
            for items in batches(date_records, store, batch_size):
                batch = stack_batch(items, device)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    forecast = auto_regressive_inference(
                        tokenizer,
                        model,
                        batch["x"][:, :LOOKBACK],
                        batch["stamp"][:, :LOOKBACK],
                        batch["stamp"][:, LOOKBACK:LOOKBACK + PREDICT],
                        max_context=512,
                        pred_len=PREDICT,
                        clip=5,
                        T=config["temperature"],
                        top_k=0,
                        top_p=config["top_p"],
                        sample_count=config["sample_count"],
                        verbose=False,
                        sector_id=batch["sector"],
                        size_percentile=batch["percentile"],
                    )
                close_index = FEATURES.index("close")
                normalized_close = forecast[:, -PREDICT:, close_index]
                for index, item in enumerate(items):
                    predicted = normalized_close[index] * (item["std"][close_index] + 1e-5) + item["mean"][close_index]
                    last_close = float(item["x"][LOOKBACK - 1, close_index] * (item["std"][close_index] + 1e-5) + item["mean"][close_index])
                    rows.append({
                        "identity": item["identity"],
                        "asof_date": item["asof_date"],
                        "direction": item["direction"],
                        "return_10d": item["return_10d"],
                        "predicted_return_10d": float(predicted[-1] / last_close - 1),
                        "predicted_mean_horizon_return": float(predicted.mean() / last_close - 1),
                    })
            print(f"calibration {config_index + 1}/{len(configs)} {asof_date}: {len(date_records)}", flush=True)
        frame = pd.DataFrame(rows)
        frame["daily_median_centered"] = frame["predicted_return_10d"] - frame.groupby("asof_date")["predicted_return_10d"].transform("median")
        frame["daily_mean_centered"] = frame["predicted_return_10d"] - frame.groupby("asof_date")["predicted_return_10d"].transform("mean")
        key = f"T{config['temperature']:g}_p{config['top_p']:g}_n{config['sample_count']}"
        all_results[key] = {
            "configuration": config,
            "raw": score_predictions(frame, "predicted_return_10d"),
            "daily_median_centered": score_predictions(frame, "daily_median_centered"),
            "daily_mean_centered": score_predictions(frame, "daily_mean_centered"),
        }
    return all_results


def run_calibration(
    input_root,
    output_dir,
    batch_size=64,
    seed=20260827,
    model_label="best467",
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evaluation_root, manifest = find_evaluation_root(Path(input_root))
    parent_root, best_metric, progress = find_parent_root(Path(input_root))
    with (evaluation_root / manifest["artifacts"]["panel_file"]).open("rb") as handle:
        panel = __import__("pickle").load(handle)
    groups = load_samples(evaluation_root / manifest["artifacts"]["samples_file"])
    records, dates = choose_records(groups["historical_balanced"], seed=seed)
    store = WindowStore(panel, manifest["model_contract"]["sector_labels"])
    device = torch.device("cuda:0")
    tokenizer = __import__("model").KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").to(device).eval()
    model_paths = {
        "best467": parent_root / "checkpoints/best_model",
        "last1058": parent_root / "checkpoints/last_model",
    }
    if model_label not in model_paths:
        raise ValueError(f"Unsupported calibration model: {model_label}")
    model_path = model_paths[model_label]
    model = load_model(model_path, device)
    configs = [
        {"temperature": temperature, "top_p": top_p, "sample_count": 3}
        for temperature in (0.4, 0.6, 0.8)
        for top_p in (0.7, 0.9)
    ]
    summary = {
        "evaluation_name": f"kronos_v1_beta_{model_label}_decoder_calibration_20260827",
        "training_performed": False,
        "model": model_label,
        "model_sha256": __import__("hashlib").sha256((model_path / "model.safetensors").read_bytes()).hexdigest(),
        "parent_completed_segment": int(progress["current_segment"]),
        "best_metric": best_metric,
        "sample_dates": dates,
        "sample_count": len(records),
        "teacher_forcing": teacher_forcing_metrics(model, tokenizer, records, store, device, batch_size),
        "configs": configs,
        "forecast_results": forecast_grid(model, tokenizer, records, store, device, configs, batch_size, seed),
        "interpretation": {
            "calibration_only": True,
            "future_set_not_used_for_selection": True,
            "daily_centering_uses_only_cross_sectional_predictions": True,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (output / "calibration_manifest.json").write_text(json.dumps({
        "task": "decoder_calibration",
        "training_performed": False,
        "model": model_label,
        "sample_dates": dates,
        "sample_count": len(records),
        "configs": configs,
    }, ensure_ascii=False, indent=2) + "\n")
    return summary
