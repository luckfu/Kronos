"""Evaluate v1-beta Best467 and Last1058 without training or model transfer."""

import argparse
import gc
import hashlib
import json
import math
import pickle
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import Kronos, KronosTokenizer
from model.kronos import auto_regressive_inference


FEATURES = ["open", "high", "low", "close", "volume", "amount"]
LOOKBACK = 120
PREDICT = 10
WINDOW = LOOKBACK + PREDICT + 1
EXPECTED_BEST_SEGMENT = 467
EXPECTED_LAST_SEGMENT = 1058
EXPECTED_EVALUATION_NAME = "kronos_v1_beta_checkpoint_evaluation_20260826"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_identity(record):
    return "|".join(
        (
            str(record["symbol"]),
            str(int(record["start_index"])),
            str(record["asof_date"]),
            str(record["target_date"]),
        )
    )


def time_features(index):
    index = pd.DatetimeIndex(index)
    return np.column_stack(
        [index.minute, index.hour, index.weekday, index.day, index.month]
    ).astype(np.float32)


def find_evaluation_root(input_root):
    matches = []
    for manifest_path in Path(input_root).glob("**/evaluation_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("name") == EXPECTED_EVALUATION_NAME:
            matches.append((manifest_path.parent, manifest))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one signed evaluation input, found {len(matches)}")
    root, manifest = matches[0]
    artifacts = manifest["artifacts"]
    for key, sha_key in (("panel_file", "panel_sha256"), ("samples_file", "samples_sha256")):
        path = root / artifacts[key]
        actual = sha256_file(path)
        if actual != artifacts[sha_key]:
            raise RuntimeError(f"Evaluation artifact SHA mismatch: {path}")
    isolation = manifest["temporal_isolation"]
    if not isolation.get("strictly_after_parent_latest_training_target"):
        raise RuntimeError("Future evaluation is not temporally isolated")
    if isolation["future_signal_start"] <= isolation["parent_latest_training_target"]:
        raise RuntimeError("Future signal dates overlap the parent training labels")
    return root, manifest


def find_parent_root(input_root):
    matches = []
    for metric_path in Path(input_root).glob("**/checkpoints/best_model/best_metric.json"):
        root = metric_path.parents[2]
        progress_path = root / "progress.json"
        last_model = root / "checkpoints/last_model/model.safetensors"
        if not progress_path.is_file() or not last_model.is_file():
            continue
        try:
            metric = json.loads(metric_path.read_text())
            progress = json.loads(progress_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            int(metric.get("segment", -1)) == EXPECTED_BEST_SEGMENT
            and int(progress.get("current_segment", -1)) == EXPECTED_LAST_SEGMENT
        ):
            matches.append((root, metric, progress))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Best467/Last1058 parent, found {len(matches)}")
    return matches[0]


def load_samples(path):
    groups = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        groups.pop("", None)
        groups[record.pop("set")].append(record)
    return dict(groups)


class WindowStore:
    def __init__(self, panel, sector_labels):
        self.panel = panel
        self.sector_map = {value: index for index, value in enumerate(sector_labels)}
        observed = sorted(
            {
                str(value)
                for frame in panel.values()
                for value in frame["sector"].dropna().unique()
            }
        )
        if observed != list(sector_labels):
            raise RuntimeError("Evaluation sector vocabulary differs from training")

    def prepare(self, record):
        frame = self.panel[str(record["symbol"])]
        start = int(record["start_index"])
        window = frame.iloc[start : start + WINDOW]
        if len(window) != WINDOW:
            raise RuntimeError(f"Incomplete window: {sample_identity(record)}")
        asof = window.index[LOOKBACK - 1]
        target = window.index[LOOKBACK - 1 + PREDICT]
        if str(asof.date()) != record["asof_date"] or str(target.date()) != record["target_date"]:
            raise RuntimeError(f"Sample identity drift: {sample_identity(record)}")
        values = window[FEATURES].to_numpy(dtype=np.float32)
        mean = values[:LOOKBACK].mean(axis=0)
        std = values[:LOOKBACK].std(axis=0)
        normalized = np.clip((values - mean) / (std + 1e-5), -5, 5).astype(np.float32)
        sector = str(window["sector"].iloc[LOOKBACK - 1])
        percentile = float(window["size_percentile"].iloc[LOOKBACK - 1])
        if not math.isfinite(percentile):
            percentile = 0.5
        return {
            "identity": sample_identity(record),
            "symbol": str(record["symbol"]),
            "asof_date": record["asof_date"],
            "target_date": record["target_date"],
            "direction": record["direction"],
            "sector": sector,
            "size_decile": int(record["size_decile"]),
            "return_10d": float(record["return_10d"]),
            "x": normalized,
            "stamp": time_features(window.index),
            "sector_id": self.sector_map.get(sector, len(self.sector_map)),
            "size_percentile": float(np.clip(percentile, 0.0, 1.0)),
            "mean": mean,
            "std": std,
        }


def batches(records, store, batch_size):
    for offset in range(0, len(records), batch_size):
        yield [store.prepare(record) for record in records[offset : offset + batch_size]]


def stack_batch(items, device):
    return {
        "x": torch.as_tensor(np.stack([item["x"] for item in items]), device=device),
        "stamp": torch.as_tensor(np.stack([item["stamp"] for item in items]), device=device),
        "sector": torch.as_tensor([item["sector_id"] for item in items], device=device),
        "percentile": torch.as_tensor(
            [item["size_percentile"] for item in items], device=device
        ),
    }


def per_sample_cross_entropy(logits, targets):
    return F.cross_entropy(logits.transpose(1, 2), targets, reduction="none").mean(1)


def per_sample_losses(logits, targets):
    history = slice(0, LOOKBACK - 1)
    forecast = slice(LOOKBACK - 1, LOOKBACK - 1 + PREDICT)

    def dual_loss(selected):
        return (
            per_sample_cross_entropy(logits[0][:, selected], targets[0][:, selected])
            + per_sample_cross_entropy(logits[1][:, selected], targets[1][:, selected])
        ) / 2

    history_loss = dual_loss(history)
    forecast_loss = dual_loss(forecast)
    full_loss = dual_loss(slice(None))
    return {
        "objective_loss": forecast_loss + 0.02 * history_loss,
        "forecast_loss": forecast_loss,
        "history_loss": history_loss,
        "full_loss": full_loss,
    }


def forward_losses(model, token_in, token_out, stamp, sector, percentile, use_amp):
    with torch.autocast(
        device_type=stamp.device.type, dtype=torch.float16, enabled=use_amp
    ):
        logits = model(
            token_in[0],
            token_in[1],
            stamp[:, :-1],
            sector_id=sector,
            size_percentile=percentile,
            use_teacher_forcing=True,
            s1_targets=token_out[0],
        )
    return per_sample_losses(logits, token_out)


def evaluate_token_sets(
    label, model, tokenizer, groups, store, device, batch_size, use_amp
):
    rows = []
    evaluated_groups = ("future_all", "historical_natural", "historical_balanced")
    for group_name in evaluated_groups:
        records = groups[group_name]
        for batch_index, items in enumerate(batches(records, store, batch_size), 1):
            batch = stack_batch(items, device)
            with torch.no_grad():
                encoded = tokenizer.encode(batch["x"], half=True)
                token_in = [encoded[0][:, :-1], encoded[1][:, :-1]]
                token_out = [encoded[0][:, 1:], encoded[1][:, 1:]]
                full = forward_losses(
                    model,
                    token_in,
                    token_out,
                    batch["stamp"],
                    batch["sector"],
                    batch["percentile"],
                    use_amp,
                )
                none = shuffled = None
                if group_name == "future_all":
                    none = forward_losses(
                        model,
                        token_in,
                        token_out,
                        batch["stamp"],
                        None,
                        None,
                        use_amp,
                    )
                    shuffled = forward_losses(
                        model,
                        token_in,
                        token_out,
                        batch["stamp"],
                        torch.roll(batch["sector"], 1),
                        torch.roll(batch["percentile"], 1),
                        use_amp,
                    )
            for index, item in enumerate(items):
                row = {
                    "model": label,
                    "set": group_name,
                    **{
                        key: item[key]
                        for key in (
                            "identity",
                            "symbol",
                            "asof_date",
                            "target_date",
                            "direction",
                            "sector",
                            "size_decile",
                            "return_10d",
                        )
                    },
                }
                row.update({key: float(value[index].item()) for key, value in full.items()})
                if none is not None:
                    row["condition_none_forecast_loss"] = float(
                        none["forecast_loss"][index].item()
                    )
                    row["condition_shuffled_forecast_loss"] = float(
                        shuffled["forecast_loss"][index].item()
                    )
                rows.append(row)
            if batch_index % 100 == 0 or batch_index * batch_size >= len(records):
                print(
                    f"{label} token {group_name}: "
                    f"{min(batch_index * batch_size, len(records)):,}/{len(records):,}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def evaluate_predictions(
    label,
    model,
    tokenizer,
    records,
    store,
    device,
    batch_size,
    sample_count,
    seed,
    use_amp,
):
    rows = []
    by_date = defaultdict(list)
    for record in records:
        by_date[record["asof_date"]].append(record)
    for date_index, (asof_date, date_records) in enumerate(sorted(by_date.items())):
        torch.manual_seed(seed + date_index)
        np.random.seed(seed + date_index)
        for items in batches(date_records, store, batch_size):
            batch = stack_batch(items, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                forecast = auto_regressive_inference(
                    tokenizer,
                    model,
                    batch["x"][:, :LOOKBACK],
                    batch["stamp"][:, :LOOKBACK],
                    batch["stamp"][:, LOOKBACK : LOOKBACK + PREDICT],
                    max_context=512,
                    pred_len=PREDICT,
                    clip=5,
                    T=0.6,
                    top_k=0,
                    top_p=0.9,
                    sample_count=sample_count,
                    verbose=False,
                    sector_id=batch["sector"],
                    size_percentile=batch["percentile"],
                )
            close_index = FEATURES.index("close")
            normalized_future = forecast[:, -PREDICT:, close_index]
            for index, item in enumerate(items):
                predicted_closes = (
                    normalized_future[index]
                    * (item["std"][close_index] + 1e-5)
                    + item["mean"][close_index]
                )
                last_close = float(
                    item["x"][LOOKBACK - 1, close_index]
                    * (item["std"][close_index] + 1e-5)
                    + item["mean"][close_index]
                )
                actual_closes = (
                    item["x"][LOOKBACK : LOOKBACK + PREDICT, close_index]
                    * (item["std"][close_index] + 1e-5)
                    + item["mean"][close_index]
                )
                row = {
                        "model": label,
                        **{
                            key: item[key]
                            for key in (
                                "identity",
                                "symbol",
                                "asof_date",
                                "target_date",
                                "direction",
                                "sector",
                                "size_decile",
                                "return_10d",
                            )
                        },
                        "predicted_return_10d": float(predicted_closes[-1] / last_close - 1),
                        "predicted_mean_horizon_return": float(
                            predicted_closes.mean() / last_close - 1
                        ),
                }
                for horizon in range(PREDICT):
                    suffix = horizon + 1
                    row[f"predicted_return_d{suffix}"] = float(
                        predicted_closes[horizon] / last_close - 1
                    )
                    row[f"actual_return_d{suffix}"] = float(
                        actual_closes[horizon] / last_close - 1
                    )
                rows.append(row)
        print(
            f"{label} forecast: {date_index + 1}/{len(by_date)} {asof_date} "
            f"({len(date_records):,} symbols)",
            flush=True,
        )
    return pd.DataFrame(rows)


def three_class(values):
    values = np.asarray(values)
    return np.where(values < -0.01, "short", np.where(values > 0.01, "long", "neutral"))


def prediction_summary(frame):
    frame = frame.copy()
    frame["predicted_direction"] = three_class(frame["predicted_return_10d"])
    per_date = []
    for asof_date, rows in frame.groupby("asof_date"):
        ranked = rows.sort_values("predicted_return_10d", ascending=False)
        tail = max(1, len(ranked) // 5)
        per_date.append(
            {
                "asof_date": asof_date,
                "rank_ic": float(
                    rows["predicted_return_10d"].corr(rows["return_10d"], method="spearman")
                ),
                "direction_accuracy": float(
                    (rows["predicted_direction"] == rows["direction"]).mean()
                ),
                "return_mae": float(
                    np.abs(rows["predicted_return_10d"] - rows["return_10d"]).mean()
                ),
                "top_bottom_spread": float(
                    ranked.head(tail)["return_10d"].mean()
                    - ranked.tail(tail)["return_10d"].mean()
                ),
            }
        )
    per_date = pd.DataFrame(per_date)
    recalls = {}
    for direction in ("short", "neutral", "long"):
        actual = frame["direction"] == direction
        recalls[direction] = float(
            (frame.loc[actual, "predicted_direction"] == direction).mean()
        )
    rank_std = float(per_date["rank_ic"].std(ddof=1))
    return {
        "observations": len(frame),
        "dates": int(frame["asof_date"].nunique()),
        "mean_rank_ic": float(per_date["rank_ic"].mean()),
        "rank_ic_std": rank_std,
        "rank_icir": float(per_date["rank_ic"].mean() / rank_std) if rank_std else None,
        "rank_ic_positive_rate": float((per_date["rank_ic"] > 0).mean()),
        "direction_accuracy": float(
            (frame["predicted_direction"] == frame["direction"]).mean()
        ),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "direction_recall": recalls,
        "return_mae": float(
            np.abs(frame["predicted_return_10d"] - frame["return_10d"]).mean()
        ),
        "predicted_return_std": float(frame["predicted_return_10d"].std()),
        "mean_top_bottom_spread": float(per_date["top_bottom_spread"].mean()),
        "per_date": per_date.to_dict("records"),
    }


def token_summary(frame, balanced_identities):
    result = {}
    for set_name, rows in frame.groupby("set"):
        subsets = {set_name: rows}
        if set_name == "future_all":
            subsets["future_balanced"] = rows[rows["identity"].isin(balanced_identities)]
        for name, selected in subsets.items():
            metrics = {
                key: float(selected[key].mean())
                for key in ("objective_loss", "forecast_loss", "history_loss", "full_loss")
            }
            if "condition_none_forecast_loss" in selected:
                metrics.update(
                    {
                        "condition_none_forecast_loss": float(
                            selected["condition_none_forecast_loss"].mean()
                        ),
                        "condition_shuffled_forecast_loss": float(
                            selected["condition_shuffled_forecast_loss"].mean()
                        ),
                    }
                )
                metrics["condition_full_minus_none_forecast_loss"] = (
                    metrics["forecast_loss"] - metrics["condition_none_forecast_loss"]
                )
                metrics["condition_full_minus_shuffled_forecast_loss"] = (
                    metrics["forecast_loss"]
                    - metrics["condition_shuffled_forecast_loss"]
                )
            metrics["samples"] = len(selected)
            metrics["by_direction"] = {
                direction: {
                    "samples": len(group),
                    "objective_loss": float(group["objective_loss"].mean()),
                    "forecast_loss": float(group["forecast_loss"].mean()),
                }
                for direction, group in selected.groupby("direction")
            }
            result[name] = metrics
    return result


def bootstrap_date_difference(frame, value, draws, seed):
    table = frame.pivot_table(index="asof_date", columns="model", values=value).dropna()
    difference = table["last1058"] - table["best467"]
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        difference.to_numpy(), size=(draws, len(difference)), replace=True
    ).mean(axis=1)
    return {
        "definition": "Last1058 minus Best467",
        "date_clusters": len(difference),
        "mean": float(difference.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


def paired_comparison(token_rows, prediction_rows, balanced_identities, seed):
    token = token_rows.pivot_table(
        index=["set", "identity", "asof_date"], columns="model", values="objective_loss"
    ).dropna().reset_index()
    result = {"token_objective": {}}
    for set_name, rows in token.groupby("set"):
        difference = rows["last1058"] - rows["best467"]
        date_difference = difference.groupby(rows["asof_date"]).mean()
        rng = np.random.default_rng(seed + len(result["token_objective"]))
        sampled = rng.choice(
            date_difference.to_numpy(), size=(10000, len(date_difference)), replace=True
        ).mean(axis=1)
        result["token_objective"][set_name] = {
            "definition": "Last1058 minus Best467; negative favors Last1058",
            "mean_sample_difference": float(difference.mean()),
            "mean_date_cluster_difference": float(date_difference.mean()),
            "date_clusters": len(date_difference),
            "date_cluster_ci95": [
                float(np.quantile(sampled, 0.025)),
                float(np.quantile(sampled, 0.975)),
            ],
        }
        if set_name == "future_all":
            balanced = rows[rows["identity"].isin(balanced_identities)]
            balanced_difference = balanced["last1058"] - balanced["best467"]
            result["token_objective"]["future_balanced"] = {
                "definition": "Last1058 minus Best467; negative favors Last1058",
                "mean_sample_difference": float(balanced_difference.mean()),
                "samples": len(balanced),
            }

    predictions = prediction_rows.copy()
    predictions["predicted_direction"] = three_class(predictions["predicted_return_10d"])
    date_rows = []
    for (model, asof_date), rows in predictions.groupby(["model", "asof_date"]):
        ranked = rows.sort_values("predicted_return_10d", ascending=False)
        tail = max(1, len(ranked) // 5)
        date_rows.append(
            {
                "model": model,
                "asof_date": asof_date,
                "rank_ic": rows["predicted_return_10d"].corr(
                    rows["return_10d"], method="spearman"
                ),
                "direction_accuracy": (
                    rows["predicted_direction"] == rows["direction"]
                ).mean(),
                "return_mae": np.abs(
                    rows["predicted_return_10d"] - rows["return_10d"]
                ).mean(),
                "top_bottom_spread": (
                    ranked.head(tail)["return_10d"].mean()
                    - ranked.tail(tail)["return_10d"].mean()
                ),
            }
        )
    date_rows = pd.DataFrame(date_rows)
    result["future_forecast"] = {
        metric: bootstrap_date_difference(date_rows, metric, 10000, seed + index + 100)
        for index, metric in enumerate(
            ("rank_ic", "direction_accuracy", "return_mae", "top_bottom_spread")
        )
    }
    return result


def load_model(path, device):
    return Kronos.from_pretrained(
        path,
        num_sectors=86,
        num_size_buckets=0,
        context_layer=10,
        use_size_percentile=True,
        size_mlp_hidden_dim=64,
    ).to(device).eval()


def run_evaluation(
    input_root,
    output_dir,
    tokenizer_path="NeoQuasar/Kronos-Tokenizer-base",
    batch_size=64,
    forecast_batch_size=64,
    sample_count=3,
    seed=20260826,
):
    input_root = Path(input_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evaluation_root, manifest = find_evaluation_root(input_root)
    parent_root, best_metric, progress = find_parent_root(input_root)

    panel_path = evaluation_root / manifest["artifacts"]["panel_file"]
    samples_path = evaluation_root / manifest["artifacts"]["samples_file"]
    with panel_path.open("rb") as handle:
        panel = pickle.load(handle)
    groups = load_samples(samples_path)
    store = WindowStore(panel, manifest["model_contract"]["sector_labels"])
    balanced_identities = {
        sample_identity(record) for record in groups["future_balanced"]
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Formal checkpoint evaluation requires a CUDA GPU")
    use_amp = True
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path).to(device).eval()
    model_paths = {
        "best467": parent_root / "checkpoints/best_model",
        "last1058": parent_root / "checkpoints/last_model",
    }
    model_hashes = {
        label: sha256_file(path / "model.safetensors")
        for label, path in model_paths.items()
    }
    print(
        json.dumps(
            {
                "device": str(device),
                "best_metric": best_metric,
                "parent_progress": progress,
                "model_hashes": model_hashes,
                "sample_sets": {key: len(value) for key, value in groups.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    token_frames = []
    prediction_frames = []
    for label, path in model_paths.items():
        print(f"Loading {label}: {path}", flush=True)
        model = load_model(path, device)
        token_frames.append(
            evaluate_token_sets(
                label, model, tokenizer, groups, store, device, batch_size, use_amp
            )
        )
        prediction_frames.append(
            evaluate_predictions(
                label,
                model,
                tokenizer,
                groups["future_all"],
                store,
                device,
                forecast_batch_size,
                sample_count,
                seed,
                use_amp,
            )
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    token_rows = pd.concat(token_frames, ignore_index=True)
    prediction_rows = pd.concat(prediction_frames, ignore_index=True)
    token_rows.to_csv(output / "token_losses.csv.gz", index=False, compression="gzip")
    prediction_rows.to_csv(
        output / "future_predictions.csv.gz", index=False, compression="gzip"
    )

    summary = {
        "schema_version": 1,
        "evaluation_name": EXPECTED_EVALUATION_NAME,
        "configuration": {
            "batch_size": batch_size,
            "forecast_batch_size": forecast_batch_size,
            "forecast_sample_count": sample_count,
            "forecast_temperature": 0.6,
            "forecast_top_p": 0.9,
            "seed": seed,
            "device": str(device),
            "model_hashes": model_hashes,
            "best_metric": best_metric,
            "parent_completed_segment": int(progress["current_segment"]),
        },
        "token_metrics": {
            label: token_summary(rows, balanced_identities)
            for label, rows in token_rows.groupby("model")
        },
        "future_forecast_metrics": {
            label: prediction_summary(rows)
            for label, rows in prediction_rows.groupby("model")
        },
        "paired_comparison": paired_comparison(
            token_rows, prediction_rows, balanced_identities, seed
        ),
        "interpretation": {
            "primary_set": "future_all",
            "primary_reason": (
                "Signals and targets are strictly later than the parent's latest training target."
            ),
            "diagnostic_sets": ["future_balanced", "historical_natural", "historical_balanced"],
            "limitation": (
                "The future set has broad cross-sectional coverage but only six signal dates; "
                "date-cluster confidence intervals must be reported."
            ),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    shutil.copy2(evaluation_root / "evaluation_manifest.json", output)
    experiment_manifest = {
        "task": "evaluation_only",
        "training_performed": False,
        "parent_root": str(parent_root),
        "evaluation_input": str(evaluation_root),
        "model_hashes": model_hashes,
        "output_files": [
            "summary.json",
            "evaluation_manifest.json",
            "token_losses.csv.gz",
            "future_predictions.csv.gz",
        ],
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(experiment_manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument(
        "--output-dir", default="/kaggle/working/kronos_v1_beta_checkpoint_evaluation"
    )
    parser.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--forecast-batch-size", type=int, default=64)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    run_evaluation(
        args.input_root,
        args.output_dir,
        tokenizer_path=args.tokenizer,
        batch_size=args.batch_size,
        forecast_batch_size=args.forecast_batch_size,
        sample_count=args.sample_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
