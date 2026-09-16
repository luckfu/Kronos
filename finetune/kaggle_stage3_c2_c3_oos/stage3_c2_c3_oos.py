"""Compare C2 best and C3 last with the exact sealed production OOS evaluator."""
import gc
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SOURCE_COMMIT = "f617ec0cf18406dc0be287055a6c1b5e8125c023"
OUTPUT = Path("/kaggle/working/kronos_small_0_1_stage3_c2_c3_oos")
INPUT = Path("/kaggle/input")


def find_one(pattern, predicate=lambda path: True):
    matches = [path for path in INPUT.glob(pattern) if predicate(path)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def run(command, cwd=None):
    print({"command": command, "cwd": str(cwd) if cwd else None}, flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def clone_source():
    for attempt in range(1, 4):
        repo = Path(tempfile.mkdtemp(prefix="kronos-oos-source-")) / "repo"
        try:
            run(["git", "clone", "--depth", "8", "--branch", "master",
                 "https://github.com/luckfu/Kronos.git", str(repo)])
            run(["git", "checkout", "--detach", SOURCE_COMMIT], cwd=repo)
            actual = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
            if actual != SOURCE_COMMIT:
                raise RuntimeError(f"Source commit mismatch: {actual}")
            return repo
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * attempt)


def metric_or_none(value):
    value = float(value)
    return None if not math.isfinite(value) else value


def summarize(frame, top_fraction=0.10):
    import numpy as np
    import pandas as pd

    by_horizon = []
    by_date_h10 = []
    for horizon in range(1, 11):
        pred_col = f"predicted_return_d{horizon}"
        actual_col = f"actual_return_d{horizon}"
        pred, actual = frame[pred_col], frame[actual_col]
        daily = []
        for date, rows in frame.groupby("asof_date", sort=True):
            rank_ic = rows[pred_col].corr(rows[actual_col], method="spearman")
            count = max(1, int(math.ceil(len(rows) * top_fraction)))
            ranked = rows.sort_values(pred_col)
            spread = ranked.tail(count)[actual_col].mean() - ranked.head(count)[actual_col].mean()
            daily.append({
                "asof_date": str(date),
                "samples": int(len(rows)),
                "rank_ic": metric_or_none(rank_ic),
                "direction_accuracy": float((np.sign(rows[pred_col]) == np.sign(rows[actual_col])).mean()),
                "top_bottom_10pct": float(spread),
            })
        daily_ic = pd.Series([row["rank_ic"] for row in daily], dtype=float).dropna()
        ic_std = float(daily_ic.std(ddof=1)) if len(daily_ic) > 1 else float("nan")
        metrics = {
            "horizon_day": horizon,
            "samples": int(len(frame)),
            "direction_accuracy": float((np.sign(pred) == np.sign(actual)).mean()),
            "mae": float((pred - actual).abs().mean()),
            "pooled_rank_ic": metric_or_none(pred.corr(actual, method="spearman")),
            "daily_rank_ic_mean": metric_or_none(daily_ic.mean()),
            "daily_rank_ic_std": metric_or_none(ic_std),
            "daily_icir": metric_or_none(daily_ic.mean() / ic_std),
            "daily_icir_annualized_sqrt252": metric_or_none(daily_ic.mean() / ic_std * math.sqrt(252)),
            "positive_ic_ratio": float((daily_ic > 0).mean()),
            "mean_top_bottom_10pct": float(np.mean([row["top_bottom_10pct"] for row in daily])),
        }
        by_horizon.append(metrics)
        if horizon == 10:
            by_date_h10 = daily
    return {
        "samples": int(len(frame)),
        "signal_dates": int(frame["asof_date"].nunique()),
        "by_horizon": by_horizon,
        "by_signal_date_d10": by_date_h10,
    }


def main():
    os.environ["PYTHONUNBUFFERED"] = "1"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    repo = clone_source()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune"))

    import pandas as pd
    import torch
    from finetune.evaluate_v1_beta_checkpoints import WindowStore, evaluate_predictions, load_samples
    from model import Kronos, KronosTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")
    device = torch.device("cuda:0")

    c2_predictions = find_one(
        "**/kronos_small_0_1_stage2_oos/predictions.csv.gz",
        lambda path: "c2-alpha-oos-evaluation" in str(path),
    )
    c3_model_file = find_one("**/stage3_joint_path_smoke/checkpoints/last_model/model.safetensors")
    tokenizer_file = find_one("**/Kronos-Tokenizer-base/model.safetensors")
    manifests = []
    for path in INPUT.glob("**/evaluation_manifest.json"):
        if not ((path.parent / "evaluation_panel.pkl").is_file()
                and (path.parent / "evaluation_samples.jsonl").is_file()):
            continue
        candidate = json.loads(path.read_text())
        if candidate.get("temporal_isolation", {}).get("incremental_signal_start"):
            manifests.append(path)
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one sealed evaluation manifest, found {manifests}")
    evaluation_root = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    isolation = manifest["temporal_isolation"]
    if not (isolation.get("strictly_after_parent_latest_training_target")
            or isolation.get("targets_strictly_after_training_target_end")):
        raise RuntimeError("Evaluation package is not temporally isolated")

    with (evaluation_root / manifest["artifacts"]["panel_file"]).open("rb") as handle:
        panel = pickle.load(handle)
    sample_groups = load_samples(evaluation_root / manifest["artifacts"]["samples_file"])
    sample_key = "incremental_future_all" if "incremental_future_all" in sample_groups else "future_all"
    start = isolation.get("incremental_signal_start", isolation.get("future_signal_start"))
    end = isolation.get("incremental_signal_end", isolation.get("future_signal_end"))
    records = [row for row in sample_groups[sample_key] if start <= row["asof_date"] <= end]
    if not records:
        raise RuntimeError("No sealed OOS records")

    prior = pd.read_csv(c2_predictions)
    prior = prior[prior["model"] == "c2_best_segment_179"].copy()
    if len(prior) != len(records):
        raise RuntimeError(f"C2 prediction count mismatch: {len(prior)} != {len(records)}")

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_file.parent).to(device).eval()
    model = Kronos.from_pretrained(
        c3_model_file.parent,
        num_sectors=86,
        num_size_buckets=0,
        context_layer=6,
        use_size_percentile=True,
        size_mlp_hidden_dim=64,
    ).to(device).eval()
    store = WindowStore(panel, manifest["model_contract"]["sector_labels"])
    shards = OUTPUT / "shards"
    shards.mkdir(exist_ok=True)
    label = "c3_last_segment_15"
    for date in sorted({row["asof_date"] for row in records}):
        shard = shards / f"{label}_{date}.csv.gz"
        if shard.is_file():
            continue
        date_records = [row for row in records if row["asof_date"] == date]
        result = evaluate_predictions(
            label, model, tokenizer, date_records, store, device,
            64, 1, 20260906, True,
        )
        result.to_csv(shard, index=False, compression="gzip")
        print({"saved": str(shard), "rows": len(result)}, flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    c3 = pd.concat([pd.read_csv(path) for path in sorted(shards.glob("*.csv.gz"))], ignore_index=True)
    if len(c3) != len(records):
        raise RuntimeError(f"C3 prediction count mismatch: {len(c3)} != {len(records)}")
    combined = pd.concat([prior, c3], ignore_index=True)
    combined.to_csv(OUTPUT / "predictions.csv.gz", index=False, compression="gzip")
    summary = {
        "status": "complete",
        "training_performed": False,
        "source_commit": SOURCE_COMMIT,
        "evaluation_reference": "smmt315/kronos-small-0-1-c2-alpha-oos-evaluation",
        "sample_set": sample_key,
        "signal_start": start,
        "signal_end": end,
        "models": {
            "c2_best_segment_179": "reused exact reference predictions",
            "c3_last_segment_15": str(c3_model_file.parent),
        },
        "metrics": {name: summarize(rows) for name, rows in combined.groupby("model")},
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
