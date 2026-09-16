"""Re-infer C2 best on the original 6-date sealed OOS package.

The audited first-6 predictions (SHA 60095569) are not on current Kaggle
outputs. This is a documented re-inference for the 19-day money baseline,
not a bit-identical replay of that file. C2 last / base are skipped.
"""
import gc
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SOURCE_COMMIT = "7debae94d9ec2ada3d34e1de28dfe19a497e975d"
OUTPUT = Path("/kaggle/working/kronos_small_0_1_c2_first6_oos")
INPUT = Path("/kaggle/input")
HARD_LIMIT_SECONDS = 10800
EXPECTED_C2_SHA = "4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a"
EXPECTED_TOKENIZER_SHA = "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
EXPECTED_ROWS = 30930
EXPECTED_DATES = [
    "2026-08-03", "2026-08-04", "2026-08-05",
    "2026-08-06", "2026-08-07", "2026-08-10",
]
PURPOSE = (
    "exploratory_only; first-6 C2 re-inference for 19-day money baseline; "
    "not SHA 60095569 original file; 19-date set is design-contaminated"
)
LABEL = "c2_best_segment_179"


def find_one(pattern, predicate=lambda path: True):
    matches = [path for path in INPUT.glob(pattern) if predicate(path)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, cwd=None):
    print({"command": command, "cwd": str(cwd) if cwd else None}, flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def clone_source():
    for attempt in range(1, 4):
        repo = Path(tempfile.mkdtemp(prefix="kronos-c2-first6-source-")) / "repo"
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
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shards = OUTPUT / "shards"
    shards.mkdir(exist_ok=True)

    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip()
    # Original first-6 C2 ran on P100; this is already a re-inference, so T4 is
    # allowed. Record the GPU for the money-baseline provenance.
    if not any(name in gpu for name in ("P100", "T4")):
        raise RuntimeError(f"Expected P100 or T4, got {gpu}")

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

    c2_file = find_one("**/small_0.1_stage2_cosine_refinement/checkpoints/best_model/model.safetensors")
    tokenizer_file = find_one("**/Kronos-Tokenizer-base/model.safetensors")
    c2_sha = sha256_file(c2_file)
    tok_sha = sha256_file(tokenizer_file)
    if c2_sha != EXPECTED_C2_SHA:
        raise RuntimeError(f"C2 weights sha mismatch: {c2_sha}")
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"Tokenizer sha mismatch: {tok_sha}")

    package_dirs = [
        path.parent for path in INPUT.glob("**/evaluation_manifest.json")
        if (path.parent / "evaluation_panel.pkl").is_file()
        and (path.parent / "evaluation_samples.jsonl").is_file()
        and "kronos_small_0_1_time_oos_evaluation_20260826" in str(path)
    ]
    if len(package_dirs) != 1:
        raise RuntimeError(f"Expected the 20260826 6-date package, found {package_dirs}")
    evaluation_root = package_dirs[0]
    manifest = json.loads((evaluation_root / "evaluation_manifest.json").read_text())
    isolation = manifest["temporal_isolation"]
    if isolation.get("incremental_signal_start"):
        raise RuntimeError("Refusing incremental 13-date package; this kernel is first-6 only")
    if not (isolation.get("strictly_after_parent_latest_training_target")
            or isolation.get("targets_strictly_after_training_target_end")):
        raise RuntimeError("Evaluation package is not temporally isolated")
    start = isolation.get("future_signal_start")
    end = isolation.get("future_signal_end")
    if start != "2026-08-03" or end != "2026-08-10":
        raise RuntimeError(f"Unexpected first-6 window: {start}..{end}")

    with (evaluation_root / manifest["artifacts"]["panel_file"]).open("rb") as handle:
        panel = pickle.load(handle)
    sample_groups = load_samples(evaluation_root / manifest["artifacts"]["samples_file"])
    sample_key = "future_all" if "future_all" in sample_groups else "incremental_future_all"
    records = [row for row in sample_groups[sample_key] if start <= row["asof_date"] <= end]
    dates = sorted({row["asof_date"] for row in records})
    if dates != EXPECTED_DATES:
        raise RuntimeError(f"Unexpected dates: {dates}")
    if len(records) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} records, found {len(records)}")

    provenance = {
        "c2_weights": {"path": str(c2_file), "sha256": c2_sha},
        "tokenizer": {"path": str(tokenizer_file), "sha256": tok_sha},
        "evaluation_root": str(evaluation_root),
        "gpu": gpu,
        "note": "re-inference; not original first-6 SHA 60095569",
    }
    print(json.dumps({"phase": "inputs_resolved", **provenance}, ensure_ascii=False), flush=True)

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_file.parent).to(device).eval()
    store = WindowStore(panel, manifest["model_contract"]["sector_labels"])
    pending = [date for date in dates if not (shards / f"{LABEL}_{date}.csv.gz").is_file()]
    if pending:
        model = Kronos.from_pretrained(
            c2_file.parent, num_sectors=86, num_size_buckets=0, context_layer=6,
            use_size_percentile=True, size_mlp_hidden_dim=64,
        ).to(device).eval()
        for date in pending:
            if time.time() - started > HARD_LIMIT_SECONDS:
                raise RuntimeError(f"Timed out before {date}")
            date_records = [row for row in records if row["asof_date"] == date]
            result = evaluate_predictions(
                LABEL, model, tokenizer, date_records, store, device,
                64, 1, 20260906, True,
            )
            shard = shards / f"{LABEL}_{date}.csv.gz"
            result.to_csv(shard, index=False, compression="gzip")
            print({"saved": str(shard), "rows": len(result), "elapsed_sec": time.time() - started}, flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    parts = [pd.read_csv(path) for path in sorted(shards.glob(f"{LABEL}_*.csv.gz"))]
    if len(parts) != len(dates):
        raise RuntimeError(f"Missing shards: have {len(parts)} need {len(dates)}")
    frame = pd.concat(parts, ignore_index=True)
    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(f"Prediction count mismatch: {len(frame)}")
    if sorted(frame["asof_date"].astype(str).unique()) != EXPECTED_DATES:
        raise RuntimeError("Prediction dates mismatch")
    frame.to_csv(OUTPUT / "predictions.csv.gz", index=False, compression="gzip")
    metrics = summarize(frame)
    payload = {
        "status": "complete",
        "purpose": PURPOSE,
        "oos_used": True,
        "training_performed": False,
        "source_commit": SOURCE_COMMIT,
        "sample_set": sample_key,
        "signal_start": start,
        "signal_end": end,
        "elapsed_sec": time.time() - started,
        "provenance": provenance,
        "metrics": {LABEL: metrics},
        "d10": metrics["by_horizon"][9],
        "original_file_sha_not_reproduced": "600955695f3e186877b6f64a87fb26b8d0c5846e2cf0b2d6b8d68310e1933d07",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
