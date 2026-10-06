"""Kaggle runner: zero-training VAL generative-IC checkpoint reselection (dual T4).

Eval-only; never trains; never reads the sealed OOS package. Scores each forecast
checkpoint on a fixed 24-date subsample of temporal_symbol_validation_v1 with the
Baseline-2 production generative recipe (T0.65/p0.8/N5, seed 20260906) and the
teacher-forcing weighted forecast loss on the same windows. Each checkpoint is
scored and saved the moment its last shard lands.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

EMBEDDED_KRONOS_ARCHIVE_B64 = """
"""

OUTPUT_NAME = "beta_v2_1_c1_val_gen_ic"
KERNEL_ID = "luckfu/kronos-beta-v21-c1-val-gen-ic"
POLL_SECONDS = 30
MODEL_REPO = "luckfu/Kronos-A-Share-Beta-V2-1"
EXPECTED_TOKENIZER_SHA256 = "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
WORLD_SIZE = 2
HARD_LIMIT_SECONDS = 39600
EFFECTIVE_BATCH = 256
INPUT_ROOT = Path(os.environ.get("KRONOS_INPUT_ROOT", "/kaggle/input"))
WORKING_ROOT = Path(os.environ.get("KRONOS_WORKING_ROOT", "/kaggle/working"))
LOCAL_TEST = os.environ.get("KRONOS_VALGENIC_LOCAL_TEST") == "1"

# Order = scoring priority. full_val_wfl = logged weighted_forecast_loss on the
# full 123,836-window val (best_metric.json large_metrics), None if never logged
# on the C1 contract.
CHECKPOINTS = (
    {
        "label": "seg155_forecast_best",
        "source": "dataset",
        "patterns": (
            "**/kronos-beta-v21-c1-seg155-forecast-best/**/checkpoints/best_model",
            "**/kronos-beta-v21-c1-seg155-forecast-best/checkpoints/best_model",
        ),
        "sha256": "8b11a759e72d4125cb0c3307c482f1931ddc65be612023b422d66c52e4f8609c",
        "full_val_wfl": 2.312367872672933,
        "note": "C1 wc forecast floor (global Seg155 = local segment 26); forecast-only, no aux",
    },
    {
        "label": "seg8_rank_unfreeze_best",
        "source": "dataset",
        "patterns": (
            "**/kronos-beta-v21-c1-rank-unfreeze-seg8-best/**/checkpoints/best_model",
            "**/kronos-beta-v21-c1-rank-unfreeze-seg8-best/checkpoints/best_model",
        ),
        "sha256": "8a29277a856e84d9300188c1325f13c9fa88d96d38cafc1868f975dbda0a9892",
        "full_val_wfl": 2.325495002160697,
        "note": "v20 unfreeze from Seg19: trunk trained (1e-7) with ranking aux; forecast path changed",
    },
    {
        "label": "beta_v21_release_best475",
        "source": "modelscope",
        "patterns": (),
        "sha256": "e1bd55842996b7690a21c34c4d74e1128702bca9c16164788b741e3b5d052f97",
        "full_val_wfl": None,
        "note": "released Beta v2.1 Best@475 = C1 line parent before C1 forecast fine-tune",
    },
)


def emit(message: str) -> None:
    line = str(message)
    print(line, flush=True)
    log = WORKING_ROOT / OUTPUT_NAME / "run.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", buffering=1) as handle:
        handle.write(line + "\n")


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_sha(label: str, path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected and not LOCAL_TEST:
        raise RuntimeError(f"{label} SHA mismatch: {actual} != {expected}")
    emit(f"phase=sha label={label} sha256={actual} ok={actual == expected}")
    return actual


def extract_bundle(target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode(EMBEDDED_KRONOS_ARCHIVE_B64.strip())
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        archive.extractall(target)
    root = target / "Kronos"
    if not (root / "model").is_dir():
        raise RuntimeError(f"Embedded archive missing model/ under {root}")
    return root


def download_model_repo(target: Path) -> Path:
    override = os.environ.get("KRONOS_MODEL_SNAPSHOT")
    if override:
        return Path(override)
    target.mkdir(parents=True, exist_ok=True)
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "modelscope"])
        from modelscope.hub.snapshot_download import snapshot_download
    emit(f"phase=download_model_repo repo={MODEL_REPO}")
    snapshot_download(MODEL_REPO, local_dir=str(target))
    return target


def resolve_tokenizer(snapshot: Path) -> Path:
    tokenizer = snapshot / "tokenizer"
    if not (tokenizer / "config.json").is_file():
        matches = list(snapshot.glob("**/tokenizer/config.json"))
        if not matches:
            raise RuntimeError(f"Tokenizer not found under {snapshot}")
        tokenizer = matches[0].parent
    check_sha("tokenizer", tokenizer / "model.safetensors", EXPECTED_TOKENIZER_SHA256)
    return tokenizer


def resolve_release_predictor(snapshot: Path, expected: str) -> Path:
    candidates = [snapshot / "best_model", snapshot]
    candidates += [path.parent for path in snapshot.glob("**/model.safetensors")]
    for candidate in candidates:
        weights = candidate / "model.safetensors"
        if candidate.name == "tokenizer" or not weights.is_file():
            continue
        if not (candidate / "config.json").is_file():
            continue
        if LOCAL_TEST or sha256_file(weights) == expected:
            return candidate.resolve()
    raise RuntimeError(f"Release predictor with sha {expected} not found under {snapshot}")


def find_dataset_checkpoint(label: str, patterns) -> Path:
    matches = set()
    for pattern in patterns:
        for path in INPUT_ROOT.glob(pattern):
            if (path / "config.json").is_file() and (path / "model.safetensors").is_file():
                matches.add(path.resolve())
    if len(matches) != 1:
        raise RuntimeError(f"{label}: expected one checkpoint, found {sorted(matches)}")
    return next(iter(matches))


def parent_main() -> None:
    started = time.time()
    working = WORKING_ROOT / OUTPUT_NAME
    working.mkdir(parents=True, exist_ok=True)
    shards = working / "shards"
    shards.mkdir(exist_ok=True)
    results = working / "results"
    results.mkdir(exist_ok=True)

    if not LOCAL_TEST:
        gpu_names = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip().splitlines()
        if len(gpu_names) != WORLD_SIZE or any("T4" not in name for name in gpu_names):
            raise RuntimeError(f"Expected exactly two Tesla T4 GPUs, found {gpu_names}")
        emit("phase=gpu " + " | ".join(gpu_names))
    emit(f"phase=start kernel={KERNEL_ID} training=False sealed_oos_read=False")

    repo = extract_bundle(working / "src")
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune"))
    import pickle

    import pandas as pd
    from evaluate_beta_v21_val_gen_ic import (
        EXPECTED_VAL_DATES,
        EXPECTED_VAL_SAMPLES,
        EXPECTED_VAL_SHA256,
        PROD_ARM,
        VAL_SIGNAL_END,
        VAL_SIGNAL_START,
        VAL_SUBSAMPLE_DATES,
        build_val_records,
        compare_rankings,
        find_val_root,
        load_sector_labels,
        score_val_checkpoint,
        select_subsample_dates,
    )

    snapshot = download_model_repo(working / "pretrained")
    tokenizer = resolve_tokenizer(snapshot)
    checkpoints = []
    for spec in CHECKPOINTS:
        if spec["source"] == "modelscope":
            path = resolve_release_predictor(snapshot, spec["sha256"])
        else:
            path = find_dataset_checkpoint(spec["label"], spec["patterns"])
        sha = check_sha(spec["label"], path / "model.safetensors", spec["sha256"])
        config = json.loads((path / "config.json").read_text())
        checkpoints.append(
            {
                "label": spec["label"],
                "path": str(path),
                "sha256": sha,
                "full_val_wfl": spec["full_val_wfl"],
                "use_beta_v21_auxiliary": bool(config.get("use_beta_v21_auxiliary", False)),
                "note": spec["note"],
            }
        )
        emit(f"phase=checkpoint label={spec['label']} path={path}")

    val_root = find_val_root(INPUT_ROOT)
    val_data = val_root / "processed_datasets" / "val_data.pkl"
    val_sha = check_sha("val_data", val_data, EXPECTED_VAL_SHA256)
    sector_labels = load_sector_labels(val_root / "asset_metadata.csv")
    with val_data.open("rb") as handle:
        panel = pickle.load(handle)
    records = build_val_records(panel)
    all_dates = sorted({row["asof_date"] for row in records})
    if not LOCAL_TEST and (
        len(records) != EXPECTED_VAL_SAMPLES or len(all_dates) != EXPECTED_VAL_DATES
    ):
        raise RuntimeError(
            f"Val contract drift: {len(records)} windows / {len(all_dates)} dates "
            f"(expected {EXPECTED_VAL_SAMPLES} / {EXPECTED_VAL_DATES})"
        )
    dates = select_subsample_dates(all_dates, int(os.environ.get("KRONOS_VAL_DATES", VAL_SUBSAMPLE_DATES)))
    chosen = set(dates)
    subsample = [row for row in records if row["asof_date"] in chosen]
    records_file = working / "val_subsample_records.json"
    records_file.write_text(json.dumps(subsample))

    tasks = [{"checkpoint": item["label"], "date": date} for item in checkpoints for date in dates]
    subsample_contract = {
        "val_dataset": "temporal_symbol_validation_v1",
        "val_data_sha256": val_sha,
        "val_signal_range": [VAL_SIGNAL_START, VAL_SIGNAL_END],
        "val_full_windows": len(records),
        "val_full_dates": len(all_dates),
        "subsample_rule": "np.linspace(0, n_dates-1, k).round() over sorted signal dates; all symbols per date",
        "subsample_dates": dates,
        "subsample_windows": len(subsample),
        "decode": PROD_ARM,
        "score": "predicted_return_d10 = mean_N5 AR samples (denorm_close_d10/last_close - 1)",
        "label": "return_10d = close[asof+10]/close[asof] - 1",
        "wfl": "teacher-forcing CE, C1 horizon weights 1.364..0.455, per-sample mean on subsample",
    }
    plan = {
        "repo": str(repo),
        "tokenizer_dir": str(tokenizer),
        "val_data": str(val_data),
        "sector_labels": sector_labels,
        "records_file": str(records_file),
        "checkpoints": checkpoints,
        "shards": str(shards),
        "deadline": started + HARD_LIMIT_SECONDS,
        "effective_batch": int(os.environ.get("KRONOS_EFFECTIVE_BATCH", EFFECTIVE_BATCH)),
        "tasks": tasks,
        "contract": subsample_contract,
        "kernel": KERNEL_ID,
    }
    plan_path = working / "plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    emit(json.dumps({"phase": "inputs_resolved", **subsample_contract,
                     "checkpoints": [c["label"] for c in checkpoints], "tasks": len(tasks)},
                    ensure_ascii=False))

    worker_script = repo / "finetune" / "evaluate_beta_v21_val_gen_ic.py"
    world = int(os.environ.get("KRONOS_WORLD_OVERRIDE", WORLD_SIZE))
    processes = []
    for rank in range(world):
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONUNBUFFERED": "1",
                "CUDA_VISIBLE_DEVICES": str(rank),
                "KRONOS_SWEEP_RANK": str(rank),
                "KRONOS_SWEEP_WORLD": str(world),
                "PYTHONPATH": os.pathsep.join(
                    [str(repo), str(repo / "finetune"), environment.get("PYTHONPATH", "")]
                ),
            }
        )
        processes.append(
            subprocess.Popen(
                [sys.executable, "-u", str(worker_script), "--worker-plan", str(plan_path)],
                env=environment,
            )
        )
    emit(f"phase=workers_spawned n={world}")

    summaries: dict[str, dict] = {}

    def write_comparison(final: bool) -> dict:
        done = [summaries[c["label"]] for c in checkpoints if c["label"] in summaries]
        comparison = {
            "kernel": KERNEL_ID,
            "training": False,
            "sealed_oos_read": False,
            "contract": subsample_contract,
            "checkpoints": done,
            "rankings": compare_rankings(done) if done else None,
            "incomplete": [
                {
                    "checkpoint": c["label"],
                    "have": len(list(shards.glob(f"{c['label']}__*.csv.gz"))),
                    "need": len(dates),
                }
                for c in checkpoints
                if c["label"] not in summaries
            ],
            "final": final,
            "elapsed_sec": time.time() - started,
        }
        (results / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
        shutil.copy2(results / "comparison.json", working / "comparison.json")
        return comparison

    def score_ready() -> None:
        for item in checkpoints:
            label = item["label"]
            if label in summaries:
                continue
            parts = sorted(shards.glob(f"{label}__*.csv.gz"))
            if len(parts) != len(dates):
                continue
            frame = pd.concat([pd.read_csv(path) for path in parts], ignore_index=True)
            if len(frame) != len(subsample):
                raise RuntimeError(f"{label} rows {len(frame)} != {len(subsample)}")
            metrics = score_val_checkpoint(frame)
            summary = {
                "label": label,
                "model_sha256": item["sha256"],
                "use_beta_v21_auxiliary": item["use_beta_v21_auxiliary"],
                "note": item["note"],
                "full_val_weighted_forecast_loss": item["full_val_wfl"],
                **metrics,
            }
            summaries[label] = summary
            frame.to_csv(results / f"{label}_val_predictions.csv.gz", index=False, compression="gzip")
            (results / f"{label}_val_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
            )
            emit(json.dumps({
                "phase": "checkpoint_scored",
                "label": label,
                "samples": summary["samples"],
                "signal_dates": summary["signal_dates"],
                "return10d_rank_ic_daily": summary["return10d_rank_ic_daily"],
                "return10d_rank_ic_pooled": summary["return10d_rank_ic_pooled"],
                "return10d_rank_icir": summary["return10d_rank_icir"],
                "return10d_rank_ic_pos_rate": summary["return10d_rank_ic_pos_rate"],
                "return10d_rank_ic_se": summary["return10d_rank_ic_se"],
                "top_bottom_decile_return10d": summary["top_bottom_decile_return10d"],
                "utility_rank_ic_daily": summary["utility_rank_ic_daily"],
                "wfl_subsample": summary.get("weighted_forecast_loss_subsample"),
                "wfl_full_val_logged": item["full_val_wfl"],
                "elapsed_sec": round(time.time() - started, 1),
            }, ensure_ascii=False))
            comparison = write_comparison(final=False)
            emit("phase=rankings " + json.dumps(comparison["rankings"], ensure_ascii=False))

    while True:
        codes = [process.poll() for process in processes]
        score_ready()
        if all(code is not None for code in codes):
            break
        if any(code not in (None, 0) for code in codes):
            emit(f"phase=worker_failed codes={codes}")
            for process in processes:
                if process.poll() is None:
                    process.wait()
            break
        time.sleep(POLL_SECONDS)
    failures = [process.wait() for process in processes]
    score_ready()
    comparison = write_comparison(final=True)
    emit("phase=comparison " + json.dumps(
        {"rankings": comparison["rankings"], "incomplete": comparison["incomplete"]}, ensure_ascii=False))
    if any(code != 0 for code in failures):
        raise RuntimeError(f"Worker exit codes: {failures}")
    if not summaries:
        raise RuntimeError("No checkpoint completed")
    emit(f"phase=done total_seconds={time.time() - started:.1f}")


if __name__ == "__main__":
    parent_main()
