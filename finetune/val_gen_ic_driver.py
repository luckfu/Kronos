"""Reusable driver for the Step-1 VAL generative-IC eval contract.

Same contract as ``kaggle_beta_v21_c1_val_gen_ic.py`` (luckfu/kronos-beta-v21-c1-
val-gen-ic): temporal_symbol_validation_v1 val_data.pkl (sha 4cce31bc...), 24 dates
by ``np.linspace(0, 241, 24).round()``, all symbols per date (12,256 windows),
Baseline-2 prod decode T0.65/p0.8/N5 seed 20260906, score = mean close_d10/last_close-1,
label raw-close return_10d, plus teacher-forcing WFL on the same windows.

Used in-kernel by the forecast cosine pilot after training to score Seg3/6/9/12
snapshots (and best_model when distinct). Each checkpoint is scored and its
summary written the moment all its date shards land. Never reads sealed OOS.

Usage (inside a repo checkout, GPUs free):
    python finetune/val_gen_ic_driver.py --spec spec.json
spec = {"kernel", "output_dir", "tokenizer_dir", "input_root", "deadline",
        "checkpoints": [{"label", "path", "full_val_wfl", "note", ...}],
        optional "world", "effective_batch", "reference", "val_contract"}

Optional ``val_contract`` swaps the validation set while keeping decode, score,
label and WFL identical (used by the heavy-reg time-disjoint retrain):
    {"name", "val_data", "val_sha256", "sector_metadata", "signal_start",
     "signal_end", "expected_samples", "expected_dates", "identities_sha256",
     "subsample_dates": "all" | int, "same_contract_as"}
Without it the Step-1 contract above is used unchanged.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "finetune"))

POLL_SECONDS = 30


def emit(log: Path, message: str) -> None:
    print(message, flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", buffering=1) as handle:
        handle.write(message + "\n")


def run(spec: dict) -> dict:
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
        sha256_file,
    )

    started = time.time()
    local_test = os.environ.get("KRONOS_VALGENIC_LOCAL_TEST") == "1"
    working = Path(spec["output_dir"])
    shards = working / "shards"
    results = working / "results"
    shards.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    log = working / "run.log"
    kernel = spec.get("kernel", "")

    checkpoints = []
    for item in spec["checkpoints"]:
        path = Path(item["path"])
        weights = path / "model.safetensors"
        if not weights.is_file() or not (path / "config.json").is_file():
            raise RuntimeError(f"{item['label']}: checkpoint incomplete at {path}")
        config = json.loads((path / "config.json").read_text())
        checkpoints.append({
            **item,
            "path": str(path),
            "sha256": item.get("sha256") or sha256_file(weights),
            "use_beta_v21_auxiliary": bool(config.get("use_beta_v21_auxiliary", False)),
        })
        emit(log, json.dumps({"phase": "valgenic_checkpoint", "label": item["label"],
                              "path": str(path), "sha256": checkpoints[-1]["sha256"]}))

    override = spec.get("val_contract")
    if override:
        val_data = Path(override["val_data"])
        expected_sha = override["val_sha256"]
        sector_csv = Path(override["sector_metadata"])
        signal_range = [override["signal_start"], override["signal_end"]]
        expected_samples = int(override["expected_samples"])
        expected_dates = int(override["expected_dates"])
    else:
        val_root = find_val_root(Path(spec["input_root"]))
        val_data = val_root / "processed_datasets" / "val_data.pkl"
        expected_sha = EXPECTED_VAL_SHA256
        sector_csv = val_root / "asset_metadata.csv"
        signal_range = [VAL_SIGNAL_START, VAL_SIGNAL_END]
        expected_samples = EXPECTED_VAL_SAMPLES
        expected_dates = EXPECTED_VAL_DATES
    val_sha = sha256_file(val_data)
    if val_sha != expected_sha and not local_test:
        raise RuntimeError(f"val_data SHA mismatch: {val_sha} != {expected_sha}")
    sector_labels = load_sector_labels(sector_csv)
    with val_data.open("rb") as handle:
        panel = pickle.load(handle)
    records = build_val_records(panel, *signal_range)
    del panel
    all_dates = sorted({row["asof_date"] for row in records})
    if not local_test and (
        len(records) != expected_samples or len(all_dates) != expected_dates
    ):
        raise RuntimeError(
            f"Val contract drift: {len(records)} windows / {len(all_dates)} dates"
        )
    if override and override.get("identities_sha256") and not local_test:
        from build_time_disjoint_val_panel import identities_sha256

        actual_ids = identities_sha256(records)
        if actual_ids != override["identities_sha256"]:
            raise RuntimeError(
                f"Val identities drift: {actual_ids} != {override['identities_sha256']}"
            )
    subsample_rule = (override or {}).get("subsample_dates", VAL_SUBSAMPLE_DATES)
    if subsample_rule == "all":
        dates = list(all_dates)
    else:
        dates = select_subsample_dates(
            all_dates, int(os.environ.get("KRONOS_VAL_DATES", subsample_rule))
        )
    chosen = set(dates)
    subsample = [row for row in records if row["asof_date"] in chosen]
    records_file = working / "val_subsample_records.json"
    records_file.write_text(json.dumps(subsample))

    tasks = [{"checkpoint": c["label"], "date": d} for c in checkpoints for d in dates]
    contract = {
        "val_dataset": (override or {}).get("name", "temporal_symbol_validation_v1"),
        "val_data_sha256": val_sha,
        "val_signal_range": signal_range,
        "val_full_windows": len(records),
        "val_full_dates": len(all_dates),
        "subsample_rule": "np.linspace(0, n_dates-1, k).round() over sorted signal dates; all symbols per date",
        "subsample_dates": dates,
        "subsample_windows": len(subsample),
        "decode": PROD_ARM,
        "score": "predicted_return_d10 = mean_N5 AR samples (denorm_close_d10/last_close - 1)",
        "label": "return_10d = close[asof+10]/close[asof] - 1",
        "wfl": "teacher-forcing CE, C1 horizon weights 1.364..0.455, per-sample mean on subsample",
        "same_contract_as": (override or {}).get(
            "same_contract_as", "luckfu/kronos-beta-v21-c1-val-gen-ic"
        ),
    }
    if override:
        contract["subsample_rule"] = (
            "all signal dates, all symbols per date" if subsample_rule == "all"
            else contract["subsample_rule"]
        )
    plan = {
        "repo": str(REPO),
        "tokenizer_dir": str(spec["tokenizer_dir"]),
        "val_data": str(val_data),
        "sector_labels": sector_labels,
        "records_file": str(records_file),
        "checkpoints": checkpoints,
        "shards": str(shards),
        "deadline": float(spec["deadline"]),
        "effective_batch": int(spec.get("effective_batch", 256)),
        "tasks": tasks,
        "contract": contract,
        "kernel": kernel,
    }
    plan_path = working / "plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    emit(log, json.dumps({"phase": "valgenic_inputs_resolved", **contract,
                          "checkpoints": [c["label"] for c in checkpoints],
                          "tasks": len(tasks)}, ensure_ascii=False))

    world = int(spec.get("world", 2))
    worker_script = REPO / "finetune" / "evaluate_beta_v21_val_gen_ic.py"
    processes = []
    for rank in range(world):
        environment = os.environ.copy()
        environment.update({
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": str(rank),
            "KRONOS_SWEEP_RANK": str(rank),
            "KRONOS_SWEEP_WORLD": str(world),
            "PYTHONPATH": os.pathsep.join(
                [str(REPO), str(REPO / "finetune"), environment.get("PYTHONPATH", "")]
            ),
        })
        processes.append(subprocess.Popen(
            [sys.executable, "-u", str(worker_script), "--worker-plan", str(plan_path)],
            env=environment,
        ))
    emit(log, f"phase=valgenic_workers_spawned n={world}")

    summaries: dict[str, dict] = {}

    def write_comparison(final: bool) -> dict:
        done = [summaries[c["label"]] for c in checkpoints if c["label"] in summaries]
        comparison = {
            "kernel": kernel,
            "sealed_oos_read": False,
            "contract": contract,
            "reference": spec.get("reference"),
            "checkpoints": done,
            "rankings": compare_rankings(done) if done else None,
            "incomplete": [
                {"checkpoint": c["label"],
                 "have": len(list(shards.glob(f"{c['label']}__*.csv.gz"))),
                 "need": len(dates)}
                for c in checkpoints if c["label"] not in summaries
            ],
            "final": final,
            "elapsed_sec": time.time() - started,
        }
        text = json.dumps(comparison, ensure_ascii=False, indent=2) + "\n"
        (results / "comparison.json").write_text(text)
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
                "segment": item.get("segment"),
                "model_sha256": item["sha256"],
                "use_beta_v21_auxiliary": item["use_beta_v21_auxiliary"],
                "note": item.get("note"),
                "full_val_weighted_forecast_loss": item.get("full_val_wfl"),
                **metrics,
            }
            summaries[label] = summary
            frame.to_csv(results / f"{label}_val_predictions.csv.gz", index=False,
                         compression="gzip")
            (results / f"{label}_val_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
            )
            emit(log, json.dumps({
                "phase": "checkpoint_scored",
                "label": label,
                "segment": item.get("segment"),
                "samples": summary["samples"],
                "return10d_rank_ic_daily": summary["return10d_rank_ic_daily"],
                "return10d_rank_ic_pooled": summary["return10d_rank_ic_pooled"],
                "return10d_rank_icir": summary["return10d_rank_icir"],
                "return10d_rank_ic_pos_rate": summary["return10d_rank_ic_pos_rate"],
                "return10d_rank_ic_se": summary["return10d_rank_ic_se"],
                "top_bottom_decile_return10d": summary["top_bottom_decile_return10d"],
                "wfl_subsample": summary.get("weighted_forecast_loss_subsample"),
                "wfl_full_val_logged": item.get("full_val_wfl"),
                "elapsed_sec": round(time.time() - started, 1),
            }, ensure_ascii=False))
            comparison = write_comparison(final=False)
            emit(log, "phase=valgenic_rankings " + json.dumps(comparison["rankings"]))

    while True:
        codes = [process.poll() for process in processes]
        score_ready()
        if all(code is not None for code in codes):
            break
        if any(code not in (None, 0) for code in codes):
            emit(log, f"phase=valgenic_worker_failed codes={codes}")
            for process in processes:
                if process.poll() is None:
                    process.wait()
            break
        time.sleep(POLL_SECONDS)
    failures = [process.wait() for process in processes]
    score_ready()
    comparison = write_comparison(final=True)
    emit(log, "phase=valgenic_comparison " + json.dumps(
        {"rankings": comparison["rankings"], "incomplete": comparison["incomplete"]}))
    if any(code != 0 for code in failures):
        raise RuntimeError(f"val gen-IC worker exit codes: {failures}")
    emit(log, f"phase=valgenic_done total_seconds={time.time() - started:.1f}")
    return comparison


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] != "--spec":
        raise SystemExit("usage: val_gen_ic_driver.py --spec SPEC.json")
    run(json.loads(Path(sys.argv[2]).read_text()))


if __name__ == "__main__":
    main()
