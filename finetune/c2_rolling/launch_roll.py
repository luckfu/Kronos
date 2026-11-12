#!/usr/bin/env python3
"""Kernel entry point: continue C2 on new-data windows only, export fixed checkpoints.

Experiment: start from the frozen C2 best weights, train one coverage pass over
the roll package (signal dates after C2's training cutoff), and export the
weights at pre-registered segments so each can be scored out of sample.

Everything that defines an arm arrives through ``ROLL_*`` environment variables
set by ``push_roll.py``.  The trainer and its stage runner are used unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

C2_SHA256 = "4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a"
TOKENIZER_SHA256 = "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
REPO = Path(__file__).resolve().parents[2]


def phase(name: str, **values) -> None:
    print(json.dumps({"time": time.time(), "phase": name, **values}, ensure_ascii=False), flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_parent(input_root: Path) -> Path:
    explicit = os.environ.get("ROLL_PARENT_DIR", "").strip()
    candidates = [Path(explicit) / "model.safetensors"] if explicit else sorted(input_root.glob("**/model.safetensors"))
    expected = os.environ.get("ROLL_PARENT_SHA256", C2_SHA256)
    hits = [p for p in candidates if (p.parent / "config.json").is_file() and sha256_file(p) == expected]
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one parent with sha {expected[:12]}; found {hits}")
    return hits[0].parent


def main() -> None:
    arm = os.environ["ROLL_ARM"]
    input_root = Path(os.environ.get("ROLL_INPUT_ROOT", "/kaggle/input"))
    work = Path(os.environ.get("ROLL_WORK_ROOT", "/kaggle/working"))
    runtime, ckpt_root = work / "roll_runtime", work / "roll_ckpt"
    output_name = f"c2_roll_{arm}"
    phase("started", arm=arm, lr=os.environ["ROLL_LR"], warmup_start_lr=os.environ["ROLL_WARMUP_START_LR"],
          snapshot_every=os.environ["ROLL_SNAPSHOT_EVERY"])

    package = sorted(input_root.glob("**/roll_manifest.json"))
    if len(package) != 1:
        raise RuntimeError(f"Expected one roll package; found {package}")
    data_root = runtime / "data"
    subprocess.run([sys.executable, str(REPO / "finetune/c2_rolling/build_panels.py"),
                    "--package", str(package[0].parent), "--output", str(data_root)], check=True)
    contract = json.loads((data_root / "data_manifest.json").read_text())["window_contract"]
    parent = find_parent(input_root)
    parent_sha = sha256_file(parent / "model.safetensors")
    phase("parent_ready", parent=str(parent), sha256=parent_sha, contract=contract)

    env = {key: value for key, value in os.environ.items() if not key.startswith("KRONOS_")}
    env.update({
        "PYTHONUNBUFFERED": "1",
        "KRONOS_SMALL_V21_STAGE": "main",
        "KRONOS_SMALL_V21_RUNTIME": str(runtime),
        "KRONOS_SMALL_V21_DATA_ROOT": str(data_root),
        "KRONOS_SMALL_V21_PARENT_MODEL": str(parent),
        "KRONOS_SMALL_V21_OUTPUT_NAME": output_name,
        "KRONOS_SMALL_V21_DISABLE_AUTO_CONTINUATION": "1",   # fresh optimizer from the parent weights
        "KRONOS_KAGGLE_INPUT_ROOT": str(input_root),
        "KRONOS_TRAIN_SIGNAL_START": contract["train_signal_start"],
        "KRONOS_TRAIN_SIGNAL_END": contract["train_signal_end"],
        "KRONOS_EPOCHS": "1", "KRONOS_REQUIRE_FULL_COVERAGE": "1", "KRONOS_COVERAGE_PASSES": "1",
        "KRONOS_COVERAGE_SEED": os.environ.get("ROLL_COVERAGE_SEED", "20261111"),
        "KRONOS_MAX_SEGMENTS_PER_RUN": os.environ.get("ROLL_MAX_SEGMENTS", "0"),
        "KRONOS_TRAIN_SAMPLES_PER_SEGMENT": os.environ.get("ROLL_SAMPLES_PER_SEGMENT", "20000"),
        "KRONOS_TORCHRUN_NPROC_PER_NODE": os.environ.get("ROLL_NPROC", "2"),
        "KRONOS_BATCH_SIZE": os.environ.get("ROLL_BATCH_SIZE", "32"),
        "KRONOS_NUM_WORKERS": os.environ.get("ROLL_NUM_WORKERS", "2"),
        "KRONOS_SCHEDULER": "warmup_constant", "KRONOS_SCHEDULER_WARMUP_RATIO": "0.01",
        "KRONOS_PREDICTOR_LEARNING_RATE": os.environ["ROLL_LR"],
        "KRONOS_CONDITION_LEARNING_RATE": os.environ["ROLL_LR"],
        "KRONOS_PREDICTOR_WARMUP_START_LR": os.environ["ROLL_WARMUP_START_LR"],
        "KRONOS_CONDITION_WARMUP_START_LR": os.environ["ROLL_WARMUP_START_LR"],
        "KRONOS_PREDICTOR_MIN_LR": os.environ["ROLL_WARMUP_START_LR"],
        "KRONOS_CONDITION_MIN_LR": os.environ["ROLL_WARMUP_START_LR"],
        "KRONOS_SNAPSHOT_EVERY_SEGMENTS": os.environ["ROLL_SNAPSHOT_EVERY"],
        "KRONOS_SNAPSHOT_DIR": str(ckpt_root),
        "KRONOS_REQUIRE_SWANLAB": "0",
    })
    tokenizer = os.environ.get("ROLL_TOKENIZER_DIR", "").strip()
    if tokenizer:
        env["KRONOS_SMALL_V21_TOKENIZER"] = tokenizer
    phase("training_started", environment={k: v for k, v in env.items() if k.startswith("KRONOS_")})
    subprocess.run([sys.executable, "-u", str(REPO / "finetune/kaggle_kronos_small_v21.py")],
                   cwd=str(REPO), env=env, check=True)

    output = runtime / "outputs/models" / output_name
    tokenizer_dir = Path(tokenizer) if tokenizer else runtime / "models/Kronos-Tokenizer-base"
    progress = json.loads((output / "progress.json").read_text())
    # Snapshots are the deliverable (seg001, seg002, ...).  Each must differ from
    # the parent and from every other one, or an evaluation would silently score
    # the wrong weights (a probe once exported the untouched parent as "best").
    checkpoints = {path.parent.name: sha256_file(path) for path in sorted(ckpt_root.glob("seg*/model.safetensors"))}
    checks = {
        "snapshots_distinct_from_parent_and_each_other":
            bool(checkpoints) and len(set(checkpoints.values()) | {parent_sha}) == len(checkpoints) + 1,
        "final_snapshot_is_last_model": bool(checkpoints) and checkpoints[max(checkpoints)]
            == sha256_file(output / "checkpoints/last_model/model.safetensors"),
        "tokenizer_is_evaluation_tokenizer": sha256_file(tokenizer_dir / "model.safetensors") == TOKENIZER_SHA256,
        "coverage_completed": progress.get("status") == "completed",
    }
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines() if line.strip()]
    validation = [{k: r.get(k) for k in ("segment", "forecast_loss", "history_loss", "full_sequence_loss")}
                  for r in records if r.get("type") in {"validation", "validation_large"}]
    summary = {"arm": arm, "parent_sha256": parent_sha, "contract": contract,
               "lr": os.environ["ROLL_LR"], "warmup_start_lr": os.environ["ROLL_WARMUP_START_LR"],
               "completed_segments": progress.get("completed_segments", progress.get("current_segment")),
               "total_segments": progress.get("total_segments"), "status": progress.get("status"),
               "total_train_windows": progress.get("total_train_windows"),
               "checks": checks, "checkpoints": checkpoints, "validation": validation}
    (work / "roll_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    for name in ("run.log", "metrics.jsonl", "progress.json"):
        shutil.copy2(output / name, work / f"roll_{name}")
    shutil.rmtree(runtime)        # panels, optimizer state and the tokenizer copy are not deliverables
    phase("finished", summary=summary)
    # Only an unusable result fails the kernel; a failed kernel cannot be mounted
    # by the evaluation kernels, so softer findings stay in the summary instead.
    if not checks["snapshots_distinct_from_parent_and_each_other"] or not checks["tokenizer_is_evaluation_tokenizer"]:
        raise RuntimeError(f"Unusable training output: {checks}")


if __name__ == "__main__":
    main()
