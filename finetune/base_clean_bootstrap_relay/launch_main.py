"""Kaggle entry for Base-clean **Main** (stage 2), mirroring Small→Stage2.

Bootstrap only warms up the condition branch with a conservative backbone LR
(1e-6).  Main is the real stage: weights from Bootstrap's `last_model`,
**fresh optimizer/scheduler**, every parameter at peak 1e-5, its own cosine
over one full coverage.

Two modes:
  * fresh cross-stage start (default): read Bootstrap output, take
    `checkpoints/last_model` only (no last_state → optimizer/scheduler are
    rebuilt).  This is why the chain is "weights-only" and differs from a
    same-stage relay.
  * same-stage relay chunk 2+ (`KRONOS_MAIN_CONTINUE=1`): mount the previous
    Main chunk's complete output, copy the whole tree, resume `last_state.pt`
    (optimizer/scheduler/RNG continue).

Validation defaults for Main (Base is ~4x Small, full holdout is expensive):
  * quick 2048 fixed subset every 5 segments   → best-by-forecast, paired
  * large 123,836 every 20 segments + chunk end → milestone metric
  * Best updates ON (unlike Bootstrap), so the stage produces a real `best_model`.
Override any of these with KRONOS_VALIDATION_* if the review wants full-val.

Wall clock: KRONOS_MAIN_RUNTIME_BUDGET_SECONDS (default 36000 = 10h) is passed
to the trainer as KRONOS_MAX_RUNTIME_SECONDS, measured from process start, so a
chunk always ends on a complete segment with a resumable `last_state.pt`.
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

STARTED = time.monotonic()

EXPERIMENT = "base_clean_0.1_main"
RUN_ID = "base_clean_0_1_main"
OUTPUT_NAME = "base_clean_0.1_main"
BOOTSTRAP_OUTPUT_NAME = "base_clean_0.1_bootstrap"
DATASET = "smmt315/a-share-120d-temporal-symbol-holdout"
DATA_SHA = "b85ccc45f7cd0e895e607e6e75c7a603427bd62a353a016742a52b2943d652cd"
TORCH_VERSION = "2.6.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
BATCH_SIZE = 32
ACCUMULATION_STEPS = 8
NPROC = 2
# Upper bound; the wall-clock budget normally fires first (~145 segs at 4m/seg).
MAX_SEGMENTS = int(os.environ.get("KRONOS_MAIN_MAX_SEGMENTS", "150"))
RUNTIME_BUDGET_SECONDS = int(
    os.environ.get("KRONOS_MAIN_RUNTIME_BUDGET_SECONDS", "36000")
)
# Optional pins for the Bootstrap → Main handoff.
EXPECTED_BOOTSTRAP_COMPLETED = int(
    os.environ.get("KRONOS_MAIN_EXPECTED_BOOTSTRAP_COMPLETED", "0") or 0
)
PARENT_LAST_SHA256 = os.environ.get("KRONOS_MAIN_PARENT_LAST_SHA256", "").strip()


def phase(name: str, **payload) -> None:
    print(json.dumps({"phase": name, "time": time.time(), **payload}, ensure_ascii=False), flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_swanlab_key() -> str:
    token = os.environ.get("SWANLAB_API_KEY", "").strip()
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient

        token = (UserSecretsClient().get_secret("SWANLAB_API_KEY") or "").strip()
    except Exception as exc:
        phase("swanlab_secret_unavailable", error=f"{type(exc).__name__}: {exc}")
        token = ""
    if token:
        os.environ["SWANLAB_API_KEY"] = token
    return token


def prepare_data(input_root: Path, target: Path) -> Path:
    """Materialize processed_datasets/ from the clean smmt315 mount."""
    candidates = [
        path for path in input_root.glob("**/data_manifest.json") if sha256(path) == DATA_SHA
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one clean data_manifest.json ({DATA_SHA[:12]}…); "
            f"found {len(candidates)}: {candidates}"
        )
    manifest_path = candidates[0]
    source = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    (target / "processed_datasets").mkdir(exist_ok=True)
    for entry in manifest["files"]:
        relative = Path(entry["name"])
        path = source / relative
        if not path.is_file():
            path = source / relative.name
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Clean dataset hash mismatch: {path}")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(path.resolve())
    manifest_dest = target / "data_manifest.json"
    if manifest_dest.exists() or manifest_dest.is_symlink():
        manifest_dest.unlink()
    manifest_dest.symlink_to(manifest_path.resolve())
    phase(
        "data_ready",
        dataset=DATASET,
        manifest_sha256=DATA_SHA,
        stats=manifest.get("stats"),
        window_contract=manifest.get("window_contract"),
        data_root=str(target),
    )
    return target


def find_output_root(input_root: Path, output_name: str, want_state: bool) -> Path:
    """Locate a complete relay output tree under /kaggle/input."""
    leaf = "last_state.pt" if want_state else "last_model/model.safetensors"
    hits = sorted(input_root.glob(f"**/{output_name}/checkpoints/{leaf}"))
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one {output_name} output; found {hits}")
    root = hits[0]
    while root.name != output_name and root.parent != root:
        root = root.parent
    if root.name != output_name:
        raise RuntimeError(f"Could not resolve {output_name} root from {hits[0]}")
    return root


def bootstrap_parent(input_root: Path) -> tuple[Path, dict]:
    """Weights-only handoff: Bootstrap last_model + a durable progress record."""
    root = find_output_root(input_root, BOOTSTRAP_OUTPUT_NAME, want_state=False)
    required = [
        "checkpoints/last_model/model.safetensors",
        "checkpoints/last_model/config.json",
        "progress.json",
        "summary.json",
        "base_clean_manifest.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete bootstrap output at {root}: {missing}")
    progress = json.loads((root / "progress.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    final = summary.get("final_result", summary)
    completed = int(final.get("completed_segments", progress.get("current_segment", 0)))
    if progress.get("status") not in {"stopped", "completed"}:
        raise RuntimeError(f"Bootstrap output is not durable: {progress}")
    if completed < 1:
        raise RuntimeError(f"Bootstrap output has no completed segments: {progress}")
    if EXPECTED_BOOTSTRAP_COMPLETED and completed != EXPECTED_BOOTSTRAP_COMPLETED:
        raise RuntimeError(
            f"Expected bootstrap completed_segments={EXPECTED_BOOTSTRAP_COMPLETED}; "
            f"got {completed}"
        )
    last = root / "checkpoints/last_model/model.safetensors"
    digest = sha256(last)
    if PARENT_LAST_SHA256 and digest != PARENT_LAST_SHA256:
        raise RuntimeError(
            f"Bootstrap last_model SHA mismatch: expected {PARENT_LAST_SHA256[:12]}… "
            f"got {digest[:12]}…"
        )
    info = {
        "parent_kernel_output": str(root),
        "approach": "weights_only",
        "bootstrap_completed_segments": completed,
        "bootstrap_resume_segment": int(progress.get("current_segment", completed)) + 1,
        "bootstrap_best_metric_segment": json.loads(
            (root / "checkpoints/best_model/best_metric.json").read_text()
        ).get("segment"),
        "last_model_sha256": digest,
        "optimizer_state_restored": False,
    }
    phase("parent_ready", **info)
    return root / "checkpoints/last_model", info


def main_relay_parent(input_root: Path) -> tuple[Path, dict]:
    """Same-stage chunk 2+: locate the previous Main output; the runner copies it."""
    source = find_output_root(input_root, OUTPUT_NAME, want_state=True)
    progress = json.loads((source / "progress.json").read_text())
    if progress.get("status") not in {"stopped", "completed"}:
        raise RuntimeError(f"Main parent not durable: {progress}")
    info = {
        "parent_kernel_output": str(source),
        "approach": "same_stage_resume",
        "resume_segment": int(progress.get("current_segment", 0)) + 1,
        "completed_segments": int(progress.get("current_segment", 0)),
        "optimizer_state_restored": True,
    }
    phase("parent_ready", **info)
    return source, info


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    finetune = repo / "finetune"
    if str(finetune) not in sys.path:
        sys.path.insert(0, str(finetune))

    continuity = os.environ.get("KRONOS_MAIN_CONTINUE", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    phase(
        "started",
        experiment=EXPERIMENT,
        stage="main",
        output=OUTPUT_NAME,
        dataset=DATASET,
        mode="same_stage_resume" if continuity else "bootstrap_weights_only",
        micro_bs=BATCH_SIZE // ACCUMULATION_STEPS,
        accum=ACCUMULATION_STEPS,
        n_gpu=NPROC,
        effective_global_batch=BATCH_SIZE * NPROC,
        updates_per_seg_approx=20000 // (BATCH_SIZE * NPROC),
        max_segments=MAX_SEGMENTS,
        runtime_budget_seconds=RUNTIME_BUDGET_SECONDS,
        swanlab_board=EXPERIMENT,
        note="Main: fresh optimizer/scheduler, peak LR 1e-5 on every parameter",
    )

    phase("install_dependencies", torch=TORCH_VERSION, note="only-binary pins; skip requirements.txt")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            f"torch=={TORCH_VERSION}",
            "--only-binary=:all:",
            "--index-url", TORCH_INDEX_URL,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            "--index-url", "https://pypi.org/simple",
            "--only-binary=numpy,pandas,safetensors,orjson,psutil,protobuf,pyyaml,hf-xet",
            "numpy==2.1.3", "pandas==2.2.3",
            "einops==0.8.1", "huggingface_hub==0.33.1", "safetensors==0.6.2",
            "tqdm==4.67.1", "swanlab==0.10.1", "psutil", "pynvml",
        ],
        check=True,
    )

    import torch

    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(devices) != 2 or any("T4" not in name for name in devices):
        raise RuntimeError(f"Expected dual T4; found {devices}")
    phase("device_ready", devices=devices, torch=torch.__version__, cuda=torch.version.cuda)

    token = resolve_swanlab_key()
    os.environ["KRONOS_REQUIRE_SWANLAB"] = "0" if not token else os.environ.get(
        "KRONOS_REQUIRE_SWANLAB", "1"
    )
    if not token:
        phase("swanlab_missing", note="Continuing with local logs only")

    for key in ("SWANLAB_PROJECT", "SWANLAB_WORKSPACE", "SWANLAB_EXPERIMENT_NAME"):
        os.environ.pop(key, None)

    runtime = Path(os.environ.get("KRONOS_BASE_CLEAN_RUNTIME", "/kaggle/working/kronos_base_clean"))
    input_root = Path(os.environ.get("KRONOS_KAGGLE_INPUT_ROOT", "/kaggle/input"))
    data_root = prepare_data(input_root, runtime / "data")
    output_root = runtime / "outputs/models" / OUTPUT_NAME
    parent_model, parent_info = (
        main_relay_parent(input_root)
        if continuity
        else bootstrap_parent(input_root)
    )

    # Wall-clock budget measured from process start, so pip/data prep count.
    remaining_budget = max(300, RUNTIME_BUDGET_SECONDS - int(time.monotonic() - STARTED))
    os.environ.update({
        "PYTHONUNBUFFERED": "1",
        "KRONOS_BASE_CLEAN_STAGE": "main",
        "KRONOS_BASE_CLEAN_OUTPUT_NAME": OUTPUT_NAME,
        "KRONOS_BASE_CLEAN_DATA_ROOT": str(data_root),
        "KRONOS_BASE_CLEAN_PARENT_MODEL": str(parent_model),
        # Same-stage relay must import the full output tree (last_state);
        # the Bootstrap → Main handoff must import weights only.
        "KRONOS_BASE_CLEAN_DISABLE_AUTO_CONTINUATION": "0" if continuity else "1",
        "KRONOS_MAX_SEGMENTS_PER_RUN": str(MAX_SEGMENTS),
        "KRONOS_MAX_RUNTIME_SECONDS": str(remaining_budget),
        "KRONOS_BATCH_SIZE": str(BATCH_SIZE),
        "KRONOS_ACCUMULATION_STEPS": str(ACCUMULATION_STEPS),
        "KRONOS_TORCHRUN_NPROC_PER_NODE": str(NPROC),
        # Main: best updates ON, quick telemetry every 5, full holdout every 20.
        "KRONOS_DISABLE_BEST_UPDATES": os.environ.get("KRONOS_DISABLE_BEST_UPDATES", "0"),
        "KRONOS_VALIDATION_FULL_ONLY": os.environ.get("KRONOS_VALIDATION_FULL_ONLY", "0"),
        "KRONOS_VALIDATION_QUICK_SAMPLES": os.environ.get("KRONOS_VALIDATION_QUICK_SAMPLES", "2048"),
        "KRONOS_VALIDATION_LARGE_SAMPLES": os.environ.get("KRONOS_VALIDATION_LARGE_SAMPLES", "123836"),
        "KRONOS_VALIDATION_INTERVAL_SEGMENTS": os.environ.get("KRONOS_VALIDATION_INTERVAL_SEGMENTS", "5"),
        "KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS": os.environ.get(
            "KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS", "20"
        ),
        "KRONOS_CONDITION_ABLATION_INTERVAL_SEGMENTS": os.environ.get(
            "KRONOS_CONDITION_ABLATION_INTERVAL_SEGMENTS", "20"
        ),
        "KRONOS_VAL_BATCH_SIZE": os.environ.get("KRONOS_VAL_BATCH_SIZE", "64"),
        "SWANLAB_PROJECT": "finance",
        "SWANLAB_WORKSPACE": "roc_fu",
        "SWANLAB_EXPERIMENT_NAME": EXPERIMENT,
        "SWANLAB_RUN_ID": RUN_ID,
    })
    if continuity:
        os.environ["KRONOS_BASE_CLEAN_CONTINUATION_ROOT"] = str(
            parent_info["parent_kernel_output"]
        )
        # The copied metrics.jsonl must not replay onto the same board.
        os.environ["KRONOS_SWANLAB_REPLAY_AFTER_SEGMENT"] = str(
            parent_info.get("completed_segments", 0)
        )
    else:
        os.environ.pop("KRONOS_BASE_CLEAN_CONTINUATION_ROOT", None)
        os.environ.pop("KRONOS_SWANLAB_REPLAY_AFTER_SEGMENT", None)

    phase(
        "launch_runner",
        runner="finetune/kaggle_kronos_base_clean.py",
        swanlab_experiment=EXPERIMENT,
        swanlab_run_id=RUN_ID,
        parent_model=str(parent_model),
        continue_same_stage=continuity,
        max_segments=MAX_SEGMENTS,
        max_runtime_seconds=remaining_budget,
        swanlab_replay_after_segment=os.environ.get("KRONOS_SWANLAB_REPLAY_AFTER_SEGMENT"),
        validation={
            "quick_samples": os.environ["KRONOS_VALIDATION_QUICK_SAMPLES"],
            "interval_segments": os.environ["KRONOS_VALIDATION_INTERVAL_SEGMENTS"],
            "large_samples": os.environ["KRONOS_VALIDATION_LARGE_SAMPLES"],
            "large_interval_segments": os.environ["KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS"],
            "disable_best_updates": os.environ["KRONOS_DISABLE_BEST_UPDATES"],
        },
    )
    subprocess.run(
        [sys.executable, "-u", str(finetune / "kaggle_kronos_base_clean.py")],
        cwd=str(repo),
        env=os.environ.copy(),
        check=True,
    )
    phase("completed", experiment=EXPERIMENT)


if __name__ == "__main__":
    main()
