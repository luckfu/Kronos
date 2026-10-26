"""Kaggle entry for Base-clean Bootstrap long-run chunk (continue timing seg1).

PREP/ready-to-push only until Elon nods. Continues same-stage bootstrap from
COMPLETE timing kernel output (resume_segment=2). New SwanLab board
base_clean_0.1_bootstrap (does NOT continue timing board).
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

EXPERIMENT = "base_clean_0.1_bootstrap"
OUTPUT_NAME = "base_clean_0.1_bootstrap"
RUN_ID = "base_clean_0_1_bootstrap"
DATASET = "smmt315/a-share-120d-temporal-symbol-holdout"
DATA_SHA = "b85ccc45f7cd0e895e607e6e75c7a603427bd62a353a016742a52b2943d652cd"
PARENT_KERNEL = "user281434/kronos-base-clean-0-1-bootstrap-timing"
PARENT_OUTPUT_NAME = "base_clean_0.1_bootstrap"
# Prefer best_model (Small pattern). After timing seg1, best_sha == last_sha.
PARENT_WEIGHTS = "best_model"
EXPECTED_RESUME_SEGMENT = 2
TORCH_VERSION = "2.6.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
BATCH_SIZE = 32
ACCUMULATION_STEPS = 8
NPROC = 2
MAX_SEGMENTS = 20  # ~29 min/seg × 20 ≈ 10h under 12h dual-T4 quota


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
    """Materialize processed_datasets/ from flat or nested smmt315 mount."""
    candidates = [
        path for path in input_root.glob("**/data_manifest.json")
        if sha256(path) == DATA_SHA
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
        split=manifest.get("split"),
        window_contract=manifest.get("window_contract"),
        data_root=str(target),
    )
    return target


def find_timing_continuation(input_root: Path) -> tuple[Path, dict]:
    """Locate COMPLETE timing output tree; prefer best_model, resume via last_state."""
    # Prefer paths under the known output name to avoid accidental matches.
    preferred = sorted(
        input_root.glob(f"**/{PARENT_OUTPUT_NAME}/checkpoints/last_state.pt")
    )
    if len(preferred) == 1:
        state_path = preferred[0]
    else:
        states = sorted(input_root.glob("**/last_state.pt"))
        if len(states) != 1:
            raise RuntimeError(
                f"Expected exactly one timing last_state.pt; "
                f"preferred={preferred} all={states}"
            )
        state_path = states[0]
    source_root = state_path.parent.parent
    required = [
        "checkpoints/last_state.pt",
        "checkpoints/best_model/model.safetensors",
        "checkpoints/best_model/config.json",
        "checkpoints/best_model/best_metric.json",
        "checkpoints/last_model/model.safetensors",
        "checkpoints/last_model/config.json",
        "metrics.jsonl",
        "progress.json",
        "summary.json",
        "base_clean_manifest.json",
    ]
    missing = [name for name in required if not (source_root / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete timing continuation at {source_root}: {missing}")

    progress = json.loads((source_root / "progress.json").read_text())
    summary = json.loads((source_root / "summary.json").read_text())
    final = summary.get("final_result", summary)
    if progress.get("status") not in {"stopped", "completed"}:
        raise RuntimeError(f"Timing parent not durable: {progress}")
    resume_segment = int(final.get("resume_segment", int(progress.get("current_segment", 0)) + 1))
    completed = int(final.get("completed_segments", progress.get("current_segment", 0)))
    if resume_segment != EXPECTED_RESUME_SEGMENT or completed != 1:
        raise RuntimeError(
            f"Expected timing seg1 → resume_segment={EXPECTED_RESUME_SEGMENT}; "
            f"got completed={completed} resume_segment={resume_segment}"
        )

    best = source_root / "checkpoints/best_model/model.safetensors"
    last = source_root / "checkpoints/last_model/model.safetensors"
    best_sha = sha256(best)
    last_sha = sha256(last)
    best_metric = json.loads(
        (source_root / "checkpoints/best_model/best_metric.json").read_text()
    )
    info = {
        "parent_kernel": PARENT_KERNEL,
        "continuation_root": str(source_root),
        "parent_weights": PARENT_WEIGHTS,
        "parent_weights_note": (
            "Same-stage continuation copies full output tree (last_state drives "
            "optimizer resume). Preferred tag is best_model (Small pattern)."
        ),
        "best_model_sha256": best_sha,
        "last_model_sha256": last_sha,
        "best_equals_last": best_sha == last_sha,
        "best_metric_segment": best_metric.get("segment"),
        "best_selection_loss": best_metric.get("selection_loss"),
        "completed_segments": completed,
        "resume_segment": resume_segment,
        "progress_status": progress.get("status"),
    }
    phase("parent_ready", **info)
    return source_root, info


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    finetune = repo / "finetune"
    if str(finetune) not in sys.path:
        sys.path.insert(0, str(finetune))

    phase(
        "started",
        experiment=EXPERIMENT,
        output=OUTPUT_NAME,
        dataset=DATASET,
        parent_kernel=PARENT_KERNEL,
        parent_weights=PARENT_WEIGHTS,
        resume_segment=EXPECTED_RESUME_SEGMENT,
        micro_bs=BATCH_SIZE // ACCUMULATION_STEPS,
        accum=ACCUMULATION_STEPS,
        n_gpu=NPROC,
        effective_global_batch=BATCH_SIZE * NPROC,
        updates_per_seg_approx=20000 // (BATCH_SIZE * NPROC),
        max_segments=MAX_SEGMENTS,
        swanlab_board=EXPERIMENT,
        note="Continue timing seg1; NEW SwanLab board (not timing); HF-clean lineage",
    )

    phase(
        "install_dependencies",
        torch=TORCH_VERSION,
        numpy="2.1.3",
        pandas="2.2.3",
        note="e01/timing-v3 only-binary pins; skip requirements.txt",
    )
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
    if not token:
        phase(
            "swanlab_missing",
            note="Continuing with local logs only; set SWANLAB_API_KEY secret for remote board",
        )
        os.environ["KRONOS_REQUIRE_SWANLAB"] = "0"
    else:
        os.environ["KRONOS_REQUIRE_SWANLAB"] = os.environ.get("KRONOS_REQUIRE_SWANLAB", "0")

    for key in ("SWANLAB_PROJECT", "SWANLAB_WORKSPACE", "SWANLAB_EXPERIMENT_NAME"):
        os.environ.pop(key, None)

    input_root = Path(os.environ.get("KRONOS_KAGGLE_INPUT_ROOT", "/kaggle/input"))
    data_root = prepare_data(input_root, Path("/kaggle/working/kronos_base_clean/data"))
    continuation_root, parent_info = find_timing_continuation(input_root)

    # Write a small sidecar so Output documents which parent weights were preferred.
    sidecar = Path("/kaggle/working/bootstrap_parent_manifest.json")
    sidecar.write_text(json.dumps(parent_info, indent=2, ensure_ascii=False) + "\n")

    # Same-board / new-board pitfall (Small already hit): copied parent metrics.jsonl
    # must NOT be replayed onto a fresh SwanLab board, or segment curves go 1→2→1
    # when total_steps in old records ≠ live updates/seg. Mirror small_power_relay:
    # skip parent segments ≤ completed_segments.
    replay_after = int(parent_info.get("completed_segments", 1))
    os.environ.update({
        "PYTHONUNBUFFERED": "1",
        "KRONOS_BASE_CLEAN_STAGE": "bootstrap",
        "KRONOS_BASE_CLEAN_OUTPUT_NAME": OUTPUT_NAME,
        # Enable same-stage continuation from timing Output (do NOT disable).
        "KRONOS_BASE_CLEAN_DISABLE_AUTO_CONTINUATION": "0",
        "KRONOS_BASE_CLEAN_CONTINUATION_ROOT": str(continuation_root),
        "KRONOS_BASE_CLEAN_DATA_ROOT": str(data_root),
        "KRONOS_MAX_SEGMENTS_PER_RUN": str(MAX_SEGMENTS),
        "KRONOS_BATCH_SIZE": str(BATCH_SIZE),
        "KRONOS_ACCUMULATION_STEPS": str(ACCUMULATION_STEPS),
        "KRONOS_TORCHRUN_NPROC_PER_NODE": str(NPROC),
        "KRONOS_SWANLAB_REPLAY_AFTER_SEGMENT": str(replay_after),
        "SWANLAB_PROJECT": "finance",
        "SWANLAB_WORKSPACE": "roc_fu",
        "SWANLAB_EXPERIMENT_NAME": EXPERIMENT,
        "SWANLAB_RUN_ID": RUN_ID,
    })

    phase(
        "launch_runner",
        runner="finetune/kaggle_kronos_base_clean.py",
        swanlab_experiment=EXPERIMENT,
        swanlab_run_id=RUN_ID,
        data_root=str(data_root),
        continuation_root=str(continuation_root),
        resume_segment=parent_info["resume_segment"],
        max_segments=MAX_SEGMENTS,
        parent_weights=PARENT_WEIGHTS,
        best_equals_last=parent_info["best_equals_last"],
        swanlab_replay_after_segment=replay_after,
    )
    subprocess.run(
        [sys.executable, "-u", str(finetune / "kaggle_kronos_base_clean.py")],
        cwd=str(repo),
        env=os.environ.copy(),
        check=True,
    )
    phase("completed", experiment=EXPERIMENT, parent_kernel=PARENT_KERNEL)


if __name__ == "__main__":
    main()
