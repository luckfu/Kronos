"""Standalone Kaggle runner for a fresh Kronos-small A-share V2.1 line.

This file intentionally does not modify or wrap the existing Beta V2.1 launch
scripts. It configures the existing ``finetune/train_predictor.py`` through
environment variables and keeps each stage in a separate output tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


EXPERIMENT_NAME = "small_0.1"
TRAIN_LOG_RE = re.compile(
    r"Segment (\d+)/(\d+), Step (\d+)/(\d+).*?"
    r"Adaptation LR ([0-9.eE+-]+), Condition LR ([0-9.eE+-]+), "
    r"Loss: ([0-9.]+), Forecast: ([0-9.]+), History: ([0-9.]+)"
)
VALIDATION_LOG_RE = re.compile(
    r"Validation Forecast/History/Full: ([0-9.]+) / ([0-9.]+) / ([0-9.]+)"
)


STAGES = {
    "bootstrap": {
        "output": "small_0.1_bootstrap",
        "parent_env": "KRONOS_SMALL_V21_BASE_MODEL",
        "predictor_lr": "1e-6",
        "condition_lr": "1e-5",
        "warmup_start_lr": "1e-7",
        "condition_warmup_start_lr": "1e-6",
        "auxiliary": "0",
        "validation_full_only": "1",
        "best_metric": "forecast",
        "max_segments": "10",
    },
    "main": {
        "output": "small_0.1_main",
        "parent_env": "KRONOS_SMALL_V21_PARENT_MODEL",
        "predictor_lr": "1e-5",
        "condition_lr": "1e-5",
        "warmup_start_lr": "1e-6",
        "condition_warmup_start_lr": "1e-6",
        "auxiliary": "0",
        "validation_full_only": "1",
        "best_metric": "forecast",
        "max_segments": "120",
    },
    "v21": {
        "output": "small_0.1_decision",
        "parent_env": "KRONOS_SMALL_V21_PARENT_MODEL",
        "predictor_lr": "5e-6",
        "condition_lr": "5e-6",
        "warmup_start_lr": "5e-7",
        "condition_warmup_start_lr": "5e-7",
        "auxiliary": "1",
        "validation_full_only": "1",
        "best_metric": "beta_v21_score",
        "max_segments": "120",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_data_root(input_root: Path) -> Path:
    """Find exactly one mounted data root containing train and val panels."""
    explicit = os.getenv("KRONOS_SMALL_V21_DATA_ROOT", "").strip()
    if explicit:
        root = Path(explicit).expanduser().resolve()
        candidates = [root]
    else:
        candidates = sorted({
            path.parent.parent
            for path in input_root.glob("**/processed_datasets/train_data.pkl")
            if (path.parent / "val_data.pkl").is_file()
        })
    if len(candidates) != 1:
        raise SystemExit(
            "Expected exactly one data root with train_data.pkl and val_data.pkl; "
            f"found {len(candidates)}: {candidates}. Set "
            "KRONOS_SMALL_V21_DATA_ROOT explicitly if the dataset has multiple bundles."
        )
    root = candidates[0]
    required = [
        root / "processed_datasets/train_data.pkl",
        root / "processed_datasets/val_data.pkl",
    ]
    if not all(path.is_file() for path in required):
        raise SystemExit(f"Incomplete data root: {root}; expected {required}")
    return root


def metadata_path(data_root: Path) -> Path:
    """Require point-in-time metadata for the static V2.1 conditions."""
    explicit = os.getenv("KRONOS_SMALL_V21_METADATA_PATH", "").strip()
    path = Path(explicit).expanduser().resolve() if explicit else data_root / "asset_metadata.csv"
    if not path.is_file() and os.getenv("KRONOS_SMALL_V21_ALLOW_MISSING_METADATA", "0") not in {"1", "true", "yes"}:
        raise SystemExit(
            "Static sector and size-percentile training requires asset_metadata.csv; "
            f"missing {path}. Set KRONOS_SMALL_V21_METADATA_PATH or explicitly "
            "opt into unknown-only conditions with KRONOS_SMALL_V21_ALLOW_MISSING_METADATA=1."
        )
    return path


def fixed_validation_manifest(repo_root: Path, data_root: Path) -> tuple[Path, dict]:
    """Resolve and verify the fixed ~20K validation selection."""
    explicit = os.getenv("KRONOS_SMALL_V21_FIXED_VALIDATION_MANIFEST", "").strip()
    path = (
        Path(explicit).expanduser().resolve()
        if explicit
        else repo_root
        / "finetune/manifests/small_0_1_balanced_validation_20k_v1/"
        "balanced_validation_20k_manifest.json"
    )
    if not path.is_file():
        raise SystemExit(f"Fixed validation manifest is missing: {path}")
    manifest = json.loads(path.read_text())
    source = manifest.get("source", {})
    expected_data_sha = source.get("data_manifest_sha256")
    expected_val_sha = source.get("val_data_sha256")
    actual_data_sha = sha256_file(data_root / "data_manifest.json")
    actual_val_sha = sha256_file(data_root / "processed_datasets/val_data.pkl")
    if expected_data_sha != actual_data_sha or expected_val_sha != actual_val_sha:
        raise SystemExit(
            "Fixed validation source mismatch: "
            f"data {actual_data_sha} != {expected_data_sha}; "
            f"val {actual_val_sha} != {expected_val_sha}"
        )
    return path, manifest


def ensure_hf_snapshot(repo_id: str, target: Path) -> Path:
    """Download a public model only when a complete local snapshot is absent."""
    if (target / "config.json").is_file() and (target / "model.safetensors").is_file():
        return target
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.33.1"],
        check=True,
    )
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id, local_dir=str(target))
    required = [target / "config.json", target / "model.safetensors"]
    if not all(path.is_file() for path in required):
        raise SystemExit(f"Hugging Face snapshot is incomplete: {target}")
    return target


def resolve_model(source_env: str, default_repo: str, target: Path) -> Path:
    explicit = os.getenv(source_env, "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not (path / "config.json").is_file() or not (path / "model.safetensors").is_file():
            raise SystemExit(f"Configured model is incomplete: {path}")
        return path
    local_small = Path(__file__).resolve().parents[1] / "small_model"
    if default_repo == "NeoQuasar/Kronos-small" and (
        local_small / "config.json"
    ).is_file() and (local_small / "model.safetensors").is_file():
        return local_small
    return ensure_hf_snapshot(default_repo, target)


def resolve_stage_predictor(stage: str, runtime: Path) -> Path:
    if stage == "bootstrap":
        return resolve_model(
            "KRONOS_SMALL_V21_BASE_MODEL",
            "NeoQuasar/Kronos-small",
            runtime / "models/Kronos-small",
        )
    explicit = os.getenv("KRONOS_SMALL_V21_PARENT_MODEL", "").strip()
    if not explicit:
        raise SystemExit(
            f"Stage {stage} requires KRONOS_SMALL_V21_PARENT_MODEL pointing to "
            "the previous stage's best_model or last_model directory"
        )
    return resolve_model(
        "KRONOS_SMALL_V21_PARENT_MODEL",
        "NeoQuasar/Kronos-small",
        runtime / "models/Kronos-small-parent",
    )


def copy_continuation_if_requested(output_root: Path) -> bool:
    """Import a complete prior output tree for same-stage Kaggle continuation."""
    if os.getenv("KRONOS_SMALL_V21_DISABLE_AUTO_CONTINUATION", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        print("Same-stage continuation disabled for cross-stage initialization.", flush=True)
        return False
    source = os.getenv("KRONOS_SMALL_V21_CONTINUATION_ROOT", "").strip()
    if source:
        source_root = Path(source).expanduser().resolve()
    else:
        input_root = Path(os.getenv("KRONOS_KAGGLE_INPUT_ROOT", "/kaggle/input"))
        state_paths = sorted(input_root.glob("**/last_state.pt"))
        if not state_paths:
            return False
        if len(state_paths) != 1:
            raise SystemExit(
                "Expected exactly one continuation last_state.pt under Kaggle input; "
                f"found {len(state_paths)}: {state_paths}"
            )
        source_root = state_paths[0].parent.parent
        print(f"Auto-discovered continuation output: {source_root}", flush=True)
    required_files = [
        "checkpoints/last_state.pt",
        "checkpoints/best_model/model.safetensors",
        "checkpoints/best_model/config.json",
        "checkpoints/best_model/best_metric.json",
        "checkpoints/last_model/model.safetensors",
        "checkpoints/last_model/config.json",
        "metrics.jsonl",
        "progress.json",
        "summary.json",
        "small_v21_manifest.json",
    ]
    missing = [name for name in required_files if not (source_root / name).is_file()]
    if missing:
        raise SystemExit(
            f"Continuation output contract is incomplete at {source_root}: {missing}"
        )
    progress = json.loads((source_root / "progress.json").read_text())
    if progress.get("status") not in {"stopped", "completed"}:
        raise SystemExit(f"Continuation output is not durable: {progress}")
    state = source_root / "checkpoints/last_state.pt"
    historical_metrics = sum(
        1 for line in (source_root / "metrics.jsonl").read_text().splitlines() if line.strip()
    )
    print(
        json.dumps({
            "continuation_output": str(source_root),
            "resume_state": str(state),
            "historical_metrics_lines": historical_metrics,
            "resume": True,
            "next_epoch": progress.get("current_segment", "unknown"),
        }, ensure_ascii=False),
        flush=True,
    )
    if output_root.exists():
        if any(output_root.iterdir()):
            raise SystemExit(f"Output already exists and is not empty: {output_root}")
    else:
        output_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, output_root, dirs_exist_ok=True)
    return True


def build_environment(stage: str, data_root: Path, predictor: Path, tokenizer: Path, runtime: Path, output_name: str) -> dict[str, str]:
    settings = STAGES[stage]
    output_root = runtime / "outputs/models" / output_name
    env = os.environ.copy()
    metadata = metadata_path(data_root)
    data_manifest_path = data_root / "data_manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text())
    window_contract = data_manifest.get("window_contract", {})
    env.update({
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "KRONOS_DATASET_PATH": str(data_root / "processed_datasets"),
        "KRONOS_TRAIN_DATA_PATHS": str(data_root / "processed_datasets/train_data.pkl"),
        "KRONOS_VAL_DATA_PATHS": str(data_root / "processed_datasets/val_data.pkl"),
        "KRONOS_METADATA_PATH": str(metadata),
        "KRONOS_DATA_MANIFEST_SHA256": sha256_file(data_manifest_path),
        "KRONOS_PREDICTOR_PATH": str(predictor),
        "KRONOS_TOKENIZER_PATH": str(tokenizer),
        "KRONOS_SAVE_PATH": str(runtime / "outputs/models"),
        "KRONOS_PREDICTOR_SAVE_FOLDER": output_name,
        "KRONOS_LOOKBACK_WINDOW": "120",
        "KRONOS_PREDICT_WINDOW": "10",
        "KRONOS_CONTEXT_LAYER": "6",
        "KRONOS_NUM_SECTORS": "86",
        "KRONOS_NUM_SIZE_BUCKETS": "0",
        "KRONOS_USE_SECTOR_FEATURES": "1",
        "KRONOS_USE_SIZE_FEATURES": "0",
        "KRONOS_USE_SIZE_PERCENTILE": "1",
        "KRONOS_TRAINABLE_TRANSFORMER_LAYERS": "-1",
        "KRONOS_RESET_SECTOR_EMBEDDING": "1" if stage == "bootstrap" else "0",
        "KRONOS_RESET_SIZE_EMBEDDING": "1" if stage == "bootstrap" else "0",
        "KRONOS_PREDICTOR_LOSS_MODE": "forecast",
        "KRONOS_HISTORY_LOSS_WEIGHT": "0.02",
        "KRONOS_FORECAST_HORIZON_WEIGHTS": "1.364,1.364,1.364,1.136,1.136,0.909,0.909,0.682,0.682,0.455",
        "KRONOS_SCHEDULER": "warmup_cosine",
        "KRONOS_SCHEDULER_WARMUP_RATIO": "0.01",
        "KRONOS_PREDICTOR_LEARNING_RATE": settings["predictor_lr"],
        "KRONOS_CONDITION_LEARNING_RATE": settings["condition_lr"],
        "KRONOS_PREDICTOR_WARMUP_START_LR": settings["warmup_start_lr"],
        "KRONOS_CONDITION_WARMUP_START_LR": settings["condition_warmup_start_lr"],
        "KRONOS_PREDICTOR_MIN_LR": "1e-7",
        "KRONOS_CONDITION_MIN_LR": "1e-7",
        "KRONOS_BEST_SELECTION_METRIC": settings["best_metric"],
        "KRONOS_USE_BETA_V21_AUXILIARY": settings["auxiliary"],
        "KRONOS_BETA_V21_AUTO_CALIBRATE": "1" if stage == "v21" else "0",
        "KRONOS_BETA_V21_AUXILIARY_WARMUP_STEPS": "1000",
        "KRONOS_BETA_V21_EMA_DECAY": "0.99",
        "KRONOS_BETA_V21_CONSISTENCY_SAMPLES": "2048",
        "KRONOS_TRAIN_SAMPLES_PER_SEGMENT": "20000",
        "KRONOS_COVERAGE_SEED": os.getenv("KRONOS_COVERAGE_SEED", "100"),
        "KRONOS_COVERAGE_PASSES": "1",
        "KRONOS_EPOCHS": "1",
        "KRONOS_REQUIRE_FULL_COVERAGE": "1",
        "KRONOS_EARLY_STOPPING_PATIENCE": "0",
        "KRONOS_VALIDATION_FULL_ONLY": settings["validation_full_only"],
        "KRONOS_VALIDATION_SAMPLES": "0",
        "KRONOS_VALIDATION_QUICK_SAMPLES": "0",
        "KRONOS_VALIDATION_LARGE_SAMPLES": "0",
        "KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS": "1000000",
        "KRONOS_BATCH_SIZE": os.getenv("KRONOS_BATCH_SIZE", "32"),
        "KRONOS_NUM_WORKERS": os.getenv("KRONOS_NUM_WORKERS", "2"),
        "KRONOS_USE_AMP": "1",
        "KRONOS_AMP_DTYPE": "float16",
        "KRONOS_MAX_SEGMENTS_PER_RUN": os.getenv(
            "KRONOS_MAX_SEGMENTS_PER_RUN", settings["max_segments"]
        ),
        "KRONOS_RESUME_TRAINING": "1" if (output_root / "checkpoints/last_state.pt").is_file() else "0",
        "PYTHONUNBUFFERED": "1",
    })
    signal_start = window_contract.get("validation_signal_start")
    signal_end = window_contract.get("validation_signal_end")
    if signal_start:
        env["KRONOS_VAL_SIGNAL_START"] = str(signal_start)
    if signal_end:
        env["KRONOS_VAL_SIGNAL_END"] = str(signal_end)
    if stage == "bootstrap":
        fixed_manifest_path, fixed_manifest = fixed_validation_manifest(
            Path(__file__).resolve().parents[1], data_root
        )
        fixed_selection = fixed_manifest["selection"]
        env.update({
            "KRONOS_FIXED_VALIDATION_MANIFEST_PATH": str(fixed_manifest_path),
            "KRONOS_FIXED_VALIDATION_MANIFEST_SHA256": sha256_file(fixed_manifest_path),
            "KRONOS_EXCLUDE_FIXED_VALIDATION_FROM_TRAINING": "1",
            "KRONOS_VALIDATION_QUICK_SAMPLES": str(fixed_selection["quick_samples"]),
            "KRONOS_VALIDATION_LARGE_SAMPLES": str(fixed_selection["large_samples"]),
        })
    else:
        env.pop("KRONOS_FIXED_VALIDATION_MANIFEST_PATH", None)
        env.pop("KRONOS_FIXED_VALIDATION_MANIFEST_SHA256", None)
        env["KRONOS_EXCLUDE_FIXED_VALIDATION_FROM_TRAINING"] = "0"
    return env


def write_manifest(output_root: Path, stage: str, data_root: Path, predictor: Path, tokenizer: Path) -> None:
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "stage": stage,
        "lineage": {
            "parent": "NeoQuasar/Kronos-small" if stage == "bootstrap" else os.getenv("KRONOS_SMALL_V21_PARENT_MODEL", ""),
            "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
            "predictor_sha256": sha256_file(predictor / "model.safetensors"),
            "tokenizer_sha256": sha256_file(tokenizer / "model.safetensors"),
        },
        "data": {
            "root": str(data_root),
            "train_sha256": sha256_file(data_root / "processed_datasets/train_data.pkl"),
            "val_sha256": sha256_file(data_root / "processed_datasets/val_data.pkl"),
        },
        "architecture": {
            "model": "Kronos-small",
            "n_layers": 8,
            "d_model": 512,
            "n_heads": 8,
            "ff_dim": 1024,
            "context_layer": 6,
            "physical_layer_expansion": False,
        },
        "validation": {
            "selection": STAGES[stage]["best_metric"],
            "full_symbol_holdout_required": stage != "bootstrap",
            "profile": "balanced_20k" if stage == "bootstrap" else "full_natural",
            "quick_validation_is_telemetry_only": True,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "small_v21_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


def run_training_with_swanlab(repo_root: Path, env: dict[str, str]) -> None:
    """Stream trainer metrics to SwanLab while preserving the trainer process."""
    # The Kaggle wrapper must not buffer either side of the pipe.  Without
    # this, the child can be healthy while the CLI appears silent for minutes.
    env["PYTHONUNBUFFERED"] = "1"
    project = env.get("SWANLAB_PROJECT", "finance")
    workspace = env.get("SWANLAB_WORKSPACE", "roc_fu")
    experiment_name = env.get("SWANLAB_EXPERIMENT_NAME", EXPERIMENT_NAME)
    run_id = env.get("SWANLAB_RUN_ID", "small_0_1_bootstrap")
    # Recent SwanLab versions parse SWANLAB_PROJECT as a structured SDK setting.
    # Keep our scalar names explicit in init() instead of exposing them to that parser.
    env.pop("SWANLAB_PROJECT", None)
    env.pop("SWANLAB_WORKSPACE", None)
    env.pop("SWANLAB_EXPERIMENT_NAME", None)
    # SwanLab reads the current process environment during init(), so clearing
    # only the child-process environment is insufficient on Kaggle.
    os.environ.pop("SWANLAB_PROJECT", None)
    os.environ.pop("SWANLAB_WORKSPACE", None)
    os.environ.pop("SWANLAB_EXPERIMENT_NAME", None)
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "swanlab"], check=True)
        import swanlab
        api_key = os.getenv("SWANLAB_API_KEY", "").strip()
        if api_key:
            swanlab.login(api_key=api_key)
        run = swanlab.init(
            id=run_id,
            resume="allow",
            project=project,
            workspace=workspace,
            experiment_name=experiment_name,
            tags=["kronos-small", "small_0.1", env.get("KRONOS_SMALL_V21_STAGE", "bootstrap")],
            config={key: env[key] for key in (
                "KRONOS_LOOKBACK_WINDOW", "KRONOS_PREDICT_WINDOW", "KRONOS_CONTEXT_LAYER",
                "KRONOS_PREDICTOR_LEARNING_RATE", "KRONOS_CONDITION_LEARNING_RATE",
            )},
        )
        metrics_path = Path(env["KRONOS_SAVE_PATH"]) / env["KRONOS_PREDICTOR_SAVE_FOLDER"] / "metrics.jsonl"
        if metrics_path.is_file():
            for line in metrics_path.read_text().splitlines():
                try:
                    record = json.loads(line)
                    record_type = record.get("type")
                    if record_type == "train":
                        step = int(record.get("step", 0))
                        segment = int(record.get("segment", 1))
                        total_steps = int(record.get("total_steps", 625))
                        run.log({
                            "train/loss": float(record["loss"]),
                            "train/forecast_loss": float(record["forecast_loss"]),
                            "train/history_loss": float(record["history_loss"]),
                            "segment": segment,
                        }, step=(segment - 1) * total_steps + step)
                    elif record_type in {"validation", "validation_large"}:
                        segment = int(record.get("segment", 1))
                        total_steps = int(record.get("total_steps", 625))
                        run.log({
                            "validation/forecast_loss": float(record["forecast_loss"]),
                            "validation/history_loss": float(record["history_loss"]),
                            "validation/full_loss": float(record["full_sequence_loss"]),
                            "segment": segment,
                        }, step=segment * total_steps)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
    except Exception as exc:
        raise SystemExit(f"SwanLab initialization failed before training: {type(exc).__name__}: {exc}") from exc

    child = subprocess.Popen(
        [sys.executable, "-u", str(repo_root / "finetune/train_predictor.py")],
        cwd=repo_root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    log_path = Path(env["KRONOS_SAVE_PATH"]) / env["KRONOS_PREDICTOR_SAVE_FOLDER"] / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", buffering=1)
    segment = 0
    total_steps = 1
    try:
        for line in child.stdout:
            # Persist first, then forward. Both writes are line-buffered and
            # the explicit flush makes the parent visible to Kaggle capture.
            log_handle.write(line)
            log_handle.flush()
            print(line, end="", flush=True)
            match = TRAIN_LOG_RE.search(line)
            if match:
                segment, _, step, total_steps, lr, condition_lr, loss, forecast, history = match.groups()
                segment, step, total_steps = int(segment), int(step), int(total_steps)
                run.log({"train/loss": float(loss), "train/forecast_loss": float(forecast),
                         "train/history_loss": float(history), "train/learning_rate": float(lr),
                         "train/condition_learning_rate": float(condition_lr), "segment": segment},
                        step=(segment - 1) * total_steps + step)
            match = VALIDATION_LOG_RE.search(line)
            if match and segment:
                forecast, history, full = map(float, match.groups())
                run.log({"validation/forecast_loss": forecast, "validation/history_loss": history,
                         "validation/full_loss": full, "segment": segment}, step=segment * total_steps)
    finally:
        return_code = child.wait()
        log_handle.close()
    swanlab.finish()
    if return_code:
        raise subprocess.CalledProcessError(return_code, child.args)


def run(stage: str | None = None) -> None:
    stage = stage or os.getenv("KRONOS_SMALL_V21_STAGE", "bootstrap").strip().lower()
    if stage not in STAGES:
        raise SystemExit(f"Unsupported stage {stage!r}; choose one of {sorted(STAGES)}")
    repo_root = Path(__file__).resolve().parents[1]
    runtime = Path(os.getenv("KRONOS_SMALL_V21_RUNTIME", "/kaggle/working/kronos_small_v21"))
    input_root = Path(os.getenv("KRONOS_KAGGLE_INPUT_ROOT", "/kaggle/input"))
    data_root = find_data_root(input_root)
    metadata_path(data_root)
    model_root = runtime / "models"
    predictor = resolve_stage_predictor(stage, runtime)
    if stage == "bootstrap":
        tokenizer = resolve_model(
            "KRONOS_SMALL_V21_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base", model_root / "Kronos-Tokenizer-base"
        )
    else:
        tokenizer = resolve_model(
            "KRONOS_SMALL_V21_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base", model_root / "Kronos-Tokenizer-base"
        )

    output_name = STAGES[stage]["output"]
    output_root = runtime / "outputs/models" / output_name
    copy_continuation_if_requested(output_root)
    env = build_environment(stage, data_root, predictor, tokenizer, runtime, output_name)
    write_manifest(output_root, stage, data_root, predictor, tokenizer)

    print(json.dumps({
        "stage": stage,
        "data_root": str(data_root),
        "predictor": str(predictor),
        "tokenizer": str(tokenizer),
        "output_root": str(output_root),
        "resume": env["KRONOS_RESUME_TRAINING"],
        "predictor_sha256": sha256_file(predictor / "model.safetensors"),
    }, ensure_ascii=False, indent=2), flush=True)
    env["KRONOS_SMALL_V21_STAGE"] = stage
    run_training_with_swanlab(repo_root, env)
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "finetune/export_last_model.py"),
            "--repo-root",
            str(repo_root),
            "--output-root",
            str(output_root),
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    run()
