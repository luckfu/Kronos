"""Standalone Kaggle runner for Kronos-base clean A-share line (no Beta weights).

Mirrors ``kaggle_kronos_base_clean.py`` but starts from HF ``NeoQuasar/Kronos-base``
with ``context_layer=10``. Does not resume Beta/V6/smce2 checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path


EXPERIMENT_NAME = "base_clean_0.1"
TRAIN_LOG_RE = re.compile(
    r"Segment (\d+)/(\d+), Step (\d+)/(\d+).*?"
    r"Adaptation LR ([0-9.eE+-]+), Condition LR ([0-9.eE+-]+), "
    r"Loss: ([0-9.]+), Forecast: ([0-9.]+), History: ([0-9.]+)"
)
VALIDATION_LOG_RE = re.compile(
    r"Validation Forecast/History/Full: ([0-9.]+) / ([0-9.]+) / ([0-9.]+)"
)
RANK_IC_LOG_RE = re.compile(
    r"Temporal Rank IC selection: daily_rank_ic_mean=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)"
)


def swanlab_chart_segment(local_segment, env=None):
    """Map a local coverage segment onto a continued SwanLab x-axis."""
    source = env if env is not None else os.environ
    offset = max(0, int(source.get("KRONOS_SWANLAB_SEGMENT_OFFSET", "0") or 0))
    return int(local_segment) + offset


def globalize_training_log(line, env=None):
    """Display staged training progress on the experiment-wide segment axis."""
    source = env if env is not None else os.environ
    offset = max(0, int(source.get("KRONOS_SWANLAB_SEGMENT_OFFSET", "0") or 0))
    if not offset:
        return line
    match = TRAIN_LOG_RE.search(line)
    if match is None:
        return line
    local_segment = int(match.group(1))
    local_total = int(match.group(2))
    start, end = match.span(0)
    matched = match.group(0).replace(
        f"Segment {local_segment}/{local_total}",
        f"Segment {local_segment + offset}/{local_total + offset}",
        1,
    )
    return line[:start] + matched + line[end:]


def swanlab_chart_step(segment, step, total_steps, env=None):
    source = env if env is not None else os.environ
    if source.get('KRONOS_TRAIN_SAMPLING') == 'capped_power_mixture_v1':
        boundary = int(source['KRONOS_SAMPLING_START_SEGMENT'])
        origin = int(source['KRONOS_SAMPLING_PARENT_GLOBAL_STEP'])
        if int(segment) <= boundary:
            raise ValueError('Cannot log sampling-stage metrics before parent boundary')
        return origin + (int(segment) - boundary - 1) * int(total_steps) + int(step)
    return (swanlab_chart_segment(segment, source) - 1) * int(total_steps) + int(step)


STAGES = {
    "bootstrap": {
        "output": "base_clean_0.1_bootstrap",
        "parent_env": "KRONOS_BASE_CLEAN_BASE_MODEL",
        "predictor_lr": "1e-6",
        "condition_lr": "1e-5",
        "warmup_start_lr": "1e-7",
        "condition_warmup_start_lr": "1e-6",
        "auxiliary": "0",
        # Timing/health: quick telemetry only; full-val deferred to later segs.
        "validation_full_only": "0",
        "best_metric": "forecast",
        "max_segments": "10",
    },
    "main": {
        "output": "base_clean_0.1_main",
        "parent_env": "KRONOS_BASE_CLEAN_PARENT_MODEL",
        "predictor_lr": "1e-5",
        "condition_lr": "1e-5",
        "warmup_start_lr": "1e-6",
        "condition_warmup_start_lr": "1e-6",
        "auxiliary": "0",
        "validation_full_only": "1",
        "best_metric": "forecast",
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
    explicit = os.getenv("KRONOS_BASE_CLEAN_DATA_ROOT", "").strip()
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
            "KRONOS_BASE_CLEAN_DATA_ROOT explicitly if the dataset has multiple bundles."
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
    explicit = os.getenv("KRONOS_BASE_CLEAN_METADATA_PATH", "").strip()
    path = Path(explicit).expanduser().resolve() if explicit else data_root / "asset_metadata.csv"
    if not path.is_file() and os.getenv("KRONOS_BASE_CLEAN_ALLOW_MISSING_METADATA", "0") not in {"1", "true", "yes"}:
        raise SystemExit(
            "Static sector and size-percentile training requires asset_metadata.csv; "
            f"missing {path}. Set KRONOS_BASE_CLEAN_METADATA_PATH or explicitly "
            "opt into unknown-only conditions with KRONOS_BASE_CLEAN_ALLOW_MISSING_METADATA=1."
        )
    return path


def fixed_validation_manifest(repo_root: Path, data_root: Path) -> tuple[Path, dict]:
    """Resolve and verify the fixed ~20K validation selection."""
    explicit = os.getenv("KRONOS_BASE_CLEAN_FIXED_VALIDATION_MANIFEST", "").strip()
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
    local_small = Path(__file__).resolve().parents[1] / "base_model"
    if default_repo == "NeoQuasar/Kronos-base" and (
        local_small / "config.json"
    ).is_file() and (local_small / "model.safetensors").is_file():
        return local_small
    return ensure_hf_snapshot(default_repo, target)


def resolve_stage_predictor(stage: str, runtime: Path) -> Path:
    if stage == "bootstrap":
        return resolve_model(
            "KRONOS_BASE_CLEAN_BASE_MODEL",
            "NeoQuasar/Kronos-base",
            runtime / "models/Kronos-base",
        )
    explicit = os.getenv("KRONOS_BASE_CLEAN_PARENT_MODEL", "").strip()
    if not explicit:
        raise SystemExit(
            f"Stage {stage} requires KRONOS_BASE_CLEAN_PARENT_MODEL pointing to "
            "the previous stage's best_model or last_model directory"
        )
    return resolve_model(
        "KRONOS_BASE_CLEAN_PARENT_MODEL",
        "NeoQuasar/Kronos-base",
        runtime / "models/Kronos-base-parent",
    )


def copy_continuation_if_requested(output_root: Path) -> bool:
    """Import a complete prior output tree for same-stage Kaggle continuation."""
    if os.getenv("KRONOS_BASE_CLEAN_DISABLE_AUTO_CONTINUATION", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        print("Same-stage continuation disabled for cross-stage initialization.", flush=True)
        return False
    source = os.getenv("KRONOS_BASE_CLEAN_CONTINUATION_ROOT", "").strip()
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
        "base_clean_manifest.json",
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
        "KRONOS_CONTEXT_LAYER": "10",
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
        "KRONOS_SCHEDULER": os.getenv("KRONOS_SCHEDULER", "warmup_cosine"),
        "KRONOS_SCHEDULER_WARMUP_RATIO": os.getenv(
            "KRONOS_SCHEDULER_WARMUP_RATIO", "0.01"
        ),
        "KRONOS_PREDICTOR_LEARNING_RATE": os.getenv(
            "KRONOS_PREDICTOR_LEARNING_RATE", settings["predictor_lr"]
        ),
        "KRONOS_CONDITION_LEARNING_RATE": os.getenv(
            "KRONOS_CONDITION_LEARNING_RATE", settings["condition_lr"]
        ),
        "KRONOS_PREDICTOR_WARMUP_START_LR": os.getenv(
            "KRONOS_PREDICTOR_WARMUP_START_LR", settings["warmup_start_lr"]
        ),
        "KRONOS_CONDITION_WARMUP_START_LR": os.getenv(
            "KRONOS_CONDITION_WARMUP_START_LR", settings["condition_warmup_start_lr"]
        ),
        "KRONOS_PREDICTOR_MIN_LR": os.getenv("KRONOS_PREDICTOR_MIN_LR", "1e-7"),
        "KRONOS_CONDITION_MIN_LR": os.getenv("KRONOS_CONDITION_MIN_LR", "1e-7"),
        "KRONOS_BEST_SELECTION_METRIC": settings["best_metric"],
        "KRONOS_USE_BETA_V21_AUXILIARY": settings["auxiliary"],
        "KRONOS_BETA_V21_AUTO_CALIBRATE": "0",
        "KRONOS_BETA_V21_AUXILIARY_WARMUP_STEPS": "1000",
        "KRONOS_BETA_V21_EMA_DECAY": "0.99",
        "KRONOS_BETA_V21_CONSISTENCY_SAMPLES": "2048",
        "KRONOS_TRAIN_SAMPLES_PER_SEGMENT": "20000",
        "KRONOS_COVERAGE_SEED": os.getenv("KRONOS_COVERAGE_SEED", "100"),
        "KRONOS_COVERAGE_PASSES": os.getenv("KRONOS_COVERAGE_PASSES", "1"),
        "KRONOS_EPOCHS": os.getenv("KRONOS_EPOCHS", "1"),
        "KRONOS_REQUIRE_FULL_COVERAGE": os.getenv(
            "KRONOS_REQUIRE_FULL_COVERAGE", "1"
        ),
        "KRONOS_EARLY_STOPPING_PATIENCE": "0",
        "KRONOS_VALIDATION_FULL_ONLY": settings["validation_full_only"],
        "KRONOS_VALIDATION_SAMPLES": "0",
        "KRONOS_VALIDATION_QUICK_SAMPLES": "0",
        "KRONOS_VALIDATION_LARGE_SAMPLES": "0",
        "KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS": "1000000",
        "KRONOS_BATCH_SIZE": os.getenv("KRONOS_BATCH_SIZE", "32"),
        "KRONOS_ACCUMULATION_STEPS": os.getenv("KRONOS_ACCUMULATION_STEPS", "8"),
        "KRONOS_TORCHRUN_NPROC_PER_NODE": os.getenv("KRONOS_TORCHRUN_NPROC_PER_NODE", "2"),
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
    # Base-clean bootstrap timing uses clean holdout (smmt315) without the
    # luckfu-pinned fixed-20k manifest. Quick val is telemetry-only; full-val
    # for Best selection comes in later Main segs (sharded dual-rank).
    env.pop("KRONOS_FIXED_VALIDATION_MANIFEST_PATH", None)
    env.pop("KRONOS_FIXED_VALIDATION_MANIFEST_SHA256", None)
    env["KRONOS_EXCLUDE_FIXED_VALIDATION_FROM_TRAINING"] = "0"
    if stage == "bootstrap":
        env.update({
            "KRONOS_VALIDATION_FULL_ONLY": "0",
            "KRONOS_VALIDATION_QUICK_SAMPLES": os.getenv(
                "KRONOS_VALIDATION_QUICK_SAMPLES", "2048"
            ),
            "KRONOS_VALIDATION_LARGE_SAMPLES": os.getenv(
                "KRONOS_VALIDATION_LARGE_SAMPLES", "2048"
            ),
            "KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS": "1000000",
        })
    return env


def write_manifest(output_root: Path, stage: str, data_root: Path, predictor: Path, tokenizer: Path) -> None:
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "stage": stage,
        "lineage": {
            "parent": "NeoQuasar/Kronos-base" if stage == "bootstrap" else os.getenv("KRONOS_BASE_CLEAN_PARENT_MODEL", ""),
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
            "model": "Kronos-base",
            "n_layers": 12,
            "d_model": 832,
            "n_heads": 16,
            "ff_dim": "from_hf_config",
            "context_layer": 10,
            "physical_layer_expansion": False,
            "params_approx": "102.4M",
        },
        "validation": {
            "selection": (
                os.getenv("KRONOS_CHECKPOINT_SELECTION_PROTOCOL", "").strip()
                or STAGES[stage]["best_metric"]
            ),
            "val_loss_logged": True,
            "val_loss_selects_best": (
                os.getenv("KRONOS_CHECKPOINT_SELECTION_PROTOCOL", "").strip().lower()
                != "temporal_rank_ic_v1"
            ),
            "full_symbol_holdout_required": stage != "bootstrap",
            "profile": "quick_telemetry_timing" if stage == "bootstrap" else "full_natural",
            "quick_validation_is_telemetry_only": True,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "base_clean_manifest.json").write_text(
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
    run_id = env.get("SWANLAB_RUN_ID", "base_clean_0_1_bootstrap")
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
    swanlab = None
    run = None
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "swanlab==0.10.1", "psutil", "pynvml"], check=True)
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
            tags=["kronos-base", "base_clean_0.1", "no-beta-weights", env.get("KRONOS_BASE_CLEAN_STAGE", "bootstrap")],
            config={key: env[key] for key in (
                "KRONOS_LOOKBACK_WINDOW", "KRONOS_PREDICT_WINDOW", "KRONOS_CONTEXT_LAYER",
                "KRONOS_PREDICTOR_LEARNING_RATE", "KRONOS_CONDITION_LEARNING_RATE",
                "KRONOS_BATCH_SIZE", "KRONOS_ACCUMULATION_STEPS",
                "KRONOS_TORCHRUN_NPROC_PER_NODE",
            )},
        )
        latest_step = {"value": 0}
        stop_gpu = threading.Event()
        gpu_thread = None

        def _gpu_monitor() -> None:
            query = ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                     "--format=csv,noheader,nounits"]
            while not stop_gpu.is_set():
                try:
                    output = subprocess.check_output(query, text=True, timeout=10)
                    payload = {}
                    for line in output.strip().splitlines():
                        parts = [part.strip() for part in line.split(",")]
                        if len(parts) < 4:
                            continue
                        index, used, total, util = parts[:4]
                        used_gb = float(used) / 1024.0
                        total_gb = float(total) / 1024.0
                        payload[f"gpu/{index}_mem_used_gb"] = round(used_gb, 3)
                        payload[f"gpu/{index}_mem_total_gb"] = round(total_gb, 3)
                        payload[f"gpu/{index}_mem_used_pct"] = round(used_gb / max(total_gb, 1e-9) * 100.0, 2)
                        payload[f"gpu/{index}_util_pct"] = float(util)
                    if payload:
                        if run is not None:
                            run.log(payload, step=latest_step["value"])
                        print(json.dumps({"phase": "gpu", **payload}), flush=True)
                except Exception:
                    pass
                stop_gpu.wait(30)

        if run is not None:
            gpu_thread = threading.Thread(target=_gpu_monitor, daemon=True)
            gpu_thread.start()
        def _finish_on_term(signum, frame):
            try:
                if run is not None:
                    swanlab.finish()
            finally:
                os._exit(143)

        try:
            signal.signal(signal.SIGTERM, _finish_on_term)
        except Exception:
            pass

        metrics_path = Path(env["KRONOS_SAVE_PATH"]) / env["KRONOS_PREDICTOR_SAVE_FOLDER"] / "metrics.jsonl"
        if metrics_path.is_file():
            for line in metrics_path.read_text().splitlines():
                try:
                    record = json.loads(line)
                    record_type = record.get("type")
                    if int(record.get('segment', 0)) <= int(env.get('KRONOS_SWANLAB_REPLAY_AFTER_SEGMENT', '-1')):
                        continue
                    if record_type == "train":
                        step = int(record.get("step", 0))
                        segment = int(record.get("segment", 1))
                        total_steps = int(record.get("total_steps", 625))
                        if run is not None:
                            chart_segment = swanlab_chart_segment(segment, env)
                            run.log({
                                "train/loss": float(record["loss"]),
                                "train/forecast_loss": float(record["forecast_loss"]),
                                "train/history_loss": float(record["history_loss"]),
                                "segment": chart_segment,
                            }, step=swanlab_chart_step(segment, step, total_steps, env))
                    elif record_type == "validation_temporal_rank_ic":
                        segment = int(record.get("segment", 1))
                        total_steps = int(record.get("total_steps", 625) or 625)
                        if run is not None and record.get("daily_rank_ic_mean") is not None:
                            chart_segment = swanlab_chart_segment(segment, env)
                            run.log({
                                "validation/daily_rank_ic_mean": float(record["daily_rank_ic_mean"]),
                                "segment": chart_segment,
                            }, step=swanlab_chart_step(segment, total_steps, total_steps, env))
                    elif record_type in {"validation", "validation_large"}:
                        segment = int(record.get("segment", 1))
                        total_steps = int(record.get("total_steps", 625))
                        if run is not None:
                            chart_segment = swanlab_chart_segment(segment, env)
                            run.log({
                                "validation/forecast_loss": float(record["forecast_loss"]),
                                "validation/history_loss": float(record["history_loss"]),
                                "validation/full_loss": float(record["full_sequence_loss"]),
                                "segment": chart_segment,
                            }, step=swanlab_chart_step(segment, total_steps, total_steps, env))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
    except Exception as exc:
        if env.get("KRONOS_REQUIRE_SWANLAB", "0") == "1":
            raise RuntimeError("SwanLab initialization failed; training was not started") from exc
        # Kaggle background jobs have no TTY and may not have a SwanLab token.
        # Training and local checkpointing must continue without remote tracking.
        print(
            f"SwanLab unavailable; continuing with local logs only: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

    training_command = [sys.executable, "-u", str(repo_root / "finetune/train_predictor.py")]
    nproc = int(env.get("KRONOS_TORCHRUN_NPROC_PER_NODE", "1"))
    if nproc > 1:
        training_command = [
            sys.executable, "-u", "-m", "torch.distributed.run",
            "--standalone", f"--nproc_per_node={nproc}",
            str(repo_root / "finetune/train_predictor.py"),
        ]
        print({"phase": "launch_training", "launcher": "torchrun", "nproc_per_node": nproc}, flush=True)
    else:
        print({"phase": "launch_training", "launcher": "python", "nproc_per_node": 1}, flush=True)
    child = subprocess.Popen(
        training_command,
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
            # Parse the trainer's local stage coordinates, but persist and
            # display experiment-wide segment numbers for continued stages.
            forwarded_line = globalize_training_log(line, env)
            log_handle.write(forwarded_line)
            log_handle.flush()
            print(forwarded_line, end="", flush=True)
            match = TRAIN_LOG_RE.search(line)
            if match:
                segment, _, step, total_steps, lr, condition_lr, loss, forecast, history = match.groups()
                segment, step, total_steps = int(segment), int(step), int(total_steps)
                if run is not None:
                    chart_segment = swanlab_chart_segment(segment, env)
                    step_value = swanlab_chart_step(segment, step, total_steps, env)
                    latest_step["value"] = step_value
                    run.log({"train/loss": float(loss), "train/forecast_loss": float(forecast),
                             "train/history_loss": float(history), "train/learning_rate": float(lr),
                             "train/condition_learning_rate": float(condition_lr), "segment": chart_segment},
                            step=step_value)
            rank_ic_match = RANK_IC_LOG_RE.search(line)
            if rank_ic_match and segment and run is not None:
                chart_segment = swanlab_chart_segment(segment, env)
                step_value = swanlab_chart_step(segment, total_steps, total_steps, env)
                latest_step["value"] = step_value
                run.log({
                    "validation/daily_rank_ic_mean": float(rank_ic_match.group(1)),
                    "segment": chart_segment,
                }, step=step_value)
            match = VALIDATION_LOG_RE.search(line)
            if match and segment:
                forecast, history, full = map(float, match.groups())
                if run is not None:
                    chart_segment = swanlab_chart_segment(segment, env)
                    step_value = swanlab_chart_step(segment, total_steps, total_steps, env)
                    latest_step["value"] = step_value
                    run.log({"validation/forecast_loss": forecast, "validation/history_loss": history,
                             "validation/full_loss": full, "segment": chart_segment},
                            step=step_value)
    finally:
        stop_gpu.set()
        if gpu_thread is not None:
            gpu_thread.join(timeout=5)
        try:
            return_code = child.wait()
        finally:
            log_handle.close()
            if swanlab is not None:
                try:
                    swanlab.finish()
                except Exception:
                    pass
    if return_code:
        raise subprocess.CalledProcessError(return_code, child.args)


def run(stage: str | None = None) -> None:
    stage = stage or os.getenv("KRONOS_BASE_CLEAN_STAGE", "bootstrap").strip().lower()
    if stage not in STAGES:
        raise SystemExit(f"Unsupported stage {stage!r}; choose one of {sorted(STAGES)}")
    repo_root = Path(__file__).resolve().parents[1]
    runtime = Path(os.getenv("KRONOS_BASE_CLEAN_RUNTIME", "/kaggle/working/kronos_base_clean"))
    input_root = Path(os.getenv("KRONOS_KAGGLE_INPUT_ROOT", "/kaggle/input"))
    data_root = find_data_root(input_root)
    metadata_path(data_root)
    model_root = runtime / "models"
    predictor = resolve_stage_predictor(stage, runtime)
    if stage == "bootstrap":
        tokenizer = resolve_model(
            "KRONOS_BASE_CLEAN_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base", model_root / "Kronos-Tokenizer-base"
        )
    else:
        tokenizer = resolve_model(
            "KRONOS_BASE_CLEAN_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base", model_root / "Kronos-Tokenizer-base"
        )

    output_name = os.getenv("KRONOS_BASE_CLEAN_OUTPUT_NAME", STAGES[stage]["output"]).strip()
    if not output_name:
        raise SystemExit("KRONOS_BASE_CLEAN_OUTPUT_NAME cannot be empty")
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
    env["KRONOS_BASE_CLEAN_STAGE"] = stage
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
