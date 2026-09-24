"""Continue Beta v1.1 with the small stage-2 warmup-constant recipe on dual T4."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import re
import time
from pathlib import Path


OUTPUT_NAME = "beta_v1_1_stage2_wc_dual_t4"
SWANLAB_PROJECT = "finance"
SWANLAB_WORKSPACE = "roc_fu"
SWANLAB_RUN_ID = OUTPUT_NAME
SWANLAB_API_KEY_FALLBACK = "fmEPDGk4IItxgqSZKGLi8"
MODEL_REPO = "luckfu/Kronos-A-Share-Beta-V1-1"
EXPECTED_BEST_SHA256 = (
    "b890771368737c6c93825165695afc16b57870f4692f87a563392cc96e405673"
)
# First post-fix run is a measured 10-segment throughput check. Increase only
# after the live segment timing and checkpoint contract have been verified.
MAX_SEGMENTS_PER_RUN = 10
MAX_RUNTIME_SECONDS = 40500  # 11h15m, leaving Kaggle export margin.
COVERAGE_SEED = "20260924"
TORCH_VERSION = "2.6.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
TRAIN_LOG_RE = re.compile(
    r"Segment (\d+)/(\d+), Step (\d+)/(\d+).*?"
    r"Adaptation LR ([0-9.eE+-]+), Condition LR ([0-9.eE+-]+), "
    r"Loss: ([0-9.]+), Forecast: ([0-9.]+), History: ([0-9.]+)"
)
VALIDATION_LOG_RE = re.compile(
    r"Validation Forecast/History/Full: ([0-9.]+) / ([0-9.]+) / ([0-9.]+)"
)
EPOCH_TIME_RE = re.compile(r"Time This Epoch: (\d+):(\d+):(\d+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phase(name: str, **payload: object) -> None:
    print(json.dumps({"phase": name, **payload}, ensure_ascii=False), flush=True)


def clone_repo(repo: Path) -> str:
    phase("clone_started", repo=str(repo))
    shutil.rmtree(repo, ignore_errors=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "3",
            "--branch",
            "master",
            "https://github.com/luckfu/Kronos.git",
            str(repo),
        ],
        check=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    phase("github_ready", commit=commit)
    return commit


def find_data_root(input_root: Path) -> Path:
    explicit = os.getenv("KRONOS_BETA_V1_1_DATA_ROOT", "").strip()
    if explicit:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        candidates = sorted(
            {
                path.parent.parent
                for path in input_root.glob("**/processed_datasets/train_data.pkl")
                if (path.parent / "val_data.pkl").is_file()
            }
        )
    if len(candidates) != 1:
        raise SystemExit(
            "Expected exactly one input data root with train_data.pkl and val_data.pkl; "
            f"found {len(candidates)}: {candidates}"
        )
    root = candidates[0]
    required = [
        root / "processed_datasets/train_data.pkl",
        root / "processed_datasets/val_data.pkl",
        root / "asset_metadata.csv",
        root / "data_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Data root is incomplete: {missing}")
    return root


def find_continuation(input_root: Path) -> Path | None:
    matches = sorted(
        input_root.glob(
            f"**/{OUTPUT_NAME}/checkpoints/last_state.pt"
        )
    )
    if not matches:
        return None
    if len(matches) != 1:
        raise SystemExit(f"Expected one continuation output, found {len(matches)}: {matches}")
    root = matches[0].parent.parent
    required = [
        "run.log",
        "metrics.jsonl",
        "progress.json",
        "summary.json",
        "experiment_manifest.json",
        "checkpoints/last_state.pt",
        "checkpoints/last_model/model.safetensors",
        "checkpoints/best_model/model.safetensors",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise SystemExit(f"Continuation output is incomplete: {missing}")
    return root


def download_model(runtime: Path) -> tuple[Path, Path]:
    target = runtime / "models" / "Kronos-A-Share-Beta-V1-1"
    phase("download_model_started", repo=MODEL_REPO, target=str(target))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
         "--progress-bar", "off", "modelscope>=1.20"],
        check=True,
    )
    from modelscope.hub.snapshot_download import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_REPO, local_dir=str(target))
    predictor = target / "best_model"
    tokenizer = target / "tokenizer"
    if not (predictor / "model.safetensors").is_file():
        predictor = target
    if not (tokenizer / "config.json").is_file():
        tokenizer = target / "tokenizer"
    required = [
        predictor / "config.json",
        predictor / "model.safetensors",
        tokenizer / "config.json",
        tokenizer / "model.safetensors",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"ModelScope snapshot is incomplete: {missing}")
    actual_sha = sha256_file(predictor / "model.safetensors")
    if actual_sha != EXPECTED_BEST_SHA256:
        raise SystemExit(
            f"Beta v1.1 Best@818 SHA mismatch: {actual_sha} != {EXPECTED_BEST_SHA256}"
        )
    phase(
        "model_ready",
        checkpoint="Best@818",
        predictor=str(predictor),
        tokenizer=str(tokenizer),
        model_sha256=actual_sha,
    )
    return predictor, tokenizer


def build_environment(
    data_root: Path,
    predictor: Path,
    tokenizer: Path,
    runtime: Path,
    output_root: Path,
    repo: Path,
    resume: bool,
) -> dict[str, str]:
    data_manifest = data_root / "data_manifest.json"
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(repo),
        "KRONOS_TRAIN_DATA_PATHS": str(data_root / "processed_datasets/train_data.pkl"),
        "KRONOS_VAL_DATA_PATHS": str(data_root / "processed_datasets/val_data.pkl"),
        "KRONOS_DATASET_PATH": str(data_root / "processed_datasets"),
        "KRONOS_METADATA_PATH": str(data_root / "asset_metadata.csv"),
        "KRONOS_DATA_MANIFEST_SHA256": sha256_file(data_manifest),
        "KRONOS_PREDICTOR_PATH": str(predictor),
        "KRONOS_TOKENIZER_PATH": str(tokenizer),
        "KRONOS_SAVE_PATH": str(output_root.parent),
        "KRONOS_PREDICTOR_SAVE_FOLDER": OUTPUT_NAME,
        "KRONOS_LOOKBACK_WINDOW": "120",
        "KRONOS_PREDICT_WINDOW": "10",
        "KRONOS_CONTEXT_LAYER": "10",
        "KRONOS_NUM_SECTORS": "86",
        "KRONOS_NUM_SIZE_BUCKETS": "0",
        "KRONOS_USE_SECTOR_FEATURES": "1",
        "KRONOS_USE_SIZE_FEATURES": "0",
        "KRONOS_USE_SIZE_PERCENTILE": "1",
        "KRONOS_SIZE_MLP_HIDDEN_DIM": "64",
        "KRONOS_RESET_SECTOR_EMBEDDING": "0",
        "KRONOS_RESET_SIZE_EMBEDDING": "0",
        "KRONOS_TRAINABLE_TRANSFORMER_LAYERS": "-1",
        "KRONOS_PREDICTOR_LOSS_MODE": "forecast",
        "KRONOS_HISTORY_LOSS_WEIGHT": "0.02",
        "KRONOS_FORECAST_HORIZON_WEIGHTS":
            "1.364,1.364,1.364,1.136,1.136,0.909,0.909,0.682,0.682,0.455",
        "KRONOS_PREDICTOR_LEARNING_RATE": "1e-5",
        "KRONOS_CONDITION_LEARNING_RATE": "1e-5",
        "KRONOS_PREDICTOR_WARMUP_START_LR": "1e-5",
        "KRONOS_CONDITION_WARMUP_START_LR": "1e-5",
        "KRONOS_PREDICTOR_MIN_LR": "1e-5",
        "KRONOS_CONDITION_MIN_LR": "1e-5",
        "KRONOS_SCHEDULER": "warmup_constant",
        "KRONOS_SCHEDULER_WARMUP_RATIO": "0",
        "KRONOS_USE_BETA_V21_AUXILIARY": "0",
        "KRONOS_VALIDATION_FULL_ONLY": "1",
        "KRONOS_VALIDATION_SAMPLES": "0",
        "KRONOS_VALIDATION_QUICK_SAMPLES": "0",
        "KRONOS_VALIDATION_LARGE_SAMPLES": "0",
        "KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS": "1",
        "KRONOS_TRAIN_SAMPLES_PER_SEGMENT": "20000",
        "KRONOS_COVERAGE_SEED": COVERAGE_SEED,
        "KRONOS_COVERAGE_PASSES": "1",
        "KRONOS_EPOCHS": "1",
        "KRONOS_REQUIRE_FULL_COVERAGE": "1",
        "KRONOS_MAX_SEGMENTS_PER_RUN": str(MAX_SEGMENTS_PER_RUN),
        "KRONOS_MAX_RUNTIME_SECONDS": str(MAX_RUNTIME_SECONDS),
        "KRONOS_TORCHRUN_NPROC_PER_NODE": "2",
        "KRONOS_BATCH_SIZE": "32",
        "KRONOS_NUM_WORKERS": "2",
        "KRONOS_USE_AMP": "1",
        "KRONOS_AMP_DTYPE": "float16",
        "KRONOS_BEST_SELECTION_METRIC": "forecast",
        "KRONOS_EARLY_STOPPING_PATIENCE": "0",
        "KRONOS_RESUME_TRAINING": "1" if resume else "0",
        "KRONOS_SWANLAB_SEGMENT_OFFSET": "0",
    }
    return env


def stream_training(repo: Path, env: dict[str, str], output_root: Path, run) -> None:
    command = [
        sys.executable,
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(repo / "finetune/train_predictor.py"),
    ]
    phase("training_started", command=" ".join(command))
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "run.log"
    current_segment = 0
    total_steps = 1
    last_train_step = 0
    live_metrics = 0
    with log_path.open("a", buffering=1) as log_handle:
        child = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert child.stdout is not None
        for line in child.stdout:
            log_handle.write(line)
            log_handle.flush()
            print(line, end="", flush=True)
            train_match = TRAIN_LOG_RE.search(line)
            if train_match:
                (
                    segment,
                    _,
                    step,
                    total_steps_value,
                    learning_rate,
                    condition_learning_rate,
                    loss,
                    forecast_loss,
                    history_loss,
                ) = train_match.groups()
                current_segment = int(segment)
                last_train_step = int(step)
                total_steps = int(total_steps_value)
                run.log(
                    {
                        "train/loss": float(loss),
                        "train/forecast_loss": float(forecast_loss),
                        "train/history_loss": float(history_loss),
                        "train/learning_rate": float(learning_rate),
                        "train/condition_learning_rate": float(condition_learning_rate),
                        "segment": current_segment,
                    },
                    step=(current_segment - 1) * total_steps + last_train_step,
                )
                live_metrics += 1

            validation_match = VALIDATION_LOG_RE.search(line)
            if validation_match and current_segment:
                forecast_loss, history_loss, full_loss = map(
                    float, validation_match.groups()
                )
                run.log(
                    {
                        "validation/forecast_loss": forecast_loss,
                        "validation/history_loss": history_loss,
                        "validation/full_loss": full_loss,
                        "validation/samples": 123836,
                        "segment": current_segment,
                    },
                    step=current_segment * total_steps,
                )
                live_metrics += 1

            epoch_match = EPOCH_TIME_RE.search(line)
            if epoch_match and current_segment:
                hours, minutes, seconds = map(int, epoch_match.groups())
                segment_seconds = hours * 3600 + minutes * 60 + seconds
                run.log(
                    {
                        "timing/segment_seconds": segment_seconds,
                        "timing/segments_completed": current_segment,
                        "timing/avg_segment_seconds": segment_seconds / current_segment,
                        "segment": current_segment,
                    },
                    step=current_segment * total_steps,
                )
                live_metrics += 1
        return_code = child.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    phase(
        "training_finished",
        output_root=str(output_root),
        live_swanlab_metrics=live_metrics,
        completed_segment=current_segment,
    )


def start_swanlab(env: dict[str, str]):
    import swanlab

    api_key = env.get("SWANLAB_API_KEY", "").strip() or SWANLAB_API_KEY_FALLBACK
    swanlab.login(api_key=api_key)
    run = swanlab.init(
        id=SWANLAB_RUN_ID,
        resume="allow",
        project=SWANLAB_PROJECT,
        workspace=SWANLAB_WORKSPACE,
        experiment_name=OUTPUT_NAME,
        tags=["kronos-beta-v1.1", "stage2", "warmup-constant", "dual-t4"],
        config={
            "parent": "Beta v1.1 Best@818",
            "predictor_learning_rate": 1e-5,
            "condition_learning_rate": 1e-5,
            "scheduler": "warmup_constant",
            "validation": "full",
            "batch_size_per_gpu": 32,
            "effective_batch_size": 64,
        },
    )
    phase(
        "swanlab_ready",
        project=SWANLAB_PROJECT,
        workspace=SWANLAB_WORKSPACE,
        run_id=SWANLAB_RUN_ID,
    )
    return swanlab, run


def backfill_swanlab(swanlab, run, metrics_path: Path) -> None:
    if not metrics_path.is_file():
        return
    for line in metrics_path.read_text().splitlines():
        try:
            record = json.loads(line)
            record_type = record.get("type")
            segment = int(record.get("segment", 0))
            step = int(record.get("step", 0))
            total_steps = int(record.get("total_steps", 1))
            if record_type == "train":
                run.log(
                    {
                        "train/loss": float(record["loss"]),
                        "train/forecast_loss": float(record["forecast_loss"]),
                        "train/history_loss": float(record["history_loss"]),
                        "train/adaptation_learning_rate": float(
                            record.get("adaptation_learning_rate", 1e-5)
                        ),
                        "train/condition_learning_rate": float(
                            record.get("condition_learning_rate", 1e-5)
                        ),
                        "segment": segment,
                    },
                    step=max(0, (segment - 1) * total_steps + step),
                )
            elif record_type in {"validation", "validation_large"}:
                run.log(
                    {
                        "validation/forecast_loss": float(record["forecast_loss"]),
                        "validation/history_loss": float(record["history_loss"]),
                        "validation/full_loss": float(record["full_sequence_loss"]),
                        "validation/samples": int(record.get("samples", 0)),
                        "segment": segment,
                    },
                    step=max(0, segment * total_steps),
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    run.log({"swanlab/backfilled_metrics": True})


def main() -> None:
    phase(
        "started",
        experiment=OUTPUT_NAME,
        parent=f"{MODEL_REPO}:Best@818",
        max_segments_this_run=MAX_SEGMENTS_PER_RUN,
        max_runtime_seconds=MAX_RUNTIME_SECONDS,
        batch_size_per_gpu=32,
        effective_batch_size=64,
        predictor_lr="1e-5",
        condition_lr="1e-5",
        scheduler="warmup_constant",
        validation="full",
    )
    runtime = Path("/kaggle/working/kronos_beta_v1_1_wc_dual_t4")
    input_root = Path("/kaggle/input")
    repo = runtime / "Kronos"
    output_root = runtime / "outputs" / "models" / OUTPUT_NAME
    data_root = find_data_root(input_root)
    continuation = find_continuation(input_root)
    if continuation:
        phase("continuation_ready", source=str(continuation))
        shutil.copytree(continuation, output_root, dirs_exist_ok=True)
        predictor = output_root / "checkpoints" / "last_model"
        tokenizer = output_root / "tokenizer"
        if not (tokenizer / "config.json").is_file():
            _, tokenizer = download_model(runtime)
        resume = True
    else:
        predictor, tokenizer = download_model(runtime)
        resume = False
    clone_repo(repo)
    phase("install_dependencies_started", torch=TORCH_VERSION)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
         "--progress-bar", "off", "-r", "requirements.txt"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
         "--progress-bar", "off", "--force-reinstall", f"torch=={TORCH_VERSION}",
         "--index-url", TORCH_INDEX_URL],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
         "--progress-bar", "off", "swanlab"],
        check=True,
    )
    phase("device_check_started")
    import torch

    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(gpu_names) != 2 or any("T4" not in name for name in gpu_names):
        raise SystemExit(f"Expected exactly two Tesla T4 GPUs, found {gpu_names}")
    phase("device_ready", torch=torch.__version__, cuda=torch.version.cuda, gpus=gpu_names)
    env = build_environment(
        data_root, predictor, tokenizer, runtime, output_root, repo, resume=resume
    )
    env["SWANLAB_PROJECT"] = SWANLAB_PROJECT
    env["SWANLAB_EXPERIMENT_NAME"] = OUTPUT_NAME
    env["SWANLAB_RUN_ID"] = SWANLAB_RUN_ID
    swanlab, swanlab_run = start_swanlab(env)
    manifest = {
        "experiment": OUTPUT_NAME,
        "parent": {
            "modelscope_repo": MODEL_REPO,
            "checkpoint": "Best@818",
            "model_sha256": EXPECTED_BEST_SHA256,
        },
        "data": {
            "root": str(data_root),
            "data_manifest_sha256": env["KRONOS_DATA_MANIFEST_SHA256"],
            "train_sha256": sha256_file(data_root / "processed_datasets/train_data.pkl"),
            "val_sha256": sha256_file(data_root / "processed_datasets/val_data.pkl"),
        },
        "training": {
            "gpu": "2x Tesla T4",
            "batch_size_per_gpu": 32,
            "effective_batch_size": 64,
            "samples_per_segment": 20000,
            "learning_rate": 1e-5,
            "scheduler": "warmup_constant",
            "coverage_passes": 1,
            "full_validation_each_segment": True,
            "validation_samples": "all val_data.pkl",
            "best_selection_metric": "forecast",
        },
        "model_contract": {
            "context_layer": 10,
            "num_sectors": 86,
            "num_size_buckets": 0,
            "use_size_percentile": True,
            "reset_conditioning": False,
        },
        "continuation": resume,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    stream_training(repo, env, output_root, swanlab_run)
    swanlab.finish()
    phase("swanlab_finished", run_id=SWANLAB_RUN_ID)
    phase("export_started")
    subprocess.run(
        [
            sys.executable,
            str(repo / "finetune/export_last_model.py"),
            "--repo-root",
            str(repo),
            "--output-root",
            str(output_root),
        ],
        cwd=repo,
        env=env,
        check=True,
    )
    phase("export_finished", output_root=str(output_root))


if __name__ == "__main__":
    main()
