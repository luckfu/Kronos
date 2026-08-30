"""Kaggle orchestration for the v1-beta uniform-LR balanced-validation stage."""

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path


EXPECTED_DATA_MANIFEST_SHA256 = (
    "214c375f47e9843b7d836e414199f444ac8cd139e2ef1bb51adf526bfdc6261c"
)
VALIDATION_PROFILE = os.getenv("KRONOS_VALIDATION_PROFILE", "balanced").strip()
if VALIDATION_PROFILE == "natural":
    EXPECTED_VALIDATION_MANIFEST_SHA256 = (
        "3db79fa5a5966f5d22f0c227b84c0e5a9293cdaaf3f19139ca85f7df427f28b6"
    )
    VALIDATION_MANIFEST_FILENAME = "natural_validation_manifest.json"
    VALIDATION_SAMPLES_FILENAME = "natural_validation_samples.jsonl"
    STAGE_ID = "v1_beta_last1058_natural_validation_v1"
    DEFAULT_OUTPUT_NAME = "a_share_v1_beta_last1058_natural_120d_to_10d"
    DEFAULT_SWANLAB_RUN_ID = "kronos-v1-beta-last1058-natural-120d-to-10d"
elif VALIDATION_PROFILE == "balanced":
    EXPECTED_VALIDATION_MANIFEST_SHA256 = (
        "14afe39e260a90e438051512167f50ee29c4d8561a27fe09896ec261c79a4ba6"
    )
    VALIDATION_MANIFEST_FILENAME = "balanced_validation_manifest.json"
    VALIDATION_SAMPLES_FILENAME = "balanced_validation_samples.jsonl"
    STAGE_ID = "v1_beta_uniform_balanced_v1"
    DEFAULT_OUTPUT_NAME = "a_share_v1_beta_uniform_balanced_120d_to_10d"
    DEFAULT_SWANLAB_RUN_ID = "kronos-v1-beta-uniform-balanced-120d-to-10d"
else:
    raise SystemExit(f"Unsupported validation profile: {VALIDATION_PROFILE}")
EXPECTED_PARENT_SEGMENT = 1058
EXPECTED_PARENT_BEST_SEGMENT = 467
PARENT_KERNEL = "wynstonliu/kronos-v1-beta-best109-official-chunk-5"
SWANLAB_STEPS_PER_SEGMENT = 1250


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def find_manifest_bound_dataset(input_root):
    matches = []
    candidates = sorted(input_root.glob("**/data_manifest.json"))
    for manifest_path in candidates:
        try:
            document = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if document.get("manifest_sha256") != EXPECTED_DATA_MANIFEST_SHA256:
            continue
        root = manifest_path.parent
        files = {
            "train": root / "processed_datasets/train_data.pkl",
            "val": root / "processed_datasets/val_data.pkl",
            "metadata": root / "asset_metadata.csv",
        }
        if all(path.is_file() for path in files.values()):
            matches.append((root, files))
    if len(matches) != 1:
        raise SystemExit(
            "Expected exactly one signed v1-beta dataset, found "
            f"{len(matches)} among {candidates}"
        )
    return matches[0]


def calculate_data_manifest(files):
    items = [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in (files["train"], files["val"], files["metadata"])
    ]
    digest = hashlib.sha256(
        json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != EXPECTED_DATA_MANIFEST_SHA256:
        raise SystemExit(
            f"Dataset content manifest mismatch: {digest} != "
            f"{EXPECTED_DATA_MANIFEST_SHA256}"
        )
    return digest, items


def find_training_source(input_root):
    matches = []
    for manifest_path in sorted(input_root.glob("**/experiment_manifest.json")):
        root = manifest_path.parent
        if (
            (root / "checkpoints/last_state.pt").is_file()
            and (root / "checkpoints/last_model/model.safetensors").is_file()
            and (root / "progress.json").is_file()
        ):
            matches.append((root, manifest_path))
    if len(matches) != 1:
        raise SystemExit(
            "Expected exactly one training output source, found "
            f"{len(matches)}: {[str(root) for root, _ in matches]}"
        )
    root, manifest_path = matches[0]
    try:
        manifest = json.loads(manifest_path.read_text())
        progress = json.loads((root / "progress.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Training source metadata is unreadable: {exc}") from exc
    return root, manifest, progress


def validate_parent_source(source_root, source_manifest, source_progress):
    completed = int(source_progress.get("current_segment", -1))
    if completed != EXPECTED_PARENT_SEGMENT:
        raise SystemExit(
            f"Uniform stage requires parent Segment {EXPECTED_PARENT_SEGMENT}, got "
            f"{completed}"
        )
    if source_progress.get("status") not in {"completed", "stopped"}:
        raise SystemExit(f"Parent training status is not durable: {source_progress}")
    config = source_manifest.get("config", {})
    if config.get("KRONOS_SCHEDULER") != "two_speed":
        raise SystemExit("First uniform-stage run requires the completed two-speed parent")
    required = [
        "run.log",
        "metrics.jsonl",
        "summary.json",
        "checkpoints/last_state.pt",
        "checkpoints/last_model/model.safetensors",
        "checkpoints/last_model/config.json",
        "checkpoints/best_model/model.safetensors",
        "checkpoints/best_model/best_metric.json",
    ]
    missing = [path for path in required if not (source_root / path).is_file()]
    if missing:
        raise SystemExit(f"Parent output contract is incomplete: {missing}")
    best_metric = json.loads(
        (source_root / "checkpoints/best_model/best_metric.json").read_text()
    )
    if int(best_metric.get("segment", -1)) != EXPECTED_PARENT_BEST_SEGMENT:
        raise SystemExit(f"Unexpected parent Best candidate: {best_metric}")
    return best_metric


def validate_continuation_source(source_manifest, source_progress):
    if source_manifest.get("stage_id") != STAGE_ID:
        raise SystemExit("Continuation source belongs to a different training stage")
    config = source_manifest.get("config", {})
    if config.get("KRONOS_SCHEDULER") != "uniform_cosine":
        raise SystemExit("Uniform-stage continuation source has the wrong scheduler")
    if (
        config.get("KRONOS_FIXED_VALIDATION_MANIFEST_SHA256")
        != EXPECTED_VALIDATION_MANIFEST_SHA256
    ):
        raise SystemExit("Continuation source has a different validation manifest")
    if source_progress.get("status") not in {"completed", "stopped"}:
        raise SystemExit(f"Continuation source is not durable: {source_progress}")


def swanlab_global_step(segment, step, total_steps):
    total_steps = max(1, int(total_steps))
    normalized = round(int(step) * SWANLAB_STEPS_PER_SEGMENT / total_steps)
    normalized = min(SWANLAB_STEPS_PER_SEGMENT, max(0, normalized))
    return max(0, (int(segment) - 1) * SWANLAB_STEPS_PER_SEGMENT + normalized)


TRAIN_LOG_RE = re.compile(
    r"Segment (\d+)/(\d+), Step (\d+)/(\d+).*?"
    r"Adaptation LR ([0-9.eE+-]+), Condition LR ([0-9.eE+-]+), "
    r"Loss: ([0-9.]+), Forecast: ([0-9.]+), History: ([0-9.]+)"
)
VALIDATION_LOG_RE = re.compile(
    r"Validation Forecast/History/Full: "
    r"([0-9.eE+-]+) / ([0-9.eE+-]+) / ([0-9.eE+-]+)"
)
VALIDATION_SUMMARY_RE = re.compile(
    r"Validation Train Average/Best: ([0-9.eE+-]+) / ([0-9.eE+-]+)"
)
LARGE_VALIDATION_LOG_RE = re.compile(
    r"Large Validation Objective/Forecast/History/Full: "
    r"([0-9.eE+-]+) / ([0-9.eE+-]+) / "
    r"([0-9.eE+-]+) / ([0-9.eE+-]+)"
)
CONDITION_MONITOR_PREFIX = "Condition Monitor JSON: "
CONDITION_ABLATION_RE = re.compile(
    r"Validation Condition Full/None/Shuffled Forecast: "
    r"([0-9.eE+-]+) / ([0-9.eE+-]+) / ([0-9.eE+-]+); "
    r"Delta Full-None/Full-Shuffled: ([0-9.eE+-]+) / ([0-9.eE+-]+)"
)


def parse_train_line(line):
    match = TRAIN_LOG_RE.search(line)
    if match is None:
        return None
    values = match.groups()
    return {
        "segment": int(values[0]),
        "total_segments": int(values[1]),
        "step": int(values[2]),
        "total_steps": int(values[3]),
        "adaptation_learning_rate": float(values[4]),
        "condition_learning_rate": float(values[5]),
        "loss": float(values[6]),
        "forecast_loss": float(values[7]),
        "history_loss": float(values[8]),
    }


def parse_validation_line(line):
    match = VALIDATION_LOG_RE.search(line)
    if match is None:
        return None
    forecast, history, full = map(float, match.groups())
    return {
        "objective_loss": forecast + 0.02 * history,
        "forecast_loss": forecast,
        "history_loss": history,
        "full_loss": full,
    }


def parse_validation_summary_line(line):
    match = VALIDATION_SUMMARY_RE.search(line)
    if match is None:
        return None
    train_average, best_loss = map(float, match.groups())
    return {"train_average": train_average, "best_loss": best_loss}


def parse_large_validation_line(line):
    match = LARGE_VALIDATION_LOG_RE.search(line)
    if match is None:
        return None
    objective, forecast, history, full = map(float, match.groups())
    return {
        "objective_loss": objective,
        "forecast_loss": forecast,
        "history_loss": history,
        "full_loss": full,
    }


def parse_condition_monitor_line(line):
    marker = line.find(CONDITION_MONITOR_PREFIX)
    if marker < 0:
        return None
    payload = json.loads(line[marker + len(CONDITION_MONITOR_PREFIX):])
    if not isinstance(payload, dict) or not all(
        isinstance(value, (int, float)) for value in payload.values()
    ):
        raise ValueError("Condition monitor payload must contain numeric values")
    return payload


def parse_condition_ablation_line(line):
    match = CONDITION_ABLATION_RE.search(line)
    if match is None:
        return None
    full, none, shuffled, delta_none, delta_shuffled = map(float, match.groups())
    return {
        "full_forecast_loss": full,
        "none_forecast_loss": none,
        "shuffled_forecast_loss": shuffled,
        "full_minus_none_forecast_loss": delta_none,
        "full_minus_shuffled_forecast_loss": delta_shuffled,
    }


def swanlab_log(run, payload, step):
    if run is None:
        return None
    try:
        run.log(payload, step=int(step))
        return run
    except Exception as exc:
        print(
            f"SwanLab logging disabled after error: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


def backfill_swanlab(run, metrics_path):
    if run is None or not metrics_path.is_file():
        return run
    count = 0
    for line in metrics_path.read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        metric_type = item.get("type")
        if metric_type not in {"train", "validation", "validation_large"}:
            continue
        segment = int(item.get("segment") or 0)
        total_steps = int(item.get("total_steps") or 1250)
        step = int(item.get("step") or 0)
        if metric_type.startswith("validation") and step == 0:
            step = total_steps
        prefix = metric_type
        payload = {"segment": segment}
        for key, value in item.items():
            if not isinstance(value, (int, float)) or key in {
                "segment", "total_segments", "step", "total_steps"
            }:
                continue
            normalized = {
                "loss": "objective_loss",
                "full_sequence_loss": "full_loss",
                "average_loss": "average_loss",
            }.get(key, key)
            metric_prefix = prefix
            if metric_type == "train" and (
                key.startswith(("condition_", "adaptation_"))
                or key in {
                    "trunk_input_norm", "sector_embedding_weight_norm",
                    "size_mlp_output_weight_norm",
                }
            ) and key not in {
                "condition_learning_rate", "adaptation_learning_rate"
            }:
                metric_prefix = "monitor"
            payload[f"{metric_prefix}/{normalized}"] = value
        run = swanlab_log(
            run, payload, swanlab_global_step(segment, step, total_steps)
        )
        if run is None:
            break
        count += 1
    print(f"SwanLab backfilled {count} persisted metric rows", flush=True)
    return run


def initialize_swanlab(output_name, config, lineage, continuation):
    subprocess.run(["python", "-m", "pip", "install", "-q", "swanlab"], check=True)
    import swanlab

    api_key = os.getenv("SWANLAB_API_KEY", "").strip()
    if not api_key:
        try:
            from kaggle_secrets import UserSecretsClient

            api_key = UserSecretsClient().get_secret("SWANLAB_API_KEY")
        except Exception:
            api_key = "fmEPDGk4IItxgqSZKGLi8"
    swanlab.login(api_key=api_key)
    kwargs = {
        "project": os.getenv("SWANLAB_PROJECT", "finance"),
        "workspace": os.getenv("SWANLAB_WORKSPACE", "roc_fu"),
        "experiment_name": os.getenv("SWANLAB_EXPERIMENT_NAME", output_name),
        "id": os.getenv("SWANLAB_RUN_ID", DEFAULT_SWANLAB_RUN_ID),
        "resume": "allow",
        "tags": [
            "kronos", "v1-beta", "uniform-lr", f"{VALIDATION_PROFILE}-validation",
            "120d-to-10d",
        ],
        "config": {
            "stage_id": STAGE_ID,
            "continuation": continuation,
            "parent_model_sha256": lineage["parent_last_model_sha256"],
            "validation_manifest_sha256": EXPECTED_VALIDATION_MANIFEST_SHA256,
            "quick_validation_samples": 3000,
            "large_validation_samples": 12000,
            "peak_lr": 1e-6,
            "warmup_start_lr": 5e-7,
            "minimum_lr": 2e-7,
            "warmup_ratio": 0.01,
            "config": config,
        },
    }
    return swanlab.init(**kwargs)


def current_training_config(
    source_root, tokenizer_root, runtime, output_name, data_files,
    validation_manifest_path, resume_training,
):
    return {
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "KRONOS_TRAIN_DATA_PATHS": str(data_files["train"]),
        "KRONOS_VAL_DATA_PATHS": str(data_files["val"]),
        "KRONOS_METADATA_PATH": str(data_files["metadata"]),
        "KRONOS_DATA_MANIFEST_SHA256": EXPECTED_DATA_MANIFEST_SHA256,
        "KRONOS_FIXED_VALIDATION_MANIFEST_PATH": str(validation_manifest_path),
        "KRONOS_FIXED_VALIDATION_MANIFEST_SHA256": (
            EXPECTED_VALIDATION_MANIFEST_SHA256
        ),
        "KRONOS_EXCLUDE_FIXED_VALIDATION_FROM_TRAINING": "1",
        "KRONOS_VALIDATION_QUICK_SAMPLES": "3000",
        "KRONOS_VALIDATION_LARGE_SAMPLES": "12000",
        "KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS": "10",
        "KRONOS_PREDICTOR_PATH": str(source_root / "checkpoints/last_model"),
        "KRONOS_TOKENIZER_PATH": str(tokenizer_root),
        "KRONOS_SAVE_PATH": str(runtime / "outputs/models"),
        "KRONOS_PREDICTOR_SAVE_FOLDER": output_name,
        "KRONOS_LOOKBACK_WINDOW": "120",
        "KRONOS_PREDICT_WINDOW": "10",
        "KRONOS_USE_SECTOR_FEATURES": "1",
        "KRONOS_NUM_SECTORS": "86",
        "KRONOS_USE_SIZE_FEATURES": "0",
        "KRONOS_NUM_SIZE_BUCKETS": "0",
        "KRONOS_USE_SIZE_PERCENTILE": "1",
        "KRONOS_CONTEXT_LAYER": "10",
        "KRONOS_TRAIN_SIGNAL_START": "2015-01-01",
        "KRONOS_TRAIN_SIGNAL_END": "2026-07-17",
        "KRONOS_VAL_SIGNAL_START": "2026-01-01",
        "KRONOS_VAL_SIGNAL_END": "2026-07-17",
        "KRONOS_TRAIN_SAMPLES_PER_SEGMENT": "20000",
        "KRONOS_VALIDATION_SAMPLES": "12000",
        "KRONOS_COVERAGE_PASSES": "1",
        "KRONOS_REQUIRE_FULL_COVERAGE": "1",
        "KRONOS_EARLY_STOPPING_PATIENCE": "0",
        "KRONOS_PREDICTOR_LEARNING_RATE": "1e-6",
        "KRONOS_CONDITION_LEARNING_RATE": "1e-6",
        "KRONOS_SCHEDULER": "uniform_cosine",
        "KRONOS_SCHEDULER_MIN_LR": "2e-7",
        "KRONOS_PREDICTOR_MIN_LR": "2e-7",
        "KRONOS_CONDITION_MIN_LR": "2e-7",
        "KRONOS_SCHEDULER_WARMUP_RATIO": "0.01",
        "KRONOS_PREDICTOR_WARMUP_START_LR": "5e-7",
        "KRONOS_CONDITION_WARMUP_START_LR": "5e-7",
        "KRONOS_PREDICTOR_LOSS_MODE": "forecast",
        "KRONOS_HISTORY_LOSS_WEIGHT": "0.02",
        "KRONOS_TRAINABLE_TRANSFORMER_LAYERS": "2",
        "KRONOS_ADAM_WEIGHT_DECAY": "0.1",
        "KRONOS_GRAD_CLIP_NORM": "1.0",
        "KRONOS_CONDITION_MONITOR_INTERVAL_STEPS": "100",
        "KRONOS_CONDITION_ABLATION_INTERVAL_SEGMENTS": "10",
        "KRONOS_BATCH_SIZE": "32",
        "KRONOS_NUM_WORKERS": "2",
        "KRONOS_USE_AMP": "1",
        "KRONOS_BOOTSTRAP_COMPLETED_SEGMENTS": "0",
        "KRONOS_MAX_SEGMENTS_PER_RUN": "150",
        "KRONOS_RESUME_TRAINING": "1" if resume_training else "0",
    }


def run(source_root):
    source_root = Path(source_root).resolve()
    input_root = Path(os.getenv("KRONOS_KAGGLE_INPUT_ROOT", "/kaggle/input"))
    runtime = Path(os.getenv(
        "KRONOS_KAGGLE_ROOT", "/kaggle/working/kronos_v1_beta_uniform_balanced"
    ))
    output_name = os.getenv(
        "KRONOS_PREDICTOR_SAVE_FOLDER",
        DEFAULT_OUTPUT_NAME,
    )
    output_root = runtime / "outputs/models" / output_name
    output_root.mkdir(parents=True, exist_ok=True)

    data_root, data_files = find_manifest_bound_dataset(input_root)
    data_manifest_sha, data_items = calculate_data_manifest(data_files)
    training_source, source_manifest, source_progress = find_training_source(input_root)
    continuation = source_manifest.get("stage_id") == STAGE_ID
    if continuation:
        validate_continuation_source(source_manifest, source_progress)
        shutil.copytree(training_source, output_root, dirs_exist_ok=True)
        parent_best = source_manifest["parent_lineage"]["parent_best"]
        parent_last_sha = source_manifest["parent_lineage"][
            "parent_last_model_sha256"
        ]
    else:
        parent_best = validate_parent_source(
            training_source, source_manifest, source_progress
        )
        if any(output_root.iterdir()):
            raise SystemExit("Fresh uniform-stage output directory is not empty")
        shutil.copy2(training_source / "run.log", output_root / "parent_run.log")
        parent_last_sha = sha256(
            training_source / "checkpoints/last_model/model.safetensors"
        )

    packaged_manifest = source_root / "finetune/manifests" / VALIDATION_MANIFEST_FILENAME
    packaged_samples = source_root / "finetune/manifests" / VALIDATION_SAMPLES_FILENAME
    if sha256(packaged_manifest) != EXPECTED_VALIDATION_MANIFEST_SHA256:
        raise SystemExit("Packaged validation manifest SHA mismatch")
    output_validation = output_root / "validation"
    output_validation.mkdir(parents=True, exist_ok=True)
    output_manifest = output_validation / packaged_manifest.name
    output_samples = output_validation / packaged_samples.name
    for source, destination in (
        (packaged_manifest, output_manifest),
        (packaged_samples, output_samples),
    ):
        if destination.exists() and sha256(destination) != sha256(source):
            raise SystemExit(f"Persisted validation artifact changed: {destination}")
        shutil.copy2(source, destination)

    lineage = {
        "parent_kernel": PARENT_KERNEL,
        "parent_completed_segment": EXPECTED_PARENT_SEGMENT,
        "parent_last_model_sha256": parent_last_sha,
        "parent_best": parent_best,
        "parent_best_model_location": (
            f"{PARENT_KERNEL}/checkpoints/best_model/model.safetensors"
        ),
    }
    (output_root / "parent_best_reference.json").write_text(
        json.dumps(lineage, indent=2, ensure_ascii=False) + "\n"
    )

    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if "P100" not in gpu_name:
        raise SystemExit(f"Formal uniform-stage training requires P100, got {gpu_name}")
    arch_check = subprocess.run([
        "python", "-c",
        "import torch; raise SystemExit(0 if 'sm_60' in torch.cuda.get_arch_list() else 1)",
    ], check=False)
    if arch_check.returncode:
        subprocess.run([
            "python", "-m", "pip", "install", "-q",
            "--index-url", "https://download.pytorch.org/whl/cu121",
            "torch==2.5.1",
        ], check=True)

    tokenizer_root = runtime / "models/tokenizer"
    tokenizer_root.mkdir(parents=True, exist_ok=True)
    if not (tokenizer_root / "config.json").is_file():
        subprocess.run(
            ["python", "-m", "pip", "install", "-q", "huggingface_hub==0.33.1"],
            check=True,
        )
        from huggingface_hub import snapshot_download

        snapshot_download("NeoQuasar/Kronos-Tokenizer-base", local_dir=tokenizer_root)

    env = os.environ.copy()
    stage_config = current_training_config(
        training_source, tokenizer_root, runtime, output_name, data_files,
        output_manifest, continuation,
    )
    env.update(stage_config)
    manifest_keys = sorted(key for key in stage_config if key.startswith("KRONOS_"))
    manifest_document = {
        "stage_id": STAGE_ID,
        "experiment": (
            "v1-beta uniform low-LR stage with fixed "
            f"{VALIDATION_PROFILE}-proportion validation"
        ),
        "data_root": str(data_root),
        "data_manifest_sha256": data_manifest_sha,
        "data_files": data_items,
        "parent_lineage": lineage,
        "validation": {
            "manifest_sha256": EXPECTED_VALIDATION_MANIFEST_SHA256,
            "quick_samples": 3000,
            "large_samples": 12000,
            "large_interval_segments": 10,
            "excluded_from_stage_training": True,
            "deployment_final_test_status": "pending_future_data",
        },
        "config": {key: stage_config[key] for key in manifest_keys},
        "continuation": continuation,
    }
    existing_manifest_path = output_root / "experiment_manifest.json"
    if continuation:
        previous = json.loads(existing_manifest_path.read_text())
        for key, value in manifest_document["config"].items():
            if key in {"KRONOS_RESUME_TRAINING", "KRONOS_MAX_SEGMENTS_PER_RUN"}:
                continue
            if previous.get("config", {}).get(key) != value:
                raise SystemExit(f"Continuation config mismatch for {key}")
    existing_manifest_path.write_text(
        json.dumps(manifest_document, indent=2, ensure_ascii=False) + "\n"
    )

    log_handle = (output_root / "run.log").open("a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)
    print(json.dumps({
        "stage_id": STAGE_ID,
        "continuation": continuation,
        "gpu": gpu_name,
        "parent_lineage": lineage,
        "validation_manifest_sha256": EXPECTED_VALIDATION_MANIFEST_SHA256,
        "validation_quick_samples": 3000,
        "validation_large_samples": 12000,
        "training_excludes_fixed_validation": True,
    }, ensure_ascii=False), flush=True)

    swanlab_run = initialize_swanlab(
        output_name, manifest_document["config"], lineage, continuation
    )
    if continuation:
        swanlab_run = backfill_swanlab(
            swanlab_run, output_root / "metrics.jsonl"
        )

    child = None

    def forward_signal(signum, frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)
        print(f"Forwarded signal {signum} to training process", flush=True)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)
    child = subprocess.Popen(
        ["python", "-u", str(source_root / "finetune/train_predictor.py")],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    latest_segment = 0
    latest_step = 0
    latest_total_steps = 625
    for line in child.stdout:
        print(line, end="")
        train_record = parse_train_line(line)
        if train_record is not None:
            latest_segment = train_record["segment"]
            latest_step = train_record["step"]
            latest_total_steps = train_record["total_steps"]
            swanlab_run = swanlab_log(swanlab_run, {
                "train/loss": train_record["loss"],
                "train/forecast_loss": train_record["forecast_loss"],
                "train/history_loss": train_record["history_loss"],
                "train/learning_rate": train_record["adaptation_learning_rate"],
                "train/adaptation_learning_rate": train_record[
                    "adaptation_learning_rate"
                ],
                "train/condition_learning_rate": train_record[
                    "condition_learning_rate"
                ],
                "segment": latest_segment,
            }, swanlab_global_step(latest_segment, latest_step, latest_total_steps))
        monitor = parse_condition_monitor_line(line)
        if monitor is not None and latest_segment:
            swanlab_run = swanlab_log(swanlab_run, {
                f"monitor/{key}": value
                for key, value in monitor.items()
                if not key.endswith("parameter_count")
            }, swanlab_global_step(latest_segment, latest_step, latest_total_steps))
        validation = parse_validation_line(line)
        if validation is not None and latest_segment:
            swanlab_run = swanlab_log(swanlab_run, {
                f"validation/{key}": value for key, value in validation.items()
            } | {"segment": latest_segment}, swanlab_global_step(
                latest_segment, latest_total_steps, latest_total_steps
            ))
        validation_summary = parse_validation_summary_line(line)
        if validation_summary is not None and latest_segment:
            swanlab_run = swanlab_log(swanlab_run, {
                f"validation/{key}": value
                for key, value in validation_summary.items()
            } | {"segment": latest_segment}, swanlab_global_step(
                latest_segment, latest_total_steps, latest_total_steps
            ))
        large_validation = parse_large_validation_line(line)
        if large_validation is not None and latest_segment:
            swanlab_run = swanlab_log(swanlab_run, {
                f"validation_large/{key}": value
                for key, value in large_validation.items()
            } | {"segment": latest_segment}, swanlab_global_step(
                latest_segment, latest_total_steps, latest_total_steps
            ))
        ablation = parse_condition_ablation_line(line)
        if ablation is not None and latest_segment:
            swanlab_run = swanlab_log(swanlab_run, {
                f"validation/condition_{key}": value
                for key, value in ablation.items()
            }, swanlab_global_step(
                latest_segment, latest_total_steps, latest_total_steps
            ))
    return_code = child.wait()
    try:
        import swanlab

        swanlab.finish()
    except Exception as exc:
        print(f"SwanLab finish skipped: {type(exc).__name__}: {exc}", flush=True)
    if return_code:
        raise subprocess.CalledProcessError(return_code, child.args)

    subprocess.run([
        "python", str(source_root / "finetune/export_last_model.py"),
        "--output-root", str(output_root),
    ], env=env, check=True)
    required = [
        "run.log",
        "metrics.jsonl",
        "progress.json",
        "summary.json",
        "experiment_manifest.json",
        "parent_run.log",
        "parent_best_reference.json",
        f"validation/{VALIDATION_MANIFEST_FILENAME}",
        f"validation/{VALIDATION_SAMPLES_FILENAME}",
        "checkpoints/last_state.pt",
        "checkpoints/best_model/model.safetensors",
        "checkpoints/best_model/config.json",
        "checkpoints/best_model/best_metric.json",
        "checkpoints/last_model/model.safetensors",
        "checkpoints/last_model/config.json",
    ]
    missing = [path for path in required if not (output_root / path).is_file()]
    if missing:
        raise SystemExit(f"Uniform-stage output contract is incomplete: {missing}")
    print("Uniform validation chunk completed; output contract verified", flush=True)


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
