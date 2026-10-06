"""Beta v2.1 C1 forecast-only cosine-annealing pilot (dual T4, 12 segments).

Question: is the C1 vs small-C2 OOS gap (Seg155 gen-return daily IC 0.117 vs
C2 0.177/0.180) explained by C2's cosine LR annealing? C2 = small stage2
cosine refinement: uniform_cosine, warmup 0, 1e-5 -> 1e-6, all params, single LR.

Recipe (C2-style, on C1 Seg155):
- Parent: Seg155 forecast-best (global Seg155 = wc local segment 26) from
  private dataset luckfu/kronos-beta-v21-c1-seg155-forecast-best,
  checkpoints/best_model, sha 8b11a759..., forecast 2.312367872672933.
  Weights only; fresh AdamW; no last_state.
- Forecast-only: KRONOS_USE_BETA_V21_AUXILIARY=0 (ranking off, aux off),
  KRONOS_SAME_DAY_RANKING_BATCHES=0 (shuffled coverage order; no within-segment
  signal_date sort).
- All parameters trainable (TRAINABLE_TRANSFORMER_LAYERS=-1, heads-only off),
  single LR family (SPLIT_TRUNK_HEAD_LR=0).
- uniform_cosine 1e-5 -> 1e-6 over exactly 12 segments (EPOCHS=12,
  REQUIRE_FULL_COVERAGE=0, warmup ratio 0, start == peak == 1e-5).
- Coverage seed 20261002, coverage offset 0, 20000 samples/segment,
  batch 32 x 2 GPUs (eff 64), AMP fp16, full val every segment.
- Best by forecast (weighted forecast loss) with Seg155 2.31236787 kept as the
  threshold; weight snapshots every 3 segments (Seg3/6/9/12) to
  outputs/.../snapshots/segNNN (KRONOS_SNAPSHOT_EVERY_SEGMENTS=3).
- After training, in-kernel: score each snapshot (and best_model if its SHA
  differs from every snapshot and from the parent) with the Step-1 val
  generative-IC contract (finetune/val_gen_ic_driver.py; same contract as
  luckfu/kronos-beta-v21-c1-val-gen-ic). Each checkpoint is scored and saved
  as soon as its 24 date shards land.

Decision rule (docs/beta_v21_c1_forecast_cosine_pilot_cn.md): if val gen IC
rises with annealing vs Seg155 -> OOS-test the best snapshot against 0.117 /
0.180; if flat -> stop, the gap is not from annealing.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path


KERNEL_ID = "luckfu/kronos-beta-v21-c1-forecast-cosine-pilot"
OUTPUT_NAME = "beta_v2_1_c1_forecast_cosine_pilot"
KERNEL_VERSION = "pilot_v1"
PARENT_DATASET = "luckfu/kronos-beta-v21-c1-seg155-forecast-best"
PARENT_LABEL = "seg155_forecast_best"
EXPECTED_PARENT_SEGMENT = 26  # wc local segment; global chart Seg155
EXPECTED_PARENT_FORECAST = 2.312367872672933
EXPECTED_PARENT_MODEL_SHA256 = (
    "8b11a759e72d4125cb0c3307c482f1931ddc65be612023b422d66c52e4f8609c"
)
# Step-1 val gen-IC result for the parent is the trend anchor (same contract).
STEP1_KERNEL = "luckfu/kronos-beta-v21-c1-val-gen-ic"
SEGMENT_OFFSET = 0
PILOT_SEGMENTS = 12
SNAPSHOT_EVERY = 3
PEAK_LR = "1e-5"
MIN_LR = "1e-6"
WARMUP_RATIO = "0"
COVERAGE_SEED = "20261002"
FORECAST_MONITOR_BASE = "2.31236787"
FORECAST_MONITOR_MARGIN = "0.015"
FORECAST_DRIFT_ALERT_BASE = "2.31236787"
FORECAST_DRIFT_ALERT_MARGIN = "0.005"
FORECAST_DRIFT_ALERT_EARLY_SEGMENTS = "3"
# Training soft-stop leaves >= 3h for the in-kernel val gen-IC eval under the
# 12h session cap. Expected training ~2-2.6h; expected eval ~20 min/checkpoint.
MAX_SEGMENTS_PER_RUN = PILOT_SEGMENTS
MAX_RUNTIME_SECONDS = 30000
SESSION_HARD_LIMIT_SECONDS = 41400  # 11.5h: eval workers stop taking shards after this
TORCH_VERSION = "2.6.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
SWANLAB_PROJECT = "finance"
SWANLAB_WORKSPACE = "roc_fu"
SWANLAB_RUN_ID = OUTPUT_NAME
SWANLAB_API_KEY_FALLBACK = "fmEPDGk4IItxgqSZKGLi8"
MODEL_REPO = "luckfu/Kronos-A-Share-Beta-V2-1"
EXPECTED_BEST_SHA256 = (
    "e1bd55842996b7690a21c34c4d74e1128702bca9c16164788b741e3b5d052f97"
)
EXPECTED_TOKENIZER_SHA256 = (
    "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
)
KERNEL_START = time.time()

# Filled by build_kaggle_beta_v21_c1_forecast_cosine_pilot_kernel.py.
EMBEDDED_KRONOS_ARCHIVE_B64 = """
PLACEHOLDER
"""

SNAPSHOT_RE = re.compile(r"Snapshot saved to (\S+) \(segment (\d+)")
BEST_SAVED_RE = re.compile(r"Best model saved to (\S+) \((\w+): ([0-9.eE+-]+)\)")


TRAIN_LOG_RE = re.compile(
    r"Segment (\d+)/(\d+), Step (\d+)/(\d+).*?"
    r"Adaptation LR ([0-9.eE+-]+), Condition LR ([0-9.eE+-]+), "
    r"Loss: ([0-9.]+), Forecast: ([0-9.]+), History: ([0-9.]+)"
)


VALIDATION_LOG_RE = re.compile(
    r"Validation Forecast/History/Full: ([0-9.]+) / ([0-9.]+) / ([0-9.]+)"
)


EPOCH_TIME_RE = re.compile(r"Time This Epoch: (\d+):(\d+):(\d+)")


BETA_V21_SCORE_RE = re.compile(
    r"Best selection metric: beta_v21_score=([0-9.eE+-]+)"
)


RANKING_SELECTION_RE = re.compile(
    r"Best selection metric: ranking=([0-9.eE+-]+)"
)


PAIRWISE_SELECTION_RE = re.compile(
    r"Best selection metric: pairwise_accuracy=([0-9.eE+-]+)"
)


PAIRWISE_METRIC_RE = re.compile(
    r"Validation Pairwise Accuracy/RankIC: (\S+) / (\S+)"
)


FORECAST_MONITOR_RE = re.compile(
    r"Forecast monitor \(not a stop\): weighted_forecast_loss=([0-9.eE+-]+) "
    r"red_line=([0-9.eE+-]+) exceeded=([01])"
)


V21_RANK_RE = re.compile(
    r"Validation v2\.1 Score/Return/Bias/Barrier/Rank: "
    r"(\S+) / ([0-9.eE+-]+) / ([0-9.eE+-]+) / ([0-9.eE+-]+) / ([0-9.eE+-]+)"
)


WEIGHTED_FORECAST_RE = re.compile(
    r"Validation Weighted Forecast: ([0-9.eE+-]+)"
)


FORECAST_DRIFT_RE = re.compile(
    r"Forecast drift monitor \(not a stop\): segment=(\d+) "
    r"weighted_forecast_loss=([0-9.eE+-]+) alert_base=([0-9.eE+-]+) "
    r"delta=([0-9.eE+-]+)"
)


FORECAST_DRIFT_ALERT_RE = re.compile(r"WARNING: (EARLY )?FORECAST DRIFT ALERT")


TRAINABLE_RE = re.compile(
    r"Trainable predictor parameters: ([0-9,]+)/([0-9,]+)"
)


BETA_V21_COMPONENTS_RE = re.compile(
    r"Validation Forecast/History/Full: ([0-9.]+) / ([0-9.]+) / ([0-9.]+)"
)


def globalize_training_log(line: str) -> str:
    """Shift local Segment N/total onto the continued wc axis (N+offset/total)."""
    if SEGMENT_OFFSET <= 0:
        return line
    match = TRAIN_LOG_RE.search(line)
    if match is None:
        return line
    local_segment = int(match.group(1))
    local_total = int(match.group(2))
    start, end = match.span(0)
    matched = match.group(0).replace(
        f"Segment {local_segment}/{local_total}",
        f"Segment {local_segment + SEGMENT_OFFSET}/{local_total}",
        1,
    )
    return line[:start] + matched + line[end:]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phase(name: str, **payload: object) -> None:
    print(json.dumps({"phase": name, **payload}, ensure_ascii=False), flush=True)


def overlay_embedded_sources(repo: Path) -> None:
    """Overwrite cloned GitHub sources with the embedded dual-T4 OOM fix bundle."""
    payload = base64.b64decode(EMBEDDED_KRONOS_ARCHIVE_B64.strip())
    if not payload or EMBEDDED_KRONOS_ARCHIVE_B64.strip() == "PLACEHOLDER":
        raise SystemExit(
            "Embedded Kronos archive is empty; run "
            "finetune/build_kaggle_beta_v21_c1_forecast_cosine_pilot_kernel.py before push"
        )
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(repo.parent, filter="data")
    marker = repo / "finetune" / "train_predictor.py"
    if not marker.is_file():
        raise SystemExit(f"Embedded overlay missing {marker}")
    # Cheap sanity: scalar-only aux collection must be present.
    source = marker.read_text(encoding="utf-8", errors="replace")
    if "collect_validation_auxiliary" not in source:
        raise SystemExit("Embedded train_predictor.py missing aux-collection gate")
    if "pairwise_accuracy" not in source:
        raise SystemExit("Embedded train_predictor.py missing pairwise_accuracy")
    if "train_beta_v21_heads_only" not in source:
        raise SystemExit("Embedded train_predictor.py missing train_beta_v21_heads_only")
    if "forecast_drift_alert_lines" not in source:
        raise SystemExit("Embedded train_predictor.py missing forecast_drift_alert_lines")
    if "KRONOS_SNAPSHOT_EVERY_SEGMENTS" not in source:
        raise SystemExit("Embedded train_predictor.py missing periodic snapshot hook")
    if not (repo / "finetune" / "val_gen_ic_driver.py").is_file():
        raise SystemExit("Embedded overlay missing finetune/val_gen_ic_driver.py")
    if "bool(config.get('collect_validation_auxiliary', False))" not in source:
        raise SystemExit(
            "Embedded train_predictor.py still uses legacy CUDA aux all-gather"
        )
    phase(
        "source_overlay_ready",
        repo=str(repo),
        train_predictor_bytes=marker.stat().st_size,
        archive_bytes=len(payload),
    )


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


def download_model(runtime: Path) -> tuple[Path, Path]:
    target = runtime / "models" / "Kronos-A-Share-Beta-V2-1"
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
            f"Beta v2.1 pretrained SHA mismatch: {actual_sha} != {EXPECTED_BEST_SHA256}"
        )
    tokenizer_sha = sha256_file(tokenizer / "model.safetensors")
    if tokenizer_sha != EXPECTED_TOKENIZER_SHA256:
        raise SystemExit(
            f"Tokenizer SHA mismatch: {tokenizer_sha} != {EXPECTED_TOKENIZER_SHA256}"
        )
    phase(
        "model_ready",
        checkpoint="pretrained_release",
        predictor=str(predictor),
        tokenizer=str(tokenizer),
        model_sha256=actual_sha,
        tokenizer_sha256=tokenizer_sha,
    )
    return predictor, tokenizer


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
            line = globalize_training_log(line)
            print(line, end="", flush=True)
            trainable_match = TRAINABLE_RE.search(line)
            if trainable_match:
                trainable = int(trainable_match.group(1).replace(",", ""))
                total = int(trainable_match.group(2).replace(",", ""))
                phase(
                    "trainable_parameters",
                    trainable=trainable,
                    total=total,
                    fully_unfrozen=trainable == total,
                )
                if trainable != total:
                    child.terminate()
                    raise SystemExit(
                        f"pilot expects every parameter trainable; got {trainable}/{total}"
                    )
            drift_match = FORECAST_DRIFT_RE.search(line)
            if drift_match and current_segment:
                run.log(
                    {
                        "validation/forecast_drift_delta": float(drift_match.group(4)),
                        "validation/forecast_drift_alert_base": float(drift_match.group(3)),
                        "segment": current_segment,
                    },
                    step=current_segment * total_steps,
                )
                live_metrics += 1
            if FORECAST_DRIFT_ALERT_RE.search(line) and current_segment:
                run.log(
                    {"validation/forecast_drift_alert": 1, "segment": current_segment},
                    step=current_segment * total_steps,
                )
                live_metrics += 1
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

            weighted_match = WEIGHTED_FORECAST_RE.search(line)
            if weighted_match and current_segment:
                weighted_forecast = float(weighted_match.group(1))
                run.log(
                    {
                        "validation/weighted_forecast_loss": weighted_forecast,
                        "segment": current_segment,
                    },
                    step=current_segment * total_steps,
                )
                live_metrics += 1

            score_match = BETA_V21_SCORE_RE.search(line)
            if score_match and current_segment:
                score = float(score_match.group(1))
                run.log(
                    {
                        "validation/beta_v21_score": score,
                        "segment/beta_v21_score": score,
                        "segment": current_segment,
                    },
                    step=current_segment * total_steps,
                )
                live_metrics += 1

            rank_components = V21_RANK_RE.search(line)
            if rank_components and current_segment:
                ranking_loss = float(rank_components.group(5))
                run.log(
                    {
                        "validation/ranking_loss": ranking_loss,
                        "validation/return_loss": float(rank_components.group(2)),
                        "validation/barrier_loss": float(rank_components.group(4)),
                        "segment": current_segment,
                    },
                    step=current_segment * total_steps,
                )
                live_metrics += 1

            ranking_selection = RANKING_SELECTION_RE.search(line)
            if ranking_selection and current_segment:
                selection_loss = float(ranking_selection.group(1))
                run.log(
                    {
                        "validation/ranking_selection_loss": selection_loss,
                        "segment": current_segment,
                    },
                    step=current_segment * total_steps,
                )
                live_metrics += 1

            pairwise_selection = PAIRWISE_SELECTION_RE.search(line)
            if pairwise_selection and current_segment:
                selection_accuracy = float(pairwise_selection.group(1))
                run.log(
                    {
                        "validation/pairwise_accuracy": selection_accuracy,
                        "segment": current_segment,
                    },
                    step=current_segment * total_steps,
                )
                live_metrics += 1

            pairwise_metric = PAIRWISE_METRIC_RE.search(line)
            if pairwise_metric and current_segment:
                accuracy_token, rank_ic_token = pairwise_metric.groups()
                payload = {"segment": current_segment}
                if accuracy_token != "None":
                    payload["validation/pairwise_accuracy"] = float(accuracy_token)
                if rank_ic_token != "None":
                    payload["validation/rank_ic"] = float(rank_ic_token)
                if len(payload) > 1:
                    run.log(payload, step=current_segment * total_steps)
                    live_metrics += 1

            forecast_monitor = FORECAST_MONITOR_RE.search(line)
            if forecast_monitor and current_segment:
                weighted, red_line, exceeded = forecast_monitor.groups()
                run.log(
                    {
                        "validation/weighted_forecast_loss": float(weighted),
                        "validation/forecast_red_line": float(red_line),
                        "validation/forecast_red_line_exceeded": int(exceeded),
                        "segment": current_segment,
                    },
                    step=current_segment * total_steps,
                )
                live_metrics += 1

            snapshot_match = SNAPSHOT_RE.search(line)
            if snapshot_match:
                phase("snapshot_saved", path=snapshot_match.group(1),
                      segment=int(snapshot_match.group(2)))
            best_match = BEST_SAVED_RE.search(line)
            if best_match:
                phase("best_saved", path=best_match.group(1),
                      metric=best_match.group(2), value=float(best_match.group(3)))
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


def is_seg155_forecast_best(metric: dict) -> bool:
    try:
        segment = int(metric.get("segment", -1))
        value = float(metric.get("selection_loss", float("nan")))
    except (TypeError, ValueError):
        return False
    return (
        str(metric.get("selection_metric") or "") == "forecast"
        and segment == EXPECTED_PARENT_SEGMENT
        and abs(value - EXPECTED_PARENT_FORECAST) <= 1e-8
    )


def find_seg155_forecast_best(input_root: Path) -> Path:
    """Only <seg155 dataset>/checkpoints/best_model with the pinned SHA is accepted."""
    slug = PARENT_DATASET.split("/", 1)[1]
    matches: list[Path] = []
    rejected: list[str] = []
    for metric_path in sorted(input_root.glob("**/best_model/best_metric.json")):
        candidate = metric_path.parent
        if not (candidate / "model.safetensors").is_file():
            continue
        try:
            metric = json.loads(metric_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if slug not in candidate.parts or not is_seg155_forecast_best(metric):
            rejected.append(f"{candidate} {metric.get('selection_metric')} "
                            f"{metric.get('segment')} {metric.get('selection_loss')}")
            continue
        matches.append(candidate.resolve())
    unique = list(dict.fromkeys(matches))
    if len(unique) != 1:
        raise SystemExit(
            f"Need exactly one Seg155 forecast best_model under {PARENT_DATASET}; "
            f"found {unique}; rejected={rejected[:8]}"
        )
    return unique[0]


def build_environment(data_root, predictor, tokenizer, output_root, repo) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(repo),
        "KRONOS_TRAIN_DATA_PATHS": str(data_root / "processed_datasets/train_data.pkl"),
        "KRONOS_VAL_DATA_PATHS": str(data_root / "processed_datasets/val_data.pkl"),
        "KRONOS_DATASET_PATH": str(data_root / "processed_datasets"),
        "KRONOS_METADATA_PATH": str(data_root / "asset_metadata.csv"),
        "KRONOS_DATA_MANIFEST_SHA256": sha256_file(data_root / "data_manifest.json"),
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
        # C2-style: every parameter trains, one LR family.
        "KRONOS_TRAINABLE_TRANSFORMER_LAYERS": "-1",
        "KRONOS_TRAIN_BETA_V21_HEADS_ONLY": "0",
        "KRONOS_SPLIT_TRUNK_HEAD_LR": "0",
        "KRONOS_PREDICTOR_LOSS_MODE": "forecast",
        "KRONOS_HISTORY_LOSS_WEIGHT": "0.02",
        "KRONOS_FORECAST_HORIZON_WEIGHTS":
            "1.364,1.364,1.364,1.136,1.136,0.909,0.909,0.682,0.682,0.455",
        "KRONOS_SCHEDULER": "uniform_cosine",
        "KRONOS_SCHEDULER_WARMUP_RATIO": WARMUP_RATIO,
        "KRONOS_PREDICTOR_LEARNING_RATE": PEAK_LR,
        "KRONOS_CONDITION_LEARNING_RATE": PEAK_LR,
        "KRONOS_PREDICTOR_WARMUP_START_LR": PEAK_LR,
        "KRONOS_CONDITION_WARMUP_START_LR": PEAK_LR,
        "KRONOS_PREDICTOR_MIN_LR": MIN_LR,
        "KRONOS_CONDITION_MIN_LR": MIN_LR,
        # Forecast-only: no aux heads, no ranking, shuffled coverage order.
        "KRONOS_USE_BETA_V21_AUXILIARY": "0",
        "KRONOS_SAME_DAY_RANKING_BATCHES": "0",
        # Config default; must be > 0 to parse but is inert with aux off.
        "KRONOS_BETA_V21_RANKING_WEIGHT": "0.05",
        "KRONOS_BETA_V21_VALIDATION_DENOMINATORS": "",
        "KRONOS_BETA_V21_AUTO_CALIBRATE": "0",
        "KRONOS_BETA_V21_FORCE_RECALIBRATE": "0",
        "KRONOS_COLLECT_VALIDATION_AUXILIARY": "0",
        "KRONOS_LOG_PAIRWISE_RANKING_METRICS": "0",
        "KRONOS_VALIDATION_FULL_ONLY": "1",
        "KRONOS_VALIDATION_SAMPLES": "0",
        "KRONOS_VALIDATION_QUICK_SAMPLES": "0",
        "KRONOS_VALIDATION_LARGE_SAMPLES": "0",
        "KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS": "1",
        "KRONOS_TRAIN_SAMPLES_PER_SEGMENT": "20000",
        "KRONOS_COVERAGE_SEED": COVERAGE_SEED,
        "KRONOS_COVERAGE_PASSES": "1",
        "KRONOS_EPOCHS": str(PILOT_SEGMENTS),
        "KRONOS_REQUIRE_FULL_COVERAGE": "0",
        "KRONOS_MAX_SEGMENTS_PER_RUN": str(MAX_SEGMENTS_PER_RUN),
        "KRONOS_MAX_RUNTIME_SECONDS": str(MAX_RUNTIME_SECONDS),
        "KRONOS_TORCHRUN_NPROC_PER_NODE": "2",
        "KRONOS_BATCH_SIZE": "32",
        "KRONOS_NUM_WORKERS": "2",
        "KRONOS_USE_AMP": "1",
        "KRONOS_AMP_DTYPE": "float16",
        "KRONOS_BEST_SELECTION_METRIC": "forecast",
        "KRONOS_FORECAST_MONITOR_BASE": FORECAST_MONITOR_BASE,
        "KRONOS_FORECAST_MONITOR_MARGIN": FORECAST_MONITOR_MARGIN,
        "KRONOS_FORECAST_DRIFT_ALERT_BASE": FORECAST_DRIFT_ALERT_BASE,
        "KRONOS_FORECAST_DRIFT_ALERT_MARGIN": FORECAST_DRIFT_ALERT_MARGIN,
        "KRONOS_FORECAST_DRIFT_ALERT_EARLY_SEGMENTS": FORECAST_DRIFT_ALERT_EARLY_SEGMENTS,
        "KRONOS_EARLY_STOPPING_PATIENCE": "0",
        "KRONOS_RESUME_TRAINING": "0",
        "KRONOS_KEEP_EXISTING_BEST": "1",
        "KRONOS_SWANLAB_SEGMENT_OFFSET": str(SEGMENT_OFFSET),
        "KRONOS_COVERAGE_EPOCH_OFFSET": str(SEGMENT_OFFSET),
        "KRONOS_SNAPSHOT_EVERY_SEGMENTS": str(SNAPSHOT_EVERY),
        "KRONOS_SNAPSHOT_DIR": str(output_root / "snapshots"),
        "SWANLAB_PROJECT": SWANLAB_PROJECT,
        "SWANLAB_EXPERIMENT_NAME": OUTPUT_NAME,
        "SWANLAB_RUN_ID": SWANLAB_RUN_ID,
    }


EXPECTED_RECIPE = {
    "KRONOS_SCHEDULER": "uniform_cosine",
    "KRONOS_SCHEDULER_WARMUP_RATIO": "0",
    "KRONOS_PREDICTOR_LEARNING_RATE": "1e-5",
    "KRONOS_CONDITION_LEARNING_RATE": "1e-5",
    "KRONOS_PREDICTOR_WARMUP_START_LR": "1e-5",
    "KRONOS_CONDITION_WARMUP_START_LR": "1e-5",
    "KRONOS_PREDICTOR_MIN_LR": "1e-6",
    "KRONOS_CONDITION_MIN_LR": "1e-6",
    "KRONOS_USE_BETA_V21_AUXILIARY": "0",
    "KRONOS_SAME_DAY_RANKING_BATCHES": "0",
    "KRONOS_SPLIT_TRUNK_HEAD_LR": "0",
    "KRONOS_TRAIN_BETA_V21_HEADS_ONLY": "0",
    "KRONOS_TRAINABLE_TRANSFORMER_LAYERS": "-1",
    "KRONOS_BEST_SELECTION_METRIC": "forecast",
    "KRONOS_EPOCHS": "12",
    "KRONOS_REQUIRE_FULL_COVERAGE": "0",
    "KRONOS_MAX_SEGMENTS_PER_RUN": "12",
    "KRONOS_COVERAGE_SEED": "20261002",
    "KRONOS_COVERAGE_EPOCH_OFFSET": "0",
    "KRONOS_TRAIN_SAMPLES_PER_SEGMENT": "20000",
    "KRONOS_BATCH_SIZE": "32",
    "KRONOS_RESUME_TRAINING": "0",
    "KRONOS_KEEP_EXISTING_BEST": "1",
    "KRONOS_SNAPSHOT_EVERY_SEGMENTS": "3",
    "KRONOS_PREDICTOR_SAVE_FOLDER": "beta_v2_1_c1_forecast_cosine_pilot",
}


def verify_recipe(env: dict[str, str]) -> None:
    actual = {key: env.get(key) for key in EXPECTED_RECIPE}
    phase("recipe_verified", kernel_version=KERNEL_VERSION, **actual)
    bad = {k: (actual[k], v) for k, v in EXPECTED_RECIPE.items() if actual[k] != v}
    if bad:
        raise SystemExit(f"Pilot recipe drift: {bad}")


def start_swanlab(env: dict[str, str]):
    import swanlab

    api_key = env.get("SWANLAB_API_KEY", "").strip() or SWANLAB_API_KEY_FALLBACK
    swanlab.login(api_key=api_key)
    run = swanlab.init(
        project=SWANLAB_PROJECT,
        workspace=SWANLAB_WORKSPACE,
        experiment_name=OUTPUT_NAME,
        id=SWANLAB_RUN_ID,
        resume="allow",
        config={key.lower(): value for key, value in EXPECTED_RECIPE.items()} | {
            "parent_dataset": PARENT_DATASET,
            "parent_segment_local": EXPECTED_PARENT_SEGMENT,
            "parent_forecast": EXPECTED_PARENT_FORECAST,
            "parent_model_sha256": EXPECTED_PARENT_MODEL_SHA256,
            "kernel": KERNEL_ID,
            "kernel_version": KERNEL_VERSION,
            "adamw": "fresh",
            "effective_batch_size": 64,
        },
    )
    phase("swanlab_ready", project=SWANLAB_PROJECT, run_id=SWANLAB_RUN_ID)
    return swanlab, run


class _NullRun:
    def log(self, *_args, **_kwargs) -> None:
        return None


def collect_eval_checkpoints(output_root: Path) -> list[dict]:
    """Snapshots in segment order, then best_model if its weights are distinct."""
    items: list[dict] = []
    seen: dict[str, str] = {EXPECTED_PARENT_MODEL_SHA256: PARENT_LABEL}
    for path in sorted((output_root / "snapshots").glob("seg[0-9][0-9][0-9]")):
        weights = path / "model.safetensors"
        if not weights.is_file():
            continue
        metric = {}
        metric_file = path / "snapshot_metric.json"
        if metric_file.is_file():
            metric = json.loads(metric_file.read_text())
        sha = sha256_file(weights)
        label = f"pilot_{path.name}"
        seen.setdefault(sha, label)
        items.append({
            "label": label,
            "path": str(path),
            "sha256": sha,
            "segment": int(metric.get("segment", int(path.name[3:]))),
            "full_val_wfl": metric.get("weighted_forecast_loss"),
            "note": f"cosine pilot snapshot; lr_end={metric.get('learning_rates')}",
        })
    best = output_root / "checkpoints" / "best_model"
    if (best / "model.safetensors").is_file():
        sha = sha256_file(best / "model.safetensors")
        metric = json.loads((best / "best_metric.json").read_text())
        phase("pilot_best_model", sha256=sha, segment=metric.get("segment"),
              selection_loss=metric.get("selection_loss"),
              same_as=seen.get(sha))
        if sha not in seen:
            items.append({
                "label": f"pilot_best_seg{int(metric.get('segment', -1)):03d}",
                "path": str(best),
                "sha256": sha,
                "segment": metric.get("segment"),
                "full_val_wfl": (metric.get("large_metrics") or {}).get(
                    "weighted_forecast_loss", metric.get("selection_loss")),
                "note": "cosine pilot best_model by forecast",
            })
    return items


def run_val_gen_ic(repo: Path, output_root: Path, tokenizer: Path, input_root: Path) -> None:
    checkpoints = collect_eval_checkpoints(output_root)
    if not checkpoints:
        phase("valgenic_skipped", reason="no snapshots or distinct best_model")
        return
    eval_dir = output_root / "val_gen_ic"
    spec = {
        "kernel": KERNEL_ID,
        "output_dir": str(eval_dir),
        "tokenizer_dir": str(tokenizer),
        "input_root": str(input_root),
        "deadline": KERNEL_START + SESSION_HARD_LIMIT_SECONDS,
        "world": 2,
        "effective_batch": 256,
        "checkpoints": checkpoints,
        "reference": {
            "parent": PARENT_LABEL,
            "parent_sha256": EXPECTED_PARENT_MODEL_SHA256,
            "parent_scored_in": STEP1_KERNEL,
            "note": "Parent val gen IC comes from the Step-1 kernel (same contract, same seed).",
        },
    }
    eval_dir.mkdir(parents=True, exist_ok=True)
    spec_path = eval_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    phase("valgenic_started", checkpoints=[c["label"] for c in checkpoints],
          remaining_seconds=round(spec["deadline"] - time.time()))
    subprocess.run(
        [sys.executable, "-u", str(repo / "finetune/val_gen_ic_driver.py"),
         "--spec", str(spec_path)],
        cwd=repo,
        env={**os.environ, "PYTHONUNBUFFERED": "1",
             "PYTHONPATH": os.pathsep.join([str(repo), str(repo / "finetune")])},
        check=True,
    )
    phase("valgenic_finished", output=str(eval_dir / "comparison.json"))


def main() -> None:
    phase(
        "started",
        kernel=KERNEL_ID,
        experiment=OUTPUT_NAME,
        kernel_version=KERNEL_VERSION,
        parent_dataset=PARENT_DATASET,
        checkpoint="seg155_forecast_best",
        adamw="fresh",
        trainable="all",
        scheduler=f"uniform_cosine {PEAK_LR}->{MIN_LR} over {PILOT_SEGMENTS} segments, warmup 0",
        aux=False,
        ranking=False,
        same_day_batches=False,
        snapshots=f"every {SNAPSHOT_EVERY} segments",
        post_training_eval="val generative IC (Step-1 contract)",
    )
    runtime = Path("/kaggle/working/kronos_beta_v21_c1_forecast_cosine_pilot")
    input_root = Path("/kaggle/input")
    repo = runtime / "Kronos"
    output_root = runtime / "outputs" / "models" / OUTPUT_NAME
    data_root = find_data_root(input_root)
    source_best = find_seg155_forecast_best(input_root)
    model_sha = sha256_file(source_best / "model.safetensors")
    if model_sha != EXPECTED_PARENT_MODEL_SHA256:
        raise SystemExit(f"Seg155 SHA mismatch: {model_sha}")
    parent_config = json.loads((source_best / "config.json").read_text())
    if parent_config.get("use_beta_v21_auxiliary"):
        raise SystemExit("Seg155 parent unexpectedly carries aux heads")
    if output_root.exists():
        shutil.rmtree(output_root)
    predictor = output_root / "checkpoints" / "best_model"
    predictor.mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "config.json", "best_metric.json", "README.md"):
        if (source_best / name).is_file():
            shutil.copy2(source_best / name, predictor / name)
    if sha256_file(predictor / "model.safetensors") != EXPECTED_PARENT_MODEL_SHA256:
        raise SystemExit("Working copy Seg155 SHA mismatch")
    if not is_seg155_forecast_best(json.loads((predictor / "best_metric.json").read_text())):
        raise SystemExit("Working copy best_metric is not Seg155 forecast best")
    phase("parent_ready", source=str(source_best), model_sha256=model_sha,
          kept_best_threshold=EXPECTED_PARENT_FORECAST, adamw="fresh",
          last_state_loaded=False)

    _, tokenizer = download_model(runtime)
    clone_repo(repo)
    overlay_embedded_sources(repo)
    phase("install_dependencies_started", torch=TORCH_VERSION)
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--progress-bar", "off", "-r", "requirements.txt"], cwd=repo, check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--progress-bar", "off", "--force-reinstall", f"torch=={TORCH_VERSION}",
                    "--index-url", TORCH_INDEX_URL], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--progress-bar", "off", "swanlab"], check=True)
    import torch

    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(gpu_names) != 2 or any("T4" not in name for name in gpu_names):
        raise SystemExit(f"Expected exactly two Tesla T4 GPUs, found {gpu_names}")
    phase("device_ready", torch=torch.__version__, gpus=gpu_names)

    env = build_environment(data_root, predictor, tokenizer, output_root, repo)
    verify_recipe(env)
    manifest = {
        "experiment": OUTPUT_NAME,
        "kernel": KERNEL_ID,
        "kernel_version": KERNEL_VERSION,
        "parent": {
            "dataset": PARENT_DATASET,
            "checkpoint": "checkpoints/best_model",
            "segment_local": EXPECTED_PARENT_SEGMENT,
            "segment_global": 155,
            "selection_metric": "forecast",
            "selection_value": EXPECTED_PARENT_FORECAST,
            "model_sha256": EXPECTED_PARENT_MODEL_SHA256,
            "adamw": "fresh",
        },
        "data": {
            "root": str(data_root),
            "data_manifest_sha256": env["KRONOS_DATA_MANIFEST_SHA256"],
            "val_sha256": sha256_file(data_root / "processed_datasets/val_data.pkl"),
        },
        "recipe": EXPECTED_RECIPE,
        "lr_at_segment_end_expected": {
            "seg3": 8.68e-6, "seg6": 5.5e-6, "seg9": 2.32e-6, "seg12": 1.0e-6,
        },
        "post_training_eval": "finetune/val_gen_ic_driver.py (Step-1 contract)",
        "decision_rule": (
            "val gen IC rises with annealing -> OOS test best vs 0.117/0.180; "
            "flat -> stop (gap not from annealing)"
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    try:
        swanlab, swanlab_run = start_swanlab(env)
    except Exception as error:  # SwanLab must never block the pilot.
        phase("swanlab_unavailable", error=repr(error))
        swanlab, swanlab_run = None, _NullRun()
    training_error = None
    try:
        stream_training(repo, env, output_root, swanlab_run)
    except subprocess.CalledProcessError as error:
        # Still score whatever snapshots landed before the failure.
        training_error = error
        phase("training_failed", returncode=error.returncode)
    if swanlab is not None:
        swanlab.finish()
    try:
        subprocess.run([sys.executable, str(repo / "finetune/export_last_model.py"),
                        "--repo-root", str(repo), "--output-root", str(output_root)],
                       cwd=repo, env=env, check=True)
        phase("export_finished", output_root=str(output_root))
    except subprocess.CalledProcessError as error:
        phase("export_failed", returncode=error.returncode)
    run_val_gen_ic(repo, output_root, tokenizer, input_root)
    # Keep /kaggle/working small: drop optimizer state (not needed for scoring).
    last_state = output_root / "checkpoints" / "last_state.pt"
    if last_state.is_file():
        last_state.unlink()
        phase("last_state_removed", reason="pilot output keeps weights only")
    if training_error is not None:
        raise training_error
    phase("done", total_seconds=round(time.time() - KERNEL_START, 1))


if __name__ == "__main__":
    main()
