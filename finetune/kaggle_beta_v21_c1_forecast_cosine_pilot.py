"""Beta v2.1 C1 forecast-only cosine-annealing pilot (dual T4, 12 segments).

Question: does C2-style cosine LR annealing raise the *validation generative
return10d IC* of a C1 checkpoint? C2 = small stage2 cosine refinement:
uniform_cosine, warmup 0, 1e-5 -> 1e-6, all params, single LR.

Parent (selectable, ``PILOT_PARENT`` is rewritten by the builder):
- ``best475`` (default, slug luckfu/kronos-beta-v21-c1-forecast-cosine-pilot-best475):
  Beta v2.1 release Best@475 (ModelScope luckfu/Kronos-A-Share-Beta-V2-1, root
  model.safetensors sha e1bd5584...), located exactly like the val-gen-ic kernel
  (``beta_v21_release_best475``). It ranked first on Step-1 val gen IC (0.3145 vs
  Seg155 0.2954). The release carries aux heads (return_head/barrier_head); this
  pilot is forecast-only, so the model is built with use_beta_v21_auxiliary=False
  and only the trunk + forecast head are loaded (the 4 aux tensors are dropped;
  any other missing/unexpected key aborts before training).
- ``seg155`` (slug luckfu/kronos-beta-v21-c1-forecast-cosine-pilot): Seg155
  forecast-best from private dataset luckfu/kronos-beta-v21-c1-seg155-forecast-best.

Recipe (unchanged across parents):
- Weights only; fresh AdamW; no parent last_state.
- Forecast-only: KRONOS_USE_BETA_V21_AUXILIARY=0, KRONOS_SAME_DAY_RANKING_BATCHES=0
  (shuffled coverage order; no within-segment signal_date sort).
- All parameters trainable, single LR family.
- uniform_cosine 1e-5 -> 1e-6 over exactly 12 segments, warmup 0.
- Coverage seed 20261002, offset 0, 20000 samples/segment, batch 32 x 2 GPUs.
- Training runs in 4 resumable chunks of 3 segments (KRONOS_MAX_SEGMENTS_PER_RUN=3 +
  last_state resume; the global cosine schedule is one 12-segment plan). Between
  chunks the GPUs are free and the fresh snapshot is scored immediately with the
  Step-1 val generative-IC contract (finetune/val_gen_ic_driver.py; same contract
  as luckfu/kronos-beta-v21-c1-val-gen-ic). The parent itself (Seg0) is scored
  with the same contract right after the first chunk, before Seg3.
- Per-checkpoint JSON + cumulative comparison.json are written as soon as each
  checkpoint's 24 date shards land; pilot_selection.json is rewritten after each.
- Best = highest val gen return10d daily IC (WFL logged alongside only).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import time
from pathlib import Path


# Rewritten by build_kaggle_beta_v21_c1_forecast_cosine_pilot_kernel.py --parent.
PILOT_PARENT = "best475"

STEP1_KERNEL = "luckfu/kronos-beta-v21-c1-val-gen-ic"
MODEL_REPO = "luckfu/Kronos-A-Share-Beta-V2-1"
RELEASE_BEST475_SHA256 = (
    "e1bd55842996b7690a21c34c4d74e1128702bca9c16164788b741e3b5d052f97"
)
SEG155_SHA256 = "8b11a759e72d4125cb0c3307c482f1931ddc65be612023b422d66c52e4f8609c"
SEG155_FORECAST = 2.312367872672933
# C1 line full-val WFL red line (Seg155 floor 2.31236787 + 0.015), log only.
C1_WFL_RED_LINE_BASE = "2.31236787"
C1_WFL_RED_LINE_MARGIN = "0.015"
AUX_KEY_PREFIXES = ("return_head.", "barrier_head.")

PARENTS = {
    "best475": {
        "kernel_id": "luckfu/kronos-beta-v21-c1-forecast-cosine-pilot-best475",
        "title": "Kronos Beta V21 C1 Forecast Cosine Pilot Best475",
        "output_name": "beta_v2_1_c1_forecast_cosine_pilot_best475",
        "runtime_dir": "kronos_beta_v21_c1_forecast_cosine_pilot_best475",
        "kernel_version": "pilot_best475_v1",
        "label": "beta_v21_release_best475",
        "source": "modelscope",
        "dataset": None,
        "sha256": RELEASE_BEST475_SHA256,
        "segment_local": None,
        "segment_global": "Best@475 (release)",
        "full_val_wfl": None,  # never logged on the C1 full-val contract
        "has_aux_heads": True,
        "keep_existing_best": "0",
        # Drift alert vs the C1 red line (Best@475 full-val WFL is unknown).
        "drift_alert_base": "2.32736787",
        "step1": {"return10d_rank_ic_daily": 0.31447, "return10d_rank_ic_se": 0.025231,
                  "weighted_forecast_loss_subsample": 2.320456},
        "dataset_sources": ["luckfu/a-share-120d-temporal-symbol-holdout"],
    },
    "seg155": {
        "kernel_id": "luckfu/kronos-beta-v21-c1-forecast-cosine-pilot",
        "title": "Kronos Beta V21 C1 Forecast Cosine Pilot",
        "output_name": "beta_v2_1_c1_forecast_cosine_pilot",
        "runtime_dir": "kronos_beta_v21_c1_forecast_cosine_pilot",
        "kernel_version": "pilot_seg155_v2",
        "label": "seg155_forecast_best",
        "source": "dataset",
        "dataset": "luckfu/kronos-beta-v21-c1-seg155-forecast-best",
        "sha256": SEG155_SHA256,
        "segment_local": 26,
        "segment_global": 155,
        "full_val_wfl": SEG155_FORECAST,
        "has_aux_heads": False,
        "keep_existing_best": "1",
        "drift_alert_base": "2.31236787",
        "step1": {"return10d_rank_ic_daily": 0.295374, "return10d_rank_ic_se": 0.025138,
                  "weighted_forecast_loss_subsample": 2.30912},
        "dataset_sources": [
            "luckfu/a-share-120d-temporal-symbol-holdout",
            "luckfu/kronos-beta-v21-c1-seg155-forecast-best",
        ],
    },
}
PROFILE = PARENTS[PILOT_PARENT]
KERNEL_ID = PROFILE["kernel_id"]
OUTPUT_NAME = PROFILE["output_name"]
KERNEL_VERSION = PROFILE["kernel_version"]
PARENT_LABEL = PROFILE["label"]
PARENT_DATASET = PROFILE["dataset"]
EXPECTED_PARENT_MODEL_SHA256 = PROFILE["sha256"]
EXPECTED_PARENT_SEGMENT = PROFILE["segment_local"]
EXPECTED_PARENT_FORECAST = SEG155_FORECAST  # seg155 best_metric contract only
SEG0_LABEL = f"seg000_{PARENT_LABEL}"

SEGMENT_OFFSET = 0
PILOT_SEGMENTS = 12
SNAPSHOT_EVERY = 3
CHUNK_SEGMENTS = SNAPSHOT_EVERY  # train 3, score, resume
PEAK_LR = "1e-5"
MIN_LR = "1e-6"
WARMUP_RATIO = "0"
COVERAGE_SEED = "20261002"
FORECAST_MONITOR_BASE = C1_WFL_RED_LINE_BASE
FORECAST_MONITOR_MARGIN = C1_WFL_RED_LINE_MARGIN
FORECAST_DRIFT_ALERT_BASE = PROFILE["drift_alert_base"]
FORECAST_DRIFT_ALERT_MARGIN = "0.005"
FORECAST_DRIFT_ALERT_EARLY_SEGMENTS = "3"
# Budget (Kaggle kills at 12h). Expected: setup ~6 min, 12 x ~7.3 min training +
# 4 x ~2.5 min chunk restarts, 5-6 scoring passes x ~27 min -> ~4-4.5h total.
TRAIN_DEADLINE_SECONDS = 34200   # 9.5h: no new training chunk after this
SESSION_HARD_LIMIT_SECONDS = 41400  # 11.5h: eval workers stop taking shards
TORCH_VERSION = "2.6.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
SWANLAB_PROJECT = "finance"
SWANLAB_WORKSPACE = "roc_fu"
SWANLAB_RUN_ID = OUTPUT_NAME
SWANLAB_URL = f"https://swanlab.cn/@{SWANLAB_WORKSPACE}/{SWANLAB_PROJECT}/runs/{SWANLAB_RUN_ID}"
SWANLAB_API_KEY_FALLBACK = "fmEPDGk4IItxgqSZKGLi8"
EXPECTED_BEST_SHA256 = RELEASE_BEST475_SHA256
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


def resolve_release_predictor(snapshot: Path, expected: str) -> Path:
    """Same lookup as kaggle_beta_v21_c1_val_gen_ic.py (beta_v21_release_best475)."""
    candidates = [snapshot / "best_model", snapshot]
    candidates += [path.parent for path in sorted(snapshot.glob("**/model.safetensors"))]
    for candidate in candidates:
        weights = candidate / "model.safetensors"
        if candidate.name == "tokenizer" or not weights.is_file():
            continue
        if not (candidate / "config.json").is_file():
            continue
        if sha256_file(weights) == expected:
            return candidate.resolve()
    raise SystemExit(f"Release predictor with sha {expected} not found under {snapshot}")


def download_model(runtime: Path) -> tuple[Path, Path]:
    """ModelScope release snapshot -> (Best@475 predictor dir, tokenizer dir)."""
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
    predictor = resolve_release_predictor(target, EXPECTED_BEST_SHA256)
    tokenizer = target / "tokenizer"
    if not (tokenizer / "config.json").is_file():
        matches = sorted(target.glob("**/tokenizer/config.json"))
        if not matches:
            raise SystemExit(f"Tokenizer not found under {target}")
        tokenizer = matches[0].parent
    tokenizer_sha = sha256_file(tokenizer / "model.safetensors")
    if tokenizer_sha != EXPECTED_TOKENIZER_SHA256:
        raise SystemExit(
            f"Tokenizer SHA mismatch: {tokenizer_sha} != {EXPECTED_TOKENIZER_SHA256}"
        )
    phase(
        "model_ready",
        checkpoint="beta_v21_release_best475",
        predictor=str(predictor),
        tokenizer=str(tokenizer),
        model_sha256=EXPECTED_BEST_SHA256,
        tokenizer_sha256=tokenizer_sha,
    )
    return predictor, tokenizer


def safetensors_keys(path: Path) -> dict[str, list[int]]:
    """Tensor names -> shapes from the safetensors header (no torch needed)."""
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    return {name: list(meta["shape"]) for name, meta in header.items()}


def is_aux_key(name: str) -> bool:
    return name.startswith(AUX_KEY_PREFIXES)


def classify_parent_keys(
    parent: dict[str, list[int]], model: dict[str, list[int]]
) -> dict[str, list[str]]:
    """Diff parent checkpoint vs model state_dict; aux-head keys are tolerated both ways."""
    unexpected = sorted(set(parent) - set(model))
    missing = sorted(set(model) - set(parent))
    shared = sorted(set(parent) & set(model))
    return {
        "dropped_aux": [key for key in unexpected if is_aux_key(key)],
        "missing_aux": [key for key in missing if is_aux_key(key)],
        "bad_unexpected": [key for key in unexpected if not is_aux_key(key)],
        "bad_missing": [key for key in missing if not is_aux_key(key)],
        "shape_mismatch": [key for key in shared if list(parent[key]) != list(model[key])],
        "loaded": shared,
    }


def check_parent_state_dict(repo: Path, parent_dir: Path, env: dict[str, str]) -> dict:
    """Build the forecast-only predictor exactly as train_predictor does and verify the
    parent loads into it: only aux-head tensors may be dropped/missing."""
    sys.path.insert(0, str(repo))
    from model.kronos import Kronos

    kwargs = json.loads((parent_dir / "config.json").read_text())
    kwargs.update({
        "num_sectors": int(env["KRONOS_NUM_SECTORS"]),
        "num_size_buckets": int(env["KRONOS_NUM_SIZE_BUCKETS"]),
        "context_layer": int(env["KRONOS_CONTEXT_LAYER"]),
        "use_size_percentile": env["KRONOS_USE_SIZE_PERCENTILE"] == "1",
        "size_mlp_hidden_dim": int(env["KRONOS_SIZE_MLP_HIDDEN_DIM"]),
        "use_beta_v21_auxiliary": env["KRONOS_USE_BETA_V21_AUXILIARY"] == "1",
    })
    model = Kronos(**kwargs)
    model_shapes = {name: list(t.shape) for name, t in model.state_dict().items()}
    del model
    parent_shapes = safetensors_keys(parent_dir / "model.safetensors")
    diff = classify_parent_keys(parent_shapes, model_shapes)
    report = {
        "parent_dir": str(parent_dir),
        "parent_config_aux": bool(json.loads(
            (parent_dir / "config.json").read_text()).get("use_beta_v21_auxiliary", False)),
        "model_aux": kwargs["use_beta_v21_auxiliary"],
        "parent_tensors": len(parent_shapes),
        "model_tensors": len(model_shapes),
        "loaded_tensors": len(diff["loaded"]),
        "dropped_aux": diff["dropped_aux"],
        "missing_aux": diff["missing_aux"],
        "bad_unexpected": diff["bad_unexpected"],
        "bad_missing": diff["bad_missing"],
        "shape_mismatch": diff["shape_mismatch"],
    }
    phase("parent_state_dict_checked", **report)
    if diff["bad_unexpected"] or diff["bad_missing"] or diff["shape_mismatch"]:
        raise SystemExit(f"Parent weights do not fit the forecast-only predictor: {report}")
    if kwargs["use_beta_v21_auxiliary"]:
        raise SystemExit("forecast-only pilot must build the predictor without aux heads")
    return report


def stream_training(
    repo: Path, env: dict[str, str], output_root: Path, run, chunk: int = 0
) -> tuple[int, int]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(repo / "finetune/train_predictor.py"),
    ]
    phase("training_started", command=" ".join(command), chunk=chunk,
          resume=env.get("KRONOS_RESUME_TRAINING"),
          max_segments_this_run=env.get("KRONOS_MAX_SEGMENTS_PER_RUN"),
          max_runtime_seconds=env.get("KRONOS_MAX_RUNTIME_SECONDS"))
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
                run.log({"snapshot/segment": int(snapshot_match.group(2))},
                        step=int(snapshot_match.group(2)) * total_steps)
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
        chunk=chunk,
    )
    return current_segment, total_steps


def is_seg155_forecast_best(metric: dict) -> bool:
    try:
        segment = int(metric.get("segment", -1))
        value = float(metric.get("selection_loss", float("nan")))
    except (TypeError, ValueError):
        return False
    return (
        str(metric.get("selection_metric") or "") == "forecast"
        and segment == PARENTS["seg155"]["segment_local"]
        and abs(value - SEG155_FORECAST) <= 1e-8
    )


def find_seg155_forecast_best(input_root: Path) -> Path:
    """Only <seg155 dataset>/checkpoints/best_model with the pinned SHA is accepted."""
    slug = PARENTS["seg155"]["dataset"].split("/", 1)[1]
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
            f"Need exactly one Seg155 forecast best_model under {PARENTS['seg155']['dataset']}; "
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
        # Chunked: 3 segments per invocation, resumed from last_state.pt.
        "KRONOS_MAX_SEGMENTS_PER_RUN": str(CHUNK_SEGMENTS),
        "KRONOS_MAX_RUNTIME_SECONDS": str(TRAIN_DEADLINE_SECONDS),
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
        "KRONOS_KEEP_EXISTING_BEST": PROFILE["keep_existing_best"],
        "KRONOS_SWANLAB_SEGMENT_OFFSET": str(SEGMENT_OFFSET),
        "KRONOS_COVERAGE_EPOCH_OFFSET": str(SEGMENT_OFFSET),
        "KRONOS_SNAPSHOT_EVERY_SEGMENTS": str(SNAPSHOT_EVERY),
        "KRONOS_SNAPSHOT_DIR": str(output_root / "snapshots"),
        "SWANLAB_PROJECT": SWANLAB_PROJECT,
        "SWANLAB_EXPERIMENT_NAME": OUTPUT_NAME,
        "SWANLAB_RUN_ID": SWANLAB_RUN_ID,
    }


def expected_recipe() -> dict[str, str]:
    return {
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
        "KRONOS_MAX_SEGMENTS_PER_RUN": "3",
        "KRONOS_COVERAGE_SEED": "20261002",
        "KRONOS_COVERAGE_EPOCH_OFFSET": "0",
        "KRONOS_TRAIN_SAMPLES_PER_SEGMENT": "20000",
        "KRONOS_BATCH_SIZE": "32",
        "KRONOS_KEEP_EXISTING_BEST": PROFILE["keep_existing_best"],
        "KRONOS_SNAPSHOT_EVERY_SEGMENTS": "3",
        "KRONOS_PREDICTOR_SAVE_FOLDER": OUTPUT_NAME,
    }


EXPECTED_RECIPE = expected_recipe()


def verify_recipe(env: dict[str, str]) -> None:
    actual = {key: env.get(key) for key in EXPECTED_RECIPE}
    phase("recipe_verified", kernel_version=KERNEL_VERSION, parent=PILOT_PARENT, **actual)
    bad = {k: (actual[k], v) for k, v in EXPECTED_RECIPE.items() if actual[k] != v}
    if bad:
        raise SystemExit(f"Pilot recipe drift: {bad}")


class SafeRun:
    """SwanLab wrapper: a logging failure must never stop the pilot."""

    def __init__(self, run=None):
        self.run = run
        self.errors = 0

    def log(self, payload, step=None) -> None:
        if self.run is None:
            return
        try:
            self.run.log(payload, step=step)
        except Exception as error:  # noqa: BLE001
            self.errors += 1
            if self.errors <= 5:
                phase("swanlab_log_error", error=repr(error), keys=sorted(payload)[:6])


def resolve_swanlab_key(env: dict[str, str]) -> tuple[str, str]:
    key = env.get("SWANLAB_API_KEY", "").strip()
    if key:
        return key, "env"
    try:
        from kaggle_secrets import UserSecretsClient

        key = (UserSecretsClient().get_secret("SWANLAB_API_KEY") or "").strip()
        if key:
            return key, "kaggle_secret"
    except Exception:  # noqa: BLE001 - secret not attached to this kernel
        pass
    return SWANLAB_API_KEY_FALLBACK, "fallback"


def start_swanlab(env: dict[str, str]):
    import swanlab

    api_key, key_source = resolve_swanlab_key(env)
    swanlab.login(api_key=api_key)
    run = swanlab.init(
        project=SWANLAB_PROJECT,
        workspace=SWANLAB_WORKSPACE,
        experiment_name=OUTPUT_NAME,
        id=SWANLAB_RUN_ID,
        resume="allow",
        tags=["kronos-base", "beta-v2.1", "c1", "forecast-only", "uniform-cosine",
              "dual-t4", "pilot", f"parent-{PILOT_PARENT}"],
        config={key.lower(): value for key, value in EXPECTED_RECIPE.items()} | {
            "parent": PILOT_PARENT,
            "parent_label": PARENT_LABEL,
            "parent_source": PROFILE["source"],
            "parent_dataset": PARENT_DATASET,
            "parent_model_sha256": EXPECTED_PARENT_MODEL_SHA256,
            "parent_has_aux_heads": PROFILE["has_aux_heads"],
            "aux_heads_loaded": False,
            "parent_step1_val_gen_ic": PROFILE["step1"]["return10d_rank_ic_daily"],
            "kernel": KERNEL_ID,
            "kernel_version": KERNEL_VERSION,
            "adamw": "fresh",
            "effective_batch_size": 64,
            "chunk_segments": CHUNK_SEGMENTS,
            "selection": "val gen return10d daily IC (WFL logged only)",
        },
    )
    phase("swanlab_ready", project=SWANLAB_PROJECT, workspace=SWANLAB_WORKSPACE,
          run_id=SWANLAB_RUN_ID, url=SWANLAB_URL, key_source=key_source)
    return swanlab, SafeRun(run)


class _NullRun(SafeRun):
    def __init__(self):
        super().__init__(None)


def red_line_value() -> float:
    return float(C1_WFL_RED_LINE_BASE) + float(C1_WFL_RED_LINE_MARGIN)


def build_selection(summaries: list[dict]) -> dict:
    """Best by val gen return10d daily IC; WFL (subsample + full val) logged alongside."""
    usable = [s for s in summaries if s.get("return10d_rank_ic_daily") is not None]
    parent = next((s for s in usable if s.get("label") == SEG0_LABEL), None)
    parent_ic = (parent or {}).get("return10d_rank_ic_daily")
    red_line = red_line_value()
    rows = []
    for summary in usable:
        full_wfl = summary.get("full_val_weighted_forecast_loss")
        ic = summary["return10d_rank_ic_daily"]
        rows.append({
            "label": summary["label"],
            "segment": summary.get("segment"),
            "return10d_rank_ic_daily": ic,
            "return10d_rank_ic_se": summary.get("return10d_rank_ic_se"),
            "return10d_rank_icir": summary.get("return10d_rank_icir"),
            "top_bottom_decile_return10d": summary.get("top_bottom_decile_return10d"),
            "wfl_subsample": summary.get("weighted_forecast_loss_subsample"),
            "wfl_full_val_logged": full_wfl,
            "wfl_full_val_above_red_line": (
                None if full_wfl is None else bool(float(full_wfl) > red_line)
            ),
            "delta_ic_vs_seg0": None if parent_ic is None else ic - parent_ic,
        })
    rows.sort(key=lambda row: (row["segment"] is None, row["segment"] or 0))
    best = max(rows, key=lambda row: row["return10d_rank_ic_daily"]) if rows else None
    return {
        "kernel": KERNEL_ID,
        "parent": PILOT_PARENT,
        "seg0_label": SEG0_LABEL,
        "selection_metric": "val generative return10d rank IC, daily mean (higher is better)",
        "wfl_role": "logged only; not a selection criterion (Step 1: WFL is not a valid IC proxy)",
        "wfl_red_line_full_val": red_line,
        "wfl_red_line_note": (
            f"C1 line red line = Seg155 full-val WFL {C1_WFL_RED_LINE_BASE} + "
            f"{C1_WFL_RED_LINE_MARGIN}; flagged per checkpoint, never used to select or stop"
        ),
        "step1_reference": {
            "kernel": STEP1_KERNEL,
            "parent_label": PARENT_LABEL,
            **PROFILE["step1"],
        },
        "candidates": rows,
        "best_by_ic": None if best is None else best["label"],
        "best_by_ic_segment": None if best is None else best["segment"],
        "best_is_seg0": None if best is None else best["label"] == SEG0_LABEL,
        "updated_unix": time.time(),
        "elapsed_sec": round(time.time() - KERNEL_START, 1),
    }


class PilotScorer:
    """Scores checkpoints between training chunks; outputs accumulate in one dir."""

    def __init__(self, repo: Path, output_root: Path, tokenizer: Path, input_root: Path, run):
        self.repo = repo
        self.output_root = output_root
        self.tokenizer = tokenizer
        self.input_root = input_root
        self.run = run
        self.eval_dir = output_root / "val_gen_ic"
        self.items: list[dict] = []
        self.summaries: dict[str, dict] = {}
        self.scored_sha: dict[str, str] = {}
        self.total_steps = 1

    def snapshot_item(self, path: Path) -> dict:
        metric = {}
        metric_file = path / "snapshot_metric.json"
        if metric_file.is_file():
            metric = json.loads(metric_file.read_text())
        return {
            "label": f"pilot_{path.name}",
            "path": str(path),
            "sha256": sha256_file(path / "model.safetensors"),
            "segment": int(metric.get("segment", int(path.name[3:]))),
            "full_val_wfl": metric.get("weighted_forecast_loss"),
            "note": f"cosine pilot snapshot; lr_end={metric.get('learning_rates')}",
        }

    def new_snapshots(self) -> list[dict]:
        known = {item["label"] for item in self.items}
        found = []
        for path in sorted((self.output_root / "snapshots").glob("seg[0-9][0-9][0-9]")):
            if (path / "model.safetensors").is_file() and f"pilot_{path.name}" not in known:
                found.append(self.snapshot_item(path))
        return found

    def score(self, new_items: list[dict]) -> None:
        new_items = [item for item in new_items if item["label"] not in self.summaries]
        if not new_items:
            return
        remaining = KERNEL_START + SESSION_HARD_LIMIT_SECONDS - time.time()
        if remaining < 900:
            phase("valgenic_skipped", reason="session budget exhausted",
                  labels=[item["label"] for item in new_items],
                  remaining_seconds=round(remaining))
            return
        for item in new_items:
            if item["label"] not in {known["label"] for known in self.items}:
                self.items.append(item)
        spec = {
            "kernel": KERNEL_ID,
            "output_dir": str(self.eval_dir),
            "tokenizer_dir": str(self.tokenizer),
            "input_root": str(self.input_root),
            "deadline": KERNEL_START + SESSION_HARD_LIMIT_SECONDS,
            "world": 2,
            "effective_batch": 256,
            # Cumulative: already-scored checkpoints keep their shards (skipped by
            # workers) so comparison.json always ranks every checkpoint so far.
            "checkpoints": self.items,
            "reference": {
                "parent": PARENT_LABEL,
                "parent_sha256": EXPECTED_PARENT_MODEL_SHA256,
                "seg0_label": SEG0_LABEL,
                "parent_scored_in": f"this kernel (Seg0) and {STEP1_KERNEL}",
                "step1": PROFILE["step1"],
            },
        }
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        spec_path = self.eval_dir / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
        labels = [item["label"] for item in new_items]
        phase("valgenic_started", new=labels, cumulative=[c["label"] for c in self.items],
              remaining_seconds=round(remaining))
        started = time.time()
        try:
            subprocess.run(
                [sys.executable, "-u", str(self.repo / "finetune/val_gen_ic_driver.py"),
                 "--spec", str(spec_path)],
                cwd=self.repo,
                env={**os.environ, "PYTHONUNBUFFERED": "1",
                     "PYTHONPATH": os.pathsep.join([str(self.repo),
                                                    str(self.repo / "finetune")])},
                check=True,
            )
        except subprocess.CalledProcessError as error:
            phase("valgenic_failed", new=labels, returncode=error.returncode)
        self.collect(labels)
        phase("valgenic_finished", new=labels, seconds=round(time.time() - started, 1),
              comparison=str(self.eval_dir / "comparison.json"))

    def collect(self, labels: list[str]) -> None:
        results = self.eval_dir / "results"
        for label in labels:
            path = results / f"{label}_val_summary.json"
            if not path.is_file():
                phase("valgenic_missing_summary", label=label)
                continue
            summary = json.loads(path.read_text())
            self.summaries[label] = summary
            segment = int(summary.get("segment") or 0)
            ic = summary.get("return10d_rank_ic_daily")
            payload = {
                "valgenic/return10d_ic_daily": ic,
                "valgenic/return10d_ic_se": summary.get("return10d_rank_ic_se"),
                "valgenic/return10d_icir": summary.get("return10d_rank_icir"),
                "valgenic/top_bottom_decile": summary.get("top_bottom_decile_return10d"),
                "valgenic/wfl_subsample": summary.get("weighted_forecast_loss_subsample"),
                "valgenic/segment": segment,
            }
            if summary.get("full_val_weighted_forecast_loss") is not None:
                payload["valgenic/wfl_full_val"] = summary["full_val_weighted_forecast_loss"]
            seg0 = self.summaries.get(SEG0_LABEL)
            if seg0 is not None and ic is not None:
                payload["valgenic/delta_ic_vs_seg0"] = ic - seg0["return10d_rank_ic_daily"]
            self.run.log({k: v for k, v in payload.items() if v is not None},
                         step=segment * self.total_steps)
            phase("checkpoint_val_gen_ic", label=label, segment=segment,
                  return10d_ic_daily=ic, se=summary.get("return10d_rank_ic_se"),
                  wfl_subsample=summary.get("weighted_forecast_loss_subsample"),
                  wfl_full_val=summary.get("full_val_weighted_forecast_loss"))
        selection = build_selection(list(self.summaries.values()))
        target = self.output_root / "pilot_selection.json"
        target.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
        phase("pilot_selection_updated", best_by_ic=selection["best_by_ic"],
              best_segment=selection["best_by_ic_segment"],
              best_is_seg0=selection["best_is_seg0"], scored=len(selection["candidates"]))


def prepare_parent(input_root: Path, output_root: Path, release_predictor: Path) -> tuple[Path, Path]:
    """Return (training predictor path, Seg0 scoring path)."""
    if output_root.exists():
        shutil.rmtree(output_root)
    if PILOT_PARENT == "best475":
        model_sha = sha256_file(release_predictor / "model.safetensors")
        if model_sha != EXPECTED_PARENT_MODEL_SHA256:
            raise SystemExit(f"Best@475 SHA mismatch: {model_sha}")
        config = json.loads((release_predictor / "config.json").read_text())
        phase("parent_ready", parent=PILOT_PARENT, label=PARENT_LABEL,
              source=f"modelscope:{MODEL_REPO}", path=str(release_predictor),
              model_sha256=model_sha, parent_config_aux=bool(config.get("use_beta_v21_auxiliary")),
              aux_heads_trained=False, adamw="fresh", last_state_loaded=False,
              kept_best_threshold=None,
              note="release weights loaded with use_beta_v21_auxiliary=False; "
                   "return_head/barrier_head tensors are dropped")
        return release_predictor, release_predictor
    source_best = find_seg155_forecast_best(input_root)
    model_sha = sha256_file(source_best / "model.safetensors")
    if model_sha != EXPECTED_PARENT_MODEL_SHA256:
        raise SystemExit(f"Seg155 SHA mismatch: {model_sha}")
    predictor = output_root / "checkpoints" / "best_model"
    predictor.mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "config.json", "best_metric.json", "README.md"):
        if (source_best / name).is_file():
            shutil.copy2(source_best / name, predictor / name)
    if sha256_file(predictor / "model.safetensors") != EXPECTED_PARENT_MODEL_SHA256:
        raise SystemExit("Working copy Seg155 SHA mismatch")
    if not is_seg155_forecast_best(json.loads((predictor / "best_metric.json").read_text())):
        raise SystemExit("Working copy best_metric is not Seg155 forecast best")
    phase("parent_ready", parent=PILOT_PARENT, label=PARENT_LABEL, source=str(source_best),
          model_sha256=model_sha, kept_best_threshold=SEG155_FORECAST, adamw="fresh",
          last_state_loaded=False)
    return predictor, source_best


def chunk_environment(env: dict[str, str], chunk: int) -> dict[str, str]:
    remaining = int(KERNEL_START + TRAIN_DEADLINE_SECONDS - time.time())
    return {
        **env,
        "KRONOS_RESUME_TRAINING": "1" if chunk > 0 else "0",
        "KRONOS_MAX_RUNTIME_SECONDS": str(max(900, remaining)),
    }


def main() -> None:
    phase(
        "started",
        kernel=KERNEL_ID,
        experiment=OUTPUT_NAME,
        kernel_version=KERNEL_VERSION,
        parent=PILOT_PARENT,
        parent_label=PARENT_LABEL,
        parent_source=PROFILE["source"],
        parent_dataset=PARENT_DATASET,
        parent_sha256=EXPECTED_PARENT_MODEL_SHA256,
        adamw="fresh",
        trainable="all",
        scheduler=f"uniform_cosine {PEAK_LR}->{MIN_LR} over {PILOT_SEGMENTS} segments, warmup 0",
        aux=False,
        ranking=False,
        same_day_batches=False,
        chunks=f"{PILOT_SEGMENTS // CHUNK_SEGMENTS} x {CHUNK_SEGMENTS} segments (resume)",
        snapshots=f"every {SNAPSHOT_EVERY} segments",
        scoring="Seg0 (parent) + each snapshot right after its chunk; best_model if distinct",
        swanlab=SWANLAB_URL,
    )
    runtime = Path("/kaggle/working") / PROFILE["runtime_dir"]
    input_root = Path("/kaggle/input")
    repo = runtime / "Kronos"
    output_root = runtime / "outputs" / "models" / OUTPUT_NAME
    data_root = find_data_root(input_root)
    release_predictor, tokenizer = download_model(runtime)
    predictor, seg0_path = prepare_parent(input_root, output_root, release_predictor)

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
    key_report = check_parent_state_dict(repo, predictor, env)
    manifest = {
        "experiment": OUTPUT_NAME,
        "kernel": KERNEL_ID,
        "kernel_version": KERNEL_VERSION,
        "parent": {
            "name": PILOT_PARENT,
            "label": PARENT_LABEL,
            "source": PROFILE["source"],
            "dataset": PARENT_DATASET,
            "modelscope_repo": MODEL_REPO if PROFILE["source"] == "modelscope" else None,
            "path": str(seg0_path),
            "segment_local": EXPECTED_PARENT_SEGMENT,
            "segment_global": PROFILE["segment_global"],
            "model_sha256": EXPECTED_PARENT_MODEL_SHA256,
            "full_val_wfl": PROFILE["full_val_wfl"],
            "has_aux_heads": PROFILE["has_aux_heads"],
            "aux_heads_trained": False,
            "dropped_aux_tensors": key_report["dropped_aux"],
            "adamw": "fresh",
            "step1_val_gen_ic": PROFILE["step1"],
        },
        "data": {
            "root": str(data_root),
            "data_manifest_sha256": env["KRONOS_DATA_MANIFEST_SHA256"],
            "val_sha256": sha256_file(data_root / "processed_datasets/val_data.pkl"),
        },
        "recipe": EXPECTED_RECIPE,
        "chunks": {"segments_per_chunk": CHUNK_SEGMENTS, "resume": "last_state.pt"},
        "lr_at_segment_end_expected": {
            "seg3": 8.68e-6, "seg6": 5.5e-6, "seg9": 2.32e-6, "seg12": 1.0e-6,
        },
        "scoring": {
            "contract": "finetune/val_gen_ic_driver.py (Step-1 contract, 24 dates, N5)",
            "order": [SEG0_LABEL, "pilot_seg003", "pilot_seg006", "pilot_seg009",
                      "pilot_seg012", "best_model if distinct"],
            "selection": "max val gen return10d daily IC; WFL logged only",
            "wfl_red_line_full_val": red_line_value(),
        },
        "swanlab": SWANLAB_URL,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    try:
        swanlab, swanlab_run = start_swanlab(env)
    except Exception as error:  # SwanLab must never block the pilot.
        phase("swanlab_unavailable", error=repr(error))
        swanlab, swanlab_run = None, _NullRun()
    scorer = PilotScorer(repo, output_root, tokenizer, input_root, swanlab_run)
    seg0_item = {
        "label": SEG0_LABEL,
        "path": str(seg0_path),
        "sha256": EXPECTED_PARENT_MODEL_SHA256,
        "segment": 0,
        "full_val_wfl": PROFILE["full_val_wfl"],
        "note": f"Seg0 = parent {PARENT_LABEL} (same contract as {STEP1_KERNEL})",
    }

    training_error = None
    completed = 0
    for chunk in range(PILOT_SEGMENTS // CHUNK_SEGMENTS):
        if completed >= PILOT_SEGMENTS:
            break
        if time.time() > KERNEL_START + TRAIN_DEADLINE_SECONDS:
            phase("training_deadline", completed_segment=completed)
            break
        try:
            segment, total_steps = stream_training(
                repo, chunk_environment(env, chunk), output_root, swanlab_run, chunk=chunk
            )
            scorer.total_steps = max(1, total_steps)
        except subprocess.CalledProcessError as error:
            # Still score whatever snapshots landed before the failure.
            training_error = error
            phase("training_failed", chunk=chunk, returncode=error.returncode)
            segment = completed
        progressed = segment > completed
        completed = max(completed, segment)
        new_items = ([seg0_item] if chunk == 0 else []) + scorer.new_snapshots()
        scorer.score(new_items)
        if training_error is not None or not progressed:
            break

    try:
        subprocess.run([sys.executable, str(repo / "finetune/export_last_model.py"),
                        "--repo-root", str(repo), "--output-root", str(output_root)],
                       cwd=repo, env=env, check=True)
        phase("export_finished", output_root=str(output_root))
    except subprocess.CalledProcessError as error:
        phase("export_failed", returncode=error.returncode)
    # Late snapshots (e.g. after a failure) and best-by-forecast if it is distinct.
    extra = scorer.new_snapshots()
    best = output_root / "checkpoints" / "best_model"
    if (best / "model.safetensors").is_file() and (best / "best_metric.json").is_file():
        sha = sha256_file(best / "model.safetensors")
        metric = json.loads((best / "best_metric.json").read_text())
        known = {item["sha256"]: item["label"] for item in scorer.items + extra}
        known.setdefault(EXPECTED_PARENT_MODEL_SHA256, SEG0_LABEL)
        phase("pilot_best_model", sha256=sha, segment=metric.get("segment"),
              selection_loss=metric.get("selection_loss"), same_as=known.get(sha))
        if sha not in known:
            extra.append({
                "label": f"pilot_best_seg{int(metric.get('segment', -1)):03d}",
                "path": str(best),
                "sha256": sha,
                "segment": metric.get("segment"),
                "full_val_wfl": (metric.get("large_metrics") or {}).get(
                    "weighted_forecast_loss", metric.get("selection_loss")),
                "note": "cosine pilot best_model by full-val forecast loss",
            })
    scorer.score(extra)
    if swanlab is not None:
        swanlab.finish()
    # Keep /kaggle/working small: drop optimizer state (not needed for scoring).
    last_state = output_root / "checkpoints" / "last_state.pt"
    if last_state.is_file():
        last_state.unlink()
        phase("last_state_removed", reason="pilot output keeps weights only")
    if training_error is not None:
        raise training_error
    phase("done", total_seconds=round(time.time() - KERNEL_START, 1),
          completed_segment=completed, swanlab=SWANLAB_URL)


if __name__ == "__main__":
    main()
