"""Beta v2.1 102M heavy-regularization retrain, selected on a TIME-DISJOINT val (dual T4).

Pre-registration: finetune/docs/beta_v21_heavy_reg_time_val_cn.md.

Hypothesis: Best@475 (sealed OOS 0.155) under-performs Small C2 (0.177) because the
102M model memorises the training period; temporal_symbol_validation_v1 rewards
that (val->OOS IC drop -0.160 for Best@475 vs -0.058 for C2). C2-level
regularization (resid/ffn 0.25, attn 0.1, token 0.1) + a C2-style WSD schedule,
selected on a time-disjoint val, should transfer better.

Parent: Beta v2.1 release Best@475 (ModelScope luckfu/Kronos-A-Share-Beta-V2-1,
root model.safetensors sha e1bd5584...). Forecast-only: built with
use_beta_v21_auxiliary=False, the 4 aux tensors are dropped, any other key drift
aborts. Dropout overrides are applied via KRONOS_*_DROPOUT_P on top of the parent
config.json (weights unchanged); the runner verifies the built modules (CPU,
pre-training) and the training log line before trusting the run.

Recipe:
- Weights only; fresh AdamW (weight decay 0.1, same as lineage/C2); all params
  trainable; single LR family (condition layers are already trained).
- Training data unchanged: temporal_symbol_validation_v1 train_data.pkl (4,678
  train symbols, signals <= 2026-07-02), coverage permutation (shuffled, NOT
  date-sorted; KRONOS_SAME_DAY_RANKING_BATCHES=0), 20,000 samples/segment,
  batch 32 x 2 GPUs, AMP fp16.
- WSD over 48 segments (15,024 optimizer steps): warmup 1e-6 -> 1e-5 in segment 1,
  hold 1e-5 to the end of segment 32, cosine 1e-5 -> 1e-6 over segments 33-48.
- 12 resumable chunks of 4 segments (last_state.pt); snapshot every 4 segments.
- Training-time val (WFL, log only) AND selection val = time-disjoint panel
  (520 holdout symbols x signals 2026-07-17..2026-08-10 = 17 dates / 8,784
  windows; finetune/build_time_disjoint_val_panel.py). Sealed root files of the
  mounted dataset are never opened.
- Scoring (val_gen_ic_driver.py with val_contract override; prod decode T0.65 /
  top_p 0.8 / N5 / seed 20260906, label raw-close return_10d, all 17 dates):
  Seg0 (= Best@475) and reference Seg9 (cosine pilot) right after chunk 1, then
  each new snapshot right after its chunk. Per-checkpoint summary JSON +
  comparison.json + heavy_reg_selection.json are written immediately.
- Selection: C* = snapshot with max daily return10d rank IC. Val gate (pre-registered):
  C* - Seg0 paired t >= 2.0 over the 17 dates, >= 10/17 date wins, and the
  endpoint Seg48 - Seg0 > 0. Passing only earns one-shot confirmation on the next
  sealed window after 2026-09-03; nothing here touches 2026-08-11..09-03.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import time
from pathlib import Path

KERNEL_ID = "wynstonliu/kronos-beta-v21-heavy-reg-time-val"
KERNEL_TITLE = "Kronos Beta V21 Heavy Reg Time Val"
OUTPUT_NAME = "beta_v2_1_heavy_reg_time_val_best475"
RUNTIME_DIR = "kronos_beta_v21_heavy_reg_time_val"
KERNEL_VERSION = "heavy_reg_time_val_v1"
DATASET_SOURCES = (
    "luckfu/a-share-120d-temporal-symbol-holdout",
    "luckfu/kronos-beta-v21-c1-cosine-pilot-best475-seg9",
)

MODEL_REPO = "luckfu/Kronos-A-Share-Beta-V2-1"
RELEASE_BEST475_SHA256 = "e1bd55842996b7690a21c34c4d74e1128702bca9c16164788b741e3b5d052f97"
EXPECTED_TOKENIZER_SHA256 = "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
AUX_KEY_PREFIXES = ("return_head.", "barrier_head.")
SEG0_LABEL = "seg000_beta_v21_release_best475"
SNAPSHOT_PREFIX = "heavyreg_"
REFERENCE_SEG9 = {
    "label": "ref_pilot_seg009",
    "dataset": "luckfu/kronos-beta-v21-c1-cosine-pilot-best475-seg9",
    "patterns": (
        "**/kronos-beta-v21-c1-cosine-pilot-best475-seg9/**/checkpoints/best_model",
        "**/kronos-beta-v21-c1-cosine-pilot-best475-seg9/checkpoints/best_model",
    ),
    "sha256": "f9d3da03f8e5b55824bff28e31e00cc039ee021126d71891bda7b5daeb76e3c8",
    "note": ("reference only: cosine pilot Seg9 (old val 0.3254 > Seg0 0.3141; sealed OOS "
             "Seg9 0.1455 < Seg0 0.1546). Tests whether the time-disjoint val orders "
             "Seg0/Seg9 like sealed OOS. Never a candidate."),
}

# Parent (Best@475 release config.json) vs this run.
PARENT_DROPOUT = {"resid_dropout_p": 0.2, "ffn_dropout_p": 0.2,
                  "attn_dropout_p": 0.0, "token_dropout_p": 0.0}
DROPOUT = {"resid_dropout_p": 0.25, "ffn_dropout_p": 0.25,
           "attn_dropout_p": 0.1, "token_dropout_p": 0.1}
DROPOUT_ENV = {"resid_dropout_p": "KRONOS_RESID_DROPOUT_P",
               "ffn_dropout_p": "KRONOS_FFN_DROPOUT_P",
               "attn_dropout_p": "KRONOS_ATTN_DROPOUT_P",
               "token_dropout_p": "KRONOS_TOKEN_DROPOUT_P"}
WEIGHT_DECAY = "0.1"

TOTAL_SEGMENTS = 48
SNAPSHOT_EVERY = 4
CHUNK_SEGMENTS = 4
SAMPLES_PER_SEGMENT = 20000
BATCH_PER_GPU = 32
WORLD = 2
STEPS_PER_SEGMENT = math.ceil(math.ceil(SAMPLES_PER_SEGMENT / WORLD) / BATCH_PER_GPU)  # 313
TOTAL_STEPS = TOTAL_SEGMENTS * STEPS_PER_SEGMENT  # 15,024
WARMUP_SEGMENTS = 1
DECAY_START_SEGMENT = 32
PEAK_LR = "1e-5"
WARMUP_START_LR = "1e-6"
MIN_LR = "1e-6"
WARMUP_RATIO = "0.0208333333"        # 1/48 -> 313 steps (segment 1)
DECAY_START_RATIO = "0.6666666667"   # 32/48 -> step 10,016 (end of segment 32)
EXPECTED_WARMUP_STEPS = int(round(TOTAL_STEPS * float(WARMUP_RATIO)))
EXPECTED_DECAY_START_STEP = int(round(TOTAL_STEPS * float(DECAY_START_RATIO)))
COVERAGE_SEED = "20261007"
TRAIN_SIGNAL_END = "2026-07-02"

TIME_VAL_CONTRACT = "holdout520_time_disjoint_20260717_20260810_v1"
TIME_VAL_SIGNAL_START = "2026-07-17"
TIME_VAL_SIGNAL_END = "2026-08-10"
TIME_VAL_WINDOWS = 8784
TIME_VAL_DATES = 17
TIME_VAL_IDENTITIES_SHA256 = "3c629d8b41d82c230d205a551fabe43db3335e3741e5ea74a0b4cacf43e13f9c"
STRICT_SUBSET_START = "2026-07-31"   # label path entirely after lineage target 07-31
SEALED_SIGNAL_START = "2026-08-11"

# Pre-registered val gate (see doc).
GATE_MIN_T = 2.0
GATE_MIN_WINS = 10
GATE_ENDPOINT_SEGMENT = TOTAL_SEGMENTS

# Budget (Kaggle kills at 12h). Expected ~7.2h: setup ~10 min, 48 x ~2.6 min train
# + 12 x ~2.5 min chunk restarts, 14 scoring passes x ~17 min (8,784 windows N5).
TRAIN_DEADLINE_SECONDS = 34200      # 9.5h: no new training chunk after this
SESSION_HARD_LIMIT_SECONDS = 41400  # 11.5h: eval workers stop taking shards
MIN_EVAL_SECONDS = 1200             # skip a scoring pass with < 20 min left
TORCH_VERSION = "2.6.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
SWANLAB_PROJECT = "finance"
SWANLAB_WORKSPACE = "roc_fu"
SWANLAB_RUN_ID = OUTPUT_NAME
SWANLAB_URL = f"https://swanlab.cn/@{SWANLAB_WORKSPACE}/{SWANLAB_PROJECT}/runs/{SWANLAB_RUN_ID}"
SWANLAB_API_KEY_FALLBACK = "fmEPDGk4IItxgqSZKGLi8"
KERNEL_START = time.time()

# Filled by build_kaggle_beta_v21_heavy_reg_time_val_kernel.py.
EMBEDDED_KRONOS_ARCHIVE_B64 = """
PLACEHOLDER
"""

TRAIN_LOG_RE = re.compile(
    r"Segment (\d+)/(\d+), Step (\d+)/(\d+).*?"
    r"Adaptation LR ([0-9.eE+-]+), Condition LR ([0-9.eE+-]+), "
    r"Loss: ([0-9.]+), Forecast: ([0-9.]+), History: ([0-9.]+)"
)
VALIDATION_LOG_RE = re.compile(r"Validation Forecast/History/Full: ([0-9.]+) / ([0-9.]+) / ([0-9.]+)")
WEIGHTED_FORECAST_RE = re.compile(r"Validation Weighted Forecast: ([0-9.eE+-]+)")
EPOCH_TIME_RE = re.compile(r"Time This Epoch: (\d+):(\d+):(\d+)")
TRAINABLE_RE = re.compile(r"Trainable predictor parameters: ([0-9,]+)/([0-9,]+)")
SNAPSHOT_RE = re.compile(r"Snapshot saved to (\S+) \(segment (\d+)")
BEST_SAVED_RE = re.compile(r"Best model saved to (\S+) \((\w+): ([0-9.eE+-]+)\)")
DROPOUT_LINE_RE = re.compile(
    r"Predictor dropout \(modules\): resid=([0-9.na]+) ffn=([0-9.na]+) "
    r"attn=([0-9.na]+) token=([0-9.na]+)"
)
LR_PLAN_RE = re.compile(r"Learning-rate plan: ([0-9,]+) global optimizer steps; ([0-9,]+) warmup steps")
WSD_RE = re.compile(r"WSD decay start: step ([0-9,]+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phase(name: str, **payload: object) -> None:
    print(json.dumps({"phase": name, **payload}, ensure_ascii=False, default=str), flush=True)


def lr_at_step(step: int) -> float:
    """Mirror of train_predictor.warmup_constant_cosine_multiplier for this recipe."""
    peak, start, low = float(PEAK_LR), float(WARMUP_START_LR), float(MIN_LR)
    step = max(0, min(int(step), TOTAL_STEPS))
    if step <= EXPECTED_WARMUP_STEPS:
        return start + (peak - start) * step / max(1, EXPECTED_WARMUP_STEPS)
    if step <= EXPECTED_DECAY_START_STEP:
        return peak
    progress = (step - EXPECTED_DECAY_START_STEP) / max(1, TOTAL_STEPS - EXPECTED_DECAY_START_STEP)
    return low + 0.5 * (peak - low) * (1.0 + math.cos(math.pi * min(1.0, progress)))


def lr_at_segment_end(segment: int) -> float:
    return lr_at_step(segment * STEPS_PER_SEGMENT)


# --------------------------------------------------------------------------- setup


def overlay_embedded_sources(repo: Path) -> None:
    payload = base64.b64decode(EMBEDDED_KRONOS_ARCHIVE_B64.strip())
    if not payload or EMBEDDED_KRONOS_ARCHIVE_B64.strip() == "PLACEHOLDER":
        raise SystemExit("Embedded Kronos archive is empty; run "
                         "finetune/build_kaggle_beta_v21_heavy_reg_time_val_kernel.py before push")
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(repo.parent, filter="data")
    trainer = (repo / "finetune/train_predictor.py").read_text(encoding="utf-8", errors="replace")
    driver = (repo / "finetune/val_gen_ic_driver.py").read_text(encoding="utf-8", errors="replace")
    for marker in ("warmup_constant_cosine_multiplier", "apply_predictor_dropout_overrides",
                   "Predictor dropout (modules)", "KRONOS_SNAPSHOT_EVERY_SEGMENTS",
                   "bool(config.get('collect_validation_auxiliary', False))"):
        if marker not in trainer:
            raise SystemExit(f"Embedded train_predictor.py missing {marker!r}")
    if "val_contract" not in driver:
        raise SystemExit("Embedded val_gen_ic_driver.py lacks the val_contract override")
    if not (repo / "finetune/build_time_disjoint_val_panel.py").is_file():
        raise SystemExit("Embedded overlay missing build_time_disjoint_val_panel.py")
    phase("source_overlay_ready", repo=str(repo), archive_bytes=len(payload))


def clone_repo(repo: Path) -> str:
    phase("clone_started", repo=str(repo))
    shutil.rmtree(repo, ignore_errors=True)
    subprocess.run(["git", "clone", "--depth", "3", "--branch", "master",
                    "https://github.com/luckfu/Kronos.git", str(repo)],
                   check=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    phase("github_ready", commit=commit)
    return commit


def find_data_root(input_root: Path) -> Path:
    candidates = sorted({
        path.parent.parent
        for path in input_root.glob("**/*temporal_symbol_validation_v1/processed_datasets/train_data.pkl")
        if (path.parent / "val_data.pkl").is_file()
    })
    if len(candidates) != 1:
        raise SystemExit(f"Expected one temporal_symbol_validation_v1 data root, found {candidates}")
    root = candidates[0]
    for name in ("processed_datasets/train_data.pkl", "processed_datasets/val_data.pkl",
                 "asset_metadata.csv", "data_manifest.json"):
        if not (root / name).is_file():
            raise SystemExit(f"Data root incomplete: {root / name}")
    return root


def resolve_release_predictor(snapshot: Path, expected: str) -> Path:
    candidates = [snapshot / "best_model", snapshot]
    candidates += [path.parent for path in sorted(snapshot.glob("**/model.safetensors"))]
    for candidate in candidates:
        weights = candidate / "model.safetensors"
        if candidate.name == "tokenizer" or not weights.is_file():
            continue
        if (candidate / "config.json").is_file() and sha256_file(weights) == expected:
            return candidate.resolve()
    raise SystemExit(f"Release predictor with sha {expected} not found under {snapshot}")


def download_model(runtime: Path) -> tuple[Path, Path]:
    target = runtime / "models" / "Kronos-A-Share-Beta-V2-1"
    phase("download_model_started", repo=MODEL_REPO, target=str(target))
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--progress-bar", "off", "modelscope>=1.20"], check=True)
    from modelscope.hub.snapshot_download import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_REPO, local_dir=str(target))
    predictor = resolve_release_predictor(target, RELEASE_BEST475_SHA256)
    tokenizer = target / "tokenizer"
    if not (tokenizer / "config.json").is_file():
        matches = sorted(target.glob("**/tokenizer/config.json"))
        if not matches:
            raise SystemExit(f"Tokenizer not found under {target}")
        tokenizer = matches[0].parent
    tokenizer_sha = sha256_file(tokenizer / "model.safetensors")
    if tokenizer_sha != EXPECTED_TOKENIZER_SHA256:
        raise SystemExit(f"Tokenizer SHA mismatch: {tokenizer_sha}")
    config = json.loads((predictor / "config.json").read_text())
    parent_dropout = {key: float(config.get(key, -1)) for key in PARENT_DROPOUT}
    if parent_dropout != PARENT_DROPOUT:
        raise SystemExit(f"Parent dropout drift: {parent_dropout} != {PARENT_DROPOUT}")
    phase("model_ready", predictor=str(predictor), tokenizer=str(tokenizer),
          model_sha256=RELEASE_BEST475_SHA256, tokenizer_sha256=tokenizer_sha,
          parent_dropout=parent_dropout)
    return predictor, tokenizer


def find_reference_seg9(input_root: Path) -> Path | None:
    matches = set()
    for pattern in REFERENCE_SEG9["patterns"]:
        for path in input_root.glob(pattern):
            if (path / "config.json").is_file() and (path / "model.safetensors").is_file():
                matches.add(path.resolve())
    if len(matches) != 1:
        phase("reference_seg9_unavailable", found=sorted(str(m) for m in matches))
        return None
    path = next(iter(matches))
    sha = sha256_file(path / "model.safetensors")
    if sha != REFERENCE_SEG9["sha256"]:
        phase("reference_seg9_sha_mismatch", path=str(path), sha256=sha)
        return None
    phase("reference_seg9_ready", path=str(path), sha256=sha)
    return path


def build_time_val(repo: Path, input_root: Path, runtime: Path) -> dict:
    """Build the time-disjoint panel from pinned, non-sealed inputs only."""
    sys.path.insert(0, str(repo / "finetune"))
    import build_time_disjoint_val_panel as tv

    if (tv.SIGNAL_START, tv.SIGNAL_END) != (TIME_VAL_SIGNAL_START, TIME_VAL_SIGNAL_END):
        raise SystemExit("time-val module signal range drift")
    source_panel, source_manifest = tv.find_source(input_root)
    holdout_val = tv.find_holdout_val(input_root)
    manifest = tv.build(source_panel, source_manifest, holdout_val, runtime / "time_val")
    if (manifest["windows"] != TIME_VAL_WINDOWS or manifest["signal_dates"] != TIME_VAL_DATES
            or manifest["identities_sha256"] != TIME_VAL_IDENTITIES_SHA256):
        raise SystemExit(f"time-val contract drift: {manifest['windows']} windows / "
                         f"{manifest['signal_dates']} dates / {manifest['identities_sha256']}")
    if manifest["signal_date_list"][-1] >= SEALED_SIGNAL_START:
        raise SystemExit("time-val leaks into the sealed window")
    manifest["panel_path"] = str(runtime / "time_val" / manifest["panel_file"])
    manifest["source_panel_path"] = str(source_panel)
    phase("time_val_ready", **{k: v for k, v in manifest.items() if k != "windows_by_date"})
    return manifest


def safetensors_keys(path: Path) -> dict[str, list[int]]:
    with Path(path).open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    return {name: list(meta["shape"]) for name, meta in header.items()}


def classify_parent_keys(parent: dict, model: dict) -> dict:
    is_aux = lambda key: key.startswith(AUX_KEY_PREFIXES)  # noqa: E731
    unexpected = sorted(set(parent) - set(model))
    missing = sorted(set(model) - set(parent))
    shared = sorted(set(parent) & set(model))
    return {
        "dropped_aux": [k for k in unexpected if is_aux(k)],
        "bad_unexpected": [k for k in unexpected if not is_aux(k)],
        "bad_missing": [k for k in missing if not is_aux(k)],
        "shape_mismatch": [k for k in shared if list(parent[k]) != list(model[k])],
        "loaded": shared,
    }


def check_parent_and_dropout(repo: Path, parent_dir: Path, env: dict[str, str]) -> dict:
    """Build the predictor exactly like train_predictor (parent config + overrides) on CPU:
    only aux tensors may be dropped, loaded weights must equal the parent bit-for-bit,
    and every dropout module must carry the overridden probability."""
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune"))
    import torch
    from safetensors.torch import load_file
    from model.kronos import Kronos
    from train_predictor import apply_predictor_dropout_overrides, predictor_dropout_report

    kwargs = json.loads((parent_dir / "config.json").read_text())
    kwargs.update({
        "num_sectors": int(env["KRONOS_NUM_SECTORS"]),
        "num_size_buckets": int(env["KRONOS_NUM_SIZE_BUCKETS"]),
        "context_layer": int(env["KRONOS_CONTEXT_LAYER"]),
        "use_size_percentile": env["KRONOS_USE_SIZE_PERCENTILE"] == "1",
        "size_mlp_hidden_dim": int(env["KRONOS_SIZE_MLP_HIDDEN_DIM"]),
        "use_beta_v21_auxiliary": env["KRONOS_USE_BETA_V21_AUXILIARY"] == "1",
    })
    overrides = apply_predictor_dropout_overrides(kwargs, {
        f"predictor_{key}": float(env[name]) for key, name in DROPOUT_ENV.items()
    })
    try:
        model = Kronos.from_pretrained(str(parent_dir), **kwargs)
    except RuntimeError as error:  # shape mismatch etc.
        raise SystemExit(f"Parent weights do not load into the predictor: {error}") from error
    report = predictor_dropout_report(model)
    expected = {key: [value] for key, value in DROPOUT.items()}
    actual = {key: report[key] for key in DROPOUT}
    parent = load_file(str(parent_dir / "model.safetensors"))
    state = model.state_dict()
    diff = classify_parent_keys({k: list(v.shape) for k, v in parent.items()},
                                {k: list(v.shape) for k, v in state.items()})
    unequal = [k for k in diff["loaded"] if not torch.equal(state[k].cpu(), parent[k])]
    result = {
        "overrides": overrides,
        "module_dropout": actual,
        "config_dropout": report["config"],
        "parent_tensors": len(parent),
        "model_tensors": len(state),
        "loaded_tensors": len(diff["loaded"]),
        "dropped_aux": diff["dropped_aux"],
        "bad_unexpected": diff["bad_unexpected"],
        "bad_missing": diff["bad_missing"],
        "shape_mismatch": diff["shape_mismatch"],
        "loaded_not_equal_to_parent": unequal[:8],
    }
    phase("parent_and_dropout_checked", **result)
    del model, parent, state
    if diff["bad_unexpected"] or diff["bad_missing"] or diff["shape_mismatch"] or unequal:
        raise SystemExit(f"Parent weights do not load cleanly: {result}")
    if actual != expected or report["config"] != DROPOUT:
        raise SystemExit(f"Dropout override not applied: {actual} / {report['config']} != {DROPOUT}")
    if kwargs["use_beta_v21_auxiliary"]:
        raise SystemExit("forecast-only run must build the predictor without aux heads")
    return result


def build_environment(data_root: Path, time_val: dict, predictor: Path, tokenizer: Path,
                      output_root: Path, repo: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(repo),
        "KRONOS_TRAIN_DATA_PATHS": str(data_root / "processed_datasets/train_data.pkl"),
        # Training-time val (WFL, log only) = the time-disjoint panel.
        "KRONOS_VAL_DATA_PATHS": time_val["panel_path"],
        "KRONOS_DATASET_PATH": str(data_root / "processed_datasets"),
        "KRONOS_METADATA_PATH": str(data_root / "asset_metadata.csv"),
        "KRONOS_DATA_MANIFEST_SHA256": sha256_file(data_root / "data_manifest.json"),
        "KRONOS_TRAIN_SIGNAL_END": TRAIN_SIGNAL_END,
        "KRONOS_VAL_SIGNAL_START": TIME_VAL_SIGNAL_START,
        "KRONOS_VAL_SIGNAL_END": TIME_VAL_SIGNAL_END,
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
        # Heavier regularization (vs parent 0.2/0.2/0/0).
        **{DROPOUT_ENV[key]: str(value) for key, value in DROPOUT.items()},
        "KRONOS_ADAM_WEIGHT_DECAY": WEIGHT_DECAY,
        "KRONOS_TRAINABLE_TRANSFORMER_LAYERS": "-1",
        "KRONOS_TRAIN_BETA_V21_HEADS_ONLY": "0",
        "KRONOS_SPLIT_TRUNK_HEAD_LR": "0",
        "KRONOS_PREDICTOR_LOSS_MODE": "forecast",
        "KRONOS_HISTORY_LOSS_WEIGHT": "0.02",
        "KRONOS_FORECAST_HORIZON_WEIGHTS":
            "1.364,1.364,1.364,1.136,1.136,0.909,0.909,0.682,0.682,0.455",
        "KRONOS_SCHEDULER": "warmup_constant_cosine",
        "KRONOS_SCHEDULER_WARMUP_RATIO": WARMUP_RATIO,
        "KRONOS_SCHEDULER_DECAY_START_RATIO": DECAY_START_RATIO,
        "KRONOS_PREDICTOR_LEARNING_RATE": PEAK_LR,
        "KRONOS_CONDITION_LEARNING_RATE": PEAK_LR,
        "KRONOS_PREDICTOR_WARMUP_START_LR": WARMUP_START_LR,
        "KRONOS_CONDITION_WARMUP_START_LR": WARMUP_START_LR,
        "KRONOS_PREDICTOR_MIN_LR": MIN_LR,
        "KRONOS_CONDITION_MIN_LR": MIN_LR,
        "KRONOS_SCHEDULER_MIN_LR": MIN_LR,
        "KRONOS_USE_BETA_V21_AUXILIARY": "0",
        "KRONOS_SAME_DAY_RANKING_BATCHES": "0",
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
        "KRONOS_TRAIN_SAMPLES_PER_SEGMENT": str(SAMPLES_PER_SEGMENT),
        "KRONOS_COVERAGE_SEED": COVERAGE_SEED,
        "KRONOS_COVERAGE_PASSES": "1",
        "KRONOS_EPOCHS": str(TOTAL_SEGMENTS),
        "KRONOS_REQUIRE_FULL_COVERAGE": "0",
        "KRONOS_MAX_SEGMENTS_PER_RUN": str(CHUNK_SEGMENTS),
        "KRONOS_MAX_RUNTIME_SECONDS": str(TRAIN_DEADLINE_SECONDS),
        "KRONOS_TORCHRUN_NPROC_PER_NODE": str(WORLD),
        "KRONOS_BATCH_SIZE": str(BATCH_PER_GPU),
        "KRONOS_NUM_WORKERS": "2",
        "KRONOS_USE_AMP": "1",
        "KRONOS_AMP_DTYPE": "float16",
        "KRONOS_BEST_SELECTION_METRIC": "forecast",
        # Old-val red lines do not apply to the new val scale: monitors disabled.
        "KRONOS_FORECAST_MONITOR_BASE": "",
        "KRONOS_FORECAST_MONITOR_MARGIN": "",
        "KRONOS_FORECAST_DRIFT_ALERT_BASE": "",
        "KRONOS_FORECAST_DRIFT_ALERT_MARGIN": "",
        "KRONOS_EARLY_STOPPING_PATIENCE": "0",
        "KRONOS_RESUME_TRAINING": "0",
        "KRONOS_KEEP_EXISTING_BEST": "0",
        "KRONOS_SWANLAB_SEGMENT_OFFSET": "0",
        "KRONOS_COVERAGE_EPOCH_OFFSET": "0",
        "KRONOS_SNAPSHOT_EVERY_SEGMENTS": str(SNAPSHOT_EVERY),
        "KRONOS_SNAPSHOT_DIR": str(output_root / "snapshots"),
        "SWANLAB_PROJECT": SWANLAB_PROJECT,
        "SWANLAB_EXPERIMENT_NAME": OUTPUT_NAME,
        "SWANLAB_RUN_ID": SWANLAB_RUN_ID,
    }


def expected_recipe() -> dict[str, str]:
    return {
        "KRONOS_SCHEDULER": "warmup_constant_cosine",
        "KRONOS_SCHEDULER_WARMUP_RATIO": "0.0208333333",
        "KRONOS_SCHEDULER_DECAY_START_RATIO": "0.6666666667",
        "KRONOS_PREDICTOR_LEARNING_RATE": "1e-5",
        "KRONOS_CONDITION_LEARNING_RATE": "1e-5",
        "KRONOS_PREDICTOR_WARMUP_START_LR": "1e-6",
        "KRONOS_CONDITION_WARMUP_START_LR": "1e-6",
        "KRONOS_PREDICTOR_MIN_LR": "1e-6",
        "KRONOS_CONDITION_MIN_LR": "1e-6",
        "KRONOS_RESID_DROPOUT_P": "0.25",
        "KRONOS_FFN_DROPOUT_P": "0.25",
        "KRONOS_ATTN_DROPOUT_P": "0.1",
        "KRONOS_TOKEN_DROPOUT_P": "0.1",
        "KRONOS_ADAM_WEIGHT_DECAY": "0.1",
        "KRONOS_USE_BETA_V21_AUXILIARY": "0",
        "KRONOS_SAME_DAY_RANKING_BATCHES": "0",
        "KRONOS_SPLIT_TRUNK_HEAD_LR": "0",
        "KRONOS_TRAIN_BETA_V21_HEADS_ONLY": "0",
        "KRONOS_TRAINABLE_TRANSFORMER_LAYERS": "-1",
        "KRONOS_PREDICTOR_LOSS_MODE": "forecast",
        "KRONOS_BEST_SELECTION_METRIC": "forecast",
        "KRONOS_EPOCHS": "48",
        "KRONOS_MAX_SEGMENTS_PER_RUN": "4",
        "KRONOS_SNAPSHOT_EVERY_SEGMENTS": "4",
        "KRONOS_TRAIN_SAMPLES_PER_SEGMENT": "20000",
        "KRONOS_BATCH_SIZE": "32",
        "KRONOS_COVERAGE_SEED": "20261007",
        "KRONOS_COVERAGE_EPOCH_OFFSET": "0",
        "KRONOS_REQUIRE_FULL_COVERAGE": "0",
        "KRONOS_KEEP_EXISTING_BEST": "0",
        "KRONOS_TRAIN_SIGNAL_END": "2026-07-02",
        "KRONOS_VAL_SIGNAL_START": "2026-07-17",
        "KRONOS_VAL_SIGNAL_END": "2026-08-10",
        "KRONOS_VALIDATION_FULL_ONLY": "1",
        "KRONOS_PREDICTOR_SAVE_FOLDER": OUTPUT_NAME,
    }


EXPECTED_RECIPE = expected_recipe()


def verify_recipe(env: dict[str, str]) -> None:
    actual = {key: env.get(key) for key in EXPECTED_RECIPE}
    phase("recipe_verified", kernel_version=KERNEL_VERSION, **actual)
    bad = {k: (actual[k], v) for k, v in EXPECTED_RECIPE.items() if actual[k] != v}
    if bad:
        raise SystemExit(f"Recipe drift: {bad}")
    val_path = env.get("KRONOS_VAL_DATA_PATHS", "")
    if "temporal_symbol_validation_v1" in val_path or "evaluation_panel" in val_path:
        raise SystemExit(f"Training-time val must be the time-disjoint panel, got {val_path}")


# --------------------------------------------------------------------------- training


class SafeRun:
    """SwanLab wrapper: a logging failure must never stop the run."""

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


def check_dropout_line(line: str) -> bool | None:
    """None if not the dropout line; True if it matches DROPOUT; False otherwise."""
    match = DROPOUT_LINE_RE.search(line)
    if match is None:
        return None
    values = [float(v) for v in match.groups()]
    expected = [DROPOUT[k] for k in ("resid_dropout_p", "ffn_dropout_p",
                                     "attn_dropout_p", "token_dropout_p")]
    return all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(values, expected))


def check_schedule_line(line: str) -> bool | None:
    match = LR_PLAN_RE.search(line)
    if match:
        total = int(match.group(1).replace(",", ""))
        warmup = int(match.group(2).replace(",", ""))
        return total == TOTAL_STEPS and warmup == EXPECTED_WARMUP_STEPS
    match = WSD_RE.search(line)
    if match:
        return int(match.group(1).replace(",", "")) == EXPECTED_DECAY_START_STEP
    return None


def stream_training(repo: Path, env: dict[str, str], output_root: Path, run: SafeRun,
                    chunk: int = 0) -> tuple[int, int]:
    command = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={WORLD}", str(repo / "finetune/train_predictor.py")]
    phase("training_started", chunk=chunk, resume=env.get("KRONOS_RESUME_TRAINING"),
          max_segments_this_run=env.get("KRONOS_MAX_SEGMENTS_PER_RUN"))
    output_root.mkdir(parents=True, exist_ok=True)
    current_segment, total_steps, dropout_seen, schedule_checks = 0, STEPS_PER_SEGMENT, False, 0
    with (output_root / "run.log").open("a", buffering=1) as log_handle:
        child = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert child.stdout is not None

        def abort(message: str) -> None:
            child.terminate()
            child.wait()
            raise SystemExit(message)

        for line in child.stdout:
            log_handle.write(line)
            print(line, end="", flush=True)
            dropout_ok = check_dropout_line(line)
            if dropout_ok is not None:
                dropout_seen = True
                phase("dropout_line_checked", ok=dropout_ok, line=line.strip())
                if not dropout_ok:
                    abort(f"Training built the predictor with wrong dropout: {line.strip()}")
            schedule_ok = check_schedule_line(line)
            if schedule_ok is not None:
                schedule_checks += 1
                phase("schedule_line_checked", ok=schedule_ok, line=line.strip())
                if not schedule_ok:
                    abort(f"LR plan drift: {line.strip()} (expected {TOTAL_STEPS} steps, "
                          f"warmup {EXPECTED_WARMUP_STEPS}, decay {EXPECTED_DECAY_START_STEP})")
            trainable = TRAINABLE_RE.search(line)
            if trainable:
                got = int(trainable.group(1).replace(",", ""))
                total = int(trainable.group(2).replace(",", ""))
                if got != total:
                    abort(f"expected every parameter trainable; got {got}/{total}")
            train_match = TRAIN_LOG_RE.search(line)
            if train_match:
                segment, _, step, steps, lr, cond_lr, loss, forecast, history = train_match.groups()
                current_segment, total_steps = int(segment), int(steps)
                run.log({"train/loss": float(loss), "train/forecast_loss": float(forecast),
                         "train/history_loss": float(history), "train/learning_rate": float(lr),
                         "train/condition_learning_rate": float(cond_lr),
                         "segment": current_segment},
                        step=(current_segment - 1) * total_steps + int(step))
            validation = VALIDATION_LOG_RE.search(line)
            if validation and current_segment:
                forecast, history, full = map(float, validation.groups())
                run.log({"validation/forecast_loss": forecast, "validation/history_loss": history,
                         "validation/full_loss": full, "validation/samples": TIME_VAL_WINDOWS,
                         "segment": current_segment}, step=current_segment * total_steps)
            weighted = WEIGHTED_FORECAST_RE.search(line)
            if weighted and current_segment:
                run.log({"validation/weighted_forecast_loss": float(weighted.group(1)),
                         "segment": current_segment}, step=current_segment * total_steps)
            snapshot = SNAPSHOT_RE.search(line)
            if snapshot:
                phase("snapshot_saved", path=snapshot.group(1), segment=int(snapshot.group(2)))
            best = BEST_SAVED_RE.search(line)
            if best:
                phase("best_saved", path=best.group(1), metric=best.group(2),
                      value=float(best.group(3)))
            epoch = EPOCH_TIME_RE.search(line)
            if epoch and current_segment:
                hours, minutes, seconds = map(int, epoch.groups())
                run.log({"timing/segment_seconds": hours * 3600 + minutes * 60 + seconds,
                         "segment": current_segment}, step=current_segment * total_steps)
        return_code = child.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    if not dropout_seen:
        raise SystemExit("train_predictor never logged the predictor dropout line")
    if chunk == 0 and schedule_checks < 2:
        raise SystemExit("train_predictor did not log the WSD learning-rate plan")
    phase("training_finished", chunk=chunk, completed_segment=current_segment)
    return current_segment, total_steps


def chunk_environment(env: dict[str, str], chunk: int) -> dict[str, str]:
    remaining = int(KERNEL_START + TRAIN_DEADLINE_SECONDS - time.time())
    return {**env, "KRONOS_RESUME_TRAINING": "1" if chunk > 0 else "0",
            "KRONOS_MAX_RUNTIME_SECONDS": str(max(900, remaining))}


# --------------------------------------------------------------------------- selection


def by_date_ic(summary: dict) -> dict[str, float]:
    return {row["asof_date"]: row["return10d_rank_ic"]
            for row in summary.get("by_signal_date", [])
            if row.get("return10d_rank_ic") is not None
            and not (isinstance(row["return10d_rank_ic"], float) and math.isnan(row["return10d_rank_ic"]))}


def paired_stats(a: dict[str, float], b: dict[str, float], dates=None) -> dict | None:
    """a - b paired over common dates (t with n-1 df)."""
    common = sorted(set(a) & set(b))
    if dates is not None:
        common = [d for d in common if d in set(dates)]
    diffs = [float(a[d]) - float(b[d]) for d in common]
    n = len(diffs)
    if n < 2:
        return None
    mean = sum(diffs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1))
    return {"n": n, "mean_delta": mean, "sd": sd,
            "t": (mean / (sd / math.sqrt(n))) if sd > 0 else None,
            "wins": sum(1 for x in diffs if x > 0)}


def build_selection(summaries: dict[str, dict]) -> dict:
    seg0 = summaries.get(SEG0_LABEL)
    seg0_ic = by_date_ic(seg0) if seg0 else {}
    rows = []
    for label, summary in summaries.items():
        ic = by_date_ic(summary)
        dates = sorted(ic)
        strict = [d for d in dates if d >= STRICT_SUBSET_START]
        rows.append({
            "label": label,
            "segment": summary.get("segment"),
            "role": ("parent" if label == SEG0_LABEL else
                     "reference" if label == REFERENCE_SEG9["label"] else
                     "candidate" if label.startswith(SNAPSHOT_PREFIX + "seg") else "diagnostic"),
            "return10d_rank_ic_daily": summary.get("return10d_rank_ic_daily"),
            "return10d_rank_ic_se": summary.get("return10d_rank_ic_se"),
            "top_bottom_decile_return10d": summary.get("top_bottom_decile_return10d"),
            "wfl_subsample": summary.get("weighted_forecast_loss_subsample"),
            "signal_dates": len(dates),
            "vs_seg0": paired_stats(ic, seg0_ic) if seg0 and label != SEG0_LABEL else None,
            "vs_seg0_strict_subset": (paired_stats(ic, seg0_ic, strict)
                                      if seg0 and label != SEG0_LABEL else None),
        })
    rows.sort(key=lambda r: (r["segment"] is None, r["segment"] or 0, r["label"]))
    candidates = [r for r in rows if r["role"] == "candidate"
                  and r["return10d_rank_ic_daily"] is not None]
    best = max(candidates, key=lambda r: r["return10d_rank_ic_daily"]) if candidates else None
    endpoint = next((r for r in candidates if r["segment"] == GATE_ENDPOINT_SEGMENT), None)
    stats = (best or {}).get("vs_seg0") or {}
    t_value = stats.get("t")
    gate = {
        "c_star": None if best is None else best["label"],
        "c_star_segment": None if best is None else best["segment"],
        "c_star_ic": None if best is None else best["return10d_rank_ic_daily"],
        "seg0_ic": None if seg0 is None else seg0.get("return10d_rank_ic_daily"),
        "paired": stats or None,
        "t_pass": t_value is not None and t_value >= GATE_MIN_T,
        "wins_pass": int(stats.get("wins", 0)) >= GATE_MIN_WINS,
        "endpoint_label": None if endpoint is None else endpoint["label"],
        "endpoint_delta": (None if endpoint is None or not endpoint["vs_seg0"]
                           else endpoint["vs_seg0"]["mean_delta"]),
        "complete": bool(seg0) and endpoint is not None
                    and int(stats.get("n", 0)) == TIME_VAL_DATES,
    }
    gate["endpoint_pass"] = gate["endpoint_delta"] is not None and gate["endpoint_delta"] > 0
    gate["pass"] = bool(gate["complete"] and gate["t_pass"] and gate["wins_pass"]
                        and gate["endpoint_pass"])
    reference = next((r for r in rows if r["role"] == "reference"), None)
    return {
        "kernel": KERNEL_ID,
        "val_contract": TIME_VAL_CONTRACT,
        "selection_metric": "time-disjoint val generative return10d rank IC, daily mean",
        "rule": (f"C* = max daily IC over {SNAPSHOT_PREFIX}segNNN snapshots; pass iff C*-Seg0 "
                 f"paired t >= {GATE_MIN_T} over {TIME_VAL_DATES} dates AND wins >= "
                 f"{GATE_MIN_WINS}/{TIME_VAL_DATES} AND Seg{GATE_ENDPOINT_SEGMENT}-Seg0 > 0. "
                 "Pass -> one-shot confirmation on the next sealed window after 2026-09-03."),
        "caveats": [
            "17 overlapping 10d label windows ~ 2 independent periods: paired t is optimistic",
            "max over 12 snapshots inflates C*; the sealed confirmation is the real test",
            "signals 07-17..07-30 have labels partly inside the lineage's training targets (<= 07-31)",
        ],
        "gate": gate,
        "reference_seg9": None if reference is None else {
            "vs_seg0": reference["vs_seg0"],
            "sealed_oos_direction": "Seg0 > Seg9 (0.1546 vs 0.1455)",
            "old_val_direction": "Seg9 > Seg0 (0.3254 vs 0.3141)",
            "agrees_with_sealed_oos": (None if not reference["vs_seg0"]
                                       else reference["vs_seg0"]["mean_delta"] < 0),
        },
        "checkpoints": rows,
        "elapsed_sec": round(time.time() - KERNEL_START, 1),
    }


class Scorer:
    """Scores checkpoints between training chunks with the time-disjoint contract."""

    def __init__(self, repo: Path, output_root: Path, tokenizer: Path, input_root: Path,
                 time_val: dict, sector_metadata: Path, run: SafeRun):
        self.repo, self.output_root, self.tokenizer = repo, output_root, tokenizer
        self.input_root, self.time_val, self.sector_metadata = input_root, time_val, sector_metadata
        self.run = run
        self.eval_dir = output_root / "val_gen_ic_time_disjoint"
        self.items: list[dict] = []
        self.summaries: dict[str, dict] = {}
        self.total_steps = STEPS_PER_SEGMENT

    def val_contract(self) -> dict:
        return {
            "name": TIME_VAL_CONTRACT,
            "val_data": self.time_val["panel_path"],
            "val_sha256": self.time_val["panel_sha256"],
            "sector_metadata": str(self.sector_metadata),
            "signal_start": TIME_VAL_SIGNAL_START,
            "signal_end": TIME_VAL_SIGNAL_END,
            "expected_samples": TIME_VAL_WINDOWS,
            "expected_dates": TIME_VAL_DATES,
            "identities_sha256": TIME_VAL_IDENTITIES_SHA256,
            "subsample_dates": "all",
            "same_contract_as": "Step-1 decode/score/label; val set swapped to time-disjoint",
        }

    def snapshot_items(self) -> list[dict]:
        known = {item["label"] for item in self.items}
        found = []
        for path in sorted((self.output_root / "snapshots").glob("seg[0-9][0-9][0-9]")):
            label = f"{SNAPSHOT_PREFIX}{path.name}"
            if not (path / "model.safetensors").is_file() or label in known:
                continue
            metric = {}
            if (path / "snapshot_metric.json").is_file():
                metric = json.loads((path / "snapshot_metric.json").read_text())
            found.append({
                "label": label, "path": str(path),
                "sha256": sha256_file(path / "model.safetensors"),
                "segment": int(metric.get("segment", int(path.name[3:]))),
                "full_val_wfl": metric.get("forecast_loss"),
                "note": f"heavy-reg snapshot; lr_end={metric.get('learning_rates')}",
            })
        return found

    def score(self, new_items: list[dict]) -> None:
        new_items = [item for item in new_items if item["label"] not in self.summaries]
        if not new_items:
            return
        remaining = KERNEL_START + SESSION_HARD_LIMIT_SECONDS - time.time()
        if remaining < MIN_EVAL_SECONDS:
            phase("valgenic_skipped", reason="session budget", remaining_seconds=round(remaining),
                  labels=[item["label"] for item in new_items])
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
            "world": WORLD,
            "effective_batch": 256,
            "checkpoints": self.items,  # cumulative; finished shards are skipped
            "val_contract": self.val_contract(),
            "reference": {"seg0_label": SEG0_LABEL, "parent_sha256": RELEASE_BEST475_SHA256,
                          "reference_label": REFERENCE_SEG9["label"]},
        }
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        spec_path = self.eval_dir / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
        labels = [item["label"] for item in new_items]
        phase("valgenic_started", new=labels, remaining_seconds=round(remaining))
        started = time.time()
        try:
            subprocess.run([sys.executable, "-u", str(self.repo / "finetune/val_gen_ic_driver.py"),
                            "--spec", str(spec_path)], cwd=self.repo, check=True,
                           env={**os.environ, "PYTHONUNBUFFERED": "1",
                                "PYTHONPATH": os.pathsep.join([str(self.repo),
                                                               str(self.repo / "finetune")])})
        except subprocess.CalledProcessError as error:
            phase("valgenic_failed", new=labels, returncode=error.returncode)
        self.collect(labels)
        phase("valgenic_finished", new=labels, seconds=round(time.time() - started, 1))

    def collect(self, labels: list[str]) -> None:
        for label in labels:
            path = self.eval_dir / "results" / f"{label}_val_summary.json"
            if not path.is_file():
                phase("valgenic_missing_summary", label=label)
                continue
            self.summaries[label] = json.loads(path.read_text())
        selection = build_selection(self.summaries)
        (self.output_root / "heavy_reg_selection.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
        for row in selection["checkpoints"]:
            if row["label"] not in labels:
                continue
            segment = int(row["segment"] or 0)
            payload = {
                "valgenic/return10d_ic_daily": row["return10d_rank_ic_daily"],
                "valgenic/return10d_ic_se": row["return10d_rank_ic_se"],
                "valgenic/top_bottom_decile": row["top_bottom_decile_return10d"],
                "valgenic/wfl_subsample": row["wfl_subsample"],
                "valgenic/segment": segment,
            }
            if row["vs_seg0"]:
                payload["valgenic/delta_ic_vs_seg0"] = row["vs_seg0"]["mean_delta"]
                payload["valgenic/paired_t_vs_seg0"] = row["vs_seg0"]["t"]
                payload["valgenic/wins_vs_seg0"] = row["vs_seg0"]["wins"]
            prefix = "valgenic_ref" if row["role"] == "reference" else "valgenic"
            payload = {k.replace("valgenic/", f"{prefix}/"): v for k, v in payload.items()}
            self.run.log({k: v for k, v in payload.items() if v is not None},
                         step=segment * self.total_steps)
            phase("checkpoint_val_gen_ic", **row)
        phase("selection_updated", **selection["gate"])


# --------------------------------------------------------------------------- main


def start_swanlab(env: dict[str, str]):
    import swanlab

    key = env.get("SWANLAB_API_KEY", "").strip()
    source = "env"
    if not key:
        try:
            from kaggle_secrets import UserSecretsClient

            key = (UserSecretsClient().get_secret("SWANLAB_API_KEY") or "").strip()
            source = "kaggle_secret"
        except Exception:  # noqa: BLE001
            key = ""
    if not key:
        key, source = SWANLAB_API_KEY_FALLBACK, "fallback"
    swanlab.login(api_key=key)
    run = swanlab.init(
        project=SWANLAB_PROJECT, workspace=SWANLAB_WORKSPACE, experiment_name=OUTPUT_NAME,
        id=SWANLAB_RUN_ID, resume="allow",
        tags=["kronos-base", "beta-v2.1", "heavy-reg", "time-disjoint-val", "forecast-only",
              "wsd", "dual-t4", "parent-best475"],
        config={key.lower(): value for key, value in EXPECTED_RECIPE.items()} | {
            "parent": "beta_v21_release_best475", "parent_model_sha256": RELEASE_BEST475_SHA256,
            "parent_dropout": PARENT_DROPOUT, "dropout": DROPOUT, "aux_heads_loaded": False,
            "kernel": KERNEL_ID, "kernel_version": KERNEL_VERSION, "adamw": "fresh",
            "effective_batch_size": BATCH_PER_GPU * WORLD, "total_steps": TOTAL_STEPS,
            "warmup_steps": EXPECTED_WARMUP_STEPS, "decay_start_step": EXPECTED_DECAY_START_STEP,
            "val_contract": TIME_VAL_CONTRACT, "val_windows": TIME_VAL_WINDOWS,
            "val_dates": TIME_VAL_DATES,
            "selection": "time-disjoint val gen return10d daily IC; gate t>=2, wins>=10/17, Seg48>Seg0",
        },
    )
    phase("swanlab_ready", url=SWANLAB_URL, key_source=source)
    return swanlab, SafeRun(run)


def main() -> None:
    phase("started", kernel=KERNEL_ID, experiment=OUTPUT_NAME, kernel_version=KERNEL_VERSION,
          parent_sha256=RELEASE_BEST475_SHA256, dropout=DROPOUT, parent_dropout=PARENT_DROPOUT,
          scheduler=(f"WSD {WARMUP_START_LR}->{PEAK_LR} seg1, hold to seg{DECAY_START_SEGMENT}, "
                     f"cosine ->{MIN_LR} at seg{TOTAL_SEGMENTS}"),
          chunks=f"{TOTAL_SEGMENTS // CHUNK_SEGMENTS} x {CHUNK_SEGMENTS}", swanlab=SWANLAB_URL)
    runtime = Path("/kaggle/working") / RUNTIME_DIR
    input_root = Path("/kaggle/input")
    repo = runtime / "Kronos"
    output_root = runtime / "outputs" / "models" / OUTPUT_NAME
    data_root = find_data_root(input_root)
    predictor, tokenizer = download_model(runtime)
    reference_seg9 = find_reference_seg9(input_root)
    if output_root.exists():
        shutil.rmtree(output_root)

    clone_repo(repo)
    overlay_embedded_sources(repo)
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--progress-bar", "off", "-r", "requirements.txt"], cwd=repo, check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--progress-bar", "off", "--force-reinstall", f"torch=={TORCH_VERSION}",
                    "--index-url", TORCH_INDEX_URL], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--progress-bar", "off", "swanlab"], check=True)
    import torch

    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(gpus) != WORLD or any("T4" not in name for name in gpus):
        raise SystemExit(f"Expected exactly two Tesla T4 GPUs, found {gpus}")
    phase("device_ready", torch=torch.__version__, gpus=gpus)

    time_val = build_time_val(repo, input_root, runtime)
    env = build_environment(data_root, time_val, predictor, tokenizer, output_root, repo)
    verify_recipe(env)
    key_report = check_parent_and_dropout(repo, predictor, env)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": OUTPUT_NAME, "kernel": KERNEL_ID, "kernel_version": KERNEL_VERSION,
        "parent": {"label": "beta_v21_release_best475", "modelscope_repo": MODEL_REPO,
                   "path": str(predictor), "model_sha256": RELEASE_BEST475_SHA256,
                   "dropped_aux_tensors": key_report["dropped_aux"], "adamw": "fresh",
                   "dropout": PARENT_DROPOUT},
        "dropout": DROPOUT, "weight_decay": float(WEIGHT_DECAY),
        "data": {"train_root": str(data_root),
                 "data_manifest_sha256": env["KRONOS_DATA_MANIFEST_SHA256"],
                 "time_val": {k: v for k, v in time_val.items() if k != "windows_by_date"}},
        "recipe": EXPECTED_RECIPE,
        "schedule": {"total_steps": TOTAL_STEPS, "warmup_steps": EXPECTED_WARMUP_STEPS,
                     "decay_start_step": EXPECTED_DECAY_START_STEP,
                     "lr_at_segment_end": {f"seg{s:03d}": lr_at_segment_end(s)
                                           for s in range(SNAPSHOT_EVERY, TOTAL_SEGMENTS + 1,
                                                          SNAPSHOT_EVERY)}},
        "reference_seg9": None if reference_seg9 is None else str(reference_seg9),
        "selection_rule": build_selection({})["rule"],
        "swanlab": SWANLAB_URL,
    }
    (output_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    try:
        swanlab, swanlab_run = start_swanlab(env)
    except Exception as error:  # noqa: BLE001 - SwanLab must never block the run
        phase("swanlab_unavailable", error=repr(error))
        swanlab, swanlab_run = None, SafeRun(None)
    scorer = Scorer(repo, output_root, tokenizer, input_root, time_val,
                    data_root / "asset_metadata.csv", swanlab_run)
    first_items = [{"label": SEG0_LABEL, "path": str(predictor), "sha256": RELEASE_BEST475_SHA256,
                    "segment": 0, "full_val_wfl": None,
                    "note": "Seg0 = Beta v2.1 release Best@475 (parent)"}]
    if reference_seg9 is not None:
        first_items.append({"label": REFERENCE_SEG9["label"], "path": str(reference_seg9),
                            "sha256": REFERENCE_SEG9["sha256"], "segment": None,
                            "full_val_wfl": None, "note": REFERENCE_SEG9["note"]})

    training_error = None
    completed = 0
    for chunk in range(TOTAL_SEGMENTS // CHUNK_SEGMENTS):
        if time.time() > KERNEL_START + TRAIN_DEADLINE_SECONDS:
            phase("training_deadline", completed_segment=completed)
            break
        try:
            segment, steps = stream_training(repo, chunk_environment(env, chunk), output_root,
                                             swanlab_run, chunk=chunk)
            scorer.total_steps = max(1, steps)
        except subprocess.CalledProcessError as error:
            training_error = error
            phase("training_failed", chunk=chunk, returncode=error.returncode)
            segment = completed
        progressed = segment > completed
        completed = max(completed, segment)
        snapshots = scorer.snapshot_items()
        # Chunk 1: Seg0 and the reference first, then the first snapshot.
        scorer.score((first_items if chunk == 0 else []) + snapshots)
        if training_error is not None or not progressed or completed >= TOTAL_SEGMENTS:
            break

    try:
        subprocess.run([sys.executable, str(repo / "finetune/export_last_model.py"),
                        "--repo-root", str(repo), "--output-root", str(output_root)],
                       cwd=repo, env=env, check=True)
        phase("export_finished", output_root=str(output_root))
    except subprocess.CalledProcessError as error:
        phase("export_failed", returncode=error.returncode)
    extra = scorer.snapshot_items()
    best = output_root / "checkpoints" / "best_model"
    if (best / "model.safetensors").is_file() and (best / "best_metric.json").is_file():
        sha = sha256_file(best / "model.safetensors")
        metric = json.loads((best / "best_metric.json").read_text())
        known = {item["sha256"] for item in scorer.items + extra}
        phase("best_by_val_wfl", sha256=sha, segment=metric.get("segment"),
              selection_loss=metric.get("selection_loss"), already_scored=sha in known)
        if sha not in known:
            extra.append({"label": f"diag_best_wfl_seg{int(metric.get('segment', -1)):03d}",
                          "path": str(best), "sha256": sha, "segment": metric.get("segment"),
                          "full_val_wfl": metric.get("selection_loss"),
                          "note": "diagnostic only: best time-val WFL (not a candidate)"})
    scorer.score(extra)
    if swanlab is not None:
        swanlab.finish()
    last_state = output_root / "checkpoints" / "last_state.pt"
    if last_state.is_file():
        last_state.unlink()
        phase("last_state_removed", reason="output keeps weights only")
    if training_error is not None:
        raise training_error
    phase("done", total_seconds=round(time.time() - KERNEL_START, 1),
          completed_segment=completed, swanlab=SWANLAB_URL)


if __name__ == "__main__":
    main()
