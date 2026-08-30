#!/usr/bin/env bash
# Continue Beta v1.1 Last@1058 for two coverage passes at fixed terminal LRs.
set -euo pipefail

MODE="${1:-dry-run}"
if [[ "$MODE" != "baseline" && "$MODE" != "dry-run" && "$MODE" != "diagnostic" && "$MODE" != "full" ]]; then
  echo "usage: $0 {baseline|dry-run|diagnostic|full}" >&2
  exit 2
fi

BASE="/nfsdata/models/2026/kronos-v1-beta"
CODE_ROOT="$BASE/code/Kronos"
DATA_ROOT="$BASE/data/a_share_full_market_v1_beta"
VALIDATION_ROOT="$DATA_ROOT/natural_validation_2025h2_2026h1_v2"
SOURCE_RUN="$BASE/runs/v6_natural_twospeed_v2_seed100"
SOURCE_OUTPUT="$SOURCE_RUN/outputs/models/a_share_v1_beta_v6_natural_twospeed_v2_120d_to_10d"
PARENT_ROOT="$SOURCE_OUTPUT/checkpoints/last_model"
TOKENIZER_ROOT="$BASE/models/tokenizer"
RUN_ROOT="$BASE/runs/beta_v1_1_last1058_fixed_lr_2pass_seed100"
OUTPUT_NAME="a_share_beta_v1_1_last1058_fixed_lr_2pass_120d_to_10d"
OUTPUT_ROOT="$RUN_ROOT/outputs/models/$OUTPUT_NAME"
PYTHON=(sudo -n -E "$BASE/env/a800-py312/bin/python")

DATA_SHA="214c375f47e9843b7d836e414199f444ac8cd139e2ef1bb51adf526bfdc6261c"
NATURAL_SHA="f329b1ca2a6f94c373403563d941d05c66de3754e7e3206c002cb5e475be1ac7"
PARENT_SHA="59ac7999625e247e2211738ec42da4759c8fdb1991d683ec2de67e40f2a94bcc"

required=(
  "$CODE_ROOT/finetune/config.py"
  "$CODE_ROOT/finetune/dataset.py"
  "$CODE_ROOT/finetune/train_predictor.py"
  "$CODE_ROOT/finetune/evaluate_fixed_validation_baseline.py"
  "$CODE_ROOT/finetune/export_last_model.py"
  "$DATA_ROOT/data_manifest.json"
  "$DATA_ROOT/processed_datasets/train_data.pkl"
  "$DATA_ROOT/asset_metadata.csv"
  "$VALIDATION_ROOT/natural_validation_manifest.json"
  "$VALIDATION_ROOT/natural_validation_samples.jsonl"
  "$PARENT_ROOT/model.safetensors"
  "$PARENT_ROOT/config.json"
  "$TOKENIZER_ROOT/model.safetensors"
  "$TOKENIZER_ROOT/config.json"
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "missing required input: $path" >&2; exit 1; }
done

[[ "$(sha256sum "$PARENT_ROOT/model.safetensors" | awk '{print $1}')" == "$PARENT_SHA" ]] || {
  echo "Beta v1.1 Last@1058 checksum mismatch" >&2
  exit 1
}
[[ "$(sha256sum "$VALIDATION_ROOT/natural_validation_manifest.json" | awk '{print $1}')" == "$NATURAL_SHA" ]] || {
  echo "Natural Validation manifest checksum mismatch" >&2
  exit 1
}

"${PYTHON[@]}" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
PY

mkdir -p "$RUN_ROOT"
if [[ "$MODE" == "baseline" ]]; then
  SEGMENTS_THIS_RUN=0
  RESUME=0
elif [[ "$MODE" == "dry-run" ]]; then
  if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "dry-run refuses an existing output directory: $OUTPUT_ROOT" >&2
    exit 1
  fi
  SEGMENTS_THIS_RUN=1
  RESUME=0
elif [[ "$MODE" == "diagnostic" ]]; then
  [[ -f "$OUTPUT_ROOT/checkpoints/last_state.pt" ]] || {
    echo "diagnostic requires the dry-run last_state.pt" >&2
    exit 1
  }
  SEGMENTS_THIS_RUN=49
  RESUME=1
else
  [[ -f "$OUTPUT_ROOT/checkpoints/last_state.pt" ]] || {
    echo "full requires a resumable last_state.pt" >&2
    exit 1
  }
  SEGMENTS_THIS_RUN=1008
  RESUME=1
fi

if [[ ! -f "$RUN_ROOT/experiment_manifest.json" ]]; then
  cat > "$RUN_ROOT/experiment_manifest.json" <<EOF
{
  "stage_id": "beta_v1_1_last1058_fixed_lr_2pass",
  "purpose": "test terminal learning rates without a new LR peak",
  "execution": "single_gpu_a800_offline",
  "parent": {"release": "Beta v1.1", "checkpoint": "Last@1058", "path": "$PARENT_ROOT", "model_sha256": "$PARENT_SHA"},
  "initialization": {"model": "weights_only", "optimizer": "fresh", "scheduler": "fixed", "sector_embedding": "preserved", "size_percentile_mlp": "preserved"},
  "validation": {"profile": "natural", "periods": ["2025H2", "2026H1"], "manifest_sha256": "$NATURAL_SHA", "quick_samples": 6000, "large_samples": 24000, "large_interval_segments": 10},
  "training": {"global_segments": 1058, "coverage_passes": 2, "samples_per_segment": 20000, "seed": 100, "gpu": 0, "amp_dtype": "bfloat16", "predictor_fixed_lr": 5e-7, "condition_fixed_lr": 1e-6, "scheduler": "fixed"},
  "selection_policy": "Do not use the sealed 2026-08 future set for checkpoint selection or hyperparameter changes",
  "known_limitation": "The completed prior scheduler cannot be extended; this experiment starts a fresh optimizer and scheduler from Last@1058 weights at half the prior peak learning rates"
}
EOF
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$CODE_ROOT"
export KRONOS_TRAIN_DATA_PATHS="$DATA_ROOT/processed_datasets/train_data.pkl"
export KRONOS_VAL_DATA_PATHS="$DATA_ROOT/processed_datasets/train_data.pkl"
export KRONOS_METADATA_PATH="$DATA_ROOT/asset_metadata.csv"
export KRONOS_DATA_MANIFEST_SHA256="$DATA_SHA"
export KRONOS_FIXED_VALIDATION_MANIFEST_PATH="$VALIDATION_ROOT/natural_validation_manifest.json"
export KRONOS_FIXED_VALIDATION_MANIFEST_SHA256="$NATURAL_SHA"
export KRONOS_EXCLUDE_FIXED_VALIDATION_FROM_TRAINING=1
export KRONOS_VALIDATION_QUICK_SAMPLES=6000
export KRONOS_VALIDATION_LARGE_SAMPLES=24000
export KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS=10
export KRONOS_PREDICTOR_PATH="$PARENT_ROOT"
export KRONOS_TOKENIZER_PATH="$TOKENIZER_ROOT"
export KRONOS_SAVE_PATH="$RUN_ROOT/outputs/models"
export KRONOS_PREDICTOR_SAVE_FOLDER="$OUTPUT_NAME"
export KRONOS_LOOKBACK_WINDOW=120
export KRONOS_PREDICT_WINDOW=10
export KRONOS_USE_SECTOR_FEATURES=1
export KRONOS_NUM_SECTORS=86
export KRONOS_USE_SIZE_FEATURES=0
export KRONOS_NUM_SIZE_BUCKETS=0
export KRONOS_USE_SIZE_PERCENTILE=1
export KRONOS_CONTEXT_LAYER=10
export KRONOS_RESET_SECTOR_EMBEDDING=0
export KRONOS_RESET_SIZE_EMBEDDING=0
export KRONOS_TRAIN_SIGNAL_START=2015-01-01
export KRONOS_TRAIN_SIGNAL_END=2026-07-17
export KRONOS_VAL_SIGNAL_START=2025-07-01
export KRONOS_VAL_SIGNAL_END=2026-06-30
export KRONOS_TRAIN_SAMPLES_PER_SEGMENT=20000
export KRONOS_VALIDATION_SAMPLES=24000
export KRONOS_EPOCHS=1058
export KRONOS_COVERAGE_PASSES=2
export KRONOS_REQUIRE_FULL_COVERAGE=1
export KRONOS_EARLY_STOPPING_PATIENCE=0
export KRONOS_PREDICTOR_LEARNING_RATE=5e-7
export KRONOS_CONDITION_LEARNING_RATE=1e-6
export KRONOS_SCHEDULER_TYPE=fixed
export KRONOS_SCHEDULER=fixed
export KRONOS_PREDICTOR_WARMUP_START_LR=5e-7
export KRONOS_CONDITION_WARMUP_START_LR=1e-6
export KRONOS_PREDICTOR_MIN_LR=5e-7
export KRONOS_CONDITION_MIN_LR=1e-6
export KRONOS_PREDICTOR_LOSS_MODE=forecast
export KRONOS_HISTORY_LOSS_WEIGHT=0.02
export KRONOS_TRAINABLE_TRANSFORMER_LAYERS=2
export KRONOS_ADAM_WEIGHT_DECAY=0.1
export KRONOS_GRAD_CLIP_NORM=1.0
export KRONOS_CONDITION_MONITOR_INTERVAL_STEPS=100
export KRONOS_CONDITION_ABLATION_INTERVAL_SEGMENTS=10
export KRONOS_BATCH_SIZE=32
export KRONOS_NUM_WORKERS=2
export KRONOS_USE_AMP=1
export KRONOS_AMP_DTYPE=bfloat16
export KRONOS_BOOTSTRAP_COMPLETED_SEGMENTS=0
export KRONOS_MAX_SEGMENTS_PER_RUN="$SEGMENTS_THIS_RUN"
export KRONOS_RESUME_TRAINING="$RESUME"
export PYTHONUNBUFFERED=1

if [[ "$MODE" == "baseline" ]]; then
  echo "[$(date -u +%FT%TZ)] mode=baseline parent=Beta-v1.1-Last@1058 gpu=0" | tee -a "$RUN_ROOT/baseline.log"
  cd "$CODE_ROOT/finetune"
  "${PYTHON[@]}" -u "$CODE_ROOT/finetune/evaluate_fixed_validation_baseline.py" \
    --output "$RUN_ROOT/baseline_validation.json" 2>&1 | tee -a "$RUN_ROOT/baseline.log"
  exit 0
fi

echo "[$(date -u +%FT%TZ)] mode=$MODE segments_this_run=$SEGMENTS_THIS_RUN parent=Beta-v1.1-Last@1058 gpu=0 predictor_lr=5e-7 condition_lr=1e-6 scheduler=fixed" | tee -a "$RUN_ROOT/launcher.log"
cd "$CODE_ROOT/finetune"
"${PYTHON[@]}" -u "$CODE_ROOT/finetune/train_predictor.py" 2>&1 | tee -a "$RUN_ROOT/launcher.log"

if [[ "$MODE" == "full" ]]; then
  "${PYTHON[@]}" "$CODE_ROOT/finetune/export_last_model.py" \
    --repo-root "$CODE_ROOT" \
    --output-root "$OUTPUT_ROOT" 2>&1 | tee -a "$RUN_ROOT/launcher.log"
fi
