#!/usr/bin/env bash
# Fresh Beta v3 training from NeoQuasar/Kronos-base on one A800.
set -euo pipefail

MODE="${1:-fresh}"
if [[ "$MODE" != "fresh" && "$MODE" != "resume" ]]; then
  echo "usage: $0 {fresh|resume}" >&2
  exit 2
fi

BASE="/nfsdata/models/2026/kronos-beta-v3"
LEGACY_BASE="/nfsdata/models/2026/kronos-v1-beta"
CODE_ROOT="$BASE/code/Kronos"
DATA_ROOT="$LEGACY_BASE/data/a_share_full_market_v1_beta_symbol_holdout_90_10_v1"
VALIDATION_ROOT="$DATA_ROOT/natural_validation_2025h2_2026h1_symbol_holdout_24k_v1"
PARENT_ROOT="$BASE/models/Kronos-base"
TOKENIZER_ROOT="$LEGACY_BASE/models/tokenizer"
RUN_ROOT="$BASE/runs/beta_v3_base_dynamic_size_path_full24k_seed100"
OUTPUT_NAME="beta_v3_base_dynamic_size_path_full24k"
OUTPUT_ROOT="$RUN_ROOT/outputs/models/$OUTPUT_NAME"
PYTHON=(sudo -n -E "$LEGACY_BASE/env/a800-py312/bin/python")

DATA_MANIFEST_ID="17afbeede658c13787043e601aa355717dda4d11719b51fb3ce368fb138e627a"
DATA_MANIFEST_FILE_SHA="3989bddae6c76e34eb7772590e11b4605215270c178ae30eb5c2261e06fe9177"
TRAIN_SHA="b2f2a861f651321efd38761c65ffcff4d14290cd580325298c4b9a7bc915f832"
VAL_SHA="748a9205714ee9714525872e65432063edd26d6afbef05391919e8d9d8811115"
METADATA_SHA="697cadd672d53b8fa0a990f6c5b7fba2f88cb26aad88c3b91cfcc3865d0a1e3c"
VALIDATION_SHA="038bf7c10867d67ec175007f22ad7e51eaf67c4ca309663c155fbdcff42d0988"
PARENT_SHA="abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83"

required=(
  "$CODE_ROOT/finetune/config.py"
  "$CODE_ROOT/finetune/dataset.py"
  "$CODE_ROOT/finetune/train_predictor.py"
  "$CODE_ROOT/finetune/export_last_model.py"
  "$CODE_ROOT/model/module.py"
  "$DATA_ROOT/data_manifest.json"
  "$DATA_ROOT/processed_datasets/train_data.pkl"
  "$DATA_ROOT/processed_datasets/val_data.pkl"
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

verify_sha() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "checksum mismatch: $path ($actual != $expected)" >&2
    exit 1
  }
}

verify_sha "$DATA_MANIFEST_FILE_SHA" "$DATA_ROOT/data_manifest.json"
verify_sha "$TRAIN_SHA" "$DATA_ROOT/processed_datasets/train_data.pkl"
verify_sha "$VAL_SHA" "$DATA_ROOT/processed_datasets/val_data.pkl"
verify_sha "$METADATA_SHA" "$DATA_ROOT/asset_metadata.csv"
verify_sha "$VALIDATION_SHA" "$VALIDATION_ROOT/natural_validation_manifest.json"
verify_sha "$PARENT_SHA" "$PARENT_ROOT/model.safetensors"

if [[ "$MODE" == "fresh" ]]; then
  [[ ! -e "$OUTPUT_ROOT" ]] || {
    echo "fresh refuses existing output: $OUTPUT_ROOT" >&2
    exit 1
  }
  RESUME=0
  RESET_CONDITIONS=1
else
  [[ -f "$OUTPUT_ROOT/checkpoints/last_state.pt" ]] || {
    echo "resume requires $OUTPUT_ROOT/checkpoints/last_state.pt" >&2
    exit 1
  }
  RESUME=1
  RESET_CONDITIONS=0
fi

mkdir -p "$RUN_ROOT"
if [[ ! -f "$RUN_ROOT/experiment_manifest.json" ]]; then
  sed \
    -e "s|__DATA_MANIFEST_ID__|$DATA_MANIFEST_ID|g" \
    -e "s|__VALIDATION_SHA__|$VALIDATION_SHA|g" \
    -e "s|__PARENT_SHA__|$PARENT_SHA|g" \
    "$CODE_ROOT/finetune/manifests/beta_v3_a800_full24k_manifest.template.json" \
    > "$RUN_ROOT/experiment_manifest.json"
fi

export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$CODE_ROOT"
export KRONOS_TRAIN_DATA_PATHS="$DATA_ROOT/processed_datasets/train_data.pkl"
export KRONOS_VAL_DATA_PATHS="$DATA_ROOT/processed_datasets/val_data.pkl"
export KRONOS_METADATA_PATH="$DATA_ROOT/asset_metadata.csv"
export KRONOS_DATA_MANIFEST_SHA256="$DATA_MANIFEST_ID"
export KRONOS_FIXED_VALIDATION_MANIFEST_PATH="$VALIDATION_ROOT/natural_validation_manifest.json"
export KRONOS_FIXED_VALIDATION_MANIFEST_SHA256="$VALIDATION_SHA"
export KRONOS_EXCLUDE_FIXED_VALIDATION_FROM_TRAINING=1
export KRONOS_VALIDATION_FULL_ONLY=1
export KRONOS_VALIDATION_QUICK_SAMPLES=24000
export KRONOS_VALIDATION_LARGE_SAMPLES=24000
export KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS=1
export KRONOS_PREDICTOR_PATH="$PARENT_ROOT" KRONOS_TOKENIZER_PATH="$TOKENIZER_ROOT"
export KRONOS_SAVE_PATH="$RUN_ROOT/outputs/models" KRONOS_PREDICTOR_SAVE_FOLDER="$OUTPUT_NAME"
export KRONOS_LOOKBACK_WINDOW=120 KRONOS_PREDICT_WINDOW=10
export KRONOS_USE_SECTOR_FEATURES=1 KRONOS_NUM_SECTORS=86
export KRONOS_USE_SIZE_FEATURES=0 KRONOS_NUM_SIZE_BUCKETS=0 KRONOS_USE_SIZE_PERCENTILE=0
export KRONOS_USE_SIZE_PATH=1 KRONOS_SIZE_PATH_INPUT_DIM=4
export KRONOS_CONTEXT_LAYER=10
export KRONOS_RESET_SECTOR_EMBEDDING="$RESET_CONDITIONS"
export KRONOS_RESET_SIZE_EMBEDDING="$RESET_CONDITIONS"
export KRONOS_TRAIN_SIGNAL_START=2015-01-01 KRONOS_TRAIN_SIGNAL_END=2026-07-17
export KRONOS_VAL_SIGNAL_START=2025-07-01 KRONOS_VAL_SIGNAL_END=2026-06-30
export KRONOS_TRAIN_SAMPLES_PER_SEGMENT=20000 KRONOS_VALIDATION_SAMPLES=24000
export KRONOS_EPOCHS=1 KRONOS_COVERAGE_PASSES=1 KRONOS_REQUIRE_FULL_COVERAGE=1
export KRONOS_EARLY_STOPPING_PATIENCE=0 KRONOS_MAX_SEGMENTS_PER_RUN=473
export KRONOS_PREDICTOR_LEARNING_RATE=1e-6 KRONOS_CONDITION_LEARNING_RATE=1e-5
export KRONOS_SCHEDULER=warmup_cosine KRONOS_SCHEDULER_WARMUP_RATIO=0.01
export KRONOS_PREDICTOR_WARMUP_START_LR=1e-7 KRONOS_CONDITION_WARMUP_START_LR=1e-6
export KRONOS_PREDICTOR_MIN_LR=1e-7 KRONOS_CONDITION_MIN_LR=1e-6
export KRONOS_TRAINABLE_TRANSFORMER_LAYERS=-1
export KRONOS_PREDICTOR_LOSS_MODE=forecast KRONOS_HISTORY_LOSS_WEIGHT=0.02
export KRONOS_BEST_SELECTION_METRIC=validation_large_objective
export KRONOS_ADAM_WEIGHT_DECAY=0.1 KRONOS_GRAD_CLIP_NORM=1.0
export KRONOS_CONDITION_MONITOR_INTERVAL_STEPS=100 KRONOS_CONDITION_ABLATION_INTERVAL_SEGMENTS=0
export KRONOS_BATCH_SIZE=32 KRONOS_NUM_WORKERS=2 KRONOS_USE_AMP=1 KRONOS_AMP_DTYPE=float16
export KRONOS_BOOTSTRAP_COMPLETED_SEGMENTS=0 KRONOS_RESUME_TRAINING="$RESUME"
export PYTHONUNBUFFERED=1

echo "[$(date -u +%FT%TZ)] mode=$MODE parent=NeoQuasar/Kronos-base segments=473 validation=full-only-24k" \
  | tee -a "$RUN_ROOT/launcher.log"
cd "$CODE_ROOT/finetune"
"${PYTHON[@]}" -u "$CODE_ROOT/finetune/train_predictor.py" 2>&1 \
  | tee -a "$RUN_ROOT/launcher.log"
"${PYTHON[@]}" "$CODE_ROOT/finetune/export_last_model.py" \
  --repo-root "$CODE_ROOT" --output-root "$OUTPUT_ROOT" 2>&1 \
  | tee -a "$RUN_ROOT/launcher.log"
