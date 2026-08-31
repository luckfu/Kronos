#!/usr/bin/env bash
# Evaluate Beta v1.3.1 Best@895 before any Beta v1.3.2 update.
set -euo pipefail

BASE="/nfsdata/models/2026/kronos-v1-beta"
CODE_ROOT="$BASE/code/Kronos"
DATA_ROOT="$BASE/data/a_share_full_market_v1_beta_symbol_holdout_90_10_v1"
VALIDATION_ROOT="$DATA_ROOT/natural_validation_2025h2_2026h1_symbol_holdout_full_v1"
METADATA_ROOT="$BASE/data/a_share_full_market_v1_beta"
PARENT_RUN="${KRONOS_LAUNCH_PARENT_RUN:-$BASE/runs/beta_v1_3_best343_aggressive_1e5_twopass_seed100}"
PARENT_ROOT="${KRONOS_LAUNCH_PARENT_ROOT:-$PARENT_RUN/outputs/models/a_share_beta_v1_3_best343_aggressive_1e5_twopass_120d_to_10d/checkpoints/best_model}"
TOKENIZER_ROOT="$BASE/models/tokenizer"
RUN_ROOT="${KRONOS_LAUNCH_RUN_ROOT:-$BASE/runs/beta_v1_3_2_symbol_holdout_90_10_aggressive_1e5_twopass_seed100}"
OUTPUT_NAME="${KRONOS_LAUNCH_BASELINE_OUTPUT_NAME:-baseline_beta_v1_3_1_best895_symbol_holdout_full.json}"
OUTPUT="$RUN_ROOT/$OUTPUT_NAME"
PYTHON=(sudo -n -E "$BASE/env/a800-py312/bin/python")

mkdir -p "$RUN_ROOT"
[[ ! -e "$OUTPUT" ]] || {
  echo "baseline refuses existing output: $OUTPUT" >&2
  exit 1
}

export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$CODE_ROOT"
export KRONOS_TRAIN_DATA_PATHS="$DATA_ROOT/processed_datasets/train_data.pkl"
export KRONOS_VAL_DATA_PATHS="$DATA_ROOT/processed_datasets/val_data.pkl"
export KRONOS_METADATA_PATH="$METADATA_ROOT/asset_metadata.csv"
export KRONOS_DATA_MANIFEST_SHA256="17afbeede658c13787043e601aa355717dda4d11719b51fb3ce368fb138e627a"
export KRONOS_FIXED_VALIDATION_MANIFEST_PATH="$VALIDATION_ROOT/natural_validation_manifest.json"
export KRONOS_FIXED_VALIDATION_MANIFEST_SHA256="ea29ecdb318adf9789ddd47eb4c5d3df7cdadcbbd60471305241a469d357d184"
export KRONOS_EXCLUDE_FIXED_VALIDATION_FROM_TRAINING=1
export KRONOS_VALIDATION_FULL_ONLY=1
export KRONOS_VALIDATION_QUICK_SAMPLES=24000 KRONOS_VALIDATION_LARGE_SAMPLES=123982
export KRONOS_VALIDATION_SAMPLES=123982
export KRONOS_VAL_SIGNAL_START=2025-07-01 KRONOS_VAL_SIGNAL_END=2026-06-30
export KRONOS_PREDICTOR_PATH="$PARENT_ROOT" KRONOS_TOKENIZER_PATH="$TOKENIZER_ROOT"
export KRONOS_LOOKBACK_WINDOW=120 KRONOS_PREDICT_WINDOW=10
export KRONOS_USE_SECTOR_FEATURES=1 KRONOS_NUM_SECTORS=86
export KRONOS_USE_SIZE_FEATURES=0 KRONOS_NUM_SIZE_BUCKETS=0 KRONOS_USE_SIZE_PERCENTILE=1
export KRONOS_CONTEXT_LAYER=10 KRONOS_RESET_SECTOR_EMBEDDING=0 KRONOS_RESET_SIZE_EMBEDDING=0
export KRONOS_PREDICTOR_LOSS_MODE=forecast KRONOS_HISTORY_LOSS_WEIGHT=0.02
export KRONOS_BATCH_SIZE=64 KRONOS_NUM_WORKERS=2 KRONOS_USE_AMP=1 KRONOS_AMP_DTYPE=bfloat16
export PYTHONUNBUFFERED=1

cd "$CODE_ROOT/finetune"
"${PYTHON[@]}" "$CODE_ROOT/finetune/evaluate_fixed_validation_full.py" --output "$OUTPUT"
