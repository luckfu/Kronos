#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${KRONOS_KAGGLE_ROOT:-/kaggle/working/kronos_a_share_2026_incremental}"
DATA_ROOT="${KRONOS_KAGGLE_DATA_ROOT:-${RUNTIME_ROOT}/data/a_share_v3_2026_incremental}"
BASE_MODEL="${KRONOS_PREDICTOR_PATH:-${RUNTIME_ROOT}/base_model/v3_last}"
OUTPUT_NAME="${KRONOS_PREDICTOR_SAVE_FOLDER:-a_share_v3_last_2026_incremental_2pass}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/finetune"
export KMP_DUPLICATE_LIB_OK=TRUE
export KRONOS_DATASET_PATH="${DATA_ROOT}/processed_datasets"
export KRONOS_METADATA_PATH="${DATA_ROOT}/asset_metadata.csv"
export KRONOS_PREDICTOR_PATH="${BASE_MODEL}"
export KRONOS_SAVE_PATH="${RUNTIME_ROOT}/outputs/models"
export KRONOS_PREDICTOR_SAVE_FOLDER="${OUTPUT_NAME}"
export KRONOS_RESUME_TRAINING=0
export KRONOS_RESET_SIZE_EMBEDDING=0
export KRONOS_BALANCE_SIZE_BUCKETS=1
export KRONOS_USE_SIZE_PERCENTILE=0
export KRONOS_TRAIN_SAMPLES_PER_SEGMENT=20000
export KRONOS_VALIDATION_SAMPLES=4000
export KRONOS_BATCH_SIZE=32
export KRONOS_NUM_WORKERS=2
export KRONOS_EPOCHS=1
export KRONOS_COVERAGE_PASSES=2
export KRONOS_REQUIRE_FULL_COVERAGE=1
export KRONOS_EARLY_STOPPING_PATIENCE=0
export KRONOS_MAX_SEGMENTS_PER_RUN=0
export KRONOS_PREDICTOR_LEARNING_RATE=2e-6
export KRONOS_CONDITION_LEARNING_RATE=2e-4
export KRONOS_DRIVE_CONFLICT_CLEANUP=off

python finetune/verify_a_share_2026_incremental.py \
  --data-root "${DATA_ROOT}" \
  --base-model "${BASE_MODEL}"

echo "[incremental-2026] Base: ${BASE_MODEL}"
echo "[incremental-2026] Output: ${RUNTIME_ROOT}/outputs/models/${OUTPUT_NAME}"
echo "[incremental-2026] Fresh optimizer/scheduler; size embedding preserved"
echo "[incremental-2026] Learning rates: predictor=2e-6, condition=2e-4"
exec python -u finetune/train_predictor.py
