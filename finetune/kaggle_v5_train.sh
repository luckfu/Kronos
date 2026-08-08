#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${KRONOS_KAGGLE_ROOT:-/kaggle/working/kronos_a_share_v5_120d}"
[[ -f "${RUNTIME_ROOT}/kaggle_v5_paths.env" ]] && source "${RUNTIME_ROOT}/kaggle_v5_paths.env"
DATA_ROOT="${KRONOS_KAGGLE_DATA_ROOT:?Run kaggle_v5_bootstrap.sh first}"
BASE_MODEL="${KRONOS_PREDICTOR_PATH:?V4 B last_model is required}"
OUTPUT_NAME="${KRONOS_PREDICTOR_SAVE_FOLDER:-a_share_v5_context120_2pass}"
OUTPUT_ROOT="${RUNTIME_ROOT}/outputs/models/${OUTPUT_NAME}"
RESUME_PATH="${OUTPUT_ROOT}/checkpoints/last_state.pt"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/finetune"
export KMP_DUPLICATE_LIB_OK=TRUE
export KRONOS_DATASET_PATH="${DATA_ROOT}/processed_datasets"
export KRONOS_TRAIN_DATA_PATHS="${DATA_ROOT}/processed_datasets/train_data.pkl"
# V5 validation is sampled from the complete training pool, not a special
# market regime. The post-training evaluator owns the true time holdout.
export KRONOS_VAL_DATA_PATHS="${DATA_ROOT}/processed_datasets/train_data.pkl"
export KRONOS_METADATA_PATH="${DATA_ROOT}/asset_metadata.csv"
export KRONOS_PREDICTOR_PATH="${BASE_MODEL}"
export KRONOS_SAVE_PATH="${RUNTIME_ROOT}/outputs/models"
export KRONOS_PREDICTOR_SAVE_FOLDER="${OUTPUT_NAME}"
export KRONOS_LOOKBACK_WINDOW=120 KRONOS_PREDICT_WINDOW=10
export KRONOS_TRAIN_SIGNAL_START="${KRONOS_TRAIN_SIGNAL_START:-2015-01-01}"
export KRONOS_TRAIN_SIGNAL_END="${KRONOS_TRAIN_SIGNAL_END:-2025-12-17}"
export KRONOS_VAL_SIGNAL_START="${KRONOS_VAL_SIGNAL_START:-2026-01-05}"
export KRONOS_VAL_SIGNAL_END="${KRONOS_VAL_SIGNAL_END:-2026-07-16}"
export KRONOS_TRAIN_SAMPLES_PER_SEGMENT="${KRONOS_TRAIN_SAMPLES_PER_SEGMENT:-20000}"
export KRONOS_VALIDATION_SAMPLES="${KRONOS_VALIDATION_SAMPLES:-4000}"
export KRONOS_BATCH_SIZE="${KRONOS_BATCH_SIZE:-16}" KRONOS_NUM_WORKERS="${KRONOS_NUM_WORKERS:-2}"
export KRONOS_COVERAGE_PASSES="${KRONOS_COVERAGE_PASSES:-2}" KRONOS_EPOCHS="${KRONOS_EPOCHS:-${KRONOS_COVERAGE_PASSES}}"
export KRONOS_REQUIRE_FULL_COVERAGE=1 KRONOS_EARLY_STOPPING_PATIENCE=0
export KRONOS_RESET_SIZE_EMBEDDING=0 KRONOS_BALANCE_SIZE_BUCKETS=1 KRONOS_USE_SIZE_PERCENTILE=0
export KRONOS_DRIVE_CONFLICT_CLEANUP=off

mkdir -p "${OUTPUT_ROOT}"
if [[ -z "${KRONOS_RESUME_TRAINING+x}" ]]; then
  [[ -f "${RESUME_PATH}" ]] && export KRONOS_RESUME_TRAINING=1 || export KRONOS_RESUME_TRAINING=0
fi
if [[ "${KRONOS_RESUME_TRAINING}" == 0 && -f "${RESUME_PATH}" ]]; then
  echo "[kaggle-v5] Output already has a checkpoint; set KRONOS_RESUME_TRAINING=1 to continue." >&2
  exit 1
fi

python finetune/verify_a_share_context.py --data-root "${DATA_ROOT}" --base-model "${BASE_MODEL}" --lookback 120 --predict 10
echo "[kaggle-v5] Output: ${OUTPUT_ROOT}"
echo "[kaggle-v5] Resume: ${KRONOS_RESUME_TRAINING}"
python -u finetune/train_predictor.py
python finetune/export_last_model.py --output-root "${OUTPUT_ROOT}"
echo "[kaggle-v5] Finished. best_model, last_model and last_state.pt are under ${OUTPUT_ROOT}/checkpoints"
