#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${KRONOS_KAGGLE_ROOT:-/kaggle/working/kronos_a_share_v6_forecast}"
[[ -f "${RUNTIME_ROOT}/kaggle_v6_paths.env" ]] && source "${RUNTIME_ROOT}/kaggle_v6_paths.env"
DATA_ROOT="${KRONOS_KAGGLE_DATA_ROOT:?Run kaggle_v6_bootstrap.sh first}"
BASE_MODEL="${KRONOS_PREDICTOR_PATH:?V5 Last is required}"
OUTPUT_NAME="${KRONOS_PREDICTOR_SAVE_FOLDER:-a_share_v6_forecast_only_context120_2pass}"
OUTPUT_ROOT="${RUNTIME_ROOT}/outputs/models/${OUTPUT_NAME}"
RESUME_PATH="${OUTPUT_ROOT}/checkpoints/last_state.pt"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/finetune"
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export KRONOS_DATASET_PATH="${DATA_ROOT}/processed_datasets"
export KRONOS_TRAIN_DATA_PATHS="${DATA_ROOT}/processed_datasets/train_data.pkl"
export KRONOS_VAL_DATA_PATHS="${DATA_ROOT}/processed_datasets/train_data.pkl"
export KRONOS_METADATA_PATH="${DATA_ROOT}/asset_metadata.csv"
export KRONOS_PREDICTOR_PATH="${BASE_MODEL}"
export KRONOS_SAVE_PATH="${RUNTIME_ROOT}/outputs/models"
export KRONOS_PREDICTOR_SAVE_FOLDER="${OUTPUT_NAME}"
export KRONOS_LOOKBACK_WINDOW=120 KRONOS_PREDICT_WINDOW=10
export KRONOS_PREDICTOR_LOSS_MODE=forecast KRONOS_HISTORY_LOSS_WEIGHT=0
export KRONOS_TRAIN_SIGNAL_START="${KRONOS_TRAIN_SIGNAL_START:-2015-01-01}"
export KRONOS_TRAIN_SIGNAL_END="${KRONOS_TRAIN_SIGNAL_END:-2026-07-16}"
export KRONOS_VAL_SIGNAL_START="${KRONOS_VAL_SIGNAL_START:-${KRONOS_TRAIN_SIGNAL_START}}"
export KRONOS_VAL_SIGNAL_END="${KRONOS_VAL_SIGNAL_END:-${KRONOS_TRAIN_SIGNAL_END}}"
export KRONOS_TRAIN_SAMPLES_PER_SEGMENT="${KRONOS_TRAIN_SAMPLES_PER_SEGMENT:-20000}"
export KRONOS_VALIDATION_SAMPLES="${KRONOS_VALIDATION_SAMPLES:-4000}"
export KRONOS_BATCH_SIZE="${KRONOS_BATCH_SIZE:-32}"
export KRONOS_NUM_WORKERS="${KRONOS_NUM_WORKERS:-2}"
export KRONOS_COVERAGE_PASSES="${KRONOS_COVERAGE_PASSES:-2}"
export KRONOS_EPOCHS="${KRONOS_EPOCHS:-${KRONOS_COVERAGE_PASSES}}"
export KRONOS_MAX_SEGMENTS_PER_RUN="${KRONOS_MAX_SEGMENTS_PER_RUN:-120}"
export KRONOS_REQUIRE_FULL_COVERAGE=1 KRONOS_EARLY_STOPPING_PATIENCE=0
export KRONOS_RESET_SIZE_EMBEDDING=0 KRONOS_BALANCE_SIZE_BUCKETS=1
export KRONOS_USE_SIZE_PERCENTILE=0 KRONOS_DRIVE_CONFLICT_CLEANUP=off

mkdir -p "${OUTPUT_ROOT}"
if [[ -z "${KRONOS_RESUME_TRAINING+x}" ]]; then
  [[ -f "${RESUME_PATH}" ]] && export KRONOS_RESUME_TRAINING=1 || export KRONOS_RESUME_TRAINING=0
fi

echo "[kaggle-v6] Base: ${BASE_MODEL}"
echo "[kaggle-v6] Output: ${OUTPUT_ROOT}"
echo "[kaggle-v6] Batch: ${KRONOS_BATCH_SIZE}; resume: ${KRONOS_RESUME_TRAINING}"
echo "[kaggle-v6] Objective: forecast-only; history weight: 0"
python -u finetune/train_predictor.py
python finetune/export_last_model.py --output-root "${OUTPUT_ROOT}"
echo "[kaggle-v6] Chunk finished. Checkpoints are under ${OUTPUT_ROOT}/checkpoints"
