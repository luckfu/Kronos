#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-${KRONOS_V4_VARIANT:-}}"
if [[ "${VARIANT}" != "recent_only" && "${VARIANT}" != "replay20" ]]; then
  echo "usage: $0 recent_only|replay20" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${KRONOS_KAGGLE_ROOT:-/kaggle/working/kronos_a_share_v4_ab}"
PATHS_FILE="${RUNTIME_ROOT}/kaggle_v3_paths.env"
if [[ -f "${PATHS_FILE}" ]]; then
  source "${PATHS_FILE}"
fi
DATA_ROOT="${KRONOS_KAGGLE_DATA_ROOT:-${RUNTIME_ROOT}/data/a_share_v3}"
V4_DATA_ROOT="${RUNTIME_ROOT}/data/a_share_v4"
BASE_MODEL="${KRONOS_PREDICTOR_PATH:?KRONOS_PREDICTOR_PATH is required}"

python "${REPO_ROOT}/finetune/verify_a_share_v4_data.py" \
  --data-root "${DATA_ROOT}" \
  --strict

if [[ ! -f "${V4_DATA_ROOT}/processed_datasets/temporal_val_data.pkl" ]]; then
  python "${REPO_ROOT}/finetune/prepare_a_share_v4_runtime.py" \
    --data-root "${DATA_ROOT}" \
    --output-root "${V4_DATA_ROOT}"
fi

if [[ "${VARIANT}" == "recent_only" ]]; then
  DEFAULT_OUTPUT_NAME="a_share_v4_corrected_2026_recent_only"
  TRAIN_PATHS="${V4_DATA_ROOT}/processed_datasets/recent_context_data.pkl"
  REPLAY_RATIO="0"
else
  DEFAULT_OUTPUT_NAME="a_share_v4_corrected_2026_replay20"
  TRAIN_PATHS="${DATA_ROOT}/processed_datasets/train_data.pkl:${DATA_ROOT}/processed_datasets/val_data.pkl"
  REPLAY_RATIO="0.20"
fi
OUTPUT_NAME="${KRONOS_V4_OUTPUT_NAME:-${DEFAULT_OUTPUT_NAME}}"
COVERAGE_PASSES="${KRONOS_V4_COVERAGE_PASSES:-1}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/finetune"
export KMP_DUPLICATE_LIB_OK="TRUE"
export KRONOS_DATASET_PATH="${DATA_ROOT}/processed_datasets"
export KRONOS_TRAIN_DATA_PATHS="${TRAIN_PATHS}"
export KRONOS_VAL_DATA_PATHS="${V4_DATA_ROOT}/processed_datasets/temporal_val_data.pkl"
export KRONOS_METADATA_PATH="${DATA_ROOT}/asset_metadata.csv"
export KRONOS_PREDICTOR_PATH="${BASE_MODEL}"
export KRONOS_SAVE_PATH="${RUNTIME_ROOT}/outputs/models"
export KRONOS_PREDICTOR_SAVE_FOLDER="${OUTPUT_NAME}"
export KRONOS_TRAIN_SIGNAL_START="2026-01-01"
export KRONOS_TRAIN_SIGNAL_END="2026-06-17"
export KRONOS_VAL_SIGNAL_START="2026-06-18"
export KRONOS_VAL_SIGNAL_END="2026-07-16"
export KRONOS_HISTORY_REPLAY_RATIO="${REPLAY_RATIO}"
export KRONOS_REPLAY_SIGNAL_START="2015-01-01"
export KRONOS_REPLAY_SIGNAL_END="2025-12-31"
export KRONOS_RESET_SIZE_EMBEDDING="0"
export KRONOS_BALANCE_SIZE_BUCKETS="1"
export KRONOS_USE_SIZE_PERCENTILE="0"
export KRONOS_TRAIN_SAMPLES_PER_SEGMENT="20000"
export KRONOS_VALIDATION_SAMPLES="4000"
export KRONOS_BATCH_SIZE="32"
export KRONOS_NUM_WORKERS="2"
export KRONOS_EPOCHS="${COVERAGE_PASSES}"
export KRONOS_COVERAGE_PASSES="${COVERAGE_PASSES}"
export KRONOS_REQUIRE_FULL_COVERAGE="1"
export KRONOS_EARLY_STOPPING_PATIENCE="0"
export KRONOS_MAX_SEGMENTS_PER_RUN="0"
export KRONOS_RESUME_TRAINING="0"
export KRONOS_PREDICTOR_LEARNING_RATE="1e-6"
export KRONOS_CONDITION_LEARNING_RATE="1e-4"
export KRONOS_DRIVE_CONFLICT_CLEANUP="off"

OUTPUT_ROOT="${RUNTIME_ROOT}/outputs/models/${OUTPUT_NAME}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "[v4-ab] ERROR: output already exists: ${OUTPUT_ROOT}" >&2
  exit 1
fi

echo "[v4-ab] Variant: ${VARIANT}"
echo "[v4-ab] Base: ${BASE_MODEL}"
echo "[v4-ab] Train paths: ${TRAIN_PATHS}"
echo "[v4-ab] Replay ratio: ${REPLAY_RATIO}"
echo "[v4-ab] Coverage passes: ${COVERAGE_PASSES}"
echo "[v4-ab] Output: ${OUTPUT_ROOT}"
exec python -u "${REPO_ROOT}/finetune/train_predictor.py"
