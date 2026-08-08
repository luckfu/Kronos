#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${KRONOS_COLAB_V5_ROOT:-/content/drive/MyDrive/kronos_a_share_v5}"
DATA_ROOT="${KRONOS_COLAB_V5_DATA_ROOT:-${RUNTIME_ROOT}/data/a_share_v5}"
MODEL_ROOT="${KRONOS_COLAB_V5_MODEL_ROOT:-${RUNTIME_ROOT}/models}"
BASE_MODEL="${KRONOS_PREDICTOR_PATH:-${MODEL_ROOT}/a_share_v4_corrected_2026_replay20_latest}"
OUTPUT_NAME="${KRONOS_PREDICTOR_SAVE_FOLDER:-a_share_v5_context120_2pass}"
OUTPUT_ROOT="${RUNTIME_ROOT}/outputs/models/${OUTPUT_NAME}"
RESUME_PATH="${OUTPUT_ROOT}/checkpoints/last_state.pt"
CONFIG_LOCK="${OUTPUT_ROOT}/v5_run_config.env"

LOOKBACK="${KRONOS_LOOKBACK_WINDOW:-120}"
PREDICT="${KRONOS_PREDICT_WINDOW:-10}"
TRAIN_START="${KRONOS_TRAIN_SIGNAL_START:-2015-01-01}"
# Ten trading labels after 2025-12-17 end on 2025-12-31.  This keeps 2026
# validation labels entirely outside the training panel.
TRAIN_END="${KRONOS_TRAIN_SIGNAL_END:-2025-12-17}"
VAL_START="${KRONOS_VAL_SIGNAL_START:-2026-01-05}"
VAL_END="${KRONOS_VAL_SIGNAL_END:-2026-07-16}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/finetune"
export KMP_DUPLICATE_LIB_OK="TRUE"
export KRONOS_DATASET_PATH="${DATA_ROOT}/processed_datasets"
export KRONOS_TRAIN_DATA_PATHS="${DATA_ROOT}/processed_datasets/train_data.pkl"
export KRONOS_VAL_DATA_PATHS="${DATA_ROOT}/processed_datasets/val_data.pkl"
export KRONOS_METADATA_PATH="${DATA_ROOT}/asset_metadata.csv"
export KRONOS_PREDICTOR_PATH="${BASE_MODEL}"
export KRONOS_SAVE_PATH="${RUNTIME_ROOT}/outputs/models"
export KRONOS_PREDICTOR_SAVE_FOLDER="${OUTPUT_NAME}"
export KRONOS_LOOKBACK_WINDOW="${LOOKBACK}"
export KRONOS_PREDICT_WINDOW="${PREDICT}"
export KRONOS_TRAIN_SIGNAL_START="${TRAIN_START}"
export KRONOS_TRAIN_SIGNAL_END="${TRAIN_END}"
export KRONOS_VAL_SIGNAL_START="${VAL_START}"
export KRONOS_VAL_SIGNAL_END="${VAL_END}"
export KRONOS_RESET_SIZE_EMBEDDING="0"
export KRONOS_BALANCE_SIZE_BUCKETS="1"
export KRONOS_USE_SIZE_PERCENTILE="0"
export KRONOS_TRAIN_SAMPLES_PER_SEGMENT="20000"
export KRONOS_VALIDATION_SAMPLES="4000"
export KRONOS_COVERAGE_PASSES="${KRONOS_COVERAGE_PASSES:-2}"
export KRONOS_EPOCHS="${KRONOS_EPOCHS:-${KRONOS_COVERAGE_PASSES}}"
export KRONOS_REQUIRE_FULL_COVERAGE="1"
export KRONOS_EARLY_STOPPING_PATIENCE="0"
export KRONOS_BATCH_SIZE="${KRONOS_BATCH_SIZE:-16}"
export KRONOS_NUM_WORKERS="${KRONOS_NUM_WORKERS:-2}"
export KRONOS_DRIVE_CONFLICT_CLEANUP="${KRONOS_DRIVE_CONFLICT_CLEANUP:-api}"

if [[ -z "${KRONOS_RESUME_TRAINING+x}" ]]; then
  if [[ -f "${RESUME_PATH}" ]]; then
    export KRONOS_RESUME_TRAINING=1
  else
    export KRONOS_RESUME_TRAINING=0
  fi
fi

mkdir -p "${OUTPUT_ROOT}"
CURRENT_CONFIG="$(mktemp)"
trap 'rm -f "${CURRENT_CONFIG}"' EXIT
{
  echo "lookback=${KRONOS_LOOKBACK_WINDOW}"
  echo "predict=${KRONOS_PREDICT_WINDOW}"
  echo "train_signal_start=${KRONOS_TRAIN_SIGNAL_START}"
  echo "train_signal_end=${KRONOS_TRAIN_SIGNAL_END}"
  echo "val_signal_start=${KRONOS_VAL_SIGNAL_START}"
  echo "val_signal_end=${KRONOS_VAL_SIGNAL_END}"
  echo "batch_size=${KRONOS_BATCH_SIZE}"
  echo "coverage_passes=${KRONOS_COVERAGE_PASSES}"
  echo "train_samples_per_segment=${KRONOS_TRAIN_SAMPLES_PER_SEGMENT}"
  echo "validation_samples=${KRONOS_VALIDATION_SAMPLES}"
  echo "balance_size_buckets=${KRONOS_BALANCE_SIZE_BUCKETS}"
} > "${CURRENT_CONFIG}"

if [[ -f "${CONFIG_LOCK}" ]]; then
  if ! cmp -s "${CURRENT_CONFIG}" "${CONFIG_LOCK}"; then
    echo "[colab-v5] ERROR: resume configuration differs from ${CONFIG_LOCK}" >&2
    diff -u "${CONFIG_LOCK}" "${CURRENT_CONFIG}" >&2 || true
    exit 1
  fi
else
  cp "${CURRENT_CONFIG}" "${CONFIG_LOCK}"
fi

if [[ "${KRONOS_RESUME_TRAINING}" == "1" && ! -f "${RESUME_PATH}" ]]; then
  echo "[colab-v5] ERROR: resume requested but ${RESUME_PATH} does not exist" >&2
  exit 1
fi
if [[ "${KRONOS_RESUME_TRAINING}" == "0" && -f "${RESUME_PATH}" ]]; then
  echo "[colab-v5] ERROR: output already contains a checkpoint; resume or choose a new output" >&2
  exit 1
fi

cd "${REPO_ROOT}"
python finetune/verify_a_share_context.py \
  --data-root "${DATA_ROOT}" \
  --base-model "${BASE_MODEL}" \
  --lookback "${LOOKBACK}" \
  --predict "${PREDICT}"

if [[ "${KRONOS_DRIVE_CONFLICT_CLEANUP}" == "api" ]]; then
  python finetune/drive_cleanup.py
fi

echo "[colab-v5] Data: ${DATA_ROOT}"
echo "[colab-v5] Base: ${BASE_MODEL}"
echo "[colab-v5] Context: ${LOOKBACK} trading days; prediction: ${PREDICT} trading days"
echo "[colab-v5] Train signals: ${TRAIN_START}..${TRAIN_END}"
echo "[colab-v5] Validation signals: ${VAL_START}..${VAL_END}"
echo "[colab-v5] Output: ${OUTPUT_ROOT}"
echo "[colab-v5] Resume: ${KRONOS_RESUME_TRAINING}"
exec python -u finetune/train_predictor.py
