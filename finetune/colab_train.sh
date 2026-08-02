#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${KRONOS_COLAB_ROOT:-/content/kronos_runtime}"
DATA_ROOT="${KRONOS_COLAB_DATA_ROOT:-${RUNTIME_ROOT}/data/a_share}"
MODEL_ROOT="${KRONOS_COLAB_MODEL_ROOT:-${RUNTIME_ROOT}/models}"
BASE_KIND="${KRONOS_COLAB_BASE_MODEL:-production}"
OUTPUT_NAME="${KRONOS_PREDICTOR_SAVE_FOLDER:-a_share_size_full_coverage_colab_v1}"
RESUME_PATH="${RUNTIME_ROOT}/outputs/models/${OUTPUT_NAME}/checkpoints/last_state.pt"

if [[ "${BASE_KIND}" == 'original' ]]; then
  BASE_MODEL="${MODEL_ROOT}/Kronos-base"
else
  BASE_MODEL="${MODEL_ROOT}/a_share_size_kronos_base_earlystop50"
fi

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/finetune"
export KMP_DUPLICATE_LIB_OK="TRUE"
export KRONOS_DATASET_PATH="${DATA_ROOT}/processed_datasets"
export KRONOS_METADATA_PATH="${DATA_ROOT}/asset_metadata.csv"
export KRONOS_PREDICTOR_PATH="${BASE_MODEL}"
export KRONOS_SAVE_PATH="${RUNTIME_ROOT}/outputs/models"
export KRONOS_PREDICTOR_SAVE_FOLDER="${OUTPUT_NAME}"
export KRONOS_USE_SIZE_PERCENTILE="${KRONOS_USE_SIZE_PERCENTILE:-0}"
export KRONOS_TRAIN_SAMPLES_PER_SEGMENT="${KRONOS_TRAIN_SAMPLES_PER_SEGMENT:-20000}"
export KRONOS_VALIDATION_SAMPLES="${KRONOS_VALIDATION_SAMPLES:-2000}"
export KRONOS_COVERAGE_PASSES="${KRONOS_COVERAGE_PASSES:-1}"
export KRONOS_REQUIRE_FULL_COVERAGE="${KRONOS_REQUIRE_FULL_COVERAGE:-1}"
export KRONOS_EARLY_STOPPING_PATIENCE="${KRONOS_EARLY_STOPPING_PATIENCE:-5}"
export KRONOS_BATCH_SIZE="${KRONOS_BATCH_SIZE:-32}"
export KRONOS_NUM_WORKERS="${KRONOS_NUM_WORKERS:-2}"
if [[ -z "${KRONOS_RESUME_TRAINING+x}" ]]; then
  if [[ -f "${RESUME_PATH}" ]]; then
    export KRONOS_RESUME_TRAINING=1
  else
    export KRONOS_RESUME_TRAINING=0
  fi
else
  export KRONOS_RESUME_TRAINING
fi

echo "[colab] Output: ${KRONOS_SAVE_PATH}/${OUTPUT_NAME}"
echo "[colab] Resume: ${KRONOS_RESUME_TRAINING}"

cd "${REPO_ROOT}"
python finetune/verify_colab_setup.py \
  --data-dir "${KRONOS_DATASET_PATH}" \
  --base-model "${KRONOS_PREDICTOR_PATH}"
exec python -u finetune/train_predictor.py
