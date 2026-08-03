#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${KRONOS_COLAB_V3_ROOT:-/content/drive/MyDrive/kronos_a_share_v3}"
DATA_ROOT="${KRONOS_COLAB_V3_DATA_ROOT:-${RUNTIME_ROOT}/data/a_share_v3}"
MODEL_ROOT="${KRONOS_COLAB_V3_MODEL_ROOT:-${RUNTIME_ROOT}/models}"
BASE_MODEL="${KRONOS_PREDICTOR_PATH:-${MODEL_ROOT}/a_share_size_full_coverage_colab_bs32_latest}"
OUTPUT_NAME="${KRONOS_PREDICTOR_SAVE_FOLDER:-a_share_size_full_market_v3_colab_bs32}"
OUTPUT_ROOT="${RUNTIME_ROOT}/outputs/models/${OUTPUT_NAME}"
RESUME_PATH="${OUTPUT_ROOT}/checkpoints/last_state.pt"
CONFIG_LOCK="${OUTPUT_ROOT}/v3_run_config.env"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/finetune"
export KMP_DUPLICATE_LIB_OK="TRUE"
export KRONOS_DATASET_PATH="${DATA_ROOT}/processed_datasets"
export KRONOS_METADATA_PATH="${DATA_ROOT}/asset_metadata.csv"
export KRONOS_PREDICTOR_PATH="${BASE_MODEL}"
export KRONOS_SAVE_PATH="${RUNTIME_ROOT}/outputs/models"
export KRONOS_PREDICTOR_SAVE_FOLDER="${OUTPUT_NAME}"
export KRONOS_RESET_SIZE_EMBEDDING="1"
export KRONOS_BALANCE_SIZE_BUCKETS="1"
export KRONOS_USE_SIZE_PERCENTILE="0"
export KRONOS_TRAIN_SAMPLES_PER_SEGMENT="20000"
export KRONOS_VALIDATION_SAMPLES="4000"
export KRONOS_COVERAGE_PASSES="2"
export KRONOS_REQUIRE_FULL_COVERAGE="1"
export KRONOS_EARLY_STOPPING_PATIENCE="5"
export KRONOS_BATCH_SIZE="32"
export KRONOS_NUM_WORKERS="${KRONOS_NUM_WORKERS:-2}"

if [[ -z "${KRONOS_RESUME_TRAINING+x}" ]]; then
  [[ -f "${RESUME_PATH}" ]] && export KRONOS_RESUME_TRAINING=1 || export KRONOS_RESUME_TRAINING=0
fi

mkdir -p "${OUTPUT_ROOT}"
CURRENT_CONFIG="$(mktemp)"
trap 'rm -f "${CURRENT_CONFIG}"' EXIT
{
  echo "batch_size=${KRONOS_BATCH_SIZE}"
  echo "train_samples_per_segment=${KRONOS_TRAIN_SAMPLES_PER_SEGMENT}"
  echo "validation_samples=${KRONOS_VALIDATION_SAMPLES}"
  echo "coverage_passes=${KRONOS_COVERAGE_PASSES}"
  echo "patience=${KRONOS_EARLY_STOPPING_PATIENCE}"
  echo "balance_size_buckets=${KRONOS_BALANCE_SIZE_BUCKETS}"
  echo "reset_size_embedding=${KRONOS_RESET_SIZE_EMBEDDING}"
} > "${CURRENT_CONFIG}"

if [[ -f "${CONFIG_LOCK}" ]]; then
  if ! cmp -s "${CURRENT_CONFIG}" "${CONFIG_LOCK}"; then
    echo "[colab-v3] ERROR: resume configuration differs from ${CONFIG_LOCK}" >&2
    diff -u "${CONFIG_LOCK}" "${CURRENT_CONFIG}" >&2 || true
    exit 1
  fi
else
  cp "${CURRENT_CONFIG}" "${CONFIG_LOCK}"
fi

if [[ "${KRONOS_RESUME_TRAINING}" == "1" && ! -f "${RESUME_PATH}" ]]; then
  echo "[colab-v3] ERROR: resume requested but ${RESUME_PATH} does not exist" >&2
  exit 1
fi
if [[ "${KRONOS_RESUME_TRAINING}" == "0" && -f "${RESUME_PATH}" ]]; then
  echo "[colab-v3] ERROR: output already contains a checkpoint; resume or choose a new output" >&2
  exit 1
fi

cd "${REPO_ROOT}"
python finetune/verify_a_share_v3_setup.py \
  --data-root "${DATA_ROOT}" \
  --base-model "${BASE_MODEL}"

echo "[colab-v3] Output: ${OUTPUT_ROOT}"
echo "[colab-v3] Resume: ${KRONOS_RESUME_TRAINING}"
exec python -u finetune/train_predictor.py
