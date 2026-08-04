#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${KRONOS_KAGGLE_ROOT:-/kaggle/working/kronos_a_share_v3}"
INPUT_ROOT="${KRONOS_KAGGLE_DATASET_INPUT:-/kaggle/input/kronos-train-set-a}"
DATA_ROOT="${KRONOS_KAGGLE_DATA_ROOT:-${RUNTIME_ROOT}/data/a_share_v3}"
MODEL_ROOT="${KRONOS_KAGGLE_MODEL_ROOT:-${RUNTIME_ROOT}/models}"
BUNDLE="${KRONOS_KAGGLE_BUNDLE:-${INPUT_ROOT}/kronos_a_share_v3_colab_data.tar.gz}"
BASE_MODEL="${KRONOS_PREDICTOR_PATH:-${MODEL_ROOT}/a_share_size_full_coverage_colab_bs32_latest}"

cd "${REPO_ROOT}"
python -m pip install -q -r finetune/requirements-colab.txt
mkdir -p "${RUNTIME_ROOT}" "${MODEL_ROOT}"

if [[ ! -f "${DATA_ROOT}/processed_datasets/train_data.pkl" ]]; then
  if [[ -f "${INPUT_ROOT}/data/a_share_v3/processed_datasets/train_data.pkl" ]]; then
    DATA_ROOT="${INPUT_ROOT}/data/a_share_v3"
  else
    if [[ ! -f "${BUNDLE}" ]]; then
      BUNDLE="$(find "${INPUT_ROOT}" -maxdepth 2 -type f \
        -name 'kronos_a_share_v3*.tar.gz' -print -quit 2>/dev/null || true)"
    fi
    if [[ -f "${BUNDLE}" ]]; then
      echo "[kaggle-v3] Extracting ${BUNDLE} to local working storage"
      mkdir -p "${RUNTIME_ROOT}/data"
      tar -xzf "${BUNDLE}" -C "${RUNTIME_ROOT}"
    else
      echo "[kaggle-v3] ERROR: no V3 data bundle or extracted Kaggle Dataset found" >&2
      exit 1
    fi
  fi
fi

if [[ ! -f "${BASE_MODEL}/model.safetensors" ]]; then
  if [[ -f "${INPUT_ROOT}/base_model/model.safetensors" ]]; then
    BASE_MODEL="${INPUT_ROOT}/base_model"
  else
    echo "[kaggle-v3] Downloading the production base model from ModelScope"
    python - <<PY
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    'luckfu/a-share-size-kronos-base-earlystop50',
    repo_type='model',
    local_dir='${BASE_MODEL}',
)
PY
  fi
fi

export KRONOS_DATASET_PATH="${DATA_ROOT}/processed_datasets"
export KRONOS_METADATA_PATH="${DATA_ROOT}/asset_metadata.csv"
export KRONOS_PREDICTOR_PATH="${BASE_MODEL}"
export KRONOS_DRIVE_CONFLICT_CLEANUP="off"

{
  printf 'export KRONOS_KAGGLE_DATA_ROOT=%q\n' "${DATA_ROOT}"
  printf 'export KRONOS_PREDICTOR_PATH=%q\n' "${BASE_MODEL}"
} > "${RUNTIME_ROOT}/kaggle_v3_paths.env"

nvidia-smi -L || true
python finetune/verify_a_share_v3_setup.py \
  --data-root "${DATA_ROOT}" \
  --base-model "${BASE_MODEL}"

cat <<EOF
[kaggle-v3] Workspace ready.
[kaggle-v3] Data: ${DATA_ROOT}
[kaggle-v3] Base: ${BASE_MODEL}
[kaggle-v3] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unavailable)
[kaggle-v3] Start: bash finetune/kaggle_v3_train.sh
EOF
