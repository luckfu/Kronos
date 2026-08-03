#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${KRONOS_COLAB_ROOT:-/content/kronos_runtime}"
DATA_ROOT="${KRONOS_COLAB_DATA_ROOT:-${RUNTIME_ROOT}/data/a_share}"
RAW_DATA="${DATA_ROOT}/a_share_daily.csv"
DATASET_ROOT="${DATA_ROOT}/processed_datasets"
METADATA_PATH="${DATA_ROOT}/asset_metadata.csv"
MODEL_ROOT="${KRONOS_COLAB_MODEL_ROOT:-${RUNTIME_ROOT}/models}"
BASE_KIND="${KRONOS_COLAB_BASE_MODEL:-production}"
PREPARE_ONLY="${KRONOS_PREPARE_ONLY:-0}"

cd "${REPO_ROOT}"
python -m pip install -q -r finetune/requirements-colab.txt
mkdir -p "${DATA_ROOT}" "${MODEL_ROOT}"

if [[ ! -f "${DATASET_ROOT}/train_data.pkl" || ! -f "${DATASET_ROOT}/val_data.pkl" ]]; then
  echo '[colab] Downloading the real BaoStock A-share panel...'
  python finetune/download_a_share_baostock.py \
    --universe csi800 \
    --start 2020-01-01 \
    --end 2026-07-31 \
    --output "${RAW_DATA}" \
    --resume
  python finetune/prepare_a_share.py \
    --input "${RAW_DATA}" \
    --output-dir "${DATASET_ROOT}" \
    --metadata-out "${METADATA_PATH}" \
    --size-reference-out "${DATA_ROOT}/size_reference.json"
else
  echo '[colab] Prepared dataset already exists; reusing it.'
fi

# CPU runtimes can prepare and persist the data without downloading model
# weights or requiring CUDA. The GPU phase runs this script again and then
# continues through model setup and validation.
if [[ "${PREPARE_ONLY}" == '1' ]]; then
  cat <<EOF
[colab] Data preparation complete.
[colab] Data: ${DATASET_ROOT}
[colab] Switch to a GPU runtime, remount Google Drive, then run:
  KRONOS_COLAB_ROOT='${RUNTIME_ROOT}' bash finetune/colab_bootstrap.sh
  KRONOS_COLAB_ROOT='${RUNTIME_ROOT}' bash finetune/colab_train.sh
EOF
  exit 0
fi

if [[ "${BASE_KIND}" == 'original' ]]; then
  BASE_MODEL="${MODEL_ROOT}/Kronos-base"
  if [[ ! -f "${BASE_MODEL}/model.safetensors" ]]; then
    python - <<PY
from huggingface_hub import snapshot_download
snapshot_download('NeoQuasar/Kronos-base', local_dir='${BASE_MODEL}')
PY
  fi
else
  BASE_MODEL="${MODEL_ROOT}/a_share_size_full_coverage_colab_bs32_latest"
  if [[ ! -f "${BASE_MODEL}/model.safetensors" ]]; then
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

python finetune/verify_colab_setup.py \
  --data-dir "${DATASET_ROOT}" \
  --base-model "${BASE_MODEL}"

cat <<EOF
[colab] Workspace ready.
[colab] Data: ${DATASET_ROOT}
[colab] Base: ${BASE_MODEL}
[colab] Start training with:
  KRONOS_COLAB_ROOT='${RUNTIME_ROOT}' bash finetune/colab_train.sh
EOF
