#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${KRONOS_COLAB_V3_ROOT:-/content/drive/MyDrive/kronos_a_share_v3}"
DATA_ROOT="${KRONOS_COLAB_V3_DATA_ROOT:-${RUNTIME_ROOT}/data/a_share_v3}"
MODEL_ROOT="${KRONOS_COLAB_V3_MODEL_ROOT:-${RUNTIME_ROOT}/models}"
BUNDLE="${KRONOS_COLAB_V3_BUNDLE:-${RUNTIME_ROOT}/kronos_a_share_v3_colab_data.tar.gz}"
BASE_MODEL="${KRONOS_PREDICTOR_PATH:-${MODEL_ROOT}/a_share_size_full_coverage_colab_bs32_latest}"

cd "${REPO_ROOT}"
python -m pip install -q -r finetune/requirements-colab.txt
mkdir -p "${RUNTIME_ROOT}" "${MODEL_ROOT}"

if [[ ! -f "${DATA_ROOT}/processed_datasets/train_data.pkl" ]]; then
  if [[ ! -f "${BUNDLE}" ]]; then
    echo "[colab-v3] ERROR: upload the V3 data bundle to ${BUNDLE}" >&2
    exit 1
  fi
  if [[ -f "${BUNDLE}.sha256" ]]; then
    (
      cd "$(dirname "${BUNDLE}")"
      sha256sum -c "$(basename "${BUNDLE}").sha256"
    )
  else
    echo "[colab-v3] WARNING: ${BUNDLE}.sha256 is missing; relying on internal checksums."
  fi
  echo "[colab-v3] Extracting ${BUNDLE}"
  tar -xzf "${BUNDLE}" -C "${RUNTIME_ROOT}"
fi

if [[ -f "${RUNTIME_ROOT}/SHA256SUMS" ]]; then
  (
    cd "${RUNTIME_ROOT}"
    sha256sum -c SHA256SUMS
  )
fi

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

python finetune/verify_a_share_v3_setup.py \
  --data-root "${DATA_ROOT}" \
  --base-model "${BASE_MODEL}"

cat <<EOF
[colab-v3] Workspace ready.
[colab-v3] Data: ${DATA_ROOT}
[colab-v3] Base: ${BASE_MODEL}
[colab-v3] Start: bash finetune/colab_v3_train.sh
EOF
