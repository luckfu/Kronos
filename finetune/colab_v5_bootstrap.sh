#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${KRONOS_COLAB_V5_ROOT:-/content/drive/MyDrive/kronos_a_share_v5}"
DATA_ROOT="${KRONOS_COLAB_V5_DATA_ROOT:-${RUNTIME_ROOT}/data/a_share_v5}"
MODEL_ROOT="${KRONOS_COLAB_V5_MODEL_ROOT:-${RUNTIME_ROOT}/models}"
BUNDLE="${KRONOS_COLAB_V5_BUNDLE:-${RUNTIME_ROOT}/kronos_a_share_v5_context_120d.tar.gz}"
BASE_MODEL="${KRONOS_PREDICTOR_PATH:-${MODEL_ROOT}/a_share_v4_corrected_2026_replay20_latest}"
BASE_BUNDLE="${KRONOS_COLAB_V5_BASE_BUNDLE:-${RUNTIME_ROOT}/a_share_v4_production_last.tar.gz}"

cd "${REPO_ROOT}"
python -m pip install -q -r finetune/requirements-colab.txt
mkdir -p "${RUNTIME_ROOT}" "${MODEL_ROOT}"

if [[ ! -f "${DATA_ROOT}/processed_datasets/train_data.pkl" ]]; then
  if [[ ! -f "${BUNDLE}" ]]; then
    echo "[colab-v5] ERROR: upload the V5 context bundle to ${BUNDLE}" >&2
    exit 1
  fi
  if [[ -f "${BUNDLE}.sha256" ]]; then
    (
      cd "$(dirname "${BUNDLE}")"
      shasum -a 256 -c "$(basename "${BUNDLE}").sha256"
    )
  fi
  echo "[colab-v5] Extracting ${BUNDLE}"
  tar -xzf "${BUNDLE}" -C "${RUNTIME_ROOT}"
fi

if [[ -f "${RUNTIME_ROOT}/SHA256SUMS" ]]; then
  (
    cd "${RUNTIME_ROOT}"
    shasum -a 256 -c SHA256SUMS
  )
fi

if [[ ! -f "${BASE_MODEL}/model.safetensors" ]]; then
  if [[ -f "${BASE_BUNDLE}" ]]; then
    if [[ -f "${BASE_BUNDLE}.sha256" ]]; then
      (
        cd "$(dirname "${BASE_BUNDLE}")"
        shasum -a 256 -c "$(basename "${BASE_BUNDLE}").sha256"
      )
    fi
    echo "[colab-v5] Extracting the uploaded V4 B production base"
    tar -xzf "${BASE_BUNDLE}" -C "${MODEL_ROOT}"
  elif [[ "${KRONOS_COLAB_V5_BASE_SOURCE:-local}" == "modelscope" ]]; then
    echo "[colab-v5] WARNING: explicitly using the ModelScope base; verify its SHA-256 and context length"
    python - <<PY
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    'luckfu/a-share-size-kronos-base-earlystop50',
    repo_type='model',
    local_dir='${BASE_MODEL}',
)
PY
  else
    echo "[colab-v5] ERROR: current V4 B base model is missing: ${BASE_MODEL}" >&2
    echo "[colab-v5] Upload a_share_v4_production_last.tar.gz or set KRONOS_PREDICTOR_PATH." >&2
    exit 1
  fi
fi

# Bootstrap may be run on a CPU runtime before switching the Colab hardware.
python finetune/verify_a_share_context.py \
  --data-root "${DATA_ROOT}" \
  --base-model "${BASE_MODEL}" \
  --lookback 120 \
  --predict 10 \
  --allow-cpu

cat <<EOF
[colab-v5] Workspace ready.
[colab-v5] Data: ${DATA_ROOT}
[colab-v5] Base: ${BASE_MODEL}
[colab-v5] After switching to GPU, run:
  KRONOS_COLAB_V5_ROOT='${RUNTIME_ROOT}' bash finetune/colab_v5_train.sh
EOF
