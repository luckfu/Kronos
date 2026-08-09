#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${KRONOS_KAGGLE_ROOT:-/kaggle/working/kronos_a_share_v6_forecast}"
INPUT_ROOT="${KRONOS_KAGGLE_INPUT_ROOT:-/kaggle/input}"
DATA_ROOT="${RUNTIME_ROOT}/data/a_share_v5"
BASE_MODEL="${KRONOS_PREDICTOR_PATH:-}"
OUTPUT_NAME="${KRONOS_PREDICTOR_SAVE_FOLDER:-a_share_v6_forecast_only_context120_2pass}"
RESUME_ROOT="${RUNTIME_ROOT}/outputs/models/${OUTPUT_NAME}"

cd "${REPO_ROOT}"
python -m pip install -q -r finetune/requirements-colab.txt
mkdir -p "${RUNTIME_ROOT}"

if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi 'P100'; then
  if ! python - <<'PY'
import torch
raise SystemExit(0 if 'sm_60' in torch.cuda.get_arch_list() else 1)
PY
  then
    echo "[kaggle-v6] Installing a P100-compatible PyTorch wheel"
    python -m pip install -q --index-url https://download.pytorch.org/whl/cu121 \
      "torch==${KRONOS_KAGGLE_TORCH_VERSION:-2.5.1}"
  fi
fi

DETECTED_DATA="$(find "${INPUT_ROOT}" -type f \
  -path '*/data/a_share_v5/processed_datasets/train_data.pkl' -print -quit 2>/dev/null || true)"
if [[ -n "${DETECTED_DATA}" ]]; then
  DATA_ROOT="$(dirname "$(dirname "${DETECTED_DATA}")")"
fi

if [[ -z "${BASE_MODEL}" || ! -f "${BASE_MODEL}/model.safetensors" ]]; then
  V5_WEIGHT="$(find "${INPUT_ROOT}" -type f \
    -path '*/a_share_v5_context120_2pass/checkpoints/last_model/model.safetensors' \
    -print -quit 2>/dev/null || true)"
  if [[ -z "${V5_WEIGHT}" ]]; then
    V5_WEIGHT="$(find "${INPUT_ROOT}" -type f \
      -path '*/checkpoints/last_model/model.safetensors' -print -quit 2>/dev/null || true)"
  fi
  [[ -n "${V5_WEIGHT}" ]] && BASE_MODEL="$(dirname "${V5_WEIGHT}")"
fi

# Only import a V6 checkpoint. A V5 optimizer/scheduler state must never be
# resumed because V6 changes the objective and starts a fresh schedule.
if [[ ! -f "${RESUME_ROOT}/checkpoints/last_state.pt" ]]; then
  V6_STATE="$(find "${INPUT_ROOT}" -type f \
    -path '*/a_share_v6_forecast_only_context120_2pass/checkpoints/last_state.pt' \
    -print -quit 2>/dev/null || true)"
  if [[ -n "${V6_STATE}" ]]; then
    mkdir -p "${RESUME_ROOT}/checkpoints"
    cp -a "$(dirname "${V6_STATE}")/." "${RESUME_ROOT}/checkpoints/"
    echo "[kaggle-v6] Imported V6 resume checkpoint from $(dirname "${V6_STATE}")"
  fi
fi

if [[ ! -f "${DATA_ROOT}/processed_datasets/train_data.pkl" ]]; then
  echo "[kaggle-v6] Missing V5 120-day training data in /kaggle/input" >&2
  exit 1
fi
if [[ ! -f "${BASE_MODEL}/model.safetensors" ]]; then
  echo "[kaggle-v6] Missing V5 Last model in /kaggle/input" >&2
  exit 1
fi

export KRONOS_KAGGLE_ROOT="${RUNTIME_ROOT}"
export KRONOS_KAGGLE_DATA_ROOT="${DATA_ROOT}"
export KRONOS_PREDICTOR_PATH="${BASE_MODEL}"
python finetune/verify_a_share_context.py \
  --data-root "${DATA_ROOT}" --base-model "${BASE_MODEL}" \
  --lookback 120 --predict 10
printf 'export KRONOS_KAGGLE_DATA_ROOT=%q\nexport KRONOS_PREDICTOR_PATH=%q\n' \
  "${DATA_ROOT}" "${BASE_MODEL}" > "${RUNTIME_ROOT}/kaggle_v6_paths.env"
nvidia-smi -L || true
echo "[kaggle-v6] Workspace ready. Base model: ${BASE_MODEL}"
