#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${KRONOS_KAGGLE_ROOT:-/kaggle/working/kronos_a_share_v5_120d}"
INPUT_ROOT="${KRONOS_KAGGLE_INPUT_ROOT:-/kaggle/input}"
DATA_ROOT="${RUNTIME_ROOT}/data/a_share_v5"
MODEL_ROOT="${RUNTIME_ROOT}/models"
BASE_MODEL="${KRONOS_PREDICTOR_PATH:-${MODEL_ROOT}/a_share_v4_production_last}"
DATA_BUNDLE="${KRONOS_KAGGLE_DATA_BUNDLE:-}"
BASE_BUNDLE="${KRONOS_KAGGLE_BASE_BUNDLE:-}"
OUTPUT_NAME="${KRONOS_PREDICTOR_SAVE_FOLDER:-a_share_v5_context120_2pass}"
RESUME_ROOT="${RUNTIME_ROOT}/outputs/models/${OUTPUT_NAME}"

find_one() { find "${INPUT_ROOT}" -type f -name "$1" -print -quit 2>/dev/null || true; }

cd "${REPO_ROOT}"
python -m pip install -q -r finetune/requirements-colab.txt
mkdir -p "${RUNTIME_ROOT}" "${MODEL_ROOT}"

# Kaggle's current default wheel no longer includes Tesla P100 (sm_60).
if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi 'P100'; then
  if ! python - <<'PY'
import torch
raise SystemExit(0 if 'sm_60' in torch.cuda.get_arch_list() else 1)
PY
  then
    echo "[kaggle-v5] Installing a P100-compatible PyTorch wheel"
    python -m pip install -q --index-url https://download.pytorch.org/whl/cu121 \
      "torch==${KRONOS_KAGGLE_TORCH_VERSION:-2.5.1}"
  fi
fi

if [[ ! -f "${DATA_ROOT}/processed_datasets/train_data.pkl" ]]; then
  if [[ -z "${DATA_BUNDLE}" ]]; then
    DATA_BUNDLE="$(find_one 'kronos_a_share_v5_context120_train2015_2026.tar.gz')"
  fi
  if [[ -z "${DATA_BUNDLE}" ]]; then
    DATA_BUNDLE="$(find_one 'kronos_a_share_v5_context_120d.tar.gz')"
  fi
  if [[ -n "${DATA_BUNDLE}" ]]; then
    echo "[kaggle-v5] Extracting data bundle: ${DATA_BUNDLE}"
    tar -xzf "${DATA_BUNDLE}" -C "${RUNTIME_ROOT}"
  fi
  if [[ ! -f "${DATA_ROOT}/processed_datasets/train_data.pkl" ]]; then
    DETECTED_DATA="$(find "${INPUT_ROOT}" -type f \
      -path '*/data/a_share_v5/processed_datasets/train_data.pkl' -print -quit 2>/dev/null || true)"
    [[ -n "${DETECTED_DATA}" ]] && DATA_ROOT="$(dirname "$(dirname "${DETECTED_DATA}")")"
  fi
fi

if [[ ! -f "${BASE_MODEL}/model.safetensors" ]]; then
  [[ -n "${BASE_BUNDLE}" ]] || BASE_BUNDLE="$(find_one 'a_share_v4_production_last.tar.gz')"
  if [[ -n "${BASE_BUNDLE}" ]]; then
    echo "[kaggle-v5] Extracting V4 B base bundle: ${BASE_BUNDLE}"
    tar -xzf "${BASE_BUNDLE}" -C "${MODEL_ROOT}"
    DETECTED="$(find "${MODEL_ROOT}" -type f -name model.safetensors -print -quit)"
    [[ -n "${DETECTED}" ]] && BASE_MODEL="$(dirname "${DETECTED}")"
  fi
  if [[ ! -f "${BASE_MODEL}/model.safetensors" ]]; then
    DETECTED="$(find "${INPUT_ROOT}" -type f -path '*/a_share_v4_corrected_2026_replay20_latest/model.safetensors' -print -quit 2>/dev/null || true)"
    [[ -n "${DETECTED}" ]] && BASE_MODEL="$(dirname "${DETECTED}")"
  fi
fi

if [[ ! -f "${DATA_ROOT}/processed_datasets/train_data.pkl" || ! -f "${BASE_MODEL}/model.safetensors" ]]; then
  echo "[kaggle-v5] Missing V5 data or V4 B base model in /kaggle/input" >&2
  echo "Upload both tar.gz files as a Kaggle Dataset, then rerun this cell." >&2
  exit 1
fi

# A teammate can attach a Kaggle Dataset containing a previous run's output.
# Import it into the canonical working path so kaggle_v5_train.sh auto-resumes.
if [[ ! -f "${RESUME_ROOT}/checkpoints/last_state.pt" ]]; then
  RESUME_STATE="$(find "${INPUT_ROOT}" -type f -path '*/checkpoints/last_state.pt' -print -quit 2>/dev/null || true)"
  if [[ -n "${RESUME_STATE}" ]]; then
    SOURCE_CHECKPOINTS="$(dirname "${RESUME_STATE}")"
    mkdir -p "${RESUME_ROOT}/checkpoints"
    cp -a "${SOURCE_CHECKPOINTS}/." "${RESUME_ROOT}/checkpoints/"
    echo "[kaggle-v5] Imported resume checkpoint from ${SOURCE_CHECKPOINTS}"
  fi
fi

export KRONOS_KAGGLE_ROOT="${RUNTIME_ROOT}"
export KRONOS_KAGGLE_DATA_ROOT="${DATA_ROOT}"
export KRONOS_PREDICTOR_PATH="${BASE_MODEL}"
python finetune/verify_a_share_context.py --data-root "${DATA_ROOT}" --base-model "${BASE_MODEL}" --lookback 120 --predict 10
printf 'export KRONOS_KAGGLE_DATA_ROOT=%q\nexport KRONOS_PREDICTOR_PATH=%q\n' "${DATA_ROOT}" "${BASE_MODEL}" > "${RUNTIME_ROOT}/kaggle_v5_paths.env"
nvidia-smi -L || true
echo "[kaggle-v5] Workspace ready. Run: bash finetune/kaggle_v5_train.sh"
