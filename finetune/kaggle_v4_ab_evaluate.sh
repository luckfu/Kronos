#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${KRONOS_KAGGLE_ROOT:-/kaggle/working/kronos_a_share_v4_ab}"
PATHS_FILE="${RUNTIME_ROOT}/kaggle_v3_paths.env"
if [[ -f "${PATHS_FILE}" ]]; then
  source "${PATHS_FILE}"
fi
DATA_ROOT="${KRONOS_KAGGLE_DATA_ROOT:-${RUNTIME_ROOT}/data/a_share_v3}"
A_MODEL="${RUNTIME_ROOT}/outputs/models/a_share_v4_corrected_2026_recent_only/checkpoints/best_model"
B_MODEL="${RUNTIME_ROOT}/outputs/models/a_share_v4_corrected_2026_replay20/checkpoints/best_model"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/finetune"
python "${REPO_ROOT}/finetune/compare_kaggle_best_last.py" \
  --holdout "${DATA_ROOT}/processed_datasets/symbol_holdout_data.pkl" \
  --manifest "${DATA_ROOT}/universe_manifest.csv" \
  --best-model "${A_MODEL}" \
  --last-model "${B_MODEL}" \
  --signal-start 2026-06-18 \
  --signal-end 2026-07-16 \
  --signal-label 2026_time_holdout \
  --period-count 20 \
  --sample-count 1 \
  --batch-size 64 \
  --output-dir "${RUNTIME_ROOT}/outputs/evaluation/v4_recent_vs_replay20"
