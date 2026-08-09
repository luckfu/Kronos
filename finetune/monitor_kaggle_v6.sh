#!/usr/bin/env bash
set -euo pipefail

export PATH="${HOME}/.local/bin:/opt/miniconda3/bin:${PATH}"

KERNEL="${KRONOS_KAGGLE_V6_KERNEL:-user281434/kronos-a-share-v6-forecast-only-from-v5-last}"
INTERVAL_SECONDS="${KRONOS_KAGGLE_V6_MONITOR_INTERVAL:-1800}"
LOG_FILE="${KRONOS_KAGGLE_V6_MONITOR_LOG:-${HOME}/.codex/kaggle-v6-monitor.log}"
ONE_SHOT="${1:-}"

mkdir -p "$(dirname "${LOG_FILE}")"
while true; do
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  kernel_status="$(kaggle kernels status "${KERNEL}" 2>&1 || true)"
  quota="$(kaggle quota 2>&1 || true)"
  {
    printf '[%s] %s\n' "${timestamp}" "${kernel_status}"
    printf '%s\n' "${quota}"
  } >> "${LOG_FILE}"

  if [[ "${kernel_status}" == *'KernelWorkerStatus.COMPLETE'* || "${kernel_status}" == *'KernelWorkerStatus.ERROR'* ]]; then
    result="${kernel_status##*KernelWorkerStatus.}"
    osascript -e "display notification \"${result}\" with title \"Kronos V6 Kaggle training\"" || true
    exit 0
  fi
  [[ "${ONE_SHOT}" == "--once" ]] && exit 0
  sleep "${INTERVAL_SECONDS}"
done
