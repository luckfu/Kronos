#!/usr/bin/env bash
set -euo pipefail

KERNEL="${KRONOS_KAGGLE_V5_KERNEL:-luckfu/kronos-a-share-v5-120d-full-2015-2026-training}"
INTERVAL_SECONDS="${KRONOS_KAGGLE_V5_MONITOR_INTERVAL:-3600}"
LOG_FILE="${KRONOS_KAGGLE_V5_MONITOR_LOG:-${HOME}/.codex/kaggle-v5-monitor.log}"
IPV4_PATH="${KRONOS_KAGGLE_V5_IPV4_PATH:-/tmp/kronos_force_ipv4}"
ONE_SHOT="${1:-}"

mkdir -p "$(dirname "${LOG_FILE}")"
while true; do
  status="$(PYTHONPATH="${IPV4_PATH}" kaggle kernels status "${KERNEL}" 2>&1 || true)"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${status}" >> "${LOG_FILE}"
  if [[ "${status}" == *'KernelWorkerStatus.COMPLETE'* || "${status}" == *'KernelWorkerStatus.ERROR'* ]]; then
    result="${status##*KernelWorkerStatus.}"
    osascript -e "display notification \"${result}\" with title \"Kronos V5 Kaggle training\"" || true
    exit 0
  fi
  [[ "${ONE_SHOT}" == "--once" ]] && exit 0
  sleep "${INTERVAL_SECONDS}"
done
