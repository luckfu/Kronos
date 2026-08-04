#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${KRONOS_KAGGLE_KERNEL_STAGE:-${ROOT}/artifacts/kaggle_v3_kernel}"
USERNAME="${KAGGLE_USERNAME:-luckfu}"

mkdir -p "${STAGE}"
cp "${ROOT}/finetune/kaggle_v3_benchmark.py" "${STAGE}/"
sed "s/YOUR_KAGGLE_USERNAME/${USERNAME}/g" \
  "${ROOT}/finetune/kaggle_v3_kernel-metadata.example.json" \
  > "${STAGE}/kernel-metadata.json"

echo "[kaggle-kernel] Staging directory: ${STAGE}"
echo "[kaggle-kernel] Submit with: kaggle kernels push -p ${STAGE} --accelerator P100"
ls -lh "${STAGE}"
