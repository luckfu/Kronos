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

for SLOT in a b; do
  LONG_STAGE="${KRONOS_KAGGLE_LONG_STAGE:-${ROOT}/artifacts/kaggle_v3_long}_${SLOT}"
  mkdir -p "${LONG_STAGE}"
  cp "${ROOT}/finetune/kaggle_v3_long.py" "${LONG_STAGE}/"
  cp "${ROOT}/finetune/kaggle_v3_long_${SLOT}-metadata.json" \
    "${LONG_STAGE}/kernel-metadata.json"
  echo "[kaggle-kernel] Long-training ${SLOT^^}: ${LONG_STAGE}"
done
