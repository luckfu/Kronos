#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${KRONOS_V4_KERNEL_STAGE:-${ROOT}/artifacts/kaggle_v4_ab_kernel}"

rm -rf "${STAGE}"
mkdir -p "${STAGE}"
cp "${ROOT}/finetune/kaggle_v4_ab.py" "${STAGE}/kaggle_v4_ab.py"
cp "${ROOT}/finetune/kaggle_v4_ab-metadata.json" "${STAGE}/kernel-metadata.json"
echo "Prepared ${STAGE}"
