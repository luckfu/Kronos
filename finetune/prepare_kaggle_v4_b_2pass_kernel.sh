#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${KRONOS_V4_B_2PASS_STAGE:-${ROOT}/artifacts/kaggle_v4_b_2pass_kernel}"

rm -rf "${STAGE}"
mkdir -p "${STAGE}"
cp "${ROOT}/finetune/kaggle_v4_b_2pass.py" "${STAGE}/kaggle_v4_b_2pass.py"
cp "${ROOT}/finetune/kaggle_v4_b_2pass-metadata.json" "${STAGE}/kernel-metadata.json"
echo "Prepared ${STAGE}"
