#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESUME_DATASET="${1:?Usage: $0 <owner/v6-segment-120-dataset>}"
STAGE="${KRONOS_V6_RESUME_STAGE:-${ROOT}/artifacts/kaggle_v6_resume_kernel}"

mkdir -p "${STAGE}"
cp "${ROOT}/finetune/kaggle_v6_resume.py" "${STAGE}/kaggle_v6_resume.py"
sed "s|YOUR_V6_RESUME_DATASET|${RESUME_DATASET}|" \
  "${ROOT}/finetune/kaggle_v6_resume-metadata.template.json" \
  > "${STAGE}/kernel-metadata.json"
echo "Prepared ${STAGE} to resume from ${RESUME_DATASET} through Segment 284."
