#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${KRONOS_V6_RESUME_STAGE:-${ROOT}/artifacts/kaggle_v6_resume_kernel}"

mkdir -p "${STAGE}"
cp "${ROOT}/finetune/kaggle_v6_resume.py" "${STAGE}/kaggle_v6_resume.py"
cp "${ROOT}/finetune/kaggle_v6_resume-metadata.template.json" \
  "${STAGE}/kernel-metadata.json"
echo "Prepared ${STAGE} to resume from the Segment 120 kernel output through Segment 284."
