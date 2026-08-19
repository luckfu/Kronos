#!/usr/bin/env bash
set -euo pipefail
BASE_URL=${KRONOS_PUBLIC_URL:-https://allmoneybymehold.com/kronos}
echo "Health:"
curl --fail --silent --show-error "$BASE_URL/health"
printf '\n\nModel status:\n'
curl --fail --silent --show-error "$BASE_URL/api/model-status"
printf '\n\nSingle-stock Modal prediction:\n'
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"600519","backend":"remote","sample_count":50,"temperature":0.65,"top_p":0.8}' \
  "$BASE_URL/api/predict"
printf '\n'
