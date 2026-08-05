#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_STATE="${KRONOS_V3_LAST_STATE:-${ROOT}/artifacts/kaggle_v3_long_final_output/kronos_a_share_v3_long/outputs/models/a_share_size_full_market_v3_kaggle_bs32/checkpoints/last_state.pt}"
SOURCE_CONFIG="${KRONOS_V3_MODEL_CONFIG:-${ROOT}/artifacts/kaggle_v3_long_final_output/kronos_a_share_v3_long/outputs/models/a_share_size_full_market_v3_kaggle_bs32/checkpoints/best_model/config.json}"
STAGE="${KRONOS_2026_INCREMENTAL_STAGE:-${ROOT}/artifacts/kaggle_2026_incremental_dataset}"
BUNDLE="${STAGE}/kronos_a_share_2026_incremental.tar.gz"

rm -rf "${STAGE}"
mkdir -p "${STAGE}/payload/base_model/v3_last"

cd "${ROOT}"
python finetune/prepare_a_share_2026_incremental.py
KMP_DUPLICATE_LIB_OK=TRUE python finetune/extract_model_from_last_state.py \
  --state "${SOURCE_STATE}" \
  --config "${SOURCE_CONFIG}" \
  --output "${STAGE}/payload/base_model/v3_last"

mkdir -p "${STAGE}/payload/data"
cp -R "${ROOT}/data/a_share_v3_2026_incremental" "${STAGE}/payload/data/"
(
  cd "${STAGE}/payload"
  find base_model data -type f -print0 | LC_ALL=C sort -z \
    | xargs -0 shasum -a 256 > SHA256SUMS
  tar -czf "${BUNDLE}" SHA256SUMS base_model data
)
rm -rf "${STAGE}/payload"

cat > "${STAGE}/dataset-metadata.json" <<'JSON'
{
  "title": "Kronos A-share 2026 incremental training",
  "id": "luckfu/kronos-a-share-2026-incremental",
  "licenses": [{"name": "other"}],
  "isPrivate": true
}
JSON
shasum -a 256 "${BUNDLE}" > "${BUNDLE}.sha256"
du -h "${BUNDLE}"
cat "${BUNDLE}.sha256"
