#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${KRONOS_V3_SOURCE_ROOT:-${ROOT}/data/a_share_v3}"
OUTPUT="${KRONOS_V3_BUNDLE:-${ROOT}/artifacts/kronos_a_share_v3_colab_data.tar.gz}"
STAGE="$(mktemp -d)"

cleanup() {
  rm -rf "${STAGE}"
}
trap cleanup EXIT

required=(
  processed_datasets/train_data.pkl
  processed_datasets/val_data.pkl
  processed_datasets/symbol_holdout_data.pkl
  processed_datasets/test_data.pkl
  asset_metadata.csv
  size_reference.json
  universe_manifest.csv
  universe_summary.json
)

for relative_path in "${required[@]}"; do
  if [[ ! -f "${SOURCE_ROOT}/${relative_path}" ]]; then
    echo "[v3-package] ERROR: missing ${SOURCE_ROOT}/${relative_path}" >&2
    exit 1
  fi
done

DEST="${STAGE}/data/a_share_v3"
mkdir -p "${DEST}/processed_datasets" "$(dirname "${OUTPUT}")"

for relative_path in "${required[@]}"; do
  cp "${SOURCE_ROOT}/${relative_path}" "${DEST}/${relative_path}"
done

(
  cd "${STAGE}"
  find data/a_share_v3 -type f -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 shasum -a 256 > SHA256SUMS
)

tar -czf "${OUTPUT}" -C "${STAGE}" SHA256SUMS data/a_share_v3
(
  cd "$(dirname "${OUTPUT}")"
  shasum -a 256 "$(basename "${OUTPUT}")" > "$(basename "${OUTPUT}").sha256"
)

echo "[v3-package] Bundle: ${OUTPUT}"
du -h "${OUTPUT}"
cat "${OUTPUT}.sha256"
