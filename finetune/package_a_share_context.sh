#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${KRONOS_CONTEXT_SOURCE_ROOT:-${ROOT}/data/a_share_v5}"
OUTPUT="${KRONOS_CONTEXT_BUNDLE:-${ROOT}/artifacts/kronos_a_share_v5_context_120d.tar.gz}"
STAGE="$(mktemp -d)"

cleanup() {
  rm -rf "${STAGE}"
}
trap cleanup EXIT

required=(
  processed_datasets/train_data.pkl
  processed_datasets/val_data.pkl
  processed_datasets/symbol_holdout_data.pkl
  asset_metadata.csv
  size_reference.json
  universe_manifest.csv
  context_coverage_manifest.csv
  v5_context_summary.json
)

for relative_path in "${required[@]}"; do
  if [[ ! -f "${SOURCE_ROOT}/${relative_path}" ]]; then
    echo "[v5-package] ERROR: missing ${SOURCE_ROOT}/${relative_path}" >&2
    exit 1
  fi
done

DEST="${STAGE}/data/a_share_v5"
mkdir -p "${DEST}/processed_datasets" "$(dirname "${OUTPUT}")"
for relative_path in "${required[@]}"; do
  cp "${SOURCE_ROOT}/${relative_path}" "${DEST}/${relative_path}"
done

(
  cd "${STAGE}"
  find data/a_share_v5 -type f -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 shasum -a 256 > SHA256SUMS
)

tar -czf "${OUTPUT}" -C "${STAGE}" SHA256SUMS data/a_share_v5
(
  cd "$(dirname "${OUTPUT}")"
  shasum -a 256 "$(basename "${OUTPUT}")" > "$(basename "${OUTPUT}").sha256"
)

echo "[v5-package] Bundle: ${OUTPUT}"
du -h "${OUTPUT}"
cat "${OUTPUT}.sha256"
