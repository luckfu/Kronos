#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${KRONOS_V5_BASE_MODEL:-${ROOT}/outputs/models/a_share_v4_corrected_2026_replay20_latest/checkpoints/last_model}"
OUTPUT="${KRONOS_V5_BASE_BUNDLE:-${ROOT}/artifacts/a_share_v4_production_last.tar.gz}"
STAGE="$(mktemp -d)"

cleanup() {
  rm -rf "${STAGE}"
}
trap cleanup EXIT

for name in config.json model.safetensors README.md; do
  if [[ ! -f "${SOURCE}/${name}" ]]; then
    echo "[v5-base-package] ERROR: missing ${SOURCE}/${name}" >&2
    exit 1
  fi
done

DEST="${STAGE}/a_share_v4_corrected_2026_replay20_latest"
mkdir -p "${DEST}" "$(dirname "${OUTPUT}")"
cp "${SOURCE}/config.json" "${SOURCE}/model.safetensors" "${SOURCE}/README.md" "${DEST}/"
tar -czf "${OUTPUT}" -C "${STAGE}" "a_share_v4_corrected_2026_replay20_latest"
(
  cd "$(dirname "${OUTPUT}")"
  shasum -a 256 "$(basename "${OUTPUT}")" > "$(basename "${OUTPUT}").sha256"
)

echo "[v5-base-package] Bundle: ${OUTPUT}"
du -h "${OUTPUT}"
cat "${OUTPUT}.sha256"
