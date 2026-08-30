#!/usr/bin/env bash
set -euo pipefail

api_base_url="${1:-https://luckfu--kronos-beta-v1-2-inference-web.modal.run}"
api_base_url="${api_base_url%/}"
payload_file="$(mktemp "${TMPDIR:-/tmp}/kronos-modal-payload.XXXXXX")"
trap 'rm -f "$payload_file"' EXIT

python - "$payload_file" <<'PY'
import json
import sys
from datetime import datetime, timedelta, timezone

output_path = sys.argv[1]
current = datetime(2026, 2, 2, tzinfo=timezone.utc)
rows = []
index = 0
while len(rows) < 120:
    if current.weekday() < 5:
        base = 10.0 + index * 0.015
        close = base + ((index % 7) - 3) * 0.01
        volume = 1_000_000 + index * 2_500
        rows.append({
            "timestamp": current.isoformat(),
            "open": round(base, 4),
            "high": round(max(base, close) + 0.12, 4),
            "low": round(min(base, close) - 0.12, 4),
            "close": round(close, 4),
            "volume": volume,
            "amount": round(volume * close, 2),
        })
        index += 1
    current += timedelta(days=1)

future_timestamps = []
while len(future_timestamps) < 10:
    if current.weekday() < 5:
        future_timestamps.append(current.isoformat())
    current += timedelta(days=1)

payload = {
    "data": rows,
    "future_timestamps": future_timestamps,
    "pred_len": 10,
    "sample_count": 50,
    "temperature": 0.65,
    "top_p": 0.8,
    "sector_id": 42,
    "size_percentile": 0.5,
}
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle)
PY

authorization_args=()
if [[ -n "${KRONOS_API_KEY:-}" ]]; then
    authorization_args=(-H "Authorization: Bearer ${KRONOS_API_KEY}")
fi

echo "==> GET ${api_base_url}/health"
curl --fail-with-body --silent --show-error \
    "${authorization_args[@]}" \
    "${api_base_url}/health" | python -m json.tool

echo "==> POST ${api_base_url}/predict"
curl --fail-with-body --silent --show-error \
    -X POST \
    -H "Content-Type: application/json" \
    "${authorization_args[@]}" \
    --data-binary "@${payload_file}" \
    "${api_base_url}/predict" | python -m json.tool
