#!/usr/bin/env bash
set -Eeuo pipefail

# Start today's ETL-approved trading date after the evening ETL window.
# The script is intentionally idempotent: completed runs are left untouched and
# prediction_drill.py resumes a partially completed run from its batch files.

ROOT=/opt/kronos-web
PYTHON=/data/miniconda3/bin/python
HEALTH_SCRIPT=/data/openclaw_workspace/imperial_data_engine/scripts/etl_health_status.py
PREDICTOR=$ROOT/deploy/prediction_drill.py
SECTOR_MAP=$ROOT/webui/symbol_sector_map.json
OUTPUT_ROOT=$ROOT/data/prediction_shadow
LOG_DIR=$ROOT/logs
LOCK_FILE=$OUTPUT_ROOT/.daily-prediction.lock
POLL_SECONDS=${KRONOS_PREDICTION_POLL_SECONDS:-300}
MAX_WAIT_SECONDS=${KRONOS_PREDICTION_MAX_WAIT_SECONDS:-57600}

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  exit 0
fi

today=$(date +%F)
log_file="$LOG_DIR/daily_prediction_scheduler_${today//-/}.log"
exec >>"$log_file" 2>&1

echo "[$(date '+%F %T')] scheduler started"
set -a
# The environment contains credentials for the health check and Modal request.
. /home/opc/.openclaw/.env
set +a

health_ready() {
  local target_date=$1
  local status_file
  status_file=$(mktemp)
  if ! "$PYTHON" "$HEALTH_SCRIPT" status --date "$target_date" >"$status_file" 2>/dev/null; then
    rm -f "$status_file"
    return 1
  fi
  if "$PYTHON" - "$status_file" <<'PY'
import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("ready_for_downstream") is True else 1)
PY
  then
    local result=0
  else
    local result=$?
  fi
  rm -f "$status_file"
  return "$result"
}

elapsed=0
target_date=""
while (( elapsed <= MAX_WAIT_SECONDS )); do
  if health_ready "$today"; then
    target_date=$today
    break
  fi
  # A completed skipped status means a weekend/holiday; there is no prediction.
  status_file=$(mktemp)
  "$PYTHON" "$HEALTH_SCRIPT" status --date "$today" >"$status_file" 2>/dev/null || true
  if "$PYTHON" - "$status_file" <<'PY'
import json
import sys
from pathlib import Path
try:
    raise SystemExit(0 if json.loads(Path(sys.argv[1]).read_text()).get("status") == "skipped" else 1)
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
PY
  then
    rm -f "$status_file"
    echo "[$(date '+%F %T')] $today is a skipped non-trading day; nothing to do"
    exit 0
  fi
  rm -f "$status_file"
  echo "[$(date '+%F %T')] today's ETL is not approved yet; retry in ${POLL_SECONDS}s"
  sleep "$POLL_SECONDS"
  elapsed=$((elapsed + POLL_SECONDS))
done

if [[ -z "$target_date" ]]; then
  echo "[$(date '+%F %T')] timed out waiting for ETL approval"
  exit 1
fi

output_dir="$OUTPUT_ROOT/$target_date/full_market"
summary_file="$output_dir/summary.json"
if [[ -f "$summary_file" ]] && "$PYTHON" - "$summary_file" <<'PY'
import json
import sys
from pathlib import Path

try:
    raise SystemExit(0 if json.loads(Path(sys.argv[1]).read_text()).get("status") == "complete" else 1)
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
PY
then
  echo "[$(date '+%F %T')] $target_date already complete; nothing to do"
  exit 0
fi

if pgrep -af "[p]rediction_drill.py.*--asof $target_date" >/dev/null; then
  echo "[$(date '+%F %T')] $target_date prediction already running"
  exit 0
fi

mkdir -p "$output_dir"
echo "[$(date '+%F %T')] starting prediction for $target_date"
exec "$PYTHON" "$PREDICTOR" \
  --asof "$target_date" \
  --sector-map "$SECTOR_MAP" \
  --output-dir "$output_dir" \
  --sample-count 5 \
  --batch-size 12 \
  --max-retries 4 \
  --request-timeout 240
