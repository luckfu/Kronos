#!/usr/bin/env bash
set -euo pipefail

SSH_TARGET=${SSH_TARGET:-oracle4C24G}
SNAPSHOT_DATE=${1:-}

if [[ -n "$SNAPSHOT_DATE" && ! "$SNAPSHOT_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "Usage: $0 [YYYY-MM-DD]" >&2
  exit 2
fi

if [[ -n "$SNAPSHOT_DATE" ]]; then
  ssh "$SSH_TARGET" "/opt/kronos-web/.venv/bin/python /opt/kronos-web/webui/update_sector_mapping.py --date '$SNAPSHOT_DATE'"
else
  ssh "$SSH_TARGET" "/opt/kronos-web/.venv/bin/python /opt/kronos-web/webui/update_sector_mapping.py"
fi

ssh "$SSH_TARGET" 'sudo systemctl restart kronos-web && curl -fsS http://127.0.0.1:7072/health'
printf '\n'
