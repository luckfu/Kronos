#!/usr/bin/env bash
# Run the local A800-to-SwanLab metrics relay. Keep credentials out of argv/logs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv-swanlab-bridge/bin/python"
STATE_ROOT="$ROOT/artifacts/swanlab_v6_natural_twospeed_v2_bf16_a800"
REMOTE_RUN="/nfsdata/models/2026/kronos-v1-beta/runs/v6_natural_twospeed_v2_seed100"
REMOTE_OUTPUT="$REMOTE_RUN/outputs/models/a_share_v1_beta_v6_natural_twospeed_v2_120d_to_10d"

[[ -x "$PYTHON" ]] || {
  echo "missing relay environment: $PYTHON" >&2
  exit 1
}

mkdir -p "$STATE_ROOT"
cd "$ROOT"
exec "$PYTHON" -u finetune/relay_a800_metrics_to_swanlab.py \
  --host A800 \
  --remote-metrics "$REMOTE_OUTPUT/metrics.jsonl" \
  --remote-baseline "$REMOTE_RUN/baseline_validation.json" \
  --state "$STATE_ROOT/state.json" \
  --poll-seconds 10 \
  --project finance \
  --workspace roc_fu \
  --experiment-name a_share_v1_beta_v6_natural_twospeed_v2_bf16_a800 \
  --run-id kronos-v1-beta-v6-natural-twospeed-v2-bf16-a800
