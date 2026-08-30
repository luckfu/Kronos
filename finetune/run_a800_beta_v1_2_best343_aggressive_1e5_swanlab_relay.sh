#!/usr/bin/env bash
# Relay the Best@343 aggressive 1e-5 diagnostic metrics to SwanLab.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv-swanlab-bridge/bin/python"
STATE_ROOT="$ROOT/finetune/artifacts/swanlab_beta_v1_2_best343_aggressive_1e5_50seg_a800"
REMOTE_RUN="/nfsdata/models/2026/kronos-v1-beta/runs/beta_v1_2_best343_aggressive_1e5_50seg_seed100"
REMOTE_OUTPUT="$REMOTE_RUN/outputs/models/a_share_beta_v1_2_best343_aggressive_1e5_120d_to_10d"

[[ -x "$PYTHON" ]] || { echo "missing relay environment: $PYTHON" >&2; exit 1; }
mkdir -p "$STATE_ROOT"
cd "$ROOT"
exec "$PYTHON" -u finetune/relay_a800_metrics_to_swanlab.py \
  --host A800 \
  --remote-metrics "$REMOTE_OUTPUT/metrics.jsonl" \
  --state "$STATE_ROOT/state.json" \
  --poll-seconds 10 \
  --project finance \
  --workspace roc_fu \
  --steps-per-segment 313 \
  --experiment-name a_share_beta_v1_2_best343_aggressive_1e5_50seg_a800 \
  --run-id kronos-beta-v1-2-best343-aggressive-1e5-50seg-a800
