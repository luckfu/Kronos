#!/usr/bin/env bash
# Relay Beta v1.2 full-only-every-segment continuation metrics to SwanLab.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv-swanlab-bridge/bin/python"
STATE_ROOT="$ROOT/finetune/artifacts/swanlab_beta_v1_2_best871_full_every_segment_onecycle_1pass_a800_v2"
REMOTE_RUN="/nfsdata/models/2026/kronos-v1-beta/runs/beta_v1_2_best871_full_every_segment_onecycle_1pass_seed100"
REMOTE_OUTPUT="$REMOTE_RUN/outputs/models/a_share_beta_v1_2_best871_full_every_segment_onecycle_1pass_120d_to_10d"

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
  --experiment-name a_share_beta_v1_2_best871_full_every_segment_onecycle_1pass_a800_v2 \
  --run-id kronos-beta-v1-2-best871-full-every-segment-onecycle-1pass-a800-v2
