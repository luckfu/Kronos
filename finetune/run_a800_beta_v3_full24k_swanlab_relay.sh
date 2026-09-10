#!/usr/bin/env bash
# Relay the fresh A800 Beta v3 run to SwanLab from this Mac.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv-swanlab-bridge/bin/python"
STATE_ROOT="$ROOT/finetune/artifacts/swanlab_beta_v3_a800_full24k_v2"
REMOTE_RUN="/nfsdata/models/2026/kronos-beta-v3/runs/beta_v3_base_dynamic_size_path_full24k_seed100"
REMOTE_OUTPUT="$REMOTE_RUN/outputs/models/beta_v3_base_dynamic_size_path_full24k"

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
  --steps-per-segment 625 \
  --experiment-name beta_v3_base_dynamic_size_path_a800_full24k_v2 \
  --run-id kronos-beta-v3-base-dynamic-size-path-a800-full24k-v2
