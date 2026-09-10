#!/usr/bin/env bash
# Relay the formal A800 V3 run to one dedicated SwanLab run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv-swanlab-bridge/bin/python"
STATE_ROOT="$ROOT/finetune/artifacts/swanlab_beta_v3_base_dynamic_size_path_twopass_full_v1"
REMOTE_RUN="/nfsdata/models/2026/kronos-beta-v3/runs/beta_v3_base_dynamic_size_path_twopass_full_seed100"
REMOTE_OUTPUT="$REMOTE_RUN/outputs/models/beta_v3_base_dynamic_size_path_twopass_full_120d_to_10d"

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
  --experiment-name beta_v3_base_dynamic_size_path_twopass_full_v1 \
  --run-id kronos-beta-v3-base-dynamic-size-path-twopass-full-v1
