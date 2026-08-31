#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-swanlab-bridge"
STATE_ROOT="$ROOT/finetune/artifacts/swanlab_beta_v1_3_2_symbol_holdout_90_10_aggressive_1e5_twopass_a800"
REMOTE_RUN="/nfsdata/models/2026/kronos-v1-beta/runs/beta_v1_3_2_symbol_holdout_90_10_aggressive_1e5_twopass_seed100"
REMOTE_OUTPUT="$REMOTE_RUN/outputs/models/a_share_beta_v1_3_2_symbol_holdout_90_10_aggressive_1e5_twopass_120d_to_10d"

mkdir -p "$STATE_ROOT"
exec "$VENV/bin/python" -u "$ROOT/finetune/relay_a800_metrics_to_swanlab.py" \
  --host A800 \
  --remote-metrics "$REMOTE_OUTPUT/metrics.jsonl" \
  --state "$STATE_ROOT/state.json" \
  --poll-seconds 10 \
  --project finance \
  --workspace roc_fu \
  --steps-per-segment 313 \
  --experiment-name a_share_beta_v1_3_2_symbol_holdout_90_10_aggressive_1e5_twopass_a800 \
  --run-id kronos-beta-v1-3-2-symbol-holdout-90-10-aggressive-1e5-twopass-a800
