#!/usr/bin/env bash
set -euo pipefail

BASE="/nfsdata/models/2026/kronos-v1-beta"
export KRONOS_LAUNCH_PARENT_RUN="$BASE/runs/v6_natural_twospeed_v2_seed100"
export KRONOS_LAUNCH_PARENT_ROOT="$KRONOS_LAUNCH_PARENT_RUN/outputs/models/a_share_v1_beta_v6_natural_twospeed_v2_120d_to_10d/checkpoints/best_model"
export KRONOS_LAUNCH_RUN_ROOT="$BASE/runs/beta_v1_3_2_clean_v11_best818_symbol_holdout_90_10_aggressive_1e5_twopass_seed100"
export KRONOS_LAUNCH_BASELINE_OUTPUT_NAME="baseline_beta_v1_1_best818_symbol_holdout_full.json"

exec "$BASE/code/Kronos/finetune/run_a800_beta_v1_3_2_symbol_holdout_baseline.sh"
