#!/usr/bin/env bash
set -euo pipefail

BASE="/nfsdata/models/2026/kronos-v1-beta"
PARENT_RUN="$BASE/runs/beta_v1_3_2_clean_v11_best818_symbol_holdout_90_10_aggressive_1e5_twopass_seed100"
PARENT_OUTPUT="$PARENT_RUN/outputs/models/a_share_beta_v1_3_2_clean_v11_best818_symbol_holdout_90_10_aggressive_1e5_twopass_120d_to_10d"

export KRONOS_LAUNCH_PARENT_RUN="$PARENT_RUN"
export KRONOS_LAUNCH_PARENT_OUTPUT="$PARENT_OUTPUT"
export KRONOS_LAUNCH_PARENT_ROOT="$PARENT_OUTPUT/checkpoints/best_model"
export KRONOS_LAUNCH_PARENT_RELEASE="Beta v2.0"
export KRONOS_LAUNCH_PARENT_CHECKPOINT="Best@687"
export KRONOS_LAUNCH_PARENT_SHA="e603fe3178d61ee7feb8a5b0ad520d13166d533785f12d0a4f51d85db0a91ed3"
export KRONOS_LAUNCH_PARENT_OLD_TUNING_LOSS="2.3532605171203613"
export KRONOS_LAUNCH_RUN_ROOT="$BASE/runs/beta_v2_1_best687_decision_heads_twopass_seed100"
export KRONOS_LAUNCH_OUTPUT_NAME="a_share_beta_v2_1_best687_decision_heads_twopass_120d_to_10d"
export KRONOS_LAUNCH_STAGE_ID="beta_v2_1_best687_decision_heads_twopass"

export KRONOS_USE_BETA_V21_AUXILIARY=1
export KRONOS_BETA_V21_AUXILIARY_WARMUP_STEPS=1000
export KRONOS_BETA_V21_EMA_DECAY=0.99
export KRONOS_BETA_V21_AUTO_CALIBRATE=1
export KRONOS_BETA_V21_CONSISTENCY_SAMPLES=2048
export KRONOS_BETA_V21_CONSISTENCY_SAMPLE_COUNT=1
export KRONOS_FORECAST_HORIZON_WEIGHTS="1.364,1.364,1.364,1.136,1.136,0.909,0.909,0.682,0.682,0.455"
export KRONOS_BEST_SELECTION_METRIC=beta_v21_score

required=(
  "$BASE/code/Kronos/finetune/beta_v21.py"
  "$KRONOS_LAUNCH_PARENT_ROOT/model.safetensors"
  "$KRONOS_LAUNCH_PARENT_ROOT/config.json"
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "missing required Beta v2.1 input: $path" >&2; exit 1; }
done

mkdir -p "$KRONOS_LAUNCH_RUN_ROOT"
if [[ ! -f "$KRONOS_LAUNCH_RUN_ROOT/experiment_manifest.json" ]]; then
  cat > "$KRONOS_LAUNCH_RUN_ROOT/experiment_manifest.json" <<EOF
{
  "stage_id": "$KRONOS_LAUNCH_STAGE_ID",
  "release_candidate": "Beta v2.1",
  "parent": {"release": "Beta v2.0", "checkpoint": "Best@687", "path": "$KRONOS_LAUNCH_PARENT_ROOT", "model_sha256": "$KRONOS_LAUNCH_PARENT_SHA"},
  "architecture": {"autoregressive_path_head": "preserved", "return_head": 4, "barrier_head": 3, "ranking_head": "derived_not_independent", "h_asof_index": 119},
  "labels": {"entry": "next_day_open", "return_horizons": [1, 3, 5, 10], "return_normalization": "max(sigma20,0.005)*sqrt(h)", "barriers": {"take_profit": 0.05, "stop_loss": -0.03, "same_day_double_hit_training": "masked"}},
  "objective": {"path": 0.68, "history": 0.02, "return": 0.15, "barrier": 0.10, "ranking": 0.05, "return_bias_share": 0.20, "ema_decay": 0.99, "auxiliary_warmup_steps": 1000},
  "validation": {"mode": "full_only", "fixed_samples": 123982, "selection_metric": "beta_v21_score", "fixed_denominators": "auto_calibrated_once_from_parent_and_persisted", "return_path_consistency_samples": 2048},
  "training": {"coverage_passes": 2, "batch_size": 64, "peak_learning_rate": 1e-5, "amp_dtype": "bfloat16", "trainable_predictor": "all"}
}
EOF
fi

exec "$BASE/code/Kronos/finetune/run_a800_beta_v1_3_2_symbol_holdout_90_10_aggressive_1e5_twopass.sh" "${1:-start}"
