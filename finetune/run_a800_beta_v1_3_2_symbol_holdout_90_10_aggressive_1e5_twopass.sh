#!/usr/bin/env bash
# Beta v1.3.2: two passes over the 90/10 symbol-isolated training split.
set -euo pipefail

MODE="${1:-start}"
if [[ "$MODE" != "start" && "$MODE" != "resume" ]]; then
  echo "usage: $0 {start|resume}" >&2
  exit 2
fi

BASE="/nfsdata/models/2026/kronos-v1-beta"
CODE_ROOT="$BASE/code/Kronos"
DATA_ROOT="$BASE/data/a_share_full_market_v1_beta_symbol_holdout_90_10_v1"
VALIDATION_ROOT="$DATA_ROOT/natural_validation_2025h2_2026h1_symbol_holdout_full_v1"
METADATA_ROOT="$BASE/data/a_share_full_market_v1_beta"
PARENT_RUN="${KRONOS_LAUNCH_PARENT_RUN:-$BASE/runs/beta_v1_3_best343_aggressive_1e5_twopass_seed100}"
PARENT_OUTPUT="${KRONOS_LAUNCH_PARENT_OUTPUT:-$PARENT_RUN/outputs/models/a_share_beta_v1_3_best343_aggressive_1e5_twopass_120d_to_10d}"
PARENT_ROOT="${KRONOS_LAUNCH_PARENT_ROOT:-$PARENT_OUTPUT/checkpoints/best_model}"
TOKENIZER_ROOT="$BASE/models/tokenizer"
RUN_ROOT="${KRONOS_LAUNCH_RUN_ROOT:-$BASE/runs/beta_v1_3_2_symbol_holdout_90_10_aggressive_1e5_twopass_seed100}"
OUTPUT_NAME="${KRONOS_LAUNCH_OUTPUT_NAME:-a_share_beta_v1_3_2_symbol_holdout_90_10_aggressive_1e5_twopass_120d_to_10d}"
OUTPUT_ROOT="$RUN_ROOT/outputs/models/$OUTPUT_NAME"
PYTHON=(sudo -n -E "$BASE/env/a800-py312/bin/python")

STAGE_ID="${KRONOS_LAUNCH_STAGE_ID:-beta_v1_3_2_symbol_holdout_90_10_aggressive_1e5_twopass}"
PARENT_RELEASE="${KRONOS_LAUNCH_PARENT_RELEASE:-Beta v1.3.1}"
PARENT_CHECKPOINT="${KRONOS_LAUNCH_PARENT_CHECKPOINT:-Best@895}"
PARENT_OLD_TUNING_LOSS="${KRONOS_LAUNCH_PARENT_OLD_TUNING_LOSS:-2.3491287231445312}"

DATA_MANIFEST_ID="17afbeede658c13787043e601aa355717dda4d11719b51fb3ce368fb138e627a"
DATA_MANIFEST_FILE_SHA="3989bddae6c76e34eb7772590e11b4605215270c178ae30eb5c2261e06fe9177"
TRAIN_SHA="b2f2a861f651321efd38761c65ffcff4d14290cd580325298c4b9a7bc915f832"
VAL_SHA="748a9205714ee9714525872e65432063edd26d6afbef05391919e8d9d8811115"
VALIDATION_SHA="ea29ecdb318adf9789ddd47eb4c5d3df7cdadcbbd60471305241a469d357d184"
PARENT_SHA="${KRONOS_LAUNCH_PARENT_SHA:-ad6f2ffc84536795a5c88e0dfa74d5486aa720a2296bce4f669ff5e99bc8ed6a}"

required=(
  "$CODE_ROOT/finetune/config.py"
  "$CODE_ROOT/finetune/dataset.py"
  "$CODE_ROOT/finetune/train_predictor.py"
  "$CODE_ROOT/finetune/export_last_model.py"
  "$DATA_ROOT/data_manifest.json"
  "$DATA_ROOT/processed_datasets/train_data.pkl"
  "$DATA_ROOT/processed_datasets/val_data.pkl"
  "$METADATA_ROOT/asset_metadata.csv"
  "$VALIDATION_ROOT/natural_validation_manifest.json"
  "$VALIDATION_ROOT/natural_validation_samples.jsonl"
  "$PARENT_ROOT/model.safetensors"
  "$PARENT_ROOT/config.json"
  "$TOKENIZER_ROOT/model.safetensors"
  "$TOKENIZER_ROOT/config.json"
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "missing required input: $path" >&2; exit 1; }
done

verify_sha() {
  local expected="$1" path="$2" label="$3"
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
    echo "$label checksum mismatch: $path" >&2
    exit 1
  }
}
verify_sha "$DATA_MANIFEST_FILE_SHA" "$DATA_ROOT/data_manifest.json" "data manifest file"
verify_sha "$TRAIN_SHA" "$DATA_ROOT/processed_datasets/train_data.pkl" "training panel"
verify_sha "$VAL_SHA" "$DATA_ROOT/processed_datasets/val_data.pkl" "validation panel"
verify_sha "$VALIDATION_SHA" "$VALIDATION_ROOT/natural_validation_manifest.json" "validation manifest"
verify_sha "$PARENT_SHA" "$PARENT_ROOT/model.safetensors" "$PARENT_RELEASE $PARENT_CHECKPOINT"

if [[ "$MODE" == "start" ]]; then
  [[ ! -e "$OUTPUT_ROOT" ]] || {
    echo "start refuses existing output: $OUTPUT_ROOT" >&2
    exit 1
  }
  RESUME=0
else
  [[ -f "$OUTPUT_ROOT/checkpoints/last_state.pt" ]] || {
    echo "resume requires last_state.pt: $OUTPUT_ROOT/checkpoints/last_state.pt" >&2
    exit 1
  }
  RESUME=1
fi

mkdir -p "$RUN_ROOT"
if [[ ! -f "$RUN_ROOT/experiment_manifest.json" ]]; then
  cat > "$RUN_ROOT/experiment_manifest.json" <<EOF
{
  "stage_id": "$STAGE_ID",
  "parent": {"release": "$PARENT_RELEASE", "checkpoint": "$PARENT_CHECKPOINT", "path": "$PARENT_ROOT", "model_sha256": "$PARENT_SHA", "old_tuning_loss": $PARENT_OLD_TUNING_LOSS},
  "initialization": {"model": "weights_only", "optimizer": "fresh", "scheduler": "one_cycle", "conditioning": "preserved_from_parent_checkpoint", "reset_sector_embedding": false, "reset_size_embedding": false},
  "data": {"split": "symbol_holdout_90_10", "train_symbols": 4678, "validation_symbols": 520, "symbol_overlap": 0, "train_windows": 9457646, "manifest_id": "$DATA_MANIFEST_ID", "manifest_file_sha256": "$DATA_MANIFEST_FILE_SHA"},
  "training": {"global_segments": 946, "coverage_passes": 2, "samples_per_segment": 20000, "batch_size": 64, "seed": 100, "gpu": 0, "amp_dtype": "bfloat16", "predictor_learning_rate": 1e-5, "condition_learning_rate": 1e-5, "trainable_predictor": "all"},
  "validation": {"mode": "full_only", "manifest_sha256": "$VALIDATION_SHA", "full_samples": 123982, "full_interval_segments": 1, "quick_validation": "disabled", "role": "symbol-isolated tuning set"},
  "best_selection": {"dashboard_metric": "validation/full/loss", "implementation_metric": "validation_large_objective", "evaluated_every_segment": true}
}
EOF
fi

export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$CODE_ROOT"
export KRONOS_TRAIN_DATA_PATHS="$DATA_ROOT/processed_datasets/train_data.pkl"
export KRONOS_VAL_DATA_PATHS="$DATA_ROOT/processed_datasets/val_data.pkl"
export KRONOS_METADATA_PATH="$METADATA_ROOT/asset_metadata.csv"
export KRONOS_DATA_MANIFEST_SHA256="$DATA_MANIFEST_ID"
export KRONOS_FIXED_VALIDATION_MANIFEST_PATH="$VALIDATION_ROOT/natural_validation_manifest.json"
export KRONOS_FIXED_VALIDATION_MANIFEST_SHA256="$VALIDATION_SHA"
export KRONOS_EXCLUDE_FIXED_VALIDATION_FROM_TRAINING=1
export KRONOS_VALIDATION_FULL_ONLY=1
export KRONOS_VALIDATION_QUICK_SAMPLES=24000
export KRONOS_VALIDATION_LARGE_SAMPLES=123982
export KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS=1
export KRONOS_PREDICTOR_PATH="$PARENT_ROOT" KRONOS_TOKENIZER_PATH="$TOKENIZER_ROOT"
export KRONOS_SAVE_PATH="$RUN_ROOT/outputs/models" KRONOS_PREDICTOR_SAVE_FOLDER="$OUTPUT_NAME"
export KRONOS_LOOKBACK_WINDOW=120 KRONOS_PREDICT_WINDOW=10
export KRONOS_USE_SECTOR_FEATURES=1 KRONOS_NUM_SECTORS=86
export KRONOS_USE_SIZE_FEATURES=0 KRONOS_NUM_SIZE_BUCKETS=0 KRONOS_USE_SIZE_PERCENTILE=1
export KRONOS_CONTEXT_LAYER=10 KRONOS_RESET_SECTOR_EMBEDDING=0 KRONOS_RESET_SIZE_EMBEDDING=0
export KRONOS_TRAIN_SIGNAL_START=2015-01-01 KRONOS_TRAIN_SIGNAL_END=2026-07-17
export KRONOS_VAL_SIGNAL_START=2025-07-01 KRONOS_VAL_SIGNAL_END=2026-06-30
export KRONOS_TRAIN_SAMPLES_PER_SEGMENT=20000 KRONOS_VALIDATION_SAMPLES=123982
export KRONOS_EPOCHS=1 KRONOS_COVERAGE_PASSES=2 KRONOS_REQUIRE_FULL_COVERAGE=1
export KRONOS_EARLY_STOPPING_PATIENCE=0
export KRONOS_PREDICTOR_LEARNING_RATE=1e-5 KRONOS_CONDITION_LEARNING_RATE=1e-5
export KRONOS_SCHEDULER_TYPE=one_cycle KRONOS_SCHEDULER=one_cycle
export KRONOS_PREDICTOR_WARMUP_START_LR=1e-6 KRONOS_CONDITION_WARMUP_START_LR=1e-6
export KRONOS_PREDICTOR_MIN_LR=1e-8 KRONOS_CONDITION_MIN_LR=1e-8
export KRONOS_TRAINABLE_TRANSFORMER_LAYERS=-1
export KRONOS_PREDICTOR_LOSS_MODE=forecast KRONOS_HISTORY_LOSS_WEIGHT=0.02
export KRONOS_BEST_SELECTION_METRIC=validation_large_objective
export KRONOS_ADAM_WEIGHT_DECAY=0.1 KRONOS_GRAD_CLIP_NORM=1.0
export KRONOS_CONDITION_MONITOR_INTERVAL_STEPS=100 KRONOS_CONDITION_ABLATION_INTERVAL_SEGMENTS=0
export KRONOS_BATCH_SIZE=64 KRONOS_NUM_WORKERS=2 KRONOS_USE_AMP=1 KRONOS_AMP_DTYPE=bfloat16
export KRONOS_BOOTSTRAP_COMPLETED_SEGMENTS=0 KRONOS_MAX_SEGMENTS_PER_RUN=0
export KRONOS_RESUME_TRAINING="$RESUME" PYTHONUNBUFFERED=1

echo "[$(date -u +%FT%TZ)] mode=$MODE parent=$PARENT_RELEASE-$PARENT_CHECKPOINT split=symbol-holdout-90-10 coverage=2 peak_lr=1e-5 validation=full-only-123982-every-segment" | tee -a "$RUN_ROOT/launcher.log"
cd "$CODE_ROOT/finetune"
"${PYTHON[@]}" -u "$CODE_ROOT/finetune/train_predictor.py" 2>&1 | tee -a "$RUN_ROOT/launcher.log"
"${PYTHON[@]}" "$CODE_ROOT/finetune/export_last_model.py" --repo-root "$CODE_ROOT" --output-root "$OUTPUT_ROOT" 2>&1 | tee -a "$RUN_ROOT/launcher.log"
