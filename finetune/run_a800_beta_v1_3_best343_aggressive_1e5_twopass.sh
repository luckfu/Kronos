#!/usr/bin/env bash
# Two-coverage full-Predictor experiment from Beta v1.3 Best@343.
set -euo pipefail

MODE="${1:-start}"
if [[ "$MODE" != "start" && "$MODE" != "resume" ]]; then
  echo "usage: $0 {start|resume}" >&2
  exit 2
fi

BASE="/nfsdata/models/2026/kronos-v1-beta"
CODE_ROOT="$BASE/code/Kronos"
DATA_ROOT="$BASE/data/a_share_full_market_v1_beta"
VALIDATION_ROOT="$DATA_ROOT/natural_validation_2025h2_2026h1_v2"
PARENT_RUN="$BASE/runs/beta_v1_2_best871_full_every_segment_onecycle_1pass_seed100"
PARENT_OUTPUT="$PARENT_RUN/outputs/models/a_share_beta_v1_2_best871_full_every_segment_onecycle_1pass_120d_to_10d"
PARENT_ROOT="$PARENT_OUTPUT/checkpoints/best_model"
TOKENIZER_ROOT="$BASE/models/tokenizer"
RUN_ROOT="$BASE/runs/beta_v1_3_best343_aggressive_1e5_twopass_seed100"
OUTPUT_NAME="a_share_beta_v1_3_best343_aggressive_1e5_twopass_120d_to_10d"
OUTPUT_ROOT="$RUN_ROOT/outputs/models/$OUTPUT_NAME"
PYTHON=(sudo -n -E "$BASE/env/a800-py312/bin/python")

DATA_SHA="214c375f47e9843b7d836e414199f444ac8cd139e2ef1bb51adf526bfdc6261c"
NATURAL_SHA="f329b1ca2a6f94c373403563d941d05c66de3754e7e3206c002cb5e475be1ac7"
PARENT_SHA="b2e90710d7619f3ae3a1b488726b2885adff11dcd45834f2560766f49bbc20f3"

required=(
  "$CODE_ROOT/finetune/config.py"
  "$CODE_ROOT/finetune/dataset.py"
  "$CODE_ROOT/finetune/train_predictor.py"
  "$CODE_ROOT/finetune/export_last_model.py"
  "$DATA_ROOT/processed_datasets/train_data.pkl"
  "$DATA_ROOT/asset_metadata.csv"
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

[[ "$(sha256sum "$PARENT_ROOT/model.safetensors" | awk '{print $1}')" == "$PARENT_SHA" ]] || {
  echo "Beta v1.3 Best@343 checksum mismatch" >&2
  exit 1
}
[[ "$(sha256sum "$VALIDATION_ROOT/natural_validation_manifest.json" | awk '{print $1}')" == "$NATURAL_SHA" ]] || {
  echo "Natural Validation manifest checksum mismatch" >&2
  exit 1
}

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
  "stage_id": "beta_v1_3_best343_aggressive_1e5_twopass",
  "purpose": "two full-market coverage passes at aggressive 1e-5 peak learning rate",
  "parent": {"release": "Beta v1.3", "checkpoint": "Best@343", "path": "$PARENT_ROOT", "model_sha256": "$PARENT_SHA", "validation_full_loss": 2.4104652404785156},
  "initialization": {"model": "weights_only", "optimizer": "fresh", "scheduler": "one_cycle", "conditioning": "preserved"},
  "training": {"global_segments": 1056, "coverage_passes": 2, "samples_per_segment": 20000, "batch_size": 64, "seed": 100, "gpu": 0, "amp_dtype": "bfloat16", "predictor_learning_rate": 1e-5, "condition_learning_rate": 1e-5, "trainable_predictor": "all"},
  "validation": {"mode": "full_only", "manifest_sha256": "$NATURAL_SHA", "full_samples": 24000, "full_interval_segments": 1, "quick_validation": "disabled"},
  "best_selection": {"dashboard_metric": "validation/full/loss", "implementation_metric": "validation_large_objective", "evaluated_every_segment": true},
  "diagnostic_evidence": {"run": "beta_v1_2_best343_aggressive_1e5_50seg_seed100", "best_segment": 43, "best_validation_full_loss": 2.404633045196533}
}
EOF
fi

export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$CODE_ROOT"
export KRONOS_TRAIN_DATA_PATHS="$DATA_ROOT/processed_datasets/train_data.pkl"
export KRONOS_VAL_DATA_PATHS="$DATA_ROOT/processed_datasets/train_data.pkl"
export KRONOS_METADATA_PATH="$DATA_ROOT/asset_metadata.csv"
export KRONOS_DATA_MANIFEST_SHA256="$DATA_SHA"
export KRONOS_FIXED_VALIDATION_MANIFEST_PATH="$VALIDATION_ROOT/natural_validation_manifest.json"
export KRONOS_FIXED_VALIDATION_MANIFEST_SHA256="$NATURAL_SHA"
export KRONOS_EXCLUDE_FIXED_VALIDATION_FROM_TRAINING=1
export KRONOS_VALIDATION_FULL_ONLY=1
export KRONOS_VALIDATION_QUICK_SAMPLES=6000
export KRONOS_VALIDATION_LARGE_SAMPLES=24000
export KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS=1
export KRONOS_PREDICTOR_PATH="$PARENT_ROOT" KRONOS_TOKENIZER_PATH="$TOKENIZER_ROOT"
export KRONOS_SAVE_PATH="$RUN_ROOT/outputs/models" KRONOS_PREDICTOR_SAVE_FOLDER="$OUTPUT_NAME"
export KRONOS_LOOKBACK_WINDOW=120 KRONOS_PREDICT_WINDOW=10
export KRONOS_USE_SECTOR_FEATURES=1 KRONOS_NUM_SECTORS=86
export KRONOS_USE_SIZE_FEATURES=0 KRONOS_NUM_SIZE_BUCKETS=0 KRONOS_USE_SIZE_PERCENTILE=1
export KRONOS_CONTEXT_LAYER=10 KRONOS_RESET_SECTOR_EMBEDDING=0 KRONOS_RESET_SIZE_EMBEDDING=0
export KRONOS_TRAIN_SIGNAL_START=2015-01-01 KRONOS_TRAIN_SIGNAL_END=2026-07-17
export KRONOS_VAL_SIGNAL_START=2025-07-01 KRONOS_VAL_SIGNAL_END=2026-06-30
export KRONOS_TRAIN_SAMPLES_PER_SEGMENT=20000 KRONOS_VALIDATION_SAMPLES=24000
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

echo "[$(date -u +%FT%TZ)] mode=$MODE parent=Beta-v1.3-Best@343 coverage=2 peak_lr=1e-5 validation=full-only-24k-every-segment best=validation/full/loss" | tee -a "$RUN_ROOT/launcher.log"
cd "$CODE_ROOT/finetune"
"${PYTHON[@]}" -u "$CODE_ROOT/finetune/train_predictor.py" 2>&1 | tee -a "$RUN_ROOT/launcher.log"
"${PYTHON[@]}" "$CODE_ROOT/finetune/export_last_model.py" --repo-root "$CODE_ROOT" --output-root "$OUTPUT_ROOT" 2>&1 | tee -a "$RUN_ROOT/launcher.log"
