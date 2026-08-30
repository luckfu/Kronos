#!/usr/bin/env bash
# Short, single-GPU diagnostic: favor the ten-day objective with parent conditioning retained.
set -euo pipefail

BASE="/nfsdata/models/2026/kronos-v1-beta"
CODE_ROOT="$BASE/code/Kronos"
DATA_ROOT="$BASE/data/a_share_full_market_v1_beta"
PARENT_ROOT="$BASE/checkpoints/last1058"
TOKENIZER_ROOT="$BASE/models/tokenizer"
RUN_ROOT="$BASE/runs/last1058_late10_condition_40_seed100"
OUTPUT_ROOT="$RUN_ROOT/outputs/models/a_share_v1_beta_last1058_late10_condition_40"
PYTHON=(sudo -n -E "$BASE/env/a800-py312/bin/python")
DATA_SHA="214c375f47e9843b7d836e414199f444ac8cd139e2ef1bb51adf526bfdc6261c"
NATURAL_SHA="3db79fa5a5966f5d22f0c227b84c0e5a9293cdaaf3f19139ca85f7df427f28b6"
PARENT_SHA="b48fc9421c15ab3faf4f5c0c90974a55229982195542517913a0ea153a0927f8"

for path in "$CODE_ROOT/finetune/train_predictor.py" "$CODE_ROOT/finetune/export_last_model.py" "$DATA_ROOT/data_manifest.json" "$DATA_ROOT/processed_datasets/train_data.pkl" "$DATA_ROOT/processed_datasets/val_data.pkl" "$DATA_ROOT/asset_metadata.csv" "$DATA_ROOT/natural_validation_v1/natural_validation_manifest.json" "$DATA_ROOT/natural_validation_v1/natural_validation_samples.jsonl" "$PARENT_ROOT/model.safetensors" "$PARENT_ROOT/config.json" "$TOKENIZER_ROOT/model.safetensors" "$TOKENIZER_ROOT/config.json"; do
  [[ -f "$path" ]] || { echo "missing required input: $path" >&2; exit 1; }
done
[[ "$(sha256sum "$PARENT_ROOT/model.safetensors" | awk '{print $1}')" == "$PARENT_SHA" ]] || { echo "Last@1058 checksum mismatch" >&2; exit 1; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing existing output: $OUTPUT_ROOT" >&2; exit 1; }
"${PYTHON[@]}" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
print(torch.cuda.get_device_name(0))
PY
mkdir -p "$RUN_ROOT"
cat > "$RUN_ROOT/experiment_manifest.json" <<EOF
{"stage_id":"v1_beta_last1058_late10_condition_40_a800_v1","execution":"single_gpu_a800_offline","swanlab":{"enabled":false},"parent":{"checkpoint":"$PARENT_ROOT","model_sha256":"$PARENT_SHA","completed_segment":1058},"validation":{"profile":"natural_only","manifest_sha256":"$NATURAL_SHA","quick_samples":3000,"large_samples":12000},"training":{"segments":40,"seed":100,"single_gpu":0,"fresh_optimizer":true,"condition_inputs":"enabled","forecast_horizon_weights":[0.5,0.5,0.5,0.5,0.5,1,1,1,1,4]}}
EOF
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$CODE_ROOT" PYTHONUNBUFFERED=1
export KRONOS_TRAIN_DATA_PATHS="$DATA_ROOT/processed_datasets/train_data.pkl" KRONOS_VAL_DATA_PATHS="$DATA_ROOT/processed_datasets/val_data.pkl" KRONOS_METADATA_PATH="$DATA_ROOT/asset_metadata.csv" KRONOS_DATA_MANIFEST_SHA256="$DATA_SHA"
export KRONOS_FIXED_VALIDATION_MANIFEST_PATH="$DATA_ROOT/natural_validation_v1/natural_validation_manifest.json" KRONOS_FIXED_VALIDATION_MANIFEST_SHA256="$NATURAL_SHA" KRONOS_EXCLUDE_FIXED_VALIDATION_FROM_TRAINING=1 KRONOS_VALIDATION_QUICK_SAMPLES=3000 KRONOS_VALIDATION_LARGE_SAMPLES=12000 KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS=10
export KRONOS_PREDICTOR_PATH="$PARENT_ROOT" KRONOS_TOKENIZER_PATH="$TOKENIZER_ROOT" KRONOS_SAVE_PATH="$RUN_ROOT/outputs/models" KRONOS_PREDICTOR_SAVE_FOLDER="a_share_v1_beta_last1058_late10_condition_40"
export KRONOS_LOOKBACK_WINDOW=120 KRONOS_PREDICT_WINDOW=10 KRONOS_USE_SECTOR_FEATURES=1 KRONOS_USE_SIZE_FEATURES=0 KRONOS_USE_SIZE_PERCENTILE=1 KRONOS_DISABLE_CONDITION_INPUTS=0 KRONOS_NUM_SECTORS=86 KRONOS_NUM_SIZE_BUCKETS=0 KRONOS_CONTEXT_LAYER=10
export KRONOS_TRAIN_SIGNAL_START=2015-01-01 KRONOS_TRAIN_SIGNAL_END=2026-07-17 KRONOS_VAL_SIGNAL_START=2026-01-01 KRONOS_VAL_SIGNAL_END=2026-07-17 KRONOS_TRAIN_SAMPLES_PER_SEGMENT=20000 KRONOS_VALIDATION_SAMPLES=12000 KRONOS_EPOCHS=40 KRONOS_COVERAGE_PASSES=1 KRONOS_REQUIRE_FULL_COVERAGE=0 KRONOS_EARLY_STOPPING_PATIENCE=0 KRONOS_MAX_SEGMENTS_PER_RUN=40 KRONOS_RESUME_TRAINING=0
export KRONOS_PREDICTOR_LEARNING_RATE=1e-6 KRONOS_CONDITION_LEARNING_RATE=1e-6 KRONOS_SCHEDULER=uniform_cosine KRONOS_SCHEDULER_MIN_LR=2e-7 KRONOS_PREDICTOR_MIN_LR=2e-7 KRONOS_CONDITION_MIN_LR=2e-7 KRONOS_SCHEDULER_WARMUP_RATIO=0.01 KRONOS_PREDICTOR_WARMUP_START_LR=5e-7 KRONOS_CONDITION_WARMUP_START_LR=5e-7 KRONOS_PREDICTOR_LOSS_MODE=forecast KRONOS_HISTORY_LOSS_WEIGHT=0.02 KRONOS_FORECAST_HORIZON_WEIGHTS="0.5,0.5,0.5,0.5,0.5,1,1,1,1,4" KRONOS_TRAINABLE_TRANSFORMER_LAYERS=2 KRONOS_ADAM_WEIGHT_DECAY=0.1 KRONOS_GRAD_CLIP_NORM=1.0 KRONOS_CONDITION_MONITOR_INTERVAL_STEPS=100 KRONOS_CONDITION_ABLATION_INTERVAL_SEGMENTS=10 KRONOS_BATCH_SIZE=32 KRONOS_NUM_WORKERS=2 KRONOS_USE_AMP=1 KRONOS_BOOTSTRAP_COMPLETED_SEGMENTS=0
echo "[$(date -u +%FT%TZ)] starting natural-only late10 conditioned diagnostic" | tee -a "$RUN_ROOT/launcher.log"
cd "$CODE_ROOT/finetune"
"${PYTHON[@]}" -u "$CODE_ROOT/finetune/train_predictor.py" 2>&1 | tee -a "$RUN_ROOT/launcher.log"
"${PYTHON[@]}" "$CODE_ROOT/finetune/export_last_model.py" --repo-root "$CODE_ROOT" --output-root "$OUTPUT_ROOT" 2>&1 | tee -a "$RUN_ROOT/launcher.log"
