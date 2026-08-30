#!/usr/bin/env bash
# Fresh v1-beta dual-rate training from V6 Segment 568 on GPU 0 only.
set -euo pipefail

MODE="${1:-dry-run}"
if [[ "$MODE" != "baseline" && "$MODE" != "dry-run" && "$MODE" != "diagnostic" && "$MODE" != "full" ]]; then
  echo "usage: $0 {baseline|dry-run|diagnostic|full}" >&2
  exit 2
fi

BASE="/nfsdata/models/2026/kronos-v1-beta"
CODE_ROOT="$BASE/code/Kronos"
DATA_ROOT="$BASE/data/a_share_full_market_v1_beta"
VALIDATION_ROOT="$DATA_ROOT/natural_validation_2025h2_2026h1_v2"
PARENT_ROOT="$BASE/checkpoints/v6_segment568"
TOKENIZER_ROOT="$BASE/models/tokenizer"
RUN_ROOT="$BASE/runs/v6_natural_twospeed_v2_seed100"
OUTPUT_NAME="a_share_v1_beta_v6_natural_twospeed_v2_120d_to_10d"
OUTPUT_ROOT="$RUN_ROOT/outputs/models/$OUTPUT_NAME"
PYTHON=(sudo -n -E "$BASE/env/a800-py312/bin/python")

DATA_SHA="214c375f47e9843b7d836e414199f444ac8cd139e2ef1bb51adf526bfdc6261c"
NATURAL_SHA="f329b1ca2a6f94c373403563d941d05c66de3754e7e3206c002cb5e475be1ac7"
PARENT_SHA="69999253b35afa641d001a5e77fd53be9b5c0beb8444abce1feb173b1f99d1e0"

required=(
  "$CODE_ROOT/finetune/config.py"
  "$CODE_ROOT/finetune/dataset.py"
  "$CODE_ROOT/finetune/train_predictor.py"
  "$CODE_ROOT/finetune/evaluate_fixed_validation_baseline.py"
  "$CODE_ROOT/finetune/export_last_model.py"
  "$DATA_ROOT/data_manifest.json"
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
  echo "V6 Segment 568 model checksum mismatch" >&2
  exit 1
}
[[ "$(sha256sum "$VALIDATION_ROOT/natural_validation_manifest.json" | awk '{print $1}')" == "$NATURAL_SHA" ]] || {
  echo "Natural Validation manifest checksum mismatch" >&2
  exit 1
}

"${PYTHON[@]}" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() >= 1, "no CUDA device found"
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
PY

mkdir -p "$RUN_ROOT"
if [[ "$MODE" == "baseline" ]]; then
  SEGMENTS_THIS_RUN=0
  RESUME=0
elif [[ "$MODE" == "dry-run" ]]; then
  if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "dry-run refuses an existing output directory: $OUTPUT_ROOT" >&2
    exit 1
  fi
  SEGMENTS_THIS_RUN=1
  RESUME=0
elif [[ "$MODE" == "diagnostic" ]]; then
  [[ -f "$OUTPUT_ROOT/checkpoints/last_state.pt" ]] || {
    echo "diagnostic requires the dry-run last_state.pt" >&2
    exit 1
  }
  SEGMENTS_THIS_RUN=49
  RESUME=1
else
  [[ -f "$OUTPUT_ROOT/checkpoints/last_state.pt" ]] || {
    echo "full requires a resumable last_state.pt" >&2
    exit 1
  }
  SEGMENTS_THIS_RUN=1008
  RESUME=1
fi

if [[ ! -f "$RUN_ROOT/experiment_manifest.json" ]]; then
  cat > "$RUN_ROOT/experiment_manifest.json" <<EOF
{
  "stage_id": "v1_beta_v6_natural_twospeed_v2",
  "execution": "single_gpu_a800_offline",
  "logging": {"remote": false, "local_log": "$RUN_ROOT/launcher.log"},
  "parent": {"checkpoint": "$PARENT_ROOT", "model_sha256": "$PARENT_SHA", "lineage": "V6 Segment 568 weights only"},
  "initialization": {"fresh_optimizer": true, "fresh_scheduler": true, "fresh_amp_scaler": true, "sector_emb": "zero", "size_mlp_output": "zero"},
  "validation": {"profile": "natural", "periods": ["2025H2", "2026H1"], "best_objective": "combined forecast plus 0.02 history", "manifest_sha256": "$NATURAL_SHA", "quick_samples": 6000, "large_samples": 24000},
  "training": {"global_segments": 1058, "coverage_passes": 2, "samples_per_segment": 20000, "seed": 100, "gpu": 0, "amp_dtype": "bfloat16", "predictor_peak_lr": 0.000003, "condition_peak_lr": 0.00003, "scheduler": "two_speed"},
  "final_test_policy": "Data after 2026-06-30 is excluded from checkpoint selection and hyperparameter decisions"
}
EOF
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$CODE_ROOT"
export KRONOS_TRAIN_DATA_PATHS="$DATA_ROOT/processed_datasets/train_data.pkl"
export KRONOS_VAL_DATA_PATHS="$DATA_ROOT/processed_datasets/train_data.pkl"
export KRONOS_METADATA_PATH="$DATA_ROOT/asset_metadata.csv"
export KRONOS_DATA_MANIFEST_SHA256="$DATA_SHA"
export KRONOS_FIXED_VALIDATION_MANIFEST_PATH="$VALIDATION_ROOT/natural_validation_manifest.json"
export KRONOS_FIXED_VALIDATION_MANIFEST_SHA256="$NATURAL_SHA"
export KRONOS_EXCLUDE_FIXED_VALIDATION_FROM_TRAINING=1
export KRONOS_VALIDATION_QUICK_SAMPLES=6000
export KRONOS_VALIDATION_LARGE_SAMPLES=24000
export KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS=10
export KRONOS_PREDICTOR_PATH="$PARENT_ROOT"
export KRONOS_TOKENIZER_PATH="$TOKENIZER_ROOT"
export KRONOS_SAVE_PATH="$RUN_ROOT/outputs/models"
export KRONOS_PREDICTOR_SAVE_FOLDER="$OUTPUT_NAME"
export KRONOS_LOOKBACK_WINDOW=120
export KRONOS_PREDICT_WINDOW=10
export KRONOS_USE_SECTOR_FEATURES=1
export KRONOS_NUM_SECTORS=86
export KRONOS_USE_SIZE_FEATURES=0
export KRONOS_NUM_SIZE_BUCKETS=0
export KRONOS_USE_SIZE_PERCENTILE=1
export KRONOS_CONTEXT_LAYER=10
export KRONOS_RESET_SECTOR_EMBEDDING=1
export KRONOS_RESET_SIZE_EMBEDDING=1
export KRONOS_TRAIN_SIGNAL_START=2015-01-01
export KRONOS_TRAIN_SIGNAL_END=2026-07-17
export KRONOS_VAL_SIGNAL_START=2025-07-01
export KRONOS_VAL_SIGNAL_END=2026-06-30
export KRONOS_TRAIN_SAMPLES_PER_SEGMENT=20000
export KRONOS_VALIDATION_SAMPLES=24000
export KRONOS_EPOCHS=1058
export KRONOS_COVERAGE_PASSES=2
export KRONOS_REQUIRE_FULL_COVERAGE=1
export KRONOS_EARLY_STOPPING_PATIENCE=0
export KRONOS_PREDICTOR_LEARNING_RATE=3e-6
export KRONOS_CONDITION_LEARNING_RATE=3e-5
export KRONOS_SCHEDULER=two_speed
export KRONOS_SCHEDULER_MIN_LR=1e-6
export KRONOS_PREDICTOR_MIN_LR=5e-7
export KRONOS_CONDITION_MIN_LR=1e-6
export KRONOS_SCHEDULER_WARMUP_RATIO=0.005
export KRONOS_PREDICTOR_WARMUP_START_LR=1e-6
export KRONOS_CONDITION_WARMUP_START_LR=1e-5
export KRONOS_CONDITION_FAST_DECAY_RATIO=0.075
export KRONOS_CONDITION_FAST_DECAY_LR=1e-5
export KRONOS_PREDICTOR_LOSS_MODE=forecast
export KRONOS_HISTORY_LOSS_WEIGHT=0.02
export KRONOS_TRAINABLE_TRANSFORMER_LAYERS=2
export KRONOS_ADAM_WEIGHT_DECAY=0.1
export KRONOS_GRAD_CLIP_NORM=1.0
export KRONOS_CONDITION_MONITOR_INTERVAL_STEPS=100
export KRONOS_CONDITION_ABLATION_INTERVAL_SEGMENTS=10
export KRONOS_BATCH_SIZE=32
export KRONOS_NUM_WORKERS=2
export KRONOS_USE_AMP=1
export KRONOS_AMP_DTYPE=bfloat16
export KRONOS_BOOTSTRAP_COMPLETED_SEGMENTS=0
export KRONOS_MAX_SEGMENTS_PER_RUN="$SEGMENTS_THIS_RUN"
export KRONOS_RESUME_TRAINING="$RESUME"
export PYTHONUNBUFFERED=1

if [[ "$MODE" == "baseline" ]]; then
  echo "[$(date -u +%FT%TZ)] mode=baseline gpu=0 remote_logging=disabled" | tee -a "$RUN_ROOT/baseline.log"
  cd "$CODE_ROOT/finetune"
  "${PYTHON[@]}" -u "$CODE_ROOT/finetune/evaluate_fixed_validation_baseline.py" \
    --output "$RUN_ROOT/baseline_validation.json" 2>&1 | tee -a "$RUN_ROOT/baseline.log"
  exit 0
fi

echo "[$(date -u +%FT%TZ)] mode=$MODE segments_this_run=$SEGMENTS_THIS_RUN gpu=0 remote_logging=disabled" | tee -a "$RUN_ROOT/launcher.log"
cd "$CODE_ROOT/finetune"
"${PYTHON[@]}" -u "$CODE_ROOT/finetune/train_predictor.py" 2>&1 | tee -a "$RUN_ROOT/launcher.log"

if [[ "$MODE" == "full" ]]; then
  "${PYTHON[@]}" "$CODE_ROOT/finetune/export_last_model.py" \
    --repo-root "$CODE_ROOT" \
    --output-root "$OUTPUT_ROOT" 2>&1 | tee -a "$RUN_ROOT/launcher.log"
fi
