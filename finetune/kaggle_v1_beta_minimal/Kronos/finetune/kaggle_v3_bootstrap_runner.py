"""Run the isolated Beta v3 bootstrap stage on Kaggle."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


STAGE_ID = "beta_v3_base_dynamic_size_path_bootstrap_v1"
DATA_MANIFEST_ID = "17afbeede658c13787043e601aa355717dda4d11719b51fb3ce368fb138e627a"
DATA_MANIFEST_FILE_SHA = "3989bddae6c76e34eb7772590e11b4605215270c178ae30eb5c2261e06fe9177"
VALIDATION_MANIFEST_SHA = "ea29ecdb318adf9789ddd47eb4c5d3df7cdadcbbd60471305241a469d357d184"
ASSET_METADATA_SHA = "697cadd672d53b8fa0a990f6c5b7fba2f88cb26aad88c3b91cfcc3865d0a1e3c"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_data_root(input_root):
    matches = []
    for manifest_path in input_root.glob("**/data_manifest.json"):
        try:
            document = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if document.get("manifest_sha256") != DATA_MANIFEST_ID:
            continue
        if sha256(manifest_path) != DATA_MANIFEST_FILE_SHA:
            raise SystemExit("Beta v3 data manifest file hash mismatch")
        root = manifest_path.parent
        required = (
            root / "processed_datasets/train_data.pkl",
            root / "processed_datasets/val_data.pkl",
            root / "natural_validation_2025h2_2026h1_symbol_holdout_full_v1/natural_validation_manifest.json",
            root / "natural_validation_2025h2_2026h1_symbol_holdout_full_v1/natural_validation_samples.jsonl",
        )
        if all(path.is_file() for path in required):
            matches.append(root)
    if len(matches) != 1:
        raise SystemExit(f"Expected one Beta v3 90/10 dataset, found {matches}")
    root = matches[0]
    validation = root / "natural_validation_2025h2_2026h1_symbol_holdout_full_v1/natural_validation_manifest.json"
    if sha256(validation) != VALIDATION_MANIFEST_SHA:
        raise SystemExit("Beta v3 validation manifest hash mismatch")
    return root


def find_asset_metadata(input_root):
    matches = [
        path for path in input_root.glob("**/asset_metadata.csv")
        if sha256(path) == ASSET_METADATA_SHA
    ]
    if len(matches) != 1:
        raise SystemExit(f"Expected one signed asset_metadata.csv, found {matches}")
    return matches[0]


def download_base_models(runtime):
    subprocess.run([
        "python", "-m", "pip", "install", "-q", "huggingface_hub==0.33.1"
    ], check=True)
    from huggingface_hub import snapshot_download

    predictor = runtime / "base_models/Kronos-base"
    tokenizer = runtime / "base_models/Kronos-Tokenizer-base"
    if not (predictor / "config.json").is_file():
        snapshot_download("NeoQuasar/Kronos-base", local_dir=predictor)
    if not (tokenizer / "config.json").is_file():
        snapshot_download("NeoQuasar/Kronos-Tokenizer-base", local_dir=tokenizer)
    return predictor, tokenizer


def find_continuation(input_root):
    matches = []
    for path in input_root.glob("**/experiment_manifest.json"):
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        root = path.parent
        if (
            document.get("stage_id") == STAGE_ID
            and (root / "checkpoints/last_state.pt").is_file()
            and (root / "checkpoints/last_model/model.safetensors").is_file()
        ):
            matches.append(root)
    if len(matches) > 1:
        raise SystemExit(f"Multiple Beta v3 continuation sources found: {matches}")
    return matches[0] if matches else None


def run(source_root):
    source_root = Path(source_root).resolve()
    input_root = Path(os.getenv("KRONOS_KAGGLE_INPUT_ROOT", "/kaggle/input"))
    runtime = Path(os.getenv("KRONOS_KAGGLE_ROOT", "/kaggle/working/kronos_beta_v3_bootstrap"))
    output_name = "beta_v3_base_dynamic_size_path_bootstrap"
    output_root = runtime / "outputs/models" / output_name
    data_root = find_data_root(input_root)
    metadata_path = find_asset_metadata(input_root)
    continuation = find_continuation(input_root)
    predictor, tokenizer = download_base_models(runtime)

    if continuation is not None:
        if output_root.exists():
            raise SystemExit(f"Continuation output already exists: {output_root}")
        shutil.copytree(continuation, output_root)
        predictor = output_root / "checkpoints/last_model"
    elif output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Fresh Beta v3 output is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    validation_root = data_root / "natural_validation_2025h2_2026h1_symbol_holdout_full_v1"
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(source_root),
        "KRONOS_TRAIN_DATA_PATHS": str(data_root / "processed_datasets/train_data.pkl"),
        "KRONOS_VAL_DATA_PATHS": str(data_root / "processed_datasets/val_data.pkl"),
        "KRONOS_METADATA_PATH": str(metadata_path),
        "KRONOS_DATA_MANIFEST_SHA256": DATA_MANIFEST_ID,
        "KRONOS_FIXED_VALIDATION_MANIFEST_PATH": str(validation_root / "natural_validation_manifest.json"),
        "KRONOS_FIXED_VALIDATION_MANIFEST_SHA256": VALIDATION_MANIFEST_SHA,
        "KRONOS_EXCLUDE_FIXED_VALIDATION_FROM_TRAINING": "1",
        "KRONOS_VALIDATION_FULL_ONLY": "0",
        "KRONOS_VALIDATION_QUICK_SAMPLES": "2000",
        "KRONOS_VALIDATION_LARGE_SAMPLES": "2000",
        "KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS": "1000000",
        "KRONOS_PREDICTOR_PATH": str(predictor),
        "KRONOS_TOKENIZER_PATH": str(tokenizer),
        "KRONOS_SAVE_PATH": str(runtime / "outputs/models"),
        "KRONOS_PREDICTOR_SAVE_FOLDER": output_name,
        "KRONOS_LOOKBACK_WINDOW": "120",
        "KRONOS_PREDICT_WINDOW": "10",
        "KRONOS_USE_SECTOR_FEATURES": "1",
        "KRONOS_NUM_SECTORS": "86",
        "KRONOS_USE_SIZE_FEATURES": "0",
        "KRONOS_NUM_SIZE_BUCKETS": "0",
        "KRONOS_USE_SIZE_PERCENTILE": "0",
        "KRONOS_USE_SIZE_PATH": "1",
        "KRONOS_SIZE_PATH_INPUT_DIM": "4",
        "KRONOS_CONTEXT_LAYER": "10",
        "KRONOS_RESET_SECTOR_EMBEDDING": "1" if continuation is None else "0",
        "KRONOS_RESET_SIZE_EMBEDDING": "1" if continuation is None else "0",
        "KRONOS_TRAIN_SIGNAL_START": "2015-01-01",
        "KRONOS_TRAIN_SIGNAL_END": "2026-07-17",
        "KRONOS_VAL_SIGNAL_START": "2025-07-01",
        "KRONOS_VAL_SIGNAL_END": "2026-06-30",
        "KRONOS_TRAIN_SAMPLES_PER_SEGMENT": "20000",
        "KRONOS_VALIDATION_SAMPLES": "2000",
        "KRONOS_COVERAGE_PASSES": "1",
        "KRONOS_REQUIRE_FULL_COVERAGE": "1",
        "KRONOS_EARLY_STOPPING_PATIENCE": "0",
        "KRONOS_PREDICTOR_LEARNING_RATE": "1e-6",
        "KRONOS_CONDITION_LEARNING_RATE": "1e-5",
        "KRONOS_SCHEDULER": "warmup_cosine",
        "KRONOS_PREDICTOR_WARMUP_START_LR": "1e-7",
        "KRONOS_CONDITION_WARMUP_START_LR": "1e-6",
        "KRONOS_PREDICTOR_MIN_LR": "1e-7",
        "KRONOS_CONDITION_MIN_LR": "1e-6",
        "KRONOS_SCHEDULER_WARMUP_RATIO": "0.01",
        "KRONOS_TRAINABLE_TRANSFORMER_LAYERS": "-1",
        "KRONOS_PREDICTOR_LOSS_MODE": "forecast",
        "KRONOS_HISTORY_LOSS_WEIGHT": "0.02",
        "KRONOS_BEST_SELECTION_METRIC": "objective",
        "KRONOS_ADAM_WEIGHT_DECAY": "0.1",
        "KRONOS_GRAD_CLIP_NORM": "1.0",
        "KRONOS_CONDITION_MONITOR_INTERVAL_STEPS": "100",
        "KRONOS_CONDITION_ABLATION_INTERVAL_SEGMENTS": "0",
        "KRONOS_BATCH_SIZE": "32",
        "KRONOS_NUM_WORKERS": "2",
        "KRONOS_USE_AMP": "1",
        "KRONOS_AMP_DTYPE": "float16",
        "KRONOS_MAX_SEGMENTS_PER_RUN": "10",
        "KRONOS_RESUME_TRAINING": "1" if continuation is not None else "0",
        "PYTHONUNBUFFERED": "1",
    })
    manifest = {
        "stage_id": STAGE_ID,
        "parent": "NeoQuasar/Kronos-base",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "data_manifest_id": DATA_MANIFEST_ID,
        "split": {"train_symbols": 4678, "validation_symbols": 520, "overlap": 0},
        "conditions": {
            "sector": "zero_initialized",
            "dynamic_size_path": "120 observed + 10 causal carry, zero-output initialized",
            "static_size_percentile": False,
        },
        "learning_rates": {"pretrained_trunk": 1e-6, "new_conditions": 1e-5},
        "validation": {"fixed_windows_per_segment": 2000, "full_final_windows": 123982},
        "segments_per_invocation": 10,
        "continuation": continuation is not None,
    }
    (output_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    subprocess.run(
        ["python", "-u", str(source_root / "finetune/train_predictor.py")],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
