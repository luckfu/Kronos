"""Run the isolated Beta v3 bootstrap stage on Kaggle."""

import hashlib
import json
import os
import pickle
import re
import shutil
import signal
import subprocess
from pathlib import Path


STAGE_ID = "beta_v3_base_dynamic_size_path_bootstrap_v1"
DATA_MANIFEST_ID = "17afbeede658c13787043e601aa355717dda4d11719b51fb3ce368fb138e627a"
DATA_MANIFEST_FILE_SHA = "3989bddae6c76e34eb7772590e11b4605215270c178ae30eb5c2261e06fe9177"
VALIDATION_MANIFEST_SHA = "ea29ecdb318adf9789ddd47eb4c5d3df7cdadcbbd60471305241a469d357d184"
ASSET_METADATA_SHA = "697cadd672d53b8fa0a990f6c5b7fba2f88cb26aad88c3b91cfcc3865d0a1e3c"
SOURCE_DATA_MANIFEST_ID = "214c375f47e9843b7d836e414199f444ac8cd139e2ef1bb51adf526bfdc6261c"
SOURCE_PANEL_SHA = "7d026170517db9efe6ac21f2dccd1b9b549a03f14ff4f6b77931b696b2f86a35"
V3_TRAIN_SHA = "b2f2a861f651321efd38761c65ffcff4d14290cd580325298c4b9a7bc915f832"
V3_VAL_SHA = "748a9205714ee9714525872e65432063edd26d6afbef05391919e8d9d8811115"
SWANLAB_RUN_ID = "kronos-beta-v3-base-dynamic-size-bootstrap"
SWANLAB_STEPS_PER_SEGMENT = 1000
TRAIN_WINDOWS = 9_457_646
WINDOWS_PER_SEGMENT = 20_000

TRAIN_LOG_RE = re.compile(
    r"Segment (\d+)/(\d+), Step (\d+)/(\d+).*?"
    r"Adaptation LR ([0-9.eE+-]+), Condition LR ([0-9.eE+-]+), "
    r"Loss: ([0-9.eE+-]+), Forecast: ([0-9.eE+-]+), "
    r"History: ([0-9.eE+-]+)"
)
VALIDATION_LOG_RE = re.compile(
    r"Validation Forecast/History/Full: "
    r"([0-9.eE+-]+) / ([0-9.eE+-]+) / ([0-9.eE+-]+)"
)
CONDITION_MONITOR_PREFIX = "Condition Monitor JSON: "


class NumpyCompatibleUnpickler(pickle.Unpickler):
    """Read NumPy 2 panel pickles in either NumPy 1 or NumPy 2 runtimes."""

    def find_class(self, module, name):
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_compatible_torch():
    """Install the proven P100-compatible wheel only when the current wheel lacks sm_60."""
    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not gpu_name:
        raise SystemExit("Kaggle GPU is not available")
    needs_compatible_wheel = False
    if "P100" in gpu_name:
        arch_check = subprocess.run([
            "python", "-c",
            "import torch; raise SystemExit(0 if "
            "'sm_60' in torch.cuda.get_arch_list() else 1)",
        ], check=False)
        needs_compatible_wheel = arch_check.returncode != 0
    if needs_compatible_wheel:
        print(
            "P100 detected but the preinstalled PyTorch lacks sm_60; "
            "installing the compatible CUDA 12.1 wheel.",
            flush=True,
        )
        subprocess.run([
            "python", "-m", "pip", "install", "-q", "--index-url",
            "https://download.pytorch.org/whl/cu121", "torch==2.5.1",
        ], check=True)
    print(f"GPU runtime ready: {gpu_name}", flush=True)
    return gpu_name


def swanlab_global_step(segment, step, total_steps):
    total_steps = max(1, int(total_steps))
    normalized = round(int(step) * SWANLAB_STEPS_PER_SEGMENT / total_steps)
    normalized = min(SWANLAB_STEPS_PER_SEGMENT, max(0, normalized))
    return max(0, (int(segment) - 1) * SWANLAB_STEPS_PER_SEGMENT + normalized)


def parse_train_line(line):
    match = TRAIN_LOG_RE.search(line)
    if match is None:
        return None
    values = match.groups()
    return {
        "segment": int(values[0]),
        "total_segments": int(values[1]),
        "step": int(values[2]),
        "total_steps": int(values[3]),
        "adaptation_learning_rate": float(values[4]),
        "condition_learning_rate": float(values[5]),
        "objective_loss": float(values[6]),
        "forecast_loss": float(values[7]),
        "history_loss": float(values[8]),
    }


def parse_validation_line(line):
    match = VALIDATION_LOG_RE.search(line)
    if match is None:
        return None
    forecast, history, full = map(float, match.groups())
    return {
        "objective_loss": forecast + 0.02 * history,
        "forecast_loss": forecast,
        "history_loss": history,
        "full_loss": full,
    }


def parse_condition_monitor_line(line):
    marker = line.find(CONDITION_MONITOR_PREFIX)
    if marker < 0:
        return None
    payload = json.loads(line[marker + len(CONDITION_MONITOR_PREFIX):])
    if not isinstance(payload, dict) or not all(
        isinstance(value, (int, float)) for value in payload.values()
    ):
        raise ValueError("Condition monitor payload must contain numeric values")
    return payload


def initialize_swanlab(manifest):
    subprocess.run(["python", "-m", "pip", "install", "-q", "swanlab"], check=True)
    import swanlab

    api_key = os.getenv("SWANLAB_API_KEY", "").strip()
    if not api_key:
        try:
            from kaggle_secrets import UserSecretsClient

            api_key = UserSecretsClient().get_secret("SWANLAB_API_KEY").strip()
        except Exception:
            print(
                "SwanLab dashboard disabled: add Kaggle Secret "
                "SWANLAB_API_KEY to enable live metrics; training continues with local logs.",
                flush=True,
            )
            return None
    swanlab.login(api_key=api_key)
    run = swanlab.init(
        project=os.getenv("SWANLAB_PROJECT", "finance"),
        workspace=os.getenv("SWANLAB_WORKSPACE", "roc_fu"),
        experiment_name=os.getenv(
            "SWANLAB_EXPERIMENT_NAME", "Kronos Beta v3 dynamic size bootstrap"
        ),
        id=os.getenv("SWANLAB_RUN_ID", SWANLAB_RUN_ID),
        resume="allow",
        tags=["kronos", "beta-v3", "dynamic-size-path", "base-bootstrap"],
        config=manifest,
    )
    print(f"SwanLab dashboard initialized: run_id={SWANLAB_RUN_ID}", flush=True)
    return run


def swanlab_log(run, payload, step):
    try:
        run.log(payload, step=int(step))
        return True
    except Exception as exc:
        print(
            f"SwanLab logging disabled after error: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return False


def stream_training(command, env, run):
    child = None

    def forward_signal(signum, _frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)
    child = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    latest_segment = latest_step = 0
    latest_total_steps = 1
    dashboard_active = run is not None
    assert child.stdout is not None
    for line in child.stdout:
        print(line, end="", flush=True)
        train = parse_train_line(line)
        if train is not None:
            latest_segment = train["segment"]
            latest_step = train["step"]
            latest_total_steps = train["total_steps"]
            completed = min(
                TRAIN_WINDOWS,
                (latest_segment - 1) * WINDOWS_PER_SEGMENT
                + round(WINDOWS_PER_SEGMENT * latest_step / latest_total_steps),
            )
            step = swanlab_global_step(
                latest_segment, latest_step, latest_total_steps
            )
            if dashboard_active:
                dashboard_active = swanlab_log(run, {
                    "train/objective_loss": train["objective_loss"],
                    "train/forecast_loss": train["forecast_loss"],
                    "train/history_loss": train["history_loss"],
                    "learning_rate/trunk": train["adaptation_learning_rate"],
                    "learning_rate/condition": train["condition_learning_rate"],
                    "progress/segment": latest_segment,
                    "progress/total_segments": train["total_segments"],
                    "progress/segment_fraction": latest_step / latest_total_steps,
                    "progress/windows_covered": completed,
                    "progress/coverage_fraction": completed / TRAIN_WINDOWS,
                }, step)
        monitor = parse_condition_monitor_line(line)
        if monitor is not None and latest_segment and dashboard_active:
            dashboard_active = swanlab_log(run, {
                f"monitor/{key}": value
                for key, value in monitor.items()
                if not key.endswith("parameter_count")
            }, swanlab_global_step(latest_segment, latest_step, latest_total_steps))
        validation = parse_validation_line(line)
        if validation is not None and latest_segment and dashboard_active:
            dashboard_active = swanlab_log(run, {
                f"validation/{key}": value for key, value in validation.items()
            } | {"progress/validated_segment": latest_segment}, swanlab_global_step(
                latest_segment, latest_total_steps, latest_total_steps
            ))
    return_code = child.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


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
    if not matches:
        return None
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
    if not matches:
        raise SystemExit("Signed asset_metadata.csv was not found")
    return matches[0]


def find_source_panel(input_root):
    matches = []
    for manifest_path in input_root.glob("**/data_manifest.json"):
        try:
            document = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if document.get("manifest_sha256") != SOURCE_DATA_MANIFEST_ID:
            continue
        panel_path = manifest_path.parent / "processed_datasets/train_data.pkl"
        if panel_path.is_file() and sha256(panel_path) == SOURCE_PANEL_SHA:
            matches.append(panel_path)
    if len(matches) != 1:
        raise SystemExit(f"Expected one signed full-market source panel, found {matches}")
    return matches[0]


def materialize_v3_data(source_root, input_root, runtime):
    """Rebuild the signed 90/10 files in an isolated Kaggle working directory."""
    contract = Path(os.getenv(
        "KRONOS_V3_CONTRACT_ROOT",
        str(source_root / "finetune/v3_data_contract"),
    ))
    output = runtime / "data/a_share_full_market_v1_beta_symbol_holdout_90_10_v1"
    train_path = output / "processed_datasets/train_data.pkl"
    val_path = output / "processed_datasets/val_data.pkl"
    if train_path.is_file() and val_path.is_file():
        if (output / "data_manifest.json").is_file():
            return output
        raise SystemExit("Existing materialized Beta v3 data has no runtime manifest")

    import pandas as pd

    source_panel_path = find_source_panel(input_root)
    split = pd.read_csv(contract / "symbol_split.csv", dtype={"symbol": str})
    train_symbols = sorted(split.loc[split["split"] == "train", "symbol"])
    val_symbols = sorted(split.loc[split["split"] == "validation", "symbol"])
    if len(train_symbols) != 4678 or len(val_symbols) != 520:
        raise SystemExit("Embedded Beta v3 symbol split has unexpected counts")
    with source_panel_path.open("rb") as handle:
        panel = NumpyCompatibleUnpickler(handle).load()
    if set(train_symbols) & set(val_symbols) or set(panel) != set(train_symbols) | set(val_symbols):
        raise SystemExit("Embedded Beta v3 symbol split does not partition the source panel")

    train_path.parent.mkdir(parents=True, exist_ok=True)
    for path, symbols in ((train_path, train_symbols), (val_path, val_symbols)):
        with path.open("wb") as handle:
            pickle.dump(
                {symbol: panel[symbol] for symbol in symbols},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    del panel
    train_sha = sha256(train_path)
    val_sha = sha256(val_path)
    split_sha = sha256(contract / "symbol_split.csv")
    manifest_id = hashlib.sha256(
        json.dumps({
            "source_manifest": DATA_MANIFEST_ID,
            "split_sha256": split_sha,
            "train_sha256": train_sha,
            "val_sha256": val_sha,
        }, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = json.loads((contract / "data_manifest.json").read_text())
    manifest["manifest_sha256"] = manifest_id
    manifest["runtime_materialization"] = {
        "source_contract_manifest_sha256": DATA_MANIFEST_ID,
        "source_contract_train_sha256": V3_TRAIN_SHA,
        "source_contract_val_sha256": V3_VAL_SHA,
        "symbol_split_sha256": split_sha,
    }
    for item in manifest.get("files", []):
        if item.get("name") == "processed_datasets/train_data.pkl":
            item.update(size=train_path.stat().st_size, sha256=train_sha)
        elif item.get("name") == "processed_datasets/val_data.pkl":
            item.update(size=val_path.stat().st_size, sha256=val_sha)
    (output / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    shutil.copy2(contract / "symbol_split.csv", output / "symbol_split.csv")
    validation_output = output / "natural_validation_2025h2_2026h1_symbol_holdout_full_v1"
    validation_output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        contract / "natural_validation_samples.jsonl",
        validation_output / "natural_validation_samples.jsonl",
    )
    validation_manifest = json.loads(
        (contract / "natural_validation_manifest.json").read_text()
    )
    validation_manifest["source"]["data_manifest_sha256"] = manifest_id
    validation_manifest["source"]["val_data_sha256"] = val_sha
    validation_manifest["runtime_materialization"] = {
        "source_contract_manifest_file_sha256": VALIDATION_MANIFEST_SHA,
    }
    (validation_output / "natural_validation_manifest.json").write_text(
        json.dumps(validation_manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return output


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
    gpu_name = ensure_compatible_torch()
    input_root = Path(os.getenv("KRONOS_KAGGLE_INPUT_ROOT", "/kaggle/input"))
    runtime = Path(os.getenv("KRONOS_KAGGLE_ROOT", "/kaggle/working/kronos_beta_v3_bootstrap"))
    output_name = "beta_v3_base_dynamic_size_path_bootstrap"
    output_root = runtime / "outputs/models" / output_name
    data_root = find_data_root(input_root)
    if data_root is None:
        data_root = materialize_v3_data(source_root, input_root, runtime)
    data_manifest_id = json.loads(
        (data_root / "data_manifest.json").read_text()
    )["manifest_sha256"]
    validation_manifest_sha = sha256(
        data_root / "natural_validation_2025h2_2026h1_symbol_holdout_full_v1/natural_validation_manifest.json"
    )
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
        "KRONOS_DATA_MANIFEST_SHA256": data_manifest_id,
        "KRONOS_FIXED_VALIDATION_MANIFEST_PATH": str(validation_root / "natural_validation_manifest.json"),
        "KRONOS_FIXED_VALIDATION_MANIFEST_SHA256": validation_manifest_sha,
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
        "KRONOS_MAX_SEGMENTS_PER_RUN": "120",
        "KRONOS_RESUME_TRAINING": "1" if continuation is not None else "0",
        "PYTHONUNBUFFERED": "1",
    })
    manifest = {
        "stage_id": STAGE_ID,
        "parent": "NeoQuasar/Kronos-base",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "data_manifest_id": data_manifest_id,
        "source_data_contract_id": DATA_MANIFEST_ID,
        "validation_manifest_sha256": validation_manifest_sha,
        "split": {"train_symbols": 4678, "validation_symbols": 520, "overlap": 0},
        "conditions": {
            "sector": "zero_initialized",
            "dynamic_size_path": "120 observed + 10 causal carry, zero-output initialized",
            "static_size_percentile": False,
        },
        "learning_rates": {"pretrained_trunk": 1e-6, "new_conditions": 1e-5},
        "validation": {"fixed_windows_per_segment": 2000, "full_final_windows": 123982},
        "segments_per_invocation": 120,
        "continuation": continuation is not None,
        "gpu": gpu_name,
        "dashboard": {
            "provider": "SwanLab",
            "run_id": SWANLAB_RUN_ID,
            "resume": "allow",
        },
    }
    (output_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    swanlab_run = initialize_swanlab(manifest)
    try:
        stream_training(
            ["python", "-u", str(source_root / "finetune/train_predictor.py")],
            env,
            swanlab_run,
        )
    finally:
        finish = getattr(swanlab_run, "finish", None)
        if callable(finish):
            finish()


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
