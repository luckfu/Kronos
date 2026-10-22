"""Clean Small training: one coverage plan, exact segment-level chunk relay."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


CHUNK_CONFIG = {}
EXPERIMENT = "small_0.1_clean_bigbatch_20261022"
RUN_ID = "smce2_big"
DATASET = "smmt315/a-share-120d-temporal-symbol-holdout"
PARENT = "luckfu/kronos-small-clean-epoch2-from-c03-e05"
DATA_SHA = "b85ccc45f7cd0e895e607e6e75c7a603427bd62a353a016742a52b2943d652cd"
BEST_SHA = "68feb21bb0d2d6b385e69e95765536b25f2cad07a8efb011cd1bcdea878fda5d"
TOKENIZER_SHA = "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
COVERAGE_SEED = "20261020"
TOTAL_SEGMENTS = 520
MAX_SEGMENTS = 0  # Let elapsed time determine each chunk's completed segments.
BUDGET_SECONDS = 40500  # Original relay budget: 11h15m under the 12h cap.
REQUIRED = (
    "run.log", "metrics.jsonl", "progress.json", "summary.json",
    "small_v21_manifest.json", "experiment_manifest.json",
    "checkpoints/last_state.pt", "checkpoints/last_model/model.safetensors",
    "checkpoints/last_model/config.json", "checkpoints/best_model/model.safetensors",
    "checkpoints/best_model/config.json", "checkpoints/best_model/best_metric.json",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase(name, **values):
    print(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                      "phase": name, **values}), flush=True)


def command(args, cwd=None, env=None):
    phase("command_started", command=args)
    child = subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    finished = threading.Event()

    def heartbeat():
        while not finished.wait(30):
            phase("command_running", command=args[:4])

    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()
    try:
        for line in child.stdout:
            print(line, end="", flush=True)
    finally:
        code = child.wait()
        finished.set()
        monitor.join()
    if code:
        raise subprocess.CalledProcessError(code, args)
    phase("command_finished", command=args[:4])


def find_one(root, pattern):
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}; found {matches}")
    return matches[0]


def prepare_data(input_root, target):
    candidates = [p for p in input_root.glob("**/data_manifest.json")
                  if sha256(p) == DATA_SHA]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one clean data manifest; found {candidates}")
    manifest_path = candidates[0]
    source = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    target.mkdir(parents=True, exist_ok=True)
    (target / "processed_datasets").mkdir(exist_ok=True)
    for entry in manifest["files"]:
        relative = Path(entry["name"])
        path = source / relative
        if not path.is_file():
            path = source / relative.name
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Clean dataset hash mismatch: {path}")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(path.resolve())
    (target / "data_manifest.json").symlink_to(manifest_path.resolve())
    phase("data_ready", dataset=DATASET, manifest_sha256=DATA_SHA,
          stats=manifest["stats"], split=manifest["split"],
          window_contract=manifest["window_contract"])
    return target


def validate_output(root, torch, recipe, parent=False):
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete Output: {missing}")
    manifest = json.loads((root / "experiment_manifest.json").read_text())
    if not parent and manifest["recipe"] != recipe:
        raise RuntimeError("Continuation experiment/data/code recipe changed")
    progress = json.loads((root / "progress.json").read_text())
    summary_document = json.loads((root / "summary.json").read_text())
    summary = summary_document.get("final_result", summary_document)
    state = torch.load(root / "checkpoints/last_state.pt",
                       map_location="cpu", weights_only=False)
    boundary = int(state["next_epoch"])
    completed = int(summary.get("completed_segments", summary.get("current_segment", -1)))
    if (progress["status"] not in {"stopped", "completed"}
            or summary["status"] not in {"stopped", "completed"}
            or int(progress.get("completed_segments", progress.get("current_segment", -1))) != boundary
            or completed != boundary or not 0 < boundary <= TOTAL_SEGMENTS
            or int(state.get("resume_step", -1)) != 0):
        raise RuntimeError("Output does not describe one durable segment boundary")
    if (state["scheduler_type"] != "warmup_constant"
            or state["effective_epochs"] != (520 if parent else TOTAL_SEGMENTS)
            or not state["optimizer"]["state"]
            or "amp_scaler" not in state or not state.get("rng_state")
            or state["scheduler"]["last_epoch"] != state["batch_idx_global"]):
        raise RuntimeError("Optimizer/scheduler/AMP/RNG continuation contract failed")
    if not parent and boundary < TOTAL_SEGMENTS and summary["status"] == "completed":
        raise RuntimeError("Coverage plan ended prematurely")
    best = json.loads((root / "checkpoints/best_model/best_metric.json").read_text())
    phase("output_verified", next_epoch=boundary, next_segment=boundary + 1,
          global_step=state["batch_idx_global"], best=best,
          historical_metrics_lines=sum(1 for s in (root / "metrics.jsonl").read_text().splitlines() if s.strip()))
    del state
    return boundary


def main():
    global BUDGET_SECONDS
    if CHUNK_CONFIG.get("budget_seconds"):
        BUDGET_SECONDS = int(CHUNK_CONFIG["budget_seconds"])
    run_id = CHUNK_CONFIG.get("run_id", RUN_ID)
    batch_size = int(CHUNK_CONFIG.get("batch_size", 64))
    lr = float(CHUNK_CONFIG.get("lr", 2e-5))
    max_segments = int(CHUNK_CONFIG.get("max_segments", MAX_SEGMENTS))
    fresh_probe = bool(CHUNK_CONFIG.get("fresh_probe"))
    owner = str(CHUNK_CONFIG.get("owner", "") or "")
    started = time.monotonic()
    phase("started", chunk=CHUNK_CONFIG["chunk"], experiment=EXPERIMENT,
          owner=owner, max_gpu_seconds=43200, training_budget_seconds=BUDGET_SECONDS)
    input_root = Path("/kaggle/input")
    runtime = Path("/kaggle/working/clean_small_runtime")
    repo = Path(__file__).resolve().parents[2]
    output_name = EXPERIMENT + ("_probe" if fresh_probe else "")
    output = runtime / "outputs/models" / output_name
    entry_log = Path("/kaggle/working/entry.log")
    recipe = {
        "experiment": EXPERIMENT, "run_id": run_id, "dataset": DATASET,
        "data_manifest_sha256": DATA_SHA, "parent_best_sha256": BEST_SHA,
        "source_sha256": CHUNK_CONFIG["source_sha256"],
        "source_commit": CHUNK_CONFIG["source_commit"], "coverage_seed": COVERAGE_SEED,
        "total_segments": TOTAL_SEGMENTS, "samples_per_segment": 20000,
        "coverage_passes": 1, "batch_per_gpu": batch_size, "world_size": 2,
        "amp": "float16", "scheduler": "warmup_constant",
        "lr": lr, "warmup_start_lr": 1e-6, "warmup_ratio": 0.01,
        "loss": "forecast", "history_weight": 0.02, "oos": "evaluation_only",
    }
    command([sys.executable, "-m", "pip", "install", "torch==2.6.0",
             "--only-binary=:all:",
             "--index-url", "https://download.pytorch.org/whl/cu124"])
    command([sys.executable, "-m", "pip", "install",
             "--index-url", "https://pypi.org/simple",
             "--only-binary=numpy,pandas,safetensors,orjson,psutil,protobuf,pyyaml,hf-xet",
             "numpy==2.1.3", "pandas==2.2.3",
             "einops==0.8.1", "huggingface_hub==0.33.1", "safetensors==0.6.2",
             "tqdm==4.67.1", "swanlab==0.10.1", "psutil", "pynvml"])
    import torch
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(devices) != 2 or any("T4" not in name for name in devices):
        raise RuntimeError(f"Expected dual T4; found {devices}")
    phase("device_ready", devices=devices, torch=torch.__version__, cuda=torch.version.cuda)
    token = os.environ.get("SWANLAB_API_KEY", "").strip()
    if not token:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("SWANLAB_API_KEY")
    if not token:
        raise RuntimeError("Missing SwanLab credential")
    os.environ["SWANLAB_API_KEY"] = token
    for key in ("SWANLAB_PROJECT", "SWANLAB_WORKSPACE", "SWANLAB_EXPERIMENT_NAME"):
        os.environ.pop(key, None)
    phase("dashboard_deferred", project="finance", run_id=run_id, experiment=EXPERIMENT,
          note="single SwandLab run is created by the trainer")
    data_root = prepare_data(input_root, runtime / "data")
    tokenizer = find_one(input_root, "**/Kronos-Tokenizer-base/model.safetensors").parent
    if sha256(tokenizer / "model.safetensors") != TOKENIZER_SHA:
        raise RuntimeError("Tokenizer hash mismatch")
    if CHUNK_CONFIG["chunk"] == 1:
        parent = find_one(input_root, "**/small_0.1_clean_epoch2_from_c03_20261020/checkpoints/best_model/model.safetensors").parent
        recipe["parent_best_sha256"] = sha256(parent / "model.safetensors")
        phase("parent_model", path=str(parent), sha256=recipe["parent_best_sha256"])
        continuation = None
    else:
        state = find_one(input_root, "**/last_state.pt")
        continuation = state.parent.parent
        # Keep the original starting-parent SHA across chunks.  The module
        # constant is only a fallback for chunk 1, where it is recomputed from
        # the mounted parent weights.
        parent_manifest = json.loads((continuation / "experiment_manifest.json").read_text())
        recipe["parent_best_sha256"] = parent_manifest["recipe"]["parent_best_sha256"]
        boundary = validate_output(continuation, torch, recipe, parent=True)
        if boundary == TOTAL_SEGMENTS:
            raise RuntimeError("Parent has already completed the coverage plan")
        parent = continuation / "checkpoints/last_model"
        if fresh_probe:
            continuation = None
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("KRONOS_")}
    env.update({
        "PYTHONUNBUFFERED": "1", "KRONOS_SMALL_V21_STAGE": "main",
        "KRONOS_SMALL_V21_RUNTIME": str(runtime),
        "KRONOS_SMALL_V21_DATA_ROOT": str(data_root),
        "KRONOS_SMALL_V21_PARENT_MODEL": str(parent),
        "KRONOS_SMALL_V21_TOKENIZER": str(tokenizer),
        "KRONOS_SMALL_V21_OUTPUT_NAME": output_name,
        "KRONOS_SMALL_V21_DISABLE_AUTO_CONTINUATION": "0" if continuation else "1",
        "KRONOS_EPOCHS": "1", "KRONOS_REQUIRE_FULL_COVERAGE": "1",
        "KRONOS_COVERAGE_PASSES": "1", "KRONOS_COVERAGE_SEED": COVERAGE_SEED,
        "KRONOS_MAX_SEGMENTS_PER_RUN": str(max_segments),
        "KRONOS_MAX_RUNTIME_SECONDS": str(BUDGET_SECONDS),
        "KRONOS_TORCHRUN_NPROC_PER_NODE": "2", "KRONOS_BATCH_SIZE": str(batch_size),
        "KRONOS_NUM_WORKERS": "2", "KRONOS_SCHEDULER": "warmup_constant",
        "KRONOS_SCHEDULER_WARMUP_RATIO": "0.01",
        "KRONOS_PREDICTOR_LEARNING_RATE": str(lr), "KRONOS_CONDITION_LEARNING_RATE": str(lr),
        "KRONOS_USE_AMP": "1", "KRONOS_AMP_DTYPE": "float16",
        "KRONOS_REQUIRE_SWANLAB": "1", "SWANLAB_PROJECT": "finance",
        "SWANLAB_WORKSPACE": "roc_fu", "SWANLAB_EXPERIMENT_NAME": EXPERIMENT,
        "SWANLAB_RUN_ID": run_id,
    })
    if continuation:
        env["KRONOS_SMALL_V21_CONTINUATION_ROOT"] = str(continuation)
    # Charge setup time to the same 11h15m budget.
    env["KRONOS_MAX_RUNTIME_SECONDS"] = str(max(60, BUDGET_SECONDS - int(time.monotonic() - started)))
    phase("training_started", recipe=recipe, continuation=str(continuation),
          parent_model=str(parent), environment={k: v for k, v in env.items() if k.startswith("KRONOS_")})
    command([sys.executable, "-u", "finetune/kaggle_kronos_small_v21.py"], cwd=repo, env=env)
    (output / "experiment_manifest.json").write_text(json.dumps({
        "recipe": recipe, "chunk": CHUNK_CONFIG["chunk"],
        "parent_kernel": CHUNK_CONFIG["parent_kernel"],
        "max_runtime_seconds": env["KRONOS_MAX_RUNTIME_SECONDS"],
    }, indent=2) + "\n")
    if entry_log.is_file():
        with (output / "run.log").open("a") as handle:
            handle.write(entry_log.read_text())
    validate_output(output, torch, recipe)
    # Carry the tokenizer so later chunks mount only the previous Output.
    tokenizer_target = runtime / "models/Kronos-Tokenizer-base"
    if not tokenizer_target.exists():
        shutil.copytree(tokenizer, tokenizer_target)
    phase("finished", output=str(output), elapsed_seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
