"""Start Stage2 uniform-cosine anneal from wc_1e5 C4 last_state.

Coverage 534/534 is done; LR is still glued at 1e-5. This is a new stage:
keep AdamW moments, replace warmup_constant with uniform_cosine, new seed,
new SwanLab run. Does not overwrite the C4 kernel.

Quota is 10h40m. Trainer wall-clock cap is 10h so install/save still fit.
The cosine plan is 267 segments (83571 steps); this chunk stops on time.
"""
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


SOURCE_COMMIT = "862f35f2a1b3a90e24c4a836b289b5a31cf22531"
OUTPUT_NAME = "small_0.1_stage2_c2_best_wc_1e5_cosine"
SWANLAB_RUN_ID = "small_0.1_stage2_c2_best_wc_1e5_cosine"
COVERAGE_SEED = "20260920"
EXPECTED_PARENT_SEGMENT = 27
EXPECTED_PARENT_BEST_SEGMENT = 21
REFINEMENT_SEGMENTS = 267
MAX_SEGMENTS_PER_RUN = 170
MAX_RUNTIME_SECONDS = 36000
EXPECTED_OPTIMIZER_STEPS = 83571
EXPECTED_C4_LAST_SHA = "c285a652419841c05e3b2c7b696cb4e1d86d92bc6cfae0ec00e30f076d407f7b"
EXPECTED_TOKENIZER_SHA = "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"
TOKENIZER_CACHE = Path("/kaggle/working/kronos_tokenizer_base")
TORCH_VERSION = "2.6.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
INPUT = Path("/kaggle/input")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_one(pattern, predicate=lambda path: True):
    matches = [path for path in INPUT.glob(pattern) if predicate(path)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def run(command, cwd=None):
    print({"command": command, "cwd": str(cwd) if cwd else None}, flush=True)
    subprocess.run(command, cwd=cwd, check=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})


def clone_source():
    for attempt in range(1, 4):
        repo = Path(tempfile.mkdtemp(prefix="kronos-wc-1e5-cosine-c1-")) / "repo"
        try:
            run(["git", "init", str(repo)])
            run(["git", "remote", "add", "origin", "https://github.com/luckfu/Kronos.git"], cwd=repo)
            run(["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT], cwd=repo)
            run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo)
            actual = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
            if actual != SOURCE_COMMIT:
                raise RuntimeError(f"Source commit mismatch: {actual}")
            return repo
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * attempt)


def resolve_tokenizer():
    matches = [
        path for path in INPUT.glob("**/Kronos-Tokenizer-base/model.safetensors")
        if "cosine-refinement-c2" in str(path)
    ]
    if len(matches) == 1:
        tokenizer_dir = matches[0].parent
    elif not matches:
        run(["python", "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import snapshot_download
        tokenizer_dir = Path(snapshot_download(
            TOKENIZER_REPO,
            local_dir=str(TOKENIZER_CACHE),
            local_dir_use_symlinks=False,
        ))
    else:
        raise RuntimeError(f"Expected one tokenizer, found {matches}")
    tokenizer_file = tokenizer_dir / "model.safetensors"
    tokenizer_sha = sha256_file(tokenizer_file)
    if tokenizer_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"Tokenizer sha mismatch: {tokenizer_sha}")
    return tokenizer_dir, tokenizer_sha


def find_parent_output():
    state_file = find_one("**/small_0.1_stage2_c2_best_wc_1e5_c4/checkpoints/last_state.pt")
    source_root = state_file.parent.parent
    required = (
        "progress.json",
        "metrics.jsonl",
        "small_v21_manifest.json",
        "checkpoints/last_state.pt",
        "checkpoints/last_model/model.safetensors",
        "checkpoints/best_model/model.safetensors",
        "checkpoints/best_model/best_metric.json",
    )
    missing = [name for name in required if not (source_root / name).is_file()]
    if missing:
        raise RuntimeError(f"C4 output contract is incomplete: {missing}")
    progress = json.loads((source_root / "progress.json").read_text())
    completed = int(progress.get("completed_segments", progress.get("current_segment", 0)))
    if completed != EXPECTED_PARENT_SEGMENT or progress.get("status") != "completed":
        raise RuntimeError(f"C4 is not a completed 27-segment parent: {progress}")
    best = json.loads((source_root / "checkpoints/best_model/best_metric.json").read_text())
    if int(best.get("segment", -1)) != EXPECTED_PARENT_BEST_SEGMENT:
        raise RuntimeError(f"Unexpected C4 best checkpoint: {best}")
    last_sha = sha256_file(source_root / "checkpoints/last_model/model.safetensors")
    if last_sha != EXPECTED_C4_LAST_SHA:
        raise RuntimeError(f"C4 last sha mismatch: {last_sha}")
    return source_root, last_sha


def main():
    os.environ["PYTHONUNBUFFERED"] = "1"
    tokenizer_dir, tokenizer_sha = resolve_tokenizer()

    print(json.dumps({
        "phase": "started",
        "experiment": OUTPUT_NAME,
        "parent": "c4_last_segment_27_global_534",
        "method": "small_0.1_stage2_cosine_refinement",
        "scheduler": "uniform_cosine",
        "peak_lr": "1e-5",
        "min_lr": "1e-6",
        "warmup_ratio": 0,
        "coverage_seed": COVERAGE_SEED,
        "refinement_segments": REFINEMENT_SEGMENTS,
        "max_segments_this_chunk": MAX_SEGMENTS_PER_RUN,
        "max_runtime_seconds": MAX_RUNTIME_SECONDS,
        "expected_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "initialization": "new_stage_preserve_model_and_adamw_replace_scheduler",
        "source_commit": SOURCE_COMMIT,
        "quota_note": "10h40m GPU; trainer cap 10h",
    }, ensure_ascii=False), flush=True)

    repo = clone_source()
    subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], cwd=repo, check=True)
    subprocess.run([
        "pip", "install", "-q", "--force-reinstall", f"torch=={TORCH_VERSION}",
        "--index-url", TORCH_INDEX_URL,
    ], check=True)
    subprocess.run(["pip", "install", "-q", "swanlab"], check=True)

    import torch
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(gpu_names) != 2 or any("T4" not in name for name in gpu_names):
        raise RuntimeError(f"Expected exactly two Tesla T4 GPUs, found {gpu_names}")
    print(json.dumps({
        "phase": "device_ready",
        "torch": torch.__version__,
        "gpu_count": len(gpu_names),
        "devices": gpu_names,
    }), flush=True)

    source_root, last_sha = find_parent_output()
    source_state = source_root / "checkpoints/last_state.pt"
    parent_model = source_root / "checkpoints/last_model"
    state = torch.load(source_state, map_location="cpu", weights_only=False)
    if int(state.get("next_epoch", -1)) != EXPECTED_PARENT_SEGMENT:
        raise RuntimeError(f"C4 state boundary mismatch: {state.get('next_epoch')}")
    if state.get("scheduler_type") != "warmup_constant":
        raise RuntimeError(f"C4 scheduler mismatch: {state.get('scheduler_type')}")
    if not state.get("optimizer", {}).get("state"):
        raise RuntimeError("C4 checkpoint has no AdamW moment state")
    print(json.dumps({
        "phase": "parent_ready",
        "source_root": str(source_root),
        "parent_segment": EXPECTED_PARENT_SEGMENT,
        "parent_best_segment": EXPECTED_PARENT_BEST_SEGMENT,
        "parent_last_sha256": last_sha,
        "tokenizer_sha256": tokenizer_sha,
        "parent_scheduler": state.get("scheduler_type"),
        "parent_optimizer_step": state.get("batch_idx_global"),
        "initialization": "new_stage_preserve_model_and_adamw_replace_scheduler",
    }, ensure_ascii=False), flush=True)
    del state

    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "KRONOS_SMALL_V21_STAGE": "main",
        "KRONOS_SMALL_V21_PARENT_MODEL": str(parent_model),
        "KRONOS_SMALL_V21_TOKENIZER": str(tokenizer_dir),
        "KRONOS_SMALL_V21_DISABLE_AUTO_CONTINUATION": "1",
        "KRONOS_SMALL_V21_OUTPUT_NAME": OUTPUT_NAME,
        "KRONOS_SCHEDULER_TRANSITION_STATE": str(source_state),
        "KRONOS_EPOCHS": str(REFINEMENT_SEGMENTS),
        "KRONOS_REQUIRE_FULL_COVERAGE": "0",
        "KRONOS_MAX_SEGMENTS_PER_RUN": str(MAX_SEGMENTS_PER_RUN),
        "KRONOS_MAX_RUNTIME_SECONDS": str(MAX_RUNTIME_SECONDS),
        "KRONOS_TORCHRUN_NPROC_PER_NODE": "2",
        "KRONOS_BATCH_SIZE": "32",
        "KRONOS_NUM_WORKERS": "2",
        "KRONOS_COVERAGE_SEED": COVERAGE_SEED,
        "KRONOS_COVERAGE_PASSES": "1",
        "KRONOS_SCHEDULER": "uniform_cosine",
        "KRONOS_SCHEDULER_WARMUP_RATIO": "0",
        "KRONOS_PREDICTOR_LEARNING_RATE": "1e-5",
        "KRONOS_CONDITION_LEARNING_RATE": "1e-5",
        "KRONOS_PREDICTOR_WARMUP_START_LR": "1e-5",
        "KRONOS_CONDITION_WARMUP_START_LR": "1e-5",
        "KRONOS_PREDICTOR_MIN_LR": "1e-6",
        "KRONOS_CONDITION_MIN_LR": "1e-6",
        "KRONOS_USE_AMP": "1",
        "KRONOS_AMP_DTYPE": "float16",
        "SWANLAB_API_KEY": "fmEPDGk4IItxgqSZKGLi8",
        "SWANLAB_PROJECT": "finance",
        "SWANLAB_EXPERIMENT_NAME": SWANLAB_RUN_ID,
        "SWANLAB_RUN_ID": SWANLAB_RUN_ID,
    })
    print(json.dumps({
        "phase": "launch_refinement",
        "stage_segments": f"1-{MAX_SEGMENTS_PER_RUN}/{REFINEMENT_SEGMENTS}",
        "scheduler": "uniform_cosine",
        "warmup_steps": 0,
        "peak_lr": 1e-5,
        "min_lr": 1e-6,
        "expected_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
    }, ensure_ascii=False), flush=True)
    subprocess.run(
        ["python", "-u", "finetune/kaggle_kronos_small_v21.py"],
        cwd=repo,
        env=env,
        check=True,
    )
    print(json.dumps({"phase": "finished", "output_name": OUTPUT_NAME}), flush=True)


if __name__ == "__main__":
    main()
