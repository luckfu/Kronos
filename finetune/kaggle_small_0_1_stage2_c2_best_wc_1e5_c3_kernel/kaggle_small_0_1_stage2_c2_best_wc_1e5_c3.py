"""Finish the 1e-5 coverage pass from C2 last.

C1+C2 covered slices 0-299 of seed 20260918. Exact resume cannot extend
C2's 200-segment plan, so this chunk starts a 234-segment stage from C2
last_state, keeps AdamW, glues LR at 1e-5, and continues the same
permutation after slice 300. 234 segments is the remainder of one 534
slice coverage pass. Does not overwrite the shared C2 kernel.
"""
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


SOURCE_COMMIT = "4c35f9a781c57b354aeb6febb7f5afc81fe433bb"
OUTPUT_NAME = "small_0.1_stage2_c2_best_wc_1e5_c3"
SWANLAB_RUN_ID = "small_0.1_stage2_c2_best_wc_1e5"
COVERAGE_SEED = "20260918"
COVERAGE_EPOCH_OFFSET = 300
EXPECTED_PARENT_SEGMENT = 200
EXPECTED_PARENT_BEST_SEGMENT = 189
TARGET_SEGMENTS = 234
MAX_SEGMENTS_PER_RUN = 234
MAX_RUNTIME_SECONDS = 40500
FIXED_LR = "1e-5"
EXPECTED_C2_LAST_SHA = "7f0dba2304d26c7dd466d463b8e4ee1597370d237bd68106e73258980f448877"
EXPECTED_TOKENIZER_SHA = "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
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
        repo = Path(tempfile.mkdtemp(prefix="kronos-c2-wc-1e5-c3-")) / "repo"
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


def find_parent_output():
    state_file = find_one("**/small_0.1_stage2_c2_best_wc_1e5_c2/checkpoints/last_state.pt")
    source_root = state_file.parent.parent
    required = (
        "progress.json",
        "summary.json",
        "metrics.jsonl",
        "small_v21_manifest.json",
        "checkpoints/last_state.pt",
        "checkpoints/last_model/model.safetensors",
        "checkpoints/best_model/model.safetensors",
        "checkpoints/best_model/best_metric.json",
    )
    missing = [name for name in required if not (source_root / name).is_file()]
    if missing:
        raise RuntimeError(f"C2 output contract is incomplete: {missing}")
    progress = json.loads((source_root / "progress.json").read_text())
    completed = int(progress.get("completed_segments", progress.get("current_segment", 0)))
    if completed != EXPECTED_PARENT_SEGMENT or progress.get("status") != "completed":
        raise RuntimeError(f"C2 is not a completed 200-segment parent: {progress}")
    best = json.loads((source_root / "checkpoints/best_model/best_metric.json").read_text())
    if int(best.get("segment", -1)) != EXPECTED_PARENT_BEST_SEGMENT:
        raise RuntimeError(f"Unexpected C2 best checkpoint: {best}")
    last_sha = sha256_file(source_root / "checkpoints/last_model/model.safetensors")
    if last_sha != EXPECTED_C2_LAST_SHA:
        raise RuntimeError(f"C2 last sha mismatch: {last_sha}")
    return source_root, last_sha


def main():
    os.environ["PYTHONUNBUFFERED"] = "1"
    in_c2 = lambda path: "cosine-refinement-c2" in str(path)
    tokenizer_file = find_one("**/Kronos-Tokenizer-base/model.safetensors", in_c2)
    tokenizer_dir = tokenizer_file.parent
    tokenizer_sha = sha256_file(tokenizer_file)
    if tokenizer_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"Tokenizer sha mismatch: {tokenizer_sha}")

    print(json.dumps({
        "phase": "started",
        "experiment": OUTPUT_NAME,
        "parent": "c2_last_segment_200",
        "method": "small_0.1_stage2_wc_dual_t4",
        "scheduler": "warmup_constant",
        "fixed_lr": FIXED_LR,
        "warmup_start_lr": FIXED_LR,
        "warmup_ratio": 0,
        "coverage_seed": COVERAGE_SEED,
        "coverage_epoch_offset": COVERAGE_EPOCH_OFFSET,
        "target_segments": TARGET_SEGMENTS,
        "max_segments_this_chunk": MAX_SEGMENTS_PER_RUN,
        "initialization": "c2_last_state_preserve_adamw_new_234_segment_stage",
        "source_commit": SOURCE_COMMIT,
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
        raise RuntimeError(f"C2 state boundary mismatch: {state.get('next_epoch')}")
    if state.get("scheduler_type") != "warmup_constant":
        raise RuntimeError(f"C2 scheduler mismatch: {state.get('scheduler_type')}")
    if not state.get("optimizer", {}).get("state"):
        raise RuntimeError("C2 checkpoint has no AdamW moment state")
    print(json.dumps({
        "phase": "parent_ready",
        "source_root": str(source_root),
        "parent_segment": EXPECTED_PARENT_SEGMENT,
        "parent_best_segment": EXPECTED_PARENT_BEST_SEGMENT,
        "parent_last_sha256": last_sha,
        "tokenizer_sha256": tokenizer_sha,
        "parent_scheduler": state.get("scheduler_type"),
        "parent_optimizer_step": state.get("batch_idx_global"),
        "initialization": "c2_last_state_preserve_adamw_new_234_segment_stage",
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
        "KRONOS_EPOCHS": str(TARGET_SEGMENTS),
        "KRONOS_REQUIRE_FULL_COVERAGE": "0",
        "KRONOS_MAX_SEGMENTS_PER_RUN": str(MAX_SEGMENTS_PER_RUN),
        "KRONOS_MAX_RUNTIME_SECONDS": str(MAX_RUNTIME_SECONDS),
        "KRONOS_TORCHRUN_NPROC_PER_NODE": "2",
        "KRONOS_BATCH_SIZE": "32",
        "KRONOS_NUM_WORKERS": "2",
        "KRONOS_COVERAGE_SEED": COVERAGE_SEED,
        "KRONOS_COVERAGE_PASSES": "1",
        "KRONOS_COVERAGE_EPOCH_OFFSET": str(COVERAGE_EPOCH_OFFSET),
        "KRONOS_SWANLAB_SEGMENT_OFFSET": str(COVERAGE_EPOCH_OFFSET),
        "KRONOS_SCHEDULER": "warmup_constant",
        "KRONOS_SCHEDULER_WARMUP_RATIO": "0",
        "KRONOS_PREDICTOR_LEARNING_RATE": FIXED_LR,
        "KRONOS_CONDITION_LEARNING_RATE": FIXED_LR,
        "KRONOS_PREDICTOR_WARMUP_START_LR": FIXED_LR,
        "KRONOS_CONDITION_WARMUP_START_LR": FIXED_LR,
        "KRONOS_PREDICTOR_MIN_LR": FIXED_LR,
        "KRONOS_CONDITION_MIN_LR": FIXED_LR,
        "KRONOS_USE_AMP": "1",
        "KRONOS_AMP_DTYPE": "float16",
        "SWANLAB_API_KEY": "fmEPDGk4IItxgqSZKGLi8",
        "SWANLAB_PROJECT": "finance",
        "SWANLAB_EXPERIMENT_NAME": SWANLAB_RUN_ID,
        "SWANLAB_RUN_ID": SWANLAB_RUN_ID,
    })
    subprocess.run(
        ["python", "-u", "finetune/kaggle_kronos_small_v21.py"],
        cwd=repo,
        env=env,
        check=True,
    )
    print(json.dumps({"phase": "finished", "output_name": OUTPUT_NAME}), flush=True)


if __name__ == "__main__":
    main()
