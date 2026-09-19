"""Stage2 warmup-constant from C4 best at 2e-5 for 100 segments.

Same trainer/loss/batch as small_0.1_stage2_wc_dual_t4. Fresh AdamW.
New coverage seed. Completes a 100-segment plan; no cosine decay.
Does not continue C4 last_state or the 1e-5 SwanLab run.
"""
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path


SOURCE_COMMIT = "7e5c95e1a3dec8baec971c63579ed6045510d3e8"
OUTPUT_NAME = "small_0.1_stage2_c4_best_wc_2e5"
SWANLAB_RUN_ID = "small_0.1_stage2_c4_best_wc_2e5"
COVERAGE_SEED = "20260919"
COVERAGE_EPOCH_OFFSET = 0
MAX_SEGMENTS_PER_RUN = 100
TARGET_SEGMENTS = 100
MAX_RUNTIME_SECONDS = 39600
FIXED_LR = "2e-5"
EXPECTED_C4_BEST_SHA = "9c5605d08e4f63e223d7b5edf9c6c8d2acad88ddfe022e5b27c098588d743075"
EXPECTED_C4_BEST_SEGMENT = 21
EXPECTED_C4_BEST_FORECAST = 2.2728769779205322
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
        repo = Path(tempfile.mkdtemp(prefix="kronos-c4-wc-2e5-")) / "repo"
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


def find_parent_best():
    best_file = find_one(
        "**/small_0.1_stage2_c2_best_wc_1e5_c4/checkpoints/best_model/model.safetensors"
    )
    parent_model = best_file.parent
    best_metric = json.loads((parent_model / "best_metric.json").read_text())
    if int(best_metric.get("segment", -1)) != EXPECTED_C4_BEST_SEGMENT:
        raise RuntimeError(f"Unexpected C4 best segment: {best_metric}")
    if not math.isclose(
        float(best_metric["forecast_loss"]), EXPECTED_C4_BEST_FORECAST, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(f"Unexpected C4 best forecast CE: {best_metric}")
    best_sha = sha256_file(best_file)
    if best_sha != EXPECTED_C4_BEST_SHA:
        raise RuntimeError(f"C4 best sha mismatch: {best_sha}")
    return parent_model, best_sha


def main():
    os.environ["PYTHONUNBUFFERED"] = "1"
    tokenizer_dir, tokenizer_sha = resolve_tokenizer()
    parent_model, best_sha = find_parent_best()

    print(json.dumps({
        "phase": "started",
        "experiment": OUTPUT_NAME,
        "parent": "c4_best_local21_global528",
        "parent_model": str(parent_model),
        "parent_sha256": best_sha,
        "tokenizer_sha256": tokenizer_sha,
        "method": "small_0.1_stage2_wc_dual_t4",
        "scheduler": "warmup_constant",
        "fixed_lr": FIXED_LR,
        "warmup_start_lr": FIXED_LR,
        "warmup_ratio": 0,
        "coverage_seed": COVERAGE_SEED,
        "coverage_epoch_offset": COVERAGE_EPOCH_OFFSET,
        "target_segments": TARGET_SEGMENTS,
        "max_segments_this_chunk": MAX_SEGMENTS_PER_RUN,
        "initialization": "c4_best_weights_fresh_adamw",
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

    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "KRONOS_SMALL_V21_STAGE": "main",
        "KRONOS_SMALL_V21_PARENT_MODEL": str(parent_model),
        "KRONOS_SMALL_V21_TOKENIZER": str(tokenizer_dir),
        "KRONOS_SMALL_V21_DISABLE_AUTO_CONTINUATION": "1",
        "KRONOS_SMALL_V21_OUTPUT_NAME": OUTPUT_NAME,
        "KRONOS_EPOCHS": str(TARGET_SEGMENTS),
        "KRONOS_REQUIRE_FULL_COVERAGE": "0",
        "KRONOS_MAX_SEGMENTS_PER_RUN": str(MAX_SEGMENTS_PER_RUN),
        "KRONOS_MAX_RUNTIME_SECONDS": str(MAX_RUNTIME_SECONDS),
        "KRONOS_TORCHRUN_NPROC_PER_NODE": "2",
        "KRONOS_BATCH_SIZE": "32",
        "KRONOS_NUM_WORKERS": "2",
        "KRONOS_COVERAGE_SEED": COVERAGE_SEED,
        "KRONOS_COVERAGE_EPOCH_OFFSET": str(COVERAGE_EPOCH_OFFSET),
        "KRONOS_COVERAGE_PASSES": "1",
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
