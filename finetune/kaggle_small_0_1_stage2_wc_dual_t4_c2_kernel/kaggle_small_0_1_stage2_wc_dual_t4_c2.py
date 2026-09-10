"""Continue the dual-T4 Warmup-Constant run until the next safe time boundary."""

import json
import os
import shutil
import subprocess
from pathlib import Path


MAX_SEGMENTS_PER_RUN = 250
MAX_RUNTIME_SECONDS = 40500  # 11h15m, leaving margin under Kaggle's 12h cap.
OUTPUT_NAME = "small_0.1_stage2_wc_dual_t4_c2"
COVERAGE_SEED = "20260911"
PARENT_KERNEL = "smmt315/kronos-small-0-1-stage2-wc-dual-t4"
TORCH_VERSION = "2.6.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"


def clone_repo(repo: Path) -> str:
    shutil.rmtree(repo, ignore_errors=True)
    subprocess.run(
        ["git", "clone", "--depth", "3", "--branch", "master",
         "https://github.com/luckfu/Kronos.git", str(repo)],
        check=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def find_parent_model() -> Path:
    matches = sorted(Path("/kaggle/input").glob(
        "**/small_0.1_stage2_wc_dual_t4/checkpoints/last_model/model.safetensors"
    ))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one dual-T4 parent last_model, found {len(matches)}: {matches}")
    model_dir = matches[0].parent
    progress_path = model_dir.parent.parent / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        completed = int(progress.get("completed_segments", progress.get("current_segment", 0)))
        if completed < 130:
            raise RuntimeError(f"Parent dual-T4 run is incomplete: completed_segments={completed}")
    return model_dir


print({
    "phase": "started",
    "experiment": "small_0.1_stage2_wc_dual_t4",
    "parent_kernel": PARENT_KERNEL,
    "coverage_seed": COVERAGE_SEED,
    "batch_size_per_gpu": 32,
    "effective_batch_size": 64,
    "max_segments": MAX_SEGMENTS_PER_RUN,
    "max_runtime_seconds": MAX_RUNTIME_SECONDS,
}, flush=True)

repo = Path("/kaggle/working/Kronos")
commit = clone_repo(repo)
print({"phase": "github_ready", "commit": commit}, flush=True)
print({"phase": "install_dependencies", "torch": TORCH_VERSION}, flush=True)
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
print({
    "phase": "device_ready",
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu_count": len(gpu_names),
    "devices": gpu_names,
    "capabilities": [torch.cuda.get_device_capability(i) for i in range(len(gpu_names))],
}, flush=True)

parent_model = find_parent_model()
print({"phase": "parent_ready", "model": str(parent_model)}, flush=True)

env = os.environ.copy()
env.update({
    "PYTHONUNBUFFERED": "1",
    "KRONOS_SMALL_V21_STAGE": "main",
    "KRONOS_SMALL_V21_PARENT_MODEL": str(parent_model),
    "KRONOS_SMALL_V21_OUTPUT_NAME": OUTPUT_NAME,
    "KRONOS_SMALL_V21_DISABLE_AUTO_CONTINUATION": "1",
    "KRONOS_MAX_SEGMENTS_PER_RUN": str(MAX_SEGMENTS_PER_RUN),
    "KRONOS_MAX_RUNTIME_SECONDS": str(MAX_RUNTIME_SECONDS),
    "KRONOS_TORCHRUN_NPROC_PER_NODE": "2",
    "KRONOS_BATCH_SIZE": "32",
    "KRONOS_NUM_WORKERS": "2",
    "KRONOS_COVERAGE_SEED": COVERAGE_SEED,
    "KRONOS_SCHEDULER": "warmup_constant",
    "KRONOS_USE_AMP": "1",
    "KRONOS_AMP_DTYPE": "float16",
    "SWANLAB_API_KEY": "fmEPDGk4IItxgqSZKGLi8",
    "SWANLAB_PROJECT": "finance",
    "SWANLAB_EXPERIMENT_NAME": "small_0.1_stage2_wc_dual_t4",
    "SWANLAB_RUN_ID": "small_0.1_stage2_wc_dual_t4",
})
subprocess.run(
    ["python", "-u", "finetune/kaggle_kronos_small_v21.py"],
    cwd=repo,
    env=env,
    check=True,
)
print({"phase": "finished", "output_name": OUTPUT_NAME}, flush=True)
