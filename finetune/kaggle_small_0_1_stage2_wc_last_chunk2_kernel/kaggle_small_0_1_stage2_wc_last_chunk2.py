import json
import os
import shutil
import subprocess
import time
from pathlib import Path


MAX_SEGMENTS_PER_RUN = 130
STAGE2_EXTENSION_OUTPUT = "small_0.1_stage2_wc_last"
COVERAGE_SEED = "20260908"

repo = "/kaggle/working/Kronos"
print({"phase": "started", "coverage_seed": COVERAGE_SEED, "max_segments": MAX_SEGMENTS_PER_RUN}, flush=True)
for attempt in range(1, 6):
    shutil.rmtree(repo, ignore_errors=True)
    result = subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "3",
            "--branch",
            "master",
            "https://github.com/luckfu/Kronos.git",
            repo,
        ],
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode == 0:
        break
    print(f"GitHub clone attempt {attempt}/5 failed; retrying...", flush=True)
    time.sleep(attempt * 5)
else:
    raise RuntimeError("GitHub clone failed after 5 attempts")

commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
).strip()
training_script = Path(repo) / "finetune/kaggle_kronos_small_v21.py"
print({"git_commit": commit, "training_script": str(training_script)}, flush=True)
if not training_script.is_file():
    raise RuntimeError(f"Training implementation is missing: {training_script}")

print({"phase": "install_dependencies"}, flush=True)
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], cwd=repo, check=True)
print({"phase": "install_p100_torch"}, flush=True)
subprocess.run(
    [
        "pip",
        "install",
        "-q",
        "--force-reinstall",
        "torch==2.5.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu121",
    ],
    check=True,
)
subprocess.run(["pip", "install", "-q", "swanlab"], check=True)

import torch

if "sm_60" not in torch.cuda.get_arch_list():
    raise RuntimeError(f"PyTorch {torch.__version__} lacks P100 sm_60 support")
print({"torch": torch.__version__, "cuda_arches": torch.cuda.get_arch_list()}, flush=True)

input_root = Path("/kaggle/input")
state_paths = sorted(
    input_root.glob("**/small_0.1_stage2_wc_last/checkpoints/last_state.pt")
)
if len(state_paths) != 1:
    raise RuntimeError(
        f"Expected one extension last_state, found {len(state_paths)}: {state_paths}"
    )
state_path = state_paths[0]
continuation_root = state_path.parent.parent
progress_path = continuation_root / "progress.json"
if not progress_path.is_file():
    raise RuntimeError(f"Continuation progress is missing: {progress_path}")
progress = json.loads(progress_path.read_text())
if progress.get("status") not in {"stopped", "completed"}:
    raise RuntimeError(f"Continuation output is not durable: {progress}")
parent_model = continuation_root / "checkpoints/last_model"
if not (parent_model / "model.safetensors").is_file():
    raise RuntimeError(f"Continuation last_model is missing: {parent_model}")
print(
    {
        "continuation_output": str(continuation_root),
        "resume_state": str(state_path),
        "initialization": "segment_boundary_resume",
        "coverage_seed": COVERAGE_SEED,
        "max_segments_this_chunk": MAX_SEGMENTS_PER_RUN,
    },
    flush=True,
)

os.environ["SWANLAB_API_KEY"] = "fmEPDGk4IItxgqSZKGLi8"
os.environ.update(
    {
        "KRONOS_SMALL_V21_STAGE": "main",
        "KRONOS_SMALL_V21_PARENT_MODEL": str(parent_model),
        "KRONOS_SMALL_V21_CONTINUATION_ROOT": str(continuation_root),
        "KRONOS_SMALL_V21_OUTPUT_NAME": STAGE2_EXTENSION_OUTPUT,
        "KRONOS_MAX_SEGMENTS_PER_RUN": str(MAX_SEGMENTS_PER_RUN),
        "KRONOS_BATCH_SIZE": "64",
        "KRONOS_COVERAGE_SEED": COVERAGE_SEED,
        "KRONOS_SCHEDULER": "warmup_constant",
        "SWANLAB_PROJECT": "finance",
        "SWANLAB_EXPERIMENT_NAME": STAGE2_EXTENSION_OUTPUT,
        "SWANLAB_RUN_ID": STAGE2_EXTENSION_OUTPUT,
    }
)
subprocess.run(
    ["python", "-u", "finetune/kaggle_kronos_small_v21.py"],
    cwd=repo,
    env=os.environ.copy(),
    check=True,
)
