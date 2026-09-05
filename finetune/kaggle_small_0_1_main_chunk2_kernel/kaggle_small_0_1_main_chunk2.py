import json
import os
import shutil
import subprocess
import time
from pathlib import Path


# The training implementation is pinned to the reviewed Stage 2 commit.
EXPECTED_GIT_COMMIT = "f2470712abcef100ca3abd8c95bdf8101652ad57"
EXPECTED_PARENT_SEGMENT = 141
MAX_SEGMENTS_PER_RUN = 140

repo = "/kaggle/working/Kronos"
for attempt in range(1, 6):
    shutil.rmtree(repo, ignore_errors=True)
    result = subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
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
training_commit = subprocess.check_output(
    ["git", "log", "-1", "--format=%H", "--", "finetune/kaggle_kronos_small_v21.py"],
    cwd=repo,
    text=True,
).strip()
print({"git_commit": commit, "training_commit": training_commit}, flush=True)
if training_commit != EXPECTED_GIT_COMMIT:
    raise RuntimeError(f"Unexpected training implementation commit: {training_commit}")

subprocess.run(
    ["pip", "install", "-q", "-r", "requirements.txt"], cwd=repo, check=True
)
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
    input_root.glob("**/small_0.1_main/checkpoints/last_state.pt")
)
if len(state_paths) != 1:
    raise RuntimeError(
        f"Expected one Stage 2 continuation state, found {len(state_paths)}: {state_paths}"
    )
continuation_root = state_paths[0].parent.parent
progress_path = continuation_root / "progress.json"
if not progress_path.is_file():
    raise RuntimeError(f"Continuation progress is missing: {progress_path}")
progress = json.loads(progress_path.read_text())
if progress.get("status") not in {"stopped", "completed"}:
    raise RuntimeError(f"Continuation output is not durable: {progress}")

resume_state = torch.load(state_paths[0], map_location="cpu", weights_only=False)
next_epoch = int(resume_state["next_epoch"])
del resume_state
if next_epoch != EXPECTED_PARENT_SEGMENT:
    raise RuntimeError(
        f"Expected last_state.pt next_epoch={EXPECTED_PARENT_SEGMENT}, got {next_epoch}"
    )

parent_model = continuation_root / "checkpoints/last_model"
if not (parent_model / "model.safetensors").is_file():
    raise RuntimeError(f"Continuation last_model is missing: {parent_model}")
print(
    {
        "continuation_root": str(continuation_root),
        "resume_next_epoch": next_epoch,
        "next_segment": next_epoch + 1,
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
        "KRONOS_MAX_SEGMENTS_PER_RUN": str(MAX_SEGMENTS_PER_RUN),
        "KRONOS_BATCH_SIZE": "64",
        "SWANLAB_PROJECT": "finance",
        "SWANLAB_EXPERIMENT_NAME": "small_0.1_main",
        "SWANLAB_RUN_ID": "small_0_1_main",
    }
)
subprocess.run(
    ["python", "-u", "finetune/kaggle_kronos_small_v21.py"],
    cwd=repo,
    env=os.environ.copy(),
    check=True,
)
