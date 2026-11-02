import os
import shutil
import subprocess
import time
from pathlib import Path

repo = "/kaggle/working/Kronos"
for attempt in range(1, 6):
    shutil.rmtree(repo, ignore_errors=True)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", "master",
         "https://github.com/luckfu/Kronos.git", repo],
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
print({"git_commit": commit}, flush=True)
if commit != "f2470712abcef100ca3abd8c95bdf8101652ad57":
    raise RuntimeError(f"Unexpected Git commit: {commit}")

subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], cwd=repo, check=True)
subprocess.run(
    ["pip", "install", "-q", "--force-reinstall", "torch==2.5.1",
     "--index-url", "https://download.pytorch.org/whl/cu121"], check=True,
)
subprocess.run(["pip", "install", "-q", "swanlab"], check=True)

import torch
if "sm_60" not in torch.cuda.get_arch_list():
    raise RuntimeError(f"PyTorch {torch.__version__} lacks P100 sm_60 support")
print({"torch": torch.__version__, "cuda_arches": torch.cuda.get_arch_list()}, flush=True)

input_root = Path("/kaggle/input")
parent_weights = sorted(input_root.glob(
    "**/small_0.1_bootstrap/checkpoints/best_model/model.safetensors"
))
if len(parent_weights) != 1:
    raise RuntimeError(
        f"Expected one Bootstrap best_model, found {len(parent_weights)}: {parent_weights}"
    )
parent_model = parent_weights[0].parent
best_metric = parent_model / "best_metric.json"
if not best_metric.is_file():
    raise RuntimeError(f"Bootstrap Best metadata is missing: {best_metric}")
print({"stage2_parent": str(parent_model), "selection": "bootstrap_best"}, flush=True)

os.environ["SWANLAB_API_KEY"] = "fmEPDGk4IItxgqSZKGLi8"
os.environ.update({
    "KRONOS_SMALL_V21_STAGE": "main",
    "KRONOS_SMALL_V21_PARENT_MODEL": str(parent_model),
    "KRONOS_SMALL_V21_DISABLE_AUTO_CONTINUATION": "1",
    "KRONOS_MAX_SEGMENTS_PER_RUN": "1",
    "KRONOS_BATCH_SIZE": "64",
    "SWANLAB_PROJECT": "finance",
    "SWANLAB_EXPERIMENT_NAME": "small_0.1_main",
    "SWANLAB_RUN_ID": "small_0_1_main",
})
subprocess.run(
    ["python", "-u", "finetune/kaggle_kronos_small_v21.py"],
    cwd=repo, env=os.environ.copy(), check=True,
)
