import os
import shutil
import subprocess
import time

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

os.environ["SWANLAB_API_KEY"] = "fmEPDGk4IItxgqSZKGLi8"
os.environ.update({
    "KRONOS_SMALL_V21_STAGE": "bootstrap",
    "KRONOS_MAX_SEGMENTS_PER_RUN": "240",
    "SWANLAB_PROJECT": "finance",
    "SWANLAB_EXPERIMENT_NAME": "small_0.1_bootstrap",
    # Timing Kernel's published SwanLab run id; continuation resumes this run.
    "SWANLAB_RUN_ID": "n3sjweb7",
})
subprocess.run(
    ["python", "-u", "finetune/kaggle_kronos_small_v21.py"],
    cwd=repo, env=os.environ.copy(), check=True,
)
