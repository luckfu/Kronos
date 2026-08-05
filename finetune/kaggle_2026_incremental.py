"""Kaggle entrypoint for two-pass 2026 incremental training from V3 Last."""

import os
import shutil
import subprocess
import tarfile
from pathlib import Path


REPO = Path("/kaggle/working/Kronos")
RUNTIME = Path("/kaggle/working/kronos_a_share_2026_incremental")


if not (REPO / ".git").is_dir():
    subprocess.run(
        ["git", "clone", "https://github.com/luckfu/Kronos.git", str(REPO)],
        check=True,
    )

subprocess.run(
    ["python", "-m", "pip", "install", "-q", "-r", "finetune/requirements-colab.txt"],
    cwd=REPO,
    check=True,
)

gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    capture_output=True,
    text=True,
    check=False,
).stdout
if "P100" in gpu.upper():
    compatible = subprocess.run(
        [
            "python",
            "-c",
            "import torch; raise SystemExit(0 if 'sm_60' in torch.cuda.get_arch_list() else 1)",
        ],
        check=False,
    ).returncode == 0
    if not compatible:
        print("[incremental-2026] Installing P100-compatible PyTorch", flush=True)
        subprocess.run(
            [
                "python", "-m", "pip", "install", "-q",
                "--index-url", "https://download.pytorch.org/whl/cu121",
                "torch==2.5.1",
            ],
            check=True,
        )

env = os.environ.copy()
env["KRONOS_KAGGLE_ROOT"] = str(RUNTIME)
train_candidates = list(
    Path("/kaggle/input").rglob(
        "a_share_v3_2026_incremental/processed_datasets/train_data.pkl"
    )
)
base_candidates = list(
    Path("/kaggle/input").rglob("base_model/v3_last/model.safetensors")
)
if len(train_candidates) == 1 and len(base_candidates) == 1:
    data_root = train_candidates[0].parent.parent
    base_model = base_candidates[0].parent
    print(f"[incremental-2026] Using expanded Dataset at {data_root}", flush=True)
else:
    bundle_candidates = list(
        Path("/kaggle/input").rglob("kronos_a_share_2026_incremental.tar.gz")
    )
    if len(bundle_candidates) != 1:
        raise RuntimeError(
            "Expected one expanded incremental Dataset or bundle; found "
            f"train={train_candidates}, base={base_candidates}, bundle={bundle_candidates}"
        )
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_candidates[0], "r:gz") as archive:
        archive.extractall(RUNTIME, filter="data")
    data_root = RUNTIME / "data/a_share_v3_2026_incremental"
    base_model = RUNTIME / "base_model/v3_last"
env["KRONOS_KAGGLE_DATA_ROOT"] = str(data_root)
env["KRONOS_PREDICTOR_PATH"] = str(base_model)
subprocess.run(
    ["bash", "finetune/kaggle_2026_incremental_train.sh"],
    cwd=REPO,
    env=env,
    check=True,
)

# Keep only outputs: the input bundle remains available as a Dataset and the
# repository is reproducible from GitHub.
shutil.rmtree(RUNTIME / "data", ignore_errors=True)
shutil.rmtree(RUNTIME / "base_model", ignore_errors=True)
shutil.rmtree(REPO, ignore_errors=True)
