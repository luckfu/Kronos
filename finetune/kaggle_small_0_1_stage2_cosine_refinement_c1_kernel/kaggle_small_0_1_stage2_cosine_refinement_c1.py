"""Start Stage 2 cosine refinement from the completed WC dual-T4 last state."""

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_NAME = "small_0.1_stage2_cosine_refinement"
PARENT_KERNEL = "smmt315/kronos-small-0-1-stage2-wc-dual-t4-c4"
EXPECTED_PARENT_SEGMENT = 534
EXPECTED_PARENT_BEST_SEGMENT = 512
REFINEMENT_SEGMENTS = 267
MAX_SEGMENTS_PER_RUN = 130
MAX_RUNTIME_SECONDS = 39600  # 11 hours; Kaggle's hard limit is 12 hours.
COVERAGE_SEED = "20260912"
TORCH_VERSION = "2.6.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"


def phase(name: str, **payload) -> None:
    print(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": name,
        **payload,
    }), flush=True)


def clone_repo(repo: Path) -> str:
    shutil.rmtree(repo, ignore_errors=True)
    subprocess.run(
        ["git", "clone", "--depth", "3", "--branch", "master",
         "https://github.com/luckfu/Kronos.git", str(repo)],
        check=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def find_parent_output() -> Path:
    matches = sorted(Path("/kaggle/input").glob(
        "**/small_0.1_stage2_wc_dual_t4_c4/checkpoints/last_state.pt"
    ))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one C4 last_state.pt, found {len(matches)}: {matches}"
        )
    source_root = matches[0].parent.parent
    required = (
        "progress.json",
        "summary.json",
        "checkpoints/last_state.pt",
        "checkpoints/last_model/model.safetensors",
        "checkpoints/best_model/model.safetensors",
        "checkpoints/best_model/best_metric.json",
    )
    missing = [name for name in required if not (source_root / name).is_file()]
    if missing:
        raise RuntimeError(f"C4 output contract is incomplete: {missing}")
    progress = json.loads((source_root / "progress.json").read_text())
    completed = int(progress.get("completed_segments", 0))
    if completed != EXPECTED_PARENT_SEGMENT or progress.get("status") != "completed":
        raise RuntimeError(f"C4 is not a completed 534-segment parent: {progress}")
    best = json.loads(
        (source_root / "checkpoints/best_model/best_metric.json").read_text()
    )
    if int(best.get("segment", -1)) != EXPECTED_PARENT_BEST_SEGMENT:
        raise RuntimeError(f"Unexpected C4 best checkpoint: {best}")
    return source_root


phase(
    "started",
    experiment=OUTPUT_NAME,
    parent_kernel=PARENT_KERNEL,
    refinement_segments=REFINEMENT_SEGMENTS,
    max_segments_this_chunk=MAX_SEGMENTS_PER_RUN,
    max_runtime_seconds=MAX_RUNTIME_SECONDS,
    coverage_seed=COVERAGE_SEED,
)
repo = Path("/kaggle/working/Kronos")
commit = clone_repo(repo)
phase("github_ready", commit=commit)

phase("install_requirements_started")
subprocess.run(
    ["python", "-m", "pip", "install", "--disable-pip-version-check",
     "--progress-bar", "off", "-r", "requirements.txt"],
    cwd=repo,
    check=True,
)
phase("install_torch_started", torch=TORCH_VERSION)
subprocess.run(
    ["python", "-m", "pip", "install", "--disable-pip-version-check",
     "--progress-bar", "off", "--force-reinstall", f"torch=={TORCH_VERSION}",
     "--index-url", TORCH_INDEX_URL],
    check=True,
)
phase("install_swanlab_started")
subprocess.run(
    ["python", "-m", "pip", "install", "--disable-pip-version-check",
     "--progress-bar", "off", "swanlab"],
    check=True,
)

import torch

gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(gpu_names) != 2 or any("T4" not in name for name in gpu_names):
    raise RuntimeError(f"Expected exactly two Tesla T4 GPUs, found {gpu_names}")
phase(
    "device_ready",
    torch=torch.__version__,
    cuda=torch.version.cuda,
    gpu_count=len(gpu_names),
    devices=gpu_names,
)

source_root = find_parent_output()
source_state = source_root / "checkpoints/last_state.pt"
state = torch.load(source_state, map_location="cpu", weights_only=False)
if int(state.get("next_epoch", -1)) != EXPECTED_PARENT_SEGMENT:
    raise RuntimeError(f"C4 state boundary mismatch: {state.get('next_epoch')}")
if state.get("scheduler_type") != "warmup_constant":
    raise RuntimeError(f"C4 scheduler mismatch: {state.get('scheduler_type')}")
if not state.get("optimizer", {}).get("state"):
    raise RuntimeError("C4 checkpoint has no AdamW moment state")
parent_model = source_root / "checkpoints/last_model"
phase(
    "parent_ready",
    source_root=str(source_root),
    parent_segment=EXPECTED_PARENT_SEGMENT,
    parent_best_segment=EXPECTED_PARENT_BEST_SEGMENT,
    parent_scheduler=state.get("scheduler_type"),
    parent_optimizer_step=state.get("batch_idx_global"),
    initialization="new_stage_preserve_model_and_adamw_replace_scheduler",
)
del state

env = os.environ.copy()
env.update({
    "PYTHONUNBUFFERED": "1",
    "KRONOS_SMALL_V21_STAGE": "main",
    "KRONOS_SMALL_V21_PARENT_MODEL": str(parent_model),
    "KRONOS_SMALL_V21_OUTPUT_NAME": OUTPUT_NAME,
    "KRONOS_SMALL_V21_DISABLE_AUTO_CONTINUATION": "1",
    "KRONOS_SCHEDULER_TRANSITION_STATE": str(source_state),
    "KRONOS_EPOCHS": str(REFINEMENT_SEGMENTS),
    "KRONOS_REQUIRE_FULL_COVERAGE": "0",
    "KRONOS_MAX_SEGMENTS_PER_RUN": str(MAX_SEGMENTS_PER_RUN),
    "KRONOS_MAX_RUNTIME_SECONDS": str(MAX_RUNTIME_SECONDS),
    "KRONOS_TORCHRUN_NPROC_PER_NODE": "2",
    "KRONOS_BATCH_SIZE": "32",
    "KRONOS_NUM_WORKERS": "2",
    "KRONOS_COVERAGE_SEED": COVERAGE_SEED,
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
    "SWANLAB_EXPERIMENT_NAME": "small_0.1_stage2_cosine_refinement",
    "SWANLAB_RUN_ID": "small_0.1_stage2_cosine_refinement",
})
phase(
    "launch_refinement",
    stage_segments=f"1-{MAX_SEGMENTS_PER_RUN}/{REFINEMENT_SEGMENTS}",
    scheduler="uniform_cosine",
    warmup_steps=0,
    peak_lr=1e-5,
    min_lr=1e-6,
    expected_optimizer_steps=83571,
)
subprocess.run(
    ["python", "-u", "finetune/kaggle_kronos_small_v21.py"],
    cwd=repo,
    env=env,
    check=True,
)
phase("finished", output_name=OUTPUT_NAME)
