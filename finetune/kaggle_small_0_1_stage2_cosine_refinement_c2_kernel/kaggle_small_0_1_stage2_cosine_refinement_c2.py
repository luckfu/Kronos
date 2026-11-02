"""Finish Stage 2 cosine refinement from C1's exact segment boundary."""

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_NAME = "small_0.1_stage2_cosine_refinement"
PARENT_KERNEL = "smmt315/kronos-small-0-1-stage2-cosine-refinement-c1"
EXPECTED_PARENT_SEGMENT = 130
EXPECTED_TOTAL_SEGMENTS = 267
MAX_SEGMENTS_PER_RUN = 137
MAX_RUNTIME_SECONDS = 39600  # 11 hours; Kaggle's hard limit is 12 hours.
COVERAGE_SEED = "20260912"  # Must remain identical for exact continuation.
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
        "**/small_0.1_stage2_cosine_refinement/checkpoints/last_state.pt"
    ))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one cosine C1 last_state.pt, found {len(matches)}: {matches}"
        )
    source_root = matches[0].parent.parent
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
        raise RuntimeError(f"Cosine C1 output contract is incomplete: {missing}")
    progress = json.loads((source_root / "progress.json").read_text())
    completed = int(
        progress.get("completed_segments", progress.get("current_segment", 0))
    )
    if completed != EXPECTED_PARENT_SEGMENT or progress.get("status") != "stopped":
        raise RuntimeError(f"C1 is not a durable Segment 130 parent: {progress}")
    if int(progress.get("total_segments", -1)) != EXPECTED_TOTAL_SEGMENTS:
        raise RuntimeError(f"C1 global plan changed: {progress}")
    summary = json.loads((source_root / "summary.json").read_text())
    result = summary.get("final_result", {})
    if int(result.get("resume_segment", -1)) != EXPECTED_PARENT_SEGMENT + 1:
        raise RuntimeError(f"C1 resume boundary mismatch: {result}")
    return source_root


phase(
    "started",
    experiment=OUTPUT_NAME,
    parent_kernel=PARENT_KERNEL,
    expected_parent_segment=EXPECTED_PARENT_SEGMENT,
    target_segment=EXPECTED_TOTAL_SEGMENTS,
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
    raise RuntimeError(f"C1 state boundary mismatch: {state.get('next_epoch')}")
if int(state.get("effective_epochs", -1)) != EXPECTED_TOTAL_SEGMENTS:
    raise RuntimeError(f"C1 scheduler plan mismatch: {state.get('effective_epochs')}")
if state.get("scheduler_type") != "uniform_cosine":
    raise RuntimeError(f"C1 scheduler type mismatch: {state.get('scheduler_type')}")
if int(state.get("scheduler_warmup_steps", -1)) != 0:
    raise RuntimeError(f"C1 warmup mismatch: {state.get('scheduler_warmup_steps')}")
if int(state.get("scheduler_total_steps", -1)) != 83571:
    raise RuntimeError(f"C1 optimizer-step plan mismatch: {state.get('scheduler_total_steps')}")
phase(
    "parent_ready",
    source_root=str(source_root),
    completed_segment=EXPECTED_PARENT_SEGMENT,
    resume_segment=EXPECTED_PARENT_SEGMENT + 1,
    scheduler=state.get("scheduler_type"),
    scheduler_step=state.get("batch_idx_global"),
    scheduler_total_steps=state.get("scheduler_total_steps"),
    best_val_loss=state.get("best_val_loss"),
    initialization="exact_same_stage_resume",
)
del state

env = os.environ.copy()
env.update({
    "PYTHONUNBUFFERED": "1",
    "KRONOS_SMALL_V21_STAGE": "main",
    "KRONOS_SMALL_V21_PARENT_MODEL": str(source_root / "checkpoints/last_model"),
    "KRONOS_SMALL_V21_CONTINUATION_ROOT": str(source_root),
    "KRONOS_SMALL_V21_OUTPUT_NAME": OUTPUT_NAME,
    "KRONOS_SMALL_V21_DISABLE_AUTO_CONTINUATION": "0",
    "KRONOS_MAX_SEGMENTS_PER_RUN": str(MAX_SEGMENTS_PER_RUN),
    "KRONOS_MAX_RUNTIME_SECONDS": str(MAX_RUNTIME_SECONDS),
    "KRONOS_TORCHRUN_NPROC_PER_NODE": "2",
    "KRONOS_BATCH_SIZE": "32",
    "KRONOS_NUM_WORKERS": "2",
    "KRONOS_COVERAGE_SEED": COVERAGE_SEED,
    "KRONOS_EPOCHS": str(EXPECTED_TOTAL_SEGMENTS),
    "KRONOS_REQUIRE_FULL_COVERAGE": "0",
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
    "launch_continuation",
    stage_segments=f"131-{EXPECTED_TOTAL_SEGMENTS}/{EXPECTED_TOTAL_SEGMENTS}",
    scheduler="uniform_cosine_exact_resume",
    warmup_steps=0,
    peak_lr=1e-5,
    min_lr=1e-6,
    optimizer_steps=83571,
)
subprocess.run(
    ["python", "-u", "finetune/kaggle_kronos_small_v21.py"],
    cwd=repo,
    env=env,
    check=True,
)
phase("finished", output_name=OUTPUT_NAME)
