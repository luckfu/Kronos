"""Run one repaired continuation segment on a Kaggle T4 for a speed probe."""

import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path


MAX_SEGMENTS_PER_RUN = 1
STAGE2_EXTENSION_OUTPUT = "small_0.1_stage2_wc_last_c5_t4_probe"
COVERAGE_SEED = "20260908"
EXPECTED_NEXT_EPOCH = 498
EXPECTED_RESUME_SEGMENT = EXPECTED_NEXT_EPOCH + 1
TORCH_VERSION = "2.6.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"


def clone_repo(repo: Path) -> str:
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
                str(repo),
            ],
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        if result.returncode == 0:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
        print(f"GitHub clone attempt {attempt}/5 failed; retrying...", flush=True)
        time.sleep(attempt * 5)
    raise RuntimeError("GitHub clone failed after 5 attempts")


def locate_continuation() -> Path:
    input_root = Path("/kaggle/input")
    state_paths = sorted(
        input_root.glob("**/small_0.1_stage2_wc_last/checkpoints/last_state.pt")
    )
    if not state_paths:
        archives = sorted(input_root.glob("**/Kronos_C4_continuation_segment_498.tar"))
        if len(archives) == 1:
            extracted = Path("/kaggle/working/kronos_c4_input")
            extracted.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archives[0]) as archive:
                archive.extractall(extracted, filter="data")
            state_paths = sorted(
                extracted.glob("**/small_0.1_stage2_wc_last/checkpoints/last_state.pt")
            )
    if len(state_paths) != 1:
        raise RuntimeError(
            "Expected exactly one C4 last_state.pt, including an optional "
            f"Kronos_C4_continuation_segment_498.tar, found {len(state_paths)}: {state_paths}"
        )
    return state_paths[0].parent.parent


def repair_continuation(source_root: Path, repo: Path) -> Path:
    """Make a durable C4 cancellation snapshot acceptable to the normal runner."""
    import torch

    state_path = source_root / "checkpoints/last_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    next_epoch = int(state.get("next_epoch", -1))
    if next_epoch != EXPECTED_NEXT_EPOCH:
        raise RuntimeError(
            f"C4 durable state is at next_epoch={next_epoch}, expected {EXPECTED_NEXT_EPOCH}"
        )

    repaired_root = Path("/kaggle/working/kronos_c4_repaired")
    shutil.rmtree(repaired_root, ignore_errors=True)
    shutil.copytree(source_root, repaired_root)

    progress_path = repaired_root / "progress.json"
    progress = json.loads(progress_path.read_text())
    progress.update(
        {
            "status": "stopped",
            "phase": "complete",
            "stop_reason": "external_cancel_after_segment_boundary",
            "current_segment": EXPECTED_RESUME_SEGMENT,
            "current_step": 0,
            "observed_step": 0,
            "completed_segments": EXPECTED_NEXT_EPOCH,
            "resume_segment": EXPECTED_RESUME_SEGMENT,
        }
    )
    progress_path.write_text(json.dumps(progress, indent=2) + "\n")

    best_config = repaired_root / "checkpoints/best_model/config.json"
    last_model = repaired_root / "checkpoints/last_model"
    shutil.rmtree(last_model, ignore_errors=True)
    subprocess.run(
        [
            "python",
            str(repo / "finetune/extract_model_from_last_state.py"),
            "--state",
            str(repaired_root / "checkpoints/last_state.pt"),
            "--config",
            str(best_config),
            "--output",
            str(last_model),
        ],
        check=True,
    )

    summary_path = repaired_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    final_result = dict(summary.get("final_result", {}))
    final_result.update(
        {
            "status": "stopped",
            "stop_reason": "external_cancel_after_segment_boundary",
            "completed_segments": EXPECTED_NEXT_EPOCH,
            "resume_segment": EXPECTED_RESUME_SEGMENT,
            "total_segments": 534,
        }
    )
    summary["final_result"] = final_result
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        {
            "repaired_continuation": str(repaired_root),
            "next_epoch": next_epoch,
            "resume_segment": EXPECTED_RESUME_SEGMENT,
        },
        flush=True,
    )
    return repaired_root


repo = Path("/kaggle/working/Kronos")
print(
    {
        "phase": "started",
        "probe": "T4_one_segment",
        "coverage_seed": COVERAGE_SEED,
        "max_segments": MAX_SEGMENTS_PER_RUN,
        "expected_resume_segment": EXPECTED_RESUME_SEGMENT,
        "torch": TORCH_VERSION,
    },
    flush=True,
)
commit = clone_repo(repo)
training_script = repo / "finetune/kaggle_kronos_small_v21.py"
print({"git_commit": commit, "training_script": str(training_script)}, flush=True)

print({"phase": "install_dependencies", "torch": TORCH_VERSION}, flush=True)
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], cwd=repo, check=True)
subprocess.run(
    [
        "pip",
        "install",
        "-q",
        "--force-reinstall",
        f"torch=={TORCH_VERSION}",
        "--index-url",
        TORCH_INDEX_URL,
    ],
    check=True,
)
subprocess.run(["pip", "install", "-q", "swanlab"], check=True)

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; C5 T4 probe requires a GPU")
if torch.cuda.get_device_capability(0) != (7, 5):
    raise RuntimeError(
        f"Expected a T4 (compute capability 7.5), found {torch.cuda.get_device_name(0)} "
        f"with capability {torch.cuda.get_device_capability(0)}"
    )
print(
    {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
    },
    flush=True,
)

source_root = locate_continuation()
repaired_root = repair_continuation(source_root, repo)
parent_model = repaired_root / "checkpoints/last_model"
os.environ["SWANLAB_API_KEY"] = "fmEPDGk4IItxgqSZKGLi8"
os.environ.update(
    {
        "KRONOS_SMALL_V21_STAGE": "main",
        "KRONOS_SMALL_V21_PARENT_MODEL": str(parent_model),
        "KRONOS_SMALL_V21_CONTINUATION_ROOT": str(repaired_root),
        "KRONOS_SMALL_V21_OUTPUT_NAME": STAGE2_EXTENSION_OUTPUT,
        "KRONOS_MAX_SEGMENTS_PER_RUN": str(MAX_SEGMENTS_PER_RUN),
        "KRONOS_BATCH_SIZE": "64",
        "KRONOS_NUM_WORKERS": "4",
        "KRONOS_COVERAGE_SEED": COVERAGE_SEED,
        "KRONOS_SCHEDULER": "warmup_constant",
        "KRONOS_USE_AMP": "1",
        "KRONOS_AMP_DTYPE": "float16",
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
