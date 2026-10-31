"""Build/push a Base-clean **Main** chunk (stage 2), mirroring Small→Stage2.

First Main chunk (default):
  parent = <bootstrap kernel> (its COMPLETE Output is mounted as kernel_sources)
  mode   = bootstrap_weights_only  (last_model only, fresh optimizer/scheduler)

Later Main chunks:
  KRONOS_MAIN_CONTINUE=1 KRONOS_PARENT_KERNEL=luckfu/kronos-base-clean-0-1-main
  mode   = same_stage_resume (full Output tree + last_state)

REST first, CLI fallback — same transport fix as push_bootstrap.py.

    # prep only (no push)
    python3 finetune/base_clean_bootstrap_relay/push_main.py

    # first Main chunk, after the Bootstrap handoff point is agreed
    python3 finetune/base_clean_bootstrap_relay/push_main.py --push --force
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from push_bootstrap import (  # noqa: E402 - sibling helper reuse
    api_session,
    kernel_status,
    local_credential,
    push_kernel,
    source_bundle,
)

OWNER = os.environ.get("KRONOS_KAGGLE_OWNER", "luckfu")
DATASET = "smmt315/a-share-120d-temporal-symbol-holdout"
SLUG = "kronos-base-clean-0-1-main"
TITLE = "Kronos Base Clean 0.1 Main"
PARENT_KERNEL = os.environ.get(
    "KRONOS_PARENT_KERNEL", "luckfu/kronos-base-clean-0-1-bootstrap"
)
CONTINUE = os.environ.get("KRONOS_MAIN_CONTINUE", "0").strip().lower() in {
    "1", "true", "yes", "on",
}
SESSION_BUDGET_SECONDS = int(
    os.environ.get("KRONOS_MAIN_SESSION_BUDGET_SECONDS", "40500")
)
SESSION_OVERHEAD_SECONDS = int(
    os.environ.get("KRONOS_MAIN_SESSION_OVERHEAD_SECONDS", "900")
)
SECONDS_PER_SEGMENT = float(
    os.environ.get("KRONOS_MAIN_SECONDS_PER_SEGMENT", "566")
)
VAL_BATCH_SIZE = os.environ.get("KRONOS_VAL_BATCH_SIZE", "64")
RUN_ID = os.environ.get("KRONOS_MAIN_RUN_ID", "base_clean_0_1_main")
# LR-schedule probes switch the scheduler on the same weights, so the kernel
# must receive these explicitly (the wrapper only injects what it is given).
SCHEDULER = os.environ.get("KRONOS_SCHEDULER", "warmup_cosine")
SCHEDULER_DECAY_START = os.environ.get("KRONOS_SCHEDULER_DECAY_START_RATIO", "0.8")
MAX_SEGMENTS = int(
    os.environ.get(
        "KRONOS_MAIN_MAX_SEGMENTS",
        str(
            max(
                1,
                int(
                    (SESSION_BUDGET_SECONDS - SESSION_OVERHEAD_SECONDS)
                    // SECONDS_PER_SEGMENT
                ),
            )
        ),
    )
)
RUNTIME_BUDGET_SECONDS = SESSION_BUDGET_SECONDS
EXPECTED_BOOTSTRAP_COMPLETED = int(
    os.environ.get("KRONOS_MAIN_EXPECTED_BOOTSTRAP_COMPLETED", "0") or 0
)
PARENT_LAST_SHA256 = os.environ.get("KRONOS_MAIN_PARENT_LAST_SHA256", "").strip()
BATCH_SIZE = 32
ACCUMULATION_STEPS = 8
NPROC = 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build (and optionally push) the Base-clean Main kernel. "
            "Default is PREP-only; pass --push to submit."
        )
    )
    parser.add_argument("--push", action="store_true", help="submit to Kaggle")
    parser.add_argument("--force", action="store_true", help="overwrite an idle slug")
    parser.add_argument("--dry-run", action="store_true", help="prep only")
    args = parser.parse_args()
    do_push = bool(args.push) and not args.dry_run

    payload = source_bundle("launch_main.py")
    digest = hashlib.sha256(payload).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    build = HERE / "build" / SLUG
    build.mkdir(parents=True, exist_ok=True)

    credential = local_credential()
    code = f'''import base64, hashlib, io, json, os, runpy, sys, time, zipfile
from pathlib import Path
print(json.dumps({{"phase": "started", "time": time.time(), "slug": {SLUG!r}}}), flush=True)
os.environ["PYTHONUNBUFFERED"] = "1"
# Injected at push-time only (not committed). Prefer UserSecrets / env first.
os.environ.setdefault("SWANLAB_API_KEY", {credential!r})
os.environ.setdefault("KRONOS_MAIN_MAX_SEGMENTS", {str(MAX_SEGMENTS)!r})
os.environ.setdefault("KRONOS_MAIN_SECONDS_PER_SEGMENT", {str(SECONDS_PER_SEGMENT)!r})
os.environ.setdefault("KRONOS_MAIN_SESSION_BUDGET_SECONDS", {str(SESSION_BUDGET_SECONDS)!r})
os.environ.setdefault("KRONOS_VAL_BATCH_SIZE", {VAL_BATCH_SIZE!r})
os.environ.setdefault("KRONOS_MAIN_RUN_ID", {RUN_ID!r})
os.environ.setdefault("KRONOS_SCHEDULER", {SCHEDULER!r})
os.environ.setdefault("KRONOS_SCHEDULER_DECAY_START_RATIO", {SCHEDULER_DECAY_START!r})
os.environ.setdefault("KRONOS_MAIN_RUNTIME_BUDGET_SECONDS", {str(RUNTIME_BUDGET_SECONDS)!r})
os.environ.setdefault("KRONOS_MAIN_CONTINUE", {"1" if CONTINUE else "0"!r})
os.environ.setdefault("KRONOS_MAIN_EXPECTED_BOOTSTRAP_COMPLETED", {str(EXPECTED_BOOTSTRAP_COMPLETED)!r})
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
repo = Path("/kaggle/working/base_clean_source")
if repo.exists():
    import shutil
    shutil.rmtree(repo)
zipfile.ZipFile(io.BytesIO(payload)).extractall(repo)
class Tee:
    def __init__(self, stream):
        self.stream = stream
        self.log = open("/kaggle/working/entry.log", "a", buffering=1)
    def write(self, text):
        self.log.write(text)
        return self.stream.write(text)
    def flush(self):
        self.log.flush()
        self.stream.flush()
sys.stdout = Tee(sys.stdout)
sys.stderr = Tee(sys.stderr)
namespace = runpy.run_path(str(repo / "finetune/base_clean_bootstrap_relay/launch_main.py"))
namespace["main"]()
'''
    code_file = "train.py"
    (build / code_file).write_text(code)
    metadata = {
        "id": f"{OWNER}/{SLUG}",
        "title": TITLE,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DATASET],
        "kernel_sources": [PARENT_KERNEL],
        "competition_sources": [],
    }
    (build / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    checklist = {
        "prep_only": not do_push,
        "stage": "main",
        "slug": SLUG,
        "id": metadata["id"],
        "url": f"https://www.kaggle.com/code/{OWNER}/{SLUG}",
        "parent_kernel": PARENT_KERNEL,
        "mode": "same_stage_resume" if CONTINUE else "bootstrap_weights_only",
        "parent_model": (
            "bootstrap checkpoints/last_model (weights only, fresh optimizer)"
            if not CONTINUE
            else "previous Main Output tree (last_state resume)"
        ),
        "max_segments_per_run": MAX_SEGMENTS,
        "segment_count_basis": {
            "session_budget_seconds": SESSION_BUDGET_SECONDS,
            "overhead_seconds": SESSION_OVERHEAD_SECONDS,
            "seconds_per_segment": SECONDS_PER_SEGMENT,
            "formula": "(budget - overhead) // seconds_per_segment",
        },
        "runtime_budget_seconds": RUNTIME_BUDGET_SECONDS,
        "expected_bootstrap_completed": EXPECTED_BOOTSTRAP_COMPLETED or "unpinned",
        "parent_last_sha256": PARENT_LAST_SHA256 or "unpinned",
        "batch": {
            "batch_size": BATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "nproc": NPROC,
            "effective_global_batch": BATCH_SIZE * NPROC,
            "updates_per_seg_approx": 20000 // (BATCH_SIZE * NPROC),
        },
        "optimizer": {
            "predictor_lr": "1e-5",
            "condition_lr": "1e-5",
            "warmup_start_lr": "1e-6",
            "scheduler": "warmup_cosine (1% warmup)",
            "restored_from_bootstrap": False,
        },
        "validation": {
            "full_only": True,
            "samples_per_segment": 123836,
            "condition_ablation_interval_segments": 20,
            "val_batch_size": int(os.environ.get("KRONOS_VAL_BATCH_SIZE", "64")),
            "disable_best_updates": False,
            "note": (
                "Formal stage validates the full 123,836 holdout every segment, "
                "like Small's Main; measured 9m26s/segment on dual T4."
            ),
        },
        "swanlab": {
            "project": "finance",
            "workspace": "roc_fu",
            "experiment_name": "base_clean_0.1_main",
            "run_id": RUN_ID,
            "replay_after_segment": None,
            "note": "NEW board (stage switch), never replay bootstrap metrics.jsonl",
        },
        "swanlab_run_id": RUN_ID,
        "scheduler": {
            "type": SCHEDULER,
            "decay_start_ratio": SCHEDULER_DECAY_START,
        },
        "source_sha256": digest,
        "source_commit": commit,
        "build_dir": str(build),
        "push_gate": "default prep-only; --push submits",
    }
    (build / "build_manifest.json").write_text(
        json.dumps(checklist, indent=2, ensure_ascii=False) + "\n"
    )
    (build / "PREP_CHECKLIST.json").write_text(
        json.dumps(checklist, indent=2, ensure_ascii=False) + "\n"
    )
    compile(code, str(build / code_file), "exec")
    print(json.dumps(checklist, indent=2, ensure_ascii=False))

    if not do_push:
        print(
            json.dumps(
                {
                    "status": "READY_NOT_PUSHED",
                    "build": str(build),
                    "next_step": "python3 finetune/base_clean_bootstrap_relay/push_main.py --push --force",
                },
                indent=2,
            ),
            flush=True,
        )
        return

    session, owner = api_session()
    if session is not None and owner and owner != OWNER:
        raise RuntimeError(f"Authenticated owner {owner!r} is not {OWNER}")
    print(json.dumps({"push_transport": "rest" if session is not None else "cli"}), flush=True)

    parent_status = kernel_status(session, PARENT_KERNEL)
    if "COMPLETE" not in parent_status:
        # A failed self-parent attempt still leaves the last COMPLETE output
        # mounted as kernel_sources; --force re-pushes from that state.
        if (
            args.force
            and CONTINUE
            and PARENT_KERNEL.endswith("/" + SLUG)
            and "ERROR" in parent_status
        ):
            print(
                "Self-parent latest status ERROR; --force reuses the previous "
                "COMPLETE output as kernel_source.",
                flush=True,
            )
        else:
            raise RuntimeError(f"Parent kernel not COMPLETE: {parent_status}")
    if CONTINUE and not PARENT_KERNEL.endswith("/" + SLUG):
        raise RuntimeError("KRONOS_MAIN_CONTINUE=1 requires a self-parent Main kernel")

    status_text = kernel_status(session, metadata["id"])
    if any(state in status_text for state in ("RUNNING", "QUEUED")):
        raise RuntimeError(f"Target kernel already active: {status_text}")
    if "COMPLETE" in status_text or "ERROR" in status_text or "CANCEL" in status_text:
        if not args.force:
            raise RuntimeError(
                f"Slug already exists ({status_text.strip()}); pass --force to overwrite"
            )
        print("Slug exists and is idle; pushing update", flush=True)
    elif status_text and not any(
        token in status_text.lower() for token in ("not found", "404", "permission", "unknown")
    ):
        print(f"status probe: {status_text.strip()}", flush=True)

    message = push_kernel(session, build, metadata, code)
    print(
        json.dumps(
            {
                "pushed": metadata["id"],
                "url": f"https://www.kaggle.com/code/{OWNER}/{SLUG}",
                "server": message[:200],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
