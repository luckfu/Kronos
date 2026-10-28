"""Build/push a Base-clean Bootstrap relay chunk (default: chunk 3).

Continues from the previous COMPLETE chunk's output (resume_segment is read at
runtime from its progress.json), same SwanLab board. Pass `--push` to push.

2026-10-28 relay: the previous chunk ended at coverage segment 317
(resume 318) under user281434. This push runs under luckfu, mounting the
shared user281434 notebook; later chunks self-parent under luckfu.
Current runtime code is the cond_identity payload snapshot (the code the
training chunks actually ran), not HEAD.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OWNER = os.environ.get("KRONOS_KAGGLE_OWNER", "luckfu")
DATASET = "smmt315/a-share-120d-temporal-symbol-holdout"
# Parent output: chunk 2 lives on user281434's shared notebook. Later luckfu
# chunks should pass KRONOS_PARENT_KERNEL=luckfu/kronos-base-clean-0-1-bootstrap.
PARENT_KERNEL = os.environ.get(
    "KRONOS_PARENT_KERNEL", "user281434/kronos-base-clean-0-1-bootstrap"
)
SLUG = "kronos-base-clean-0-1-bootstrap"
TITLE = "Kronos Base Clean 0.1 Bootstrap"
# Parent chunk ended at completed=317 / resume=318 (log: "resume from coverage
# segment 318"). Pins are injected into the kernel; empty disables the guard.
RESUME_SEGMENT = int(os.environ.get("KRONOS_BOOTSTRAP_EXPECTED_RESUME", "318"))
COMPLETED_SEGMENTS = int(os.environ.get("KRONOS_BOOTSTRAP_EXPECTED_COMPLETED", "317"))
PARENT_LAST_SHA256 = os.environ.get("KRONOS_BOOTSTRAP_PARENT_LAST_SHA256", "").strip()
PARENT_BEST_SHA256 = os.environ.get("KRONOS_BOOTSTRAP_PARENT_BEST_SHA256", "").strip()
MAX_SEGMENTS = int(os.environ.get("KRONOS_BOOTSTRAP_MAX_SEGMENTS", "150"))
# Chunks are wall-clock budgeted; see launch_bootstrap.py.
RUNTIME_BUDGET_SECONDS = int(
    os.environ.get("KRONOS_BOOTSTRAP_RUNTIME_BUDGET_SECONDS", "36000")
)
BATCH_SIZE = 32
ACCUMULATION_STEPS = 8
NPROC = 2
# The training chunks actually ran with the 2026-10-27 runtime snapshot
# (Fix A/C present but default-off). Keep the relay on identical code.
RUNTIME_SNAPSHOT = ROOT / "finetune" / "cond_identity" / "payload_snapshot"


def local_credential() -> str:
    value = os.environ.get("SWANLAB_API_KEY", "").strip()
    if value:
        return value
    path = (
        ROOT
        / "finetune/kaggle_small_0_1_stage2_wc_dual_t4_kernel"
        / "kaggle_small_0_1_stage2_wc_dual_t4.py"
    )
    if not path.is_file():
        return ""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Dict):
            for key, item in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "SWANLAB_API_KEY"
                    and isinstance(item, ast.Constant)
                ):
                    return str(item.value)
    return ""


API_ROOT = os.environ.get("KRONOS_KAGGLE_API", "https://api.kaggle.com")
RPC_ROOT = f"{API_ROOT}/v1/kernels.KernelsApiService"


def api_session():
    """Kaggle REST session (auto-refreshing OAuth), or (None, "") if unavailable."""
    try:
        import relay_auth
    except ImportError:
        return None, ""
    try:
        return relay_auth.session(), relay_auth.username()
    except Exception:
        return None, ""


def kernel_status_rest(session, ref: str) -> str:
    owner, slug = ref.split("/", 1)
    response = session.post(
        f"{RPC_ROOT}/GetKernelSessionStatus",
        json={"userName": owner, "kernelSlug": slug},
        timeout=30,
    )
    response.raise_for_status()
    return str(response.json().get("status", "UNKNOWN"))


def kernel_status(session, ref: str) -> str:
    if session is not None:
        try:
            return kernel_status_rest(session, ref)
        except Exception:
            pass
    completed = subprocess.run(
        ["kaggle", "kernels", "status", ref], capture_output=True, text=True
    )
    return (completed.stdout + completed.stderr).strip()


def push_kernel(session, build: Path, metadata: dict, code: str) -> str:
    """Save (and therefore run) a new kernel version; returns the server message."""
    if session is None:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(build)], check=True)
        return "pushed via kaggle CLI"
    payload = {
        "slug": metadata["id"],
        "newTitle": metadata["title"],
        "text": code,
        "language": metadata["language"],
        "kernelType": metadata["kernel_type"],
        "isPrivate": bool(metadata.get("is_private", True)),
        "enableGpu": bool(metadata.get("enable_gpu", False)),
        "enableInternet": bool(metadata.get("enable_internet", True)),
        "datasetDataSources": metadata.get("dataset_sources", []),
        "competitionDataSources": metadata.get("competition_sources", []),
        "kernelDataSources": metadata.get("kernel_sources", []),
        "machineShape": metadata.get("machine_shape"),
    }
    response = session.post(f"{RPC_ROOT}/SaveKernel", json=payload, timeout=180)
    response.raise_for_status()
    body = response.json()
    if body.get("error"):
        raise RuntimeError(f"SaveKernel failed: {body['error']}")
    return json.dumps(body)


def _runtime_file(rel: Path) -> Path:
    """Prefer the 2026-10-27 runtime snapshot over HEAD for the files it covers."""
    snapshot = RUNTIME_SNAPSHOT / rel
    return snapshot if snapshot.is_file() else ROOT / rel


def source_bundle(launcher: str = "launch_bootstrap.py") -> bytes:
    entries: list[tuple[Path, Path]] = [(Path("requirements.txt"), ROOT / "requirements.txt")]
    for directory in ("model", "finetune/utils"):
        for path in sorted((ROOT / directory).glob("*.py")):
            rel = path.relative_to(ROOT)
            entries.append((rel, _runtime_file(rel)))
    for name in (
        "config",
        "dataset",
        "asset_metadata",
        "beta_v21",
        "drive_cleanup",
        "validation_precision",
        "train_predictor",
        "export_last_model",
        "kaggle_kronos_base_clean",
        "power_law_sampler",
        "temporal_rank_ic_selection",
    ):
        rel = Path("finetune") / f"{name}.py"
        entries.append((rel, _runtime_file(rel)))
    entries.append(
        (
            Path(f"finetune/base_clean_bootstrap_relay/{launcher}"),
            HERE / launcher,
        )
    )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for arcname, path in entries:
            if not path.is_file():
                raise FileNotFoundError(path)
            info = zipfile.ZipInfo(arcname.as_posix(), (2026, 10, 26, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return stream.getvalue()


def authenticated_owner(session) -> str:
    if session is None:
        view = subprocess.check_output(["kaggle", "config", "view"], text=True)
        for line in view.splitlines():
            if "username:" in line:
                return line.split(":", 1)[1].strip()
        return ""
    _, username = api_session()
    return username


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build (and optionally push) Base-clean Bootstrap long-run kernel. "
            "Default is PREP-only dry build; pass --push only after Elon nod."
        )
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Actually push to Kaggle (Elon nodded for 12h relay)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing idle slug (only with --push)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for default prep-only behaviour (no push)",
    )
    args = parser.parse_args()
    # Treat default and --dry-run as prep-only; --push is the only way out.
    do_push = bool(args.push) and not args.dry_run

    payload = source_bundle()
    digest = hashlib.sha256(payload).hexdigest()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    build = HERE / "build" / SLUG
    build.mkdir(parents=True, exist_ok=True)

    credential = local_credential()
    code = f'''import base64, hashlib, io, json, os, runpy, sys, time, zipfile
from pathlib import Path
print(json.dumps({{"phase": "started", "time": time.time(), "slug": {SLUG!r}}}), flush=True)
os.environ["PYTHONUNBUFFERED"] = "1"
# Injected at push-time only (not committed). Prefer UserSecrets / env first.
os.environ.setdefault("SWANLAB_API_KEY", {credential!r})
os.environ.setdefault("KRONOS_BOOTSTRAP_EXPECTED_RESUME", {str(RESUME_SEGMENT)!r})
os.environ.setdefault("KRONOS_BOOTSTRAP_EXPECTED_COMPLETED", {str(COMPLETED_SEGMENTS)!r})
os.environ.setdefault("KRONOS_BOOTSTRAP_MAX_SEGMENTS", {str(MAX_SEGMENTS)!r})
os.environ.setdefault("KRONOS_BOOTSTRAP_RUNTIME_BUDGET_SECONDS", {str(RUNTIME_BUDGET_SECONDS)!r})
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
namespace = runpy.run_path(str(repo / "finetune/base_clean_bootstrap_relay/launch_bootstrap.py"))
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
        "slug": SLUG,
        "id": metadata["id"],
        "url": f"https://www.kaggle.com/code/{OWNER}/{SLUG}",
        "title": TITLE,
        "owner": OWNER,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": metadata["dataset_sources"],
        "kernel_sources": metadata["kernel_sources"],
        "parent_kernel": PARENT_KERNEL,
        "parent_weights": "last_model",
        "parent_last_sha256": PARENT_LAST_SHA256,
        "parent_best_sha256": PARENT_BEST_SHA256,
        "parent_weights_note": (
            "Same-stage continuation via full Output tree (last_state resumes "
            "optimizer). Bootstrap continues from last only; "
            "KRONOS_DISABLE_BEST_UPDATES=1. First luckfu chunk mounts the "
            "shared user281434 notebook; later chunks self-parent under luckfu."
        ),
        "runtime_snapshot": (
            "finetune/cond_identity/payload_snapshot "
            "(model/kronos.py, finetune/config.py, finetune/train_predictor.py)"
        ),
        "resume_segment": RESUME_SEGMENT,
        "expected_completed_segments": COMPLETED_SEGMENTS,
        "max_segments_per_run": MAX_SEGMENTS,
        "runtime_budget_seconds": RUNTIME_BUDGET_SECONDS,
        "validation": {
            "quick_samples": 2048,
            "interval_segments": 5,
            "condition_ablation_interval_segments": 5,
            "val_batch_size": 64,
            "disable_best_updates": True,
            "chunk_end_required": True,
        },
        "batch": {
            "batch_size": BATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "nproc": NPROC,
            "micro_per_gpu": BATCH_SIZE // ACCUMULATION_STEPS,
            "effective_global_batch": BATCH_SIZE * NPROC,
        },
        "deps": {
            "torch": "2.6.0+cu124 only-binary",
            "numpy": "2.1.3 only-binary",
            "pandas": "2.2.3 only-binary",
            "skip_requirements_txt": True,
        },
        "data": {
            "dataset": DATASET,
            "prepare_data": "flat→processed_datasets via data_manifest SHA",
            "data_manifest_sha256": (
                "b85ccc45f7cd0e895e607e6e75c7a603427bd62a353a016742a52b2943d652cd"
            ),
        },
        "swanlab": {
            "project": "finance",
            "workspace": "roc_fu",
            "experiment_name": "base_clean_0.1_bootstrap",
            "run_id": "base_clean_0_1_bootstrap",
            "resume": "allow",
            "replay_after_segment": COMPLETED_SEGMENTS,
            "note": "SAME board — continue base_clean_0.1_bootstrap (relay chunk)",
        },
        "lineage": {
            "stage": "bootstrap",
            "hf_start": "NeoQuasar/Kronos-base",
            "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
            "no_beta_weights": True,
            "context_layer": 10,
        },
        "source_sha256": digest,
        "source_commit": commit,
        "build_dir": str(build),
        "push_gate": "requires --push after Elon nod; default is prep-only",
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
                    "reason": "prep-only; waiting for Elon nod before --push",
                    "build": str(build),
                    "next_step": (
                        f"python3 finetune/base_clean_bootstrap_relay/push_bootstrap.py "
                        f"--push"
                    ),
                },
                indent=2,
            ),
            flush=True,
        )
        return

    # ---- push path (only with --push) ----
    session, owner = api_session()
    if session is not None and owner and owner != OWNER:
        raise RuntimeError(f"Authenticated owner {owner!r} is not {OWNER}")
    print(json.dumps({"push_transport": "rest" if session is not None else "cli"}), flush=True)

    parent_status = kernel_status(session, PARENT_KERNEL)
    # Self-parent: prior COMPLETE output remains the kernel_source input even
    # when the latest attempt ended ERROR (failed resume). Allow COMPLETE or,
    # with --force, ERROR so a guard fix can overwrite the failed version.
    parent_ok = "COMPLETE" in parent_status
    if (
        not parent_ok
        and args.force
        and PARENT_KERNEL.endswith("/" + SLUG)
        and "ERROR" in parent_status
    ):
        parent_ok = True
        print(
            "Self-parent latest status ERROR; --force re-push will reuse prior "
            "COMPLETE output as kernel_source.",
            flush=True,
        )
    if not parent_ok:
        raise RuntimeError(f"Parent bootstrap kernel not COMPLETE: {parent_status}")

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

    push_message = push_kernel(session, build, metadata, code)
    print(
        json.dumps(
            {
                "pushed": metadata["id"],
                "url": f"https://www.kaggle.com/code/{OWNER}/{SLUG}",
                "server": push_message[:200],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
