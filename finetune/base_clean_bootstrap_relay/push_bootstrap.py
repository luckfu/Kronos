"""Build/push Base-clean Bootstrap long-run chunk 2 (segs 22–41).

Continues from COMPLETE self slug (resume_segment=22), same SwanLab board.
Pass `--push --force` after Elon nod (self-overwrite idle COMPLETE slug).
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
OWNER = os.environ.get("KRONOS_KAGGLE_OWNER", "user281434")
DATASET = "smmt315/a-share-120d-temporal-symbol-holdout"
# Self-parent: previous COMPLETE of this same slug (chunk 1 segs 2→21).
PARENT_KERNEL = "user281434/kronos-base-clean-0-1-bootstrap"
SLUG = "kronos-base-clean-0-1-bootstrap"
TITLE = "Kronos Base Clean 0.1 Bootstrap"
# Chunk1 last_model SHA (equals best_model at seg21; resume from last only).
PARENT_LAST_SHA256 = (
    "f618552a6f03469271c096122c86b5a883a478d43803c7fb435d0757e4b4807b"
)
PARENT_BEST_SHA256 = PARENT_LAST_SHA256  # frozen; no Best writes this chunk
MAX_SEGMENTS = 20
BATCH_SIZE = 32
ACCUMULATION_STEPS = 8
NPROC = 2


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


def source_bundle() -> bytes:
    files = [ROOT / "requirements.txt"]
    for directory in ("model", "finetune/utils"):
        files.extend(sorted((ROOT / directory).glob("*.py")))
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
        files.append(ROOT / "finetune" / f"{name}.py")
    files.append(HERE / "launch_bootstrap.py")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
            info = zipfile.ZipInfo(
                path.relative_to(ROOT).as_posix(), (2026, 10, 26, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return stream.getvalue()


def authenticated_owner() -> str:
    view = subprocess.check_output(["kaggle", "config", "view"], text=True)
    for line in view.splitlines():
        if "username:" in line:
            return line.split(":", 1)[1].strip()
    listing = subprocess.check_output(
        ["kaggle", "kernels", "list", "-m", "--page-size", "1", "--csv"],
        text=True,
    ).splitlines()
    if len(listing) >= 2:
        return listing[1].split(",")[0].split("/")[0]
    return ""


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
            "Same-stage self-parent via full Output tree (last_state resumes "
            "optimizer). Bootstrap continues from last only; "
            "KRONOS_DISABLE_BEST_UPDATES=1."
        ),
        "resume_segment": 22,
        "expected_completed_segments": 21,
        "max_segments_per_run": MAX_SEGMENTS,
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
            "replay_after_segment": 21,
            "note": "SAME board — continue base_clean_0.1_bootstrap (chunk 2)",
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
    owner = authenticated_owner()
    listing = subprocess.check_output(
        ["kaggle", "kernels", "list", "-m", "--page-size", "1"], text=True
    )
    if OWNER not in listing and owner and owner != OWNER:
        raise RuntimeError(f"Authenticated owner {owner!r} is not {OWNER}")

    parent_status = subprocess.check_output(
        ["kaggle", "kernels", "status", PARENT_KERNEL], text=True
    )
    if "COMPLETE" not in parent_status:
        raise RuntimeError(f"Parent bootstrap kernel not COMPLETE: {parent_status}")

    status = subprocess.run(
        ["kaggle", "kernels", "status", metadata["id"]],
        capture_output=True,
        text=True,
    )
    status_text = status.stdout + status.stderr
    if status.returncode == 0:
        if any(s in status_text for s in ("RUNNING", "QUEUED")):
            raise RuntimeError(f"Target kernel already active: {status_text}")
        if not args.force:
            raise RuntimeError(
                f"Slug already exists ({status_text.strip()}); pass --force to overwrite"
            )
        print("Slug exists and is idle; pushing update", flush=True)
    elif not any(s in status_text.lower() for s in ("not found", "404", "permission")):
        print(f"status probe: {status_text.strip()}", flush=True)

    subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(build)],
        check=True,
    )
    print(
        json.dumps(
            {
                "pushed": metadata["id"],
                "url": f"https://www.kaggle.com/code/{OWNER}/{SLUG}",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
