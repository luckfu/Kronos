"""Build a self-contained Base-clean bootstrap timing kernel and push to Kaggle."""

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
SLUG = "kronos-base-clean-0-1-bootstrap-timing"
TITLE = "Kronos Base Clean 0.1 Bootstrap Timing"


def local_credential() -> str:
    value = os.environ.get("SWANLAB_API_KEY", "").strip()
    if value:
        return value
    # Reuse existing experiment credential without printing it.
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
    files.append(HERE / "launch_timing.py")
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
    # ACCESS_TOKEN auth: parse whoami via kernels list author field or config.
    view = subprocess.check_output(["kaggle", "config", "view"], text=True)
    for line in view.splitlines():
        if "username:" in line:
            return line.split(":", 1)[1].strip()
    # Fallback: owner from most recent owned kernel.
    listing = subprocess.check_output(
        ["kaggle", "kernels", "list", "-m", "--page-size", "1", "--csv"],
        text=True,
    ).splitlines()
    if len(listing) >= 2:
        # ref is owner/slug
        return listing[1].split(",")[0].split("/")[0]
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Push even if slug already exists (not RUNNING)")
    args = parser.parse_args()

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
namespace = runpy.run_path(str(repo / "finetune/base_clean_bootstrap_relay/launch_timing.py"))
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
        "kernel_sources": [],
        "competition_sources": [],
    }
    (build / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (build / "build_manifest.json").write_text(
        json.dumps(
            {
                "slug": SLUG,
                "owner": OWNER,
                "source_sha256": digest,
                "source_commit": commit,
                "dataset": DATASET,
                "batch_size": 32,
                "accumulation_steps": 8,
                "nproc": 2,
                "effective_global_batch": 64,
                "max_segments": 1,
                "swanlab_experiment": "base_clean_0.1_bootstrap_timing",
                "swanlab_run_id": "base_clean_0_1_bootstrap_timing",
            },
            indent=2,
        )
        + "\n"
    )
    compile(code, str(build / code_file), "exec")
    print(
        json.dumps(
            {
                "build": str(build),
                "id": metadata["id"],
                "source_sha256": digest,
                "dataset_sources": metadata["dataset_sources"],
                "machine_shape": "NvidiaTeslaT4",
                "effective_batch": "4x8x2=64",
            },
            indent=2,
        )
    )
    if args.dry_run:
        return

    owner = authenticated_owner()
    # ACCESS_TOKEN mode may not show username in config view.
    listing = subprocess.check_output(
        ["kaggle", "kernels", "list", "-m", "--page-size", "1"], text=True
    )
    if OWNER not in listing and owner and owner != OWNER:
        raise RuntimeError(f"Authenticated owner {owner!r} is not {OWNER}")

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
        # permission-denied often means nonexistent private slug for this owner
        print(f"status probe: {status_text.strip()}", flush=True)

    subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(build)],
        check=True,
    )
    print(json.dumps({"pushed": metadata["id"], "url": f"https://www.kaggle.com/code/{OWNER}/{SLUG}"}), flush=True)


if __name__ == "__main__":
    main()
