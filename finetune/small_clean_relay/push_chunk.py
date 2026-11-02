"""Build a self-contained private Kaggle script, then submit one clean Small chunk."""

import argparse
import base64
import hashlib
import io
import json
import os
import runpy
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OWNER = "luckfu"
DATASET = "smmt315/a-share-120d-temporal-symbol-holdout"
INITIAL_PARENT = "smmt315/kronos-small-0-1-stage2-cosine-refinement-c2"


def slug(chunk):
    return f"kronos-small-clean-wc-dual-t4-c{chunk:02d}"


def source_bundle():
    files = [ROOT / "requirements.txt"]
    for directory in ("model", "finetune/utils"):
        files.extend(sorted((ROOT / directory).glob("*.py")))
    for name in (
        "config", "dataset", "asset_metadata", "beta_v21", "drive_cleanup",
        "validation_precision", "train_predictor", "export_last_model",
        "kaggle_kronos_small_v21",
    ):
        files.append(ROOT / "finetune" / f"{name}.py")
    files.append(HERE / "launch.py")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 10, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return stream.getvalue()


def local_credential():
    value = os.environ.get("SWANLAB_API_KEY", "").strip()
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient

        return (UserSecretsClient().get_secret("SWANLAB_API_KEY") or "").strip()
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry C01 only after a terminal failure with no published Output")
    args = parser.parse_args()
    if args.chunk < 1:
        parser.error("chunk must be positive")
    if args.retry_failed and args.chunk != 1:
        parser.error("--retry-failed only supports C01 setup failures")
    payload = source_bundle()
    digest = hashlib.sha256(payload).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    build = HERE / "build" / slug(args.chunk)
    if args.chunk > 1:
        previous = HERE / "build" / slug(args.chunk - 1) / "build_manifest.json"
        saved = json.loads(previous.read_text())
        if saved["source_sha256"] != digest:
            raise RuntimeError("Source changed across chunks; use the original source bundle")
        # A Git commit after submission does not change the submitted source bundle.
        commit = saved["source_commit"]
    parent = INITIAL_PARENT if args.chunk == 1 else f"{OWNER}/{slug(args.chunk - 1)}"
    config = {"chunk": args.chunk, "source_sha256": digest,
              "source_commit": commit, "parent_kernel": parent}
    build.mkdir(parents=True, exist_ok=True)
    code = f'''import base64, hashlib, io, json, os, runpy, sys, threading, time, zipfile
from pathlib import Path
print(json.dumps({{"phase": "started", "time": time.time(), "chunk": {args.chunk}}}), flush=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("SWANLAB_API_KEY", {local_credential()!r})
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
repo = Path("/kaggle/working/clean_small_source")
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
namespace = runpy.run_path(str(repo / "finetune/small_clean_relay/launch.py"))
namespace["main"].__globals__["CHUNK_CONFIG"] = {config!r}
namespace["main"]()
'''
    code_file = "train.py"
    (build / code_file).write_text(code)
    metadata = {
        "id": f"{OWNER}/{slug(args.chunk)}",
        "title": f"Kronos Small Clean WC Dual T4 C{args.chunk:02d}",
        "code_file": code_file, "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET],
        "kernel_sources": [parent],
    }
    (build / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (build / "build_manifest.json").write_text(json.dumps(config, indent=2) + "\n")
    compile(code, str(build / code_file), "exec")
    policy = runpy.run_path(str(HERE / "launch.py"))
    print(json.dumps({"build": str(build), **config, "id": metadata["id"],
                      "hard_timeout": 43200,
                      "training_budget": policy["BUDGET_SECONDS"],
                      "max_segments_per_run": policy["MAX_SEGMENTS"]}, indent=2))
    if args.dry_run:
        return
    cli = [sys.executable, "-m", "kaggle"]
    configuration = subprocess.check_output(cli + ["config", "view"], text=True)
    if "- username: luckfu" not in configuration:
        raise RuntimeError("Authenticated Kaggle owner is not luckfu")
    status = subprocess.run(cli + ["kernels", "status", metadata["id"]],
                            capture_output=True, text=True)
    status_text = status.stdout + status.stderr
    if status.returncode == 0:
        if any(s in status_text for s in ("RUNNING", "QUEUED")):
            raise RuntimeError("Target kernel is already active")
        if not args.retry_failed or not any(s in status_text for s in ("ERROR", "FAILED")):
            raise RuntimeError("Chunk slug already exists; inspect it before submitting again")
        listing = subprocess.check_output(cli + ["kernels", "files", metadata["id"],
                                                  "--page-size", "200", "--format", "json"], text=True)
        if json.loads(listing):
            raise RuntimeError("Failed chunk has published Output; inspect recovery state before retrying")
        print("Verified terminal setup failure with no published Output; retrying C01", flush=True)
    elif not any(s in status_text.lower() for s in ("not found", "404")):
        # Kaggle also returns permission-denied for an owner's nonexistent slug.
        # Cross-check the owner's complete listing before creating anything.
        if not any(s in status_text.lower() for s in ("403", "permission", "cannot access kernel")):
            raise RuntimeError(f"Could not verify target status: {status_text}")
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        page = 1
        while True:
            owned = api.kernels_list(mine=True, page=page, page_size=200) or []
            if any(item.ref == metadata["id"] for item in owned):
                raise RuntimeError("Target exists but its status could not be verified")
            if len(owned) < 200:
                break
            page += 1
        print("Verified target slug is absent from the authenticated owner's kernels", flush=True)
    if args.chunk > 1:
        parent_status = subprocess.check_output(cli + ["kernels", "status", parent], text=True)
        if "COMPLETE" not in parent_status:
            raise RuntimeError("Previous chunk is not COMPLETE")
        listing = subprocess.check_output(cli + ["kernels", "files", parent,
                                                  "--page-size", "200", "--format", "json"], text=True)
        if "last_state.pt" not in listing:
            raise RuntimeError("Published parent state missing from Output")
    subprocess.run(cli + ["kernels", "push", "-p", str(build),
                          "--timeout", "43200", "--accelerator", "NvidiaTeslaT4"], check=True)
    subprocess.run(cli + ["kernels", "status", metadata["id"]], check=True)


if __name__ == "__main__":
    main()
