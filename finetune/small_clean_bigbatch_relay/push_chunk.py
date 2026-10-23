"""Build a self-contained private Kaggle script, then submit one clean Small chunk."""

import argparse
import ast
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
OWNER = os.environ.get("KRONOS_KAGGLE_OWNER", "luckfu")
DATASET = "smmt315/a-share-120d-temporal-symbol-holdout"
INITIAL_PARENT = "wynstonliu/kronos-small-clean-bigbatch-e03b"


def slug(chunk):
    # Round 2 from e03b best; deliberately NOT e04b (closed line).
    # e01 burned; e01b/e01c NCCL; e01d 16-path too slow; e01e single-path + new SwanLab.
    # Local-only slug patch — do not commit.
    if chunk == 1:
        return "kronos-small-clean-bigbatch2-e01e-from-e03b"
    return f"kronos-small-clean-bigbatch2-e{chunk:02d}-from-e03b"


def source_bundle():
    files = [ROOT / "requirements.txt"]
    for directory in ("model", "finetune/utils"):
        files.extend(sorted((ROOT / directory).glob("*.py")))
    for name in (
        "config", "dataset", "asset_metadata", "beta_v21", "drive_cleanup",
        "validation_precision", "train_predictor", "export_last_model",
        "kaggle_kronos_small_v21", "power_law_sampler",
        "temporal_rank_ic_selection",
    ):
        files.append(ROOT / "finetune" / f"{name}.py")
    files.append(HERE / "launch.py")
    manifest = ROOT / "finetune/manifests/temporal_rank_ic_v1/selection_window.json"
    files.append(manifest)
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
    # Reuse the existing experiment credential without printing or changing it.
    path = ROOT / "finetune/kaggle_small_0_1_stage2_wc_dual_t4_kernel/kaggle_small_0_1_stage2_wc_dual_t4.py"
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Dict):
            for key, item in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant) and key.value == "SWANLAB_API_KEY"
                        and isinstance(item, ast.Constant)):
                    return item.value
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry C01 only after a terminal failure with no published Output")
    parser.add_argument("--budget-seconds", type=int, default=0,
                        help="Override the chunk wall-clock budget for this submission")
    parser.add_argument("--allow-source-change", action="store_true",
                        help="Allow a deliberate source change at a chunk boundary")
    parser.add_argument("--parent-kernel", type=str, default="",
                        help="Override the parent kernel (owner/slug)")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Override per-GPU batch size")
    parser.add_argument("--max-segments", type=int, default=0,
                        help="Stop after this many segments (short probe)")
    parser.add_argument("--fresh-probe", action="store_true",
                        help="Use the parent weights but a fresh optimizer (no state resume)")
    parser.add_argument("--run-id", type=str, default="",
                        help="Override the SwanLab run id")
    args = parser.parse_args()
    if args.chunk < 1:
        parser.error("chunk must be positive")
    sys.path.insert(0, str(ROOT / "finetune"))
    from temporal_rank_ic_selection import (
        assert_frozen_manifest_present,
        require_pinned_panel_sha,
        resolve_panel_dataset_slug,
    )
    manifest_path = assert_frozen_manifest_present(ROOT)
    selection_document = json.loads(manifest_path.read_text(encoding="utf-8"))
    require_pinned_panel_sha(selection_document)
    panel_dataset = resolve_panel_dataset_slug(
        os.environ.get("KRONOS_TEMPORAL_RANK_IC_PANEL_DATASET", ""),
        training_dataset=DATASET,
    )
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
            if not args.allow_source_change:
                raise RuntimeError("Source changed across chunks; use the original source bundle")
            print("Source changed deliberately; continuing with the new bundle", flush=True)
        # A Git commit after submission does not change the submitted source bundle.
        commit = saved["source_commit"]
    parent = args.parent_kernel or (INITIAL_PARENT if args.chunk == 1 else f"{OWNER}/{slug(args.chunk - 1)}")
    config = {"chunk": args.chunk, "source_sha256": digest,
              "source_commit": commit, "parent_kernel": parent, "owner": OWNER}
    if args.budget_seconds:
        config["budget_seconds"] = args.budget_seconds
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.max_segments:
        config["max_segments"] = args.max_segments
    if args.fresh_probe:
        config["fresh_probe"] = True
    if args.run_id:
        config["run_id"] = args.run_id
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
namespace = runpy.run_path(str(repo / "finetune/small_clean_bigbatch_relay/launch.py"))
namespace["main"].__globals__["CHUNK_CONFIG"] = {config!r}
namespace["main"]()
'''
    code_file = "train.py"
    (build / code_file).write_text(code)
    metadata = {
        "id": f"{OWNER}/{slug(args.chunk)}",
        "title": ("Kronos Small Clean Bigbatch2 E01d from e03b" if args.chunk == 1 else f"Kronos Small Clean Bigbatch2 E{args.chunk:02d} from e03b"),
        "code_file": code_file, "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        # Training prices and the temporal panel are different mounts. The
        # holdout dataset is not a fallback when the panel mount is missing.
        "dataset_sources": [DATASET, panel_dataset],
        "kernel_sources": [parent],
    }
    (build / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (build / "build_manifest.json").write_text(json.dumps(config, indent=2) + "\n")
    compile(code, str(build / code_file), "exec")
    policy = runpy.run_path(str(HERE / "launch.py"))
    print(json.dumps({"build": str(build), **config, "id": metadata["id"],
                      "dataset_sources": metadata["dataset_sources"],
                      "temporal_rank_ic_panel_dataset": panel_dataset,
                      "hard_timeout": 43200,
                      "training_budget": args.budget_seconds or policy["BUDGET_SECONDS"],
                      "max_segments_per_run": policy["MAX_SEGMENTS"]}, indent=2))
    if args.dry_run:
        return
    cli = ["kaggle"]
    configuration = subprocess.check_output(cli + ["config", "view"], text=True)
    if f"- username: {OWNER}" not in configuration:
        raise RuntimeError(f"Authenticated Kaggle owner is not {OWNER}")
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
        print("Target status is permission-like; treating the slug as absent and proceeding", flush=True)
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
