#!/usr/bin/env python3
"""Push one arm of the C2 rolling-retrain experiment as a Kaggle kernel (REST).

    KAGGLE_API_TOKEN=... KAGGLE_USERNAME=smmt315 \\
    ROLL_ARM=lr1e5 ROLL_LR=1e-5 ROLL_WARMUP_START_LR=1e-6 \\
    python3 finetune/c2_rolling/push_roll.py --push

The kernel mounts the roll package (private, same owner) and the public C2
weights, trains one coverage pass and leaves ``roll_ckpt/segNNN/`` in its output.
Without ``--push`` it only writes the build directory.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "base_clean_bootstrap_relay"))

import relay_auth  # noqa: E402

OWNER = os.environ.get("KAGGLE_USERNAME", "").strip()
ARM = os.environ.get("ROLL_ARM", "").strip()
PACKAGE_SLUG = os.environ.get("ROLL_PACKAGE_SLUG", "c2-roll-newdata-20260703-20260914")
C2_WEIGHTS = "luckfu/kronos-small-cosine-c2-best-weights"
FORWARD = ("ROLL_ARM", "ROLL_LR", "ROLL_WARMUP_START_LR", "ROLL_SNAPSHOT_EVERY",
           "ROLL_COVERAGE_SEED", "ROLL_MAX_SEGMENTS", "ROLL_BATCH_SIZE")
TRAINER_MODULES = ("config", "dataset", "asset_metadata", "beta_v21", "drive_cleanup",
                   "validation_precision", "train_predictor", "export_last_model",
                   "kaggle_kronos_small_v21", "power_law_sampler", "temporal_rank_ic_selection")


def source_bundle() -> bytes:
    files = [ROOT / "finetune" / f"{name}.py" for name in TRAINER_MODULES]
    for directory in ("model", "finetune/utils"):
        files.extend(sorted((ROOT / directory).glob("*.py")))
    files.extend([HERE / "launch_roll.py", HERE / "build_panels.py"])
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 11, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    if not OWNER or not ARM:
        raise SystemExit("KAGGLE_USERNAME and ROLL_ARM are required")
    os.environ.setdefault("ROLL_SNAPSHOT_EVERY", "1")
    for key in ("ROLL_LR", "ROLL_WARMUP_START_LR"):
        if not os.environ.get(key, "").strip():
            raise SystemExit(f"{key} is required")

    payload = source_bundle()
    digest = hashlib.sha256(payload).hexdigest()
    slug = f"c2-roll-{ARM}"
    title = f"C2 roll {ARM}"            # Kaggle derives the slug from the title
    forward = "".join(f'os.environ["{key}"] = {os.environ[key]!r}\n'
                      for key in FORWARD if os.environ.get(key, "").strip())
    code = f'''import base64, hashlib, io, json, os, runpy, shutil, sys, time, zipfile
from pathlib import Path
print(json.dumps({{"phase": "roll_kernel_started", "time": time.time(), "slug": {slug!r}}}), flush=True)
os.environ["PYTHONUNBUFFERED"] = "1"
{forward}payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
repo = Path("/kaggle/working/roll_source")
zipfile.ZipFile(io.BytesIO(payload)).extractall(repo)
try:
    runpy.run_path(str(repo / "finetune/c2_rolling/launch_roll.py"), run_name="__main__")
finally:
    shutil.rmtree(repo, ignore_errors=True)
'''
    datasets = [f"{OWNER}/{PACKAGE_SLUG}", C2_WEIGHTS]
    build = HERE / "build" / slug
    build.mkdir(parents=True, exist_ok=True)
    (build / "train.py").write_text(code)
    compile(code, str(build / "train.py"), "exec")
    metadata = {"id": f"{OWNER}/{slug}", "title": title, "datasets": datasets, "source_sha256": digest,
                "arm": {key: os.environ[key] for key in FORWARD if os.environ.get(key, "").strip()}}
    (build / "build_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    if not args.push:
        print(json.dumps({"status": "READY_NOT_PUSHED", "build": str(build)}))
        return 0
    response = relay_auth.session().post(
        "https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel",
        json={"slug": metadata["id"], "newTitle": title, "text": code, "language": "python",
              "kernelType": "script", "isPrivate": True, "enableGpu": True, "enableInternet": True,
              "machineShape": "NvidiaTeslaT4", "datasetDataSources": datasets, "kernelDataSources": []},
        timeout=180,
    )
    response.raise_for_status()
    print(response.text[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
