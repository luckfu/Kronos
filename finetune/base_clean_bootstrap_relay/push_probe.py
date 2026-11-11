#!/usr/bin/env python3
"""Push a short bootstrap probe from the pretrained HF weights (REST, no CLI).

Why: `launch_bootstrap.py` only continues an existing relay (self-parent).
A probe starts from `NeoQuasar/Kronos-base` with ONE knob changed per run, so
it needs its own pusher that skips the parent machinery entirely and calls the
stage runner (`kaggle_kronos_base_clean.py`) directly.

    KRONOS_PROBE_SLUG=bc-probe-lr1e7 \
    KRONOS_PREDICTOR_LEARNING_RATE=1e-7 \
    KRONOS_CONDITION_LEARNING_RATE=1e-7 \
    python3 finetune/base_clean_bootstrap_relay/push_probe.py --push
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import relay_auth  # noqa: E402
from push_bootstrap import source_bundle  # noqa: E402

OWNER = os.environ.get("KRONOS_KAGGLE_OWNER", "wynstonliu")
SLUG = os.environ.get("KRONOS_PROBE_SLUG", "bc-probe")
TITLE = os.environ.get("KRONOS_PROBE_TITLE", "BC probe")
DATASET = os.environ.get("KRONOS_BOOTSTRAP_DATASET", "smmt315/a-share-120d-temporal-symbol-holdout")

MAX_SEGMENTS = int(os.environ.get("KRONOS_PROBE_MAX_SEGMENTS", "10"))
RUNTIME_BUDGET_SECONDS = int(os.environ.get("KRONOS_PROBE_RUNTIME_BUDGET_SECONDS", "9000"))

# Every probe knob is forwarded only when set, so the runner's stage defaults
# apply unless the probe explicitly overrides them.
FORWARD_ENV = (
    "KRONOS_PREDICTOR_LEARNING_RATE",
    "KRONOS_CONDITION_LEARNING_RATE",
    "KRONOS_PREDICTOR_WARMUP_START_LR",
    "KRONOS_CONDITION_WARMUP_START_LR",
    "KRONOS_SCHEDULER",
    "KRONOS_SCHEDULER_WARMUP_RATIO",
    "KRONOS_FORECAST_HORIZON_WEIGHTS",
    "KRONOS_HISTORY_LOSS_WEIGHT",
    "KRONOS_PREDICTOR_LOSS_MODE",
    "KRONOS_BATCH_SIZE",
    "KRONOS_ACCUMULATION_STEPS",
    "KRONOS_COVERAGE_SEED",
    "KRONOS_RESET_SECTOR_EMBEDDING",
    "KRONOS_RESET_SIZE_EMBEDDING",
)


def credential() -> str:
    return os.environ.get("SWANLAB_API_KEY", "").strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    payload = source_bundle("launch_bootstrap.py")  # any bundled launcher; unused
    digest = hashlib.sha256(payload).hexdigest()

    forward_lines = "".join(
        f'os.environ["{key}"] = {os.environ[key]!r}\n'
        for key in FORWARD_ENV
        if os.environ.get(key, "").strip()
    )

    code = f'''import base64, hashlib, io, json, os, runpy, sys, time, zipfile
from pathlib import Path
print(json.dumps({{"phase": "probe_started", "time": time.time(), "slug": {SLUG!r}}}), flush=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("SWANLAB_API_KEY", {credential()!r})
os.environ.setdefault("KRONOS_REQUIRE_SWANLAB", "0")
os.environ["KRONOS_BASE_CLEAN_STAGE"] = "bootstrap"
os.environ["KRONOS_BASE_CLEAN_OUTPUT_NAME"] = {SLUG.replace("-", "_")!r}
os.environ["KRONOS_MAX_SEGMENTS_PER_RUN"] = {str(MAX_SEGMENTS)!r}
os.environ["KRONOS_MAIN_RUNTIME_BUDGET_SECONDS"] = {str(RUNTIME_BUDGET_SECONDS)!r}
{forward_lines}payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
repo = Path("/kaggle/working/base_clean_source")
if repo.exists():
    import shutil
    shutil.rmtree(repo)
zipfile.ZipFile(io.BytesIO(payload)).extractall(repo)
# The dataset mounts flat (train_data.pkl at root); the runner expects
# processed_datasets/train_data.pkl.  Reuse the relay's prepare step (it
# handles flat and nested layouts and verifies SHAs), then link metadata.
sys.path.insert(0, str(repo / "finetune/base_clean_bootstrap_relay"))
import launch_bootstrap
data_root = launch_bootstrap.prepare_data(
    Path("/kaggle/input"), Path("/kaggle/working/kronos_base_clean/data"))
meta = data_root / "asset_metadata.csv"
if not meta.exists():
    hits = sorted(Path("/kaggle/input").glob("**/asset_metadata.csv"))
    if hits:
        meta.symlink_to(hits[0].resolve())
os.environ["KRONOS_BASE_CLEAN_DATA_ROOT"] = str(data_root)
os.chdir(repo)
try:
    runpy.run_path(str(repo / "finetune/kaggle_kronos_base_clean.py"), run_name="__main__")
finally:
    # Top-level files land at the FRONT of the output zip; outputs/**/last_model
    # sits behind a ~1.2 GB last_state.pt and was unreachable on truncated
    # downloads.  Export it up front so probes are usable even on ERROR.
    import glob as _glob, shutil as _shutil
    hits = _glob.glob(
        "/kaggle/working/kronos_base_clean/outputs/models/*/checkpoints/last_model/model.safetensors")
    if hits:
        _shutil.copy(hits[0], "/kaggle/working/probe_last_model.safetensors")
        _shutil.copy(hits[0].replace("model.safetensors", "config.json"),
                     "/kaggle/working/probe_last_config.json")
        print(json.dumps({{"phase": "probe_exported", "src": hits[0]}}), flush=True)
'''

    build = HERE / "build" / SLUG
    build.mkdir(parents=True, exist_ok=True)
    (build / "train.py").write_text(code)
    metadata = {
        "id": f"{OWNER}/{SLUG}",
        "title": TITLE,
        "code_file": "train.py",
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
    print(json.dumps({"id": metadata["id"], "max_segments": MAX_SEGMENTS,
                      "budget_s": RUNTIME_BUDGET_SECONDS, "source_sha256": digest}, indent=2))
    if not args.push:
        print(json.dumps({"status": "READY_NOT_PUSHED", "build": str(build)}))
        return 0

    session = relay_auth.session()
    response = session.post(
        "https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel",
        json={
            "slug": metadata["id"],
            "newTitle": TITLE,
            "text": code,
            "language": "python",
            "kernelType": "script",
            "isPrivate": True,
            "enableGpu": True,
            "enableInternet": True,
            "machineShape": "NvidiaTeslaT4",
            "datasetDataSources": [DATASET],
            "kernelDataSources": [],
        },
        timeout=180,
    )
    response.raise_for_status()
    print(response.text[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
