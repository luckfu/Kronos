"""Push a self-contained sealed-OOS evaluation kernel (REST; no kaggle CLI).

Bundles the model source and the evaluator into the kernel payload, so the
kernel only needs its parent notebook for *weights + tokenizer* and the sealed
dataset for data.  That avoids depending on whether a parent output happens to
carry `model/`.

    BASE_OOS_LABEL=old_wc_c4_best \
    BASE_OOS_WEIGHTS_GLOB='**/small_0.1_stage2_c2_best_wc_1e5_c4/checkpoints/best_model/model.safetensors' \
    BASE_OOS_CONTEXT_LAYER=6 \
    KRONOS_OOS_PARENTS=user281434/kronos-small-0-1-stage2-c2-best-wc-1e5-c4 \
    KRONOS_OOS_SLUG=kronos-small-oos-old-wc-c4 \
    python3 finetune/oos_eval/base_clean_sealed_oos/push_oos_bundled.py --push
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
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "finetune" / "base_clean_bootstrap_relay"))

import relay_auth  # noqa: E402

OWNER = os.environ.get("KRONOS_OOS_OWNER", "user281434")
SLUG = os.environ.get("KRONOS_OOS_SLUG", "kronos-base-clean-sealed-oos")
TITLE = os.environ.get("KRONOS_OOS_TITLE", "Sealed OOS")
DATASET = os.environ.get(
    "KRONOS_OOS_DATASET", "wynstonliu/a-share-120d-oos-20260803-20260930"
)
PARENTS = os.environ.get("KRONOS_OOS_PARENTS", "").split(",")
PARENTS = [p.strip() for p in PARENTS if p.strip()]
# Extra dataset mounts (for example C2 weights published as a dataset rather
# than a kernel output).  The evaluator globs /kaggle/input/**, so any mount
# works as long as the weights glob resolves to exactly one file.
EXTRA_DATASETS = [
    entry.strip()
    for entry in os.environ.get("KRONOS_OOS_EXTRA_DATASETS", "").split(",")
    if entry.strip()
]
DATASETS = [DATASET, *EXTRA_DATASETS]

LABEL = os.environ.get("BASE_OOS_LABEL", "candidate")
WEIGHTS_GLOB = os.environ.get(
    "BASE_OOS_WEIGHTS_GLOB", "**/checkpoints/best_model/model.safetensors"
)
CONTEXT_LAYER = os.environ.get("BASE_OOS_CONTEXT_LAYER", "10")
SEED = os.environ.get("BASE_OOS_SEED", "20261104")
# Window contract of the mounted package.  The evaluator defaults to the sealed
# 2026-08-03..09-30 numbers, so any other package must restate its own guards.
EXPECT_KEYS = (
    "BASE_OOS_LIMIT",
    "BASE_OOS_STRIDE",
    "BASE_OOS_CONDITION",
    "BASE_OOS_HF_REPO",
    "BASE_OOS_HF_TOKENIZER",
    "BASE_OOS_TOKENIZER",
    "BASE_OOS_EXPECT_SAMPLES",
    "BASE_OOS_EXPECT_DATES",
    "BASE_OOS_EXPECT_FIRST",
    "BASE_OOS_EXPECT_LAST",
    "BASE_OOS_EXPECT_TARGET_END",
)
EXPECT_LINES = "".join(
    f'os.environ.setdefault("{key}", {os.environ[key]!r})\n'
    for key in EXPECT_KEYS
    if os.environ.get(key, "").strip()
)


def source_bundle() -> bytes:
    entries = []
    for path in sorted((ROOT / "model").glob("*.py")):
        entries.append((Path("model") / path.name, path))
    entries.append((Path("evaluate_base_sealed.py"), HERE / "evaluate_base_sealed.py"))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for arcname, path in entries:
            if not path.is_file():
                raise FileNotFoundError(path)
            info = zipfile.ZipInfo(arcname.as_posix(), (2026, 11, 4, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    payload = source_bundle()
    digest = hashlib.sha256(payload).hexdigest()
    code = f'''import base64, hashlib, io, json, os, runpy, sys, zipfile
from pathlib import Path
print(json.dumps({{"phase": "started", "time": __import__("time").time(), "slug": {SLUG!r}, "label": {LABEL!r}}}), flush=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("BASE_OOS_LABEL", {LABEL!r})
os.environ.setdefault("BASE_OOS_WEIGHTS_GLOB", {WEIGHTS_GLOB!r})
os.environ.setdefault("BASE_OOS_CONTEXT_LAYER", {CONTEXT_LAYER!r})
os.environ.setdefault("BASE_OOS_SEED", {SEED!r})
{EXPECT_LINES}payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
repo = Path("/kaggle/working/oos_src")
if repo.exists():
    import shutil
    shutil.rmtree(repo)
zipfile.ZipFile(io.BytesIO(payload)).extractall(repo)
os.environ["BASE_OOS_SOURCE"] = str(repo)
sys.path.insert(0, str(repo))
runpy.run_path(str(repo / "evaluate_base_sealed.py"), run_name="__main__")
'''
    build = HERE / "build" / SLUG
    build.mkdir(parents=True, exist_ok=True)
    (build / "script.py").write_text(code)
    compile(code, str(build / "script.py"), "exec")
    metadata = {
        "id": f"{OWNER}/{SLUG}",
        "title": TITLE,
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": DATASETS,
        "kernel_sources": PARENTS,
        "competition_sources": [],
    }
    (build / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = {
        "id": metadata["id"], "label": LABEL, "context_layer": CONTEXT_LAYER,
        "weights_glob": WEIGHTS_GLOB, "parents": PARENTS, "datasets": DATASETS, "seed": SEED,
        "source_sha256": digest,
    }
    print(json.dumps(summary, indent=2))
    if not args.push:
        print(json.dumps({"status": "READY_NOT_PUSHED", "build": str(build)}))
        return

    session = relay_auth.session()
    response = session.post(
        "https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel",
        json={
            "slug": metadata["id"],
            "newTitle": metadata["title"],
            "text": code,
            "language": "python",
            "kernelType": "script",
            "isPrivate": True,
            "enableGpu": True,
            "enableInternet": True,
            "datasetDataSources": DATASETS,
            "kernelDataSources": PARENTS,
            "machineShape": "NvidiaTeslaT4",
        },
        timeout=180,
    )
    response.raise_for_status()
    print(json.dumps(response.json())[:400])


if __name__ == "__main__":
    main()
