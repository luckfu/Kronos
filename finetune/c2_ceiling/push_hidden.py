#!/usr/bin/env python3
"""Push the frozen-C2 hidden-state extraction kernel (REST; no kaggle CLI).

    KAGGLE_API_TOKEN=... KAGGLE_USERNAME=smmt315 python3 finetune/c2_ceiling/push_hidden.py --push

The validation panel is a private dataset of smmt315, so the kernel runs there.
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

SLUG, TITLE = "c2-hidden-extract", "C2 hidden extract"
DATASETS = ["luckfu/a-share-120d-oos-20260717-20261023-v2", "luckfu/kronos-small-cosine-c2-best-weights",
            "smmt315/a-share-120d-temporal-symbol-holdout"]
SOURCES = ["finetune/config.py", "finetune/dataset.py", "finetune/asset_metadata.py",
           "finetune/power_law_sampler.py", "finetune/oos_eval/base_clean_sealed_oos/evaluate_base_sealed.py",
           "finetune/c2_ceiling/extract_hidden.py"]
OOS_CONTRACT = {"BASE_OOS_EXPECT_SAMPLES": "335312", "BASE_OOS_EXPECT_DATES": "65",
                "BASE_OOS_EXPECT_FIRST": "2026-07-17", "BASE_OOS_EXPECT_LAST": "2026-10-23",
                "BASE_OOS_EXPECT_TARGET_END": "2026-11-06"}


def source_bundle() -> bytes:
    files = [ROOT / name for name in SOURCES] + sorted((ROOT / "model").glob("*.py"))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 11, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    owner = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not owner:
        raise SystemExit("KAGGLE_USERNAME is required")
    payload = source_bundle()
    digest = hashlib.sha256(payload).hexdigest()
    contract = "".join(f'os.environ["{key}"] = {value!r}\n' for key, value in OOS_CONTRACT.items())
    code = f'''import base64, hashlib, io, json, os, runpy, shutil, time, zipfile
from pathlib import Path
print(json.dumps({{"phase": "hidden_kernel_started", "time": time.time()}}), flush=True)
os.environ["PYTHONUNBUFFERED"] = "1"
{contract}payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
repo = Path("/kaggle/working/hidden_source")
zipfile.ZipFile(io.BytesIO(payload)).extractall(repo)
try:
    runpy.run_path(str(repo / "finetune/c2_ceiling/extract_hidden.py"), run_name="__main__")
finally:
    shutil.rmtree(repo, ignore_errors=True)
    shutil.rmtree("/kaggle/working/hf_tokenizer", ignore_errors=True)
'''
    build = HERE / "build" / SLUG
    build.mkdir(parents=True, exist_ok=True)
    (build / "script.py").write_text(code)
    compile(code, str(build / "script.py"), "exec")
    print(json.dumps({"id": f"{owner}/{SLUG}", "datasets": DATASETS, "source_sha256": digest}, indent=2))
    if not args.push:
        print(json.dumps({"status": "READY_NOT_PUSHED", "build": str(build)}))
        return 0
    response = relay_auth.session().post(
        "https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel",
        json={"slug": f"{owner}/{SLUG}", "newTitle": TITLE, "text": code, "language": "python",
              "kernelType": "script", "isPrivate": True, "enableGpu": True, "enableInternet": True,
              "machineShape": "NvidiaTeslaT4", "datasetDataSources": DATASETS, "kernelDataSources": []},
        timeout=180,
    )
    response.raise_for_status()
    print(response.text[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
