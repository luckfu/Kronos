"""Push the Base-clean sealed-OOS evaluation kernel (REST; no kaggle CLI).

    python3 finetune/oos_eval/base_clean_sealed_oos/push_oos_eval.py --push

Dataset : wynstonliu/a-share-120d-oos-20260803-20260930  (sealed 42d, REFERENCE only)
Parent  : wynstonliu/kronos-base-clean-0-1-main            (checkpoints + model/ + tokenizer)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "finetune" / "base_clean_bootstrap_relay"))

import relay_auth  # noqa: E402

OWNER = os.environ.get("KRONOS_OOS_OWNER", "wynstonliu")
SLUG = os.environ.get("KRONOS_OOS_SLUG", "kronos-base-clean-sealed-oos")
TITLE = os.environ.get("KRONOS_OOS_TITLE", "Kronos Base Clean Sealed OOS")
DATASET = "wynstonliu/a-share-120d-oos-20260803-20260930"
PARENT_KERNEL = os.environ.get("KRONOS_OOS_PARENT", "wynstonliu/kronos-base-clean-0-1-main")

HEADER = '''import os, runpy, sys
from pathlib import Path
src = next(Path("/kaggle/input").glob("**/base_clean_source/model/kronos.py")).parent.parent
os.environ.setdefault("BASE_OOS_SOURCE", str(src))
sys.path.insert(0, str(src))
_here = Path("/kaggle/working/evaluate_base_sealed.py")
'''

EVALUATOR = HERE / "evaluate_base_sealed.py"


def build_code() -> str:
    body = EVALUATOR.read_text()
    return HEADER + "_here.write_text(" + repr(body) + ")\nrunpy.run_path(str(_here), run_name='__main__')\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    code = build_code()
    compile(code, "<kernel>", "exec")
    build = HERE / "build" / SLUG
    build.mkdir(parents=True, exist_ok=True)
    (build / "script.py").write_text(code)
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
        "dataset_sources": [DATASET],
        "kernel_sources": [PARENT_KERNEL],
        "competition_sources": [],
    }
    (build / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    if not args.push:
        print(json.dumps({"status": "READY_NOT_PUSHED", "build": str(build)}))
        return

    session = relay_auth.session()
    payload = {
        "slug": metadata["id"],
        "newTitle": metadata["title"],
        "text": code,
        "language": "python",
        "kernelType": "script",
        "isPrivate": True,
        "enableGpu": True,
        "enableInternet": True,
        "datasetDataSources": [DATASET],
        "kernelDataSources": [PARENT_KERNEL],
        "machineShape": "NvidiaTeslaT4",
    }
    response = session.post(
        "https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel", json=payload, timeout=180
    )
    response.raise_for_status()
    print(json.dumps(response.json())[:400])


if __name__ == "__main__":
    main()
