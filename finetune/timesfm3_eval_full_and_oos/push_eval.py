"""打包并推送到 Kaggle 执行 TimesFM 3.0 全量验证集 (12.3万条) 与 OOS (17.5万条) 终极大考。

用法:
    python finetune/timesfm3_eval_full_and_oos/push_eval.py
"""

import argparse
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).parent
OWNER = "smmt315"
DATASET_SOURCE = "smmt315/a-share-120d-temporal-symbol-holdout"
KERNEL_SOURCE = "smmt315/kronos-timesfm3-v10-alpha-c04"
KERNEL_SLUG = "kronos-timesfm3-v10-audit-full-and-oos"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    build = HERE / "build" / KERNEL_SLUG
    build.mkdir(parents=True, exist_ok=True)

    source = (HERE / "eval.py").read_text()
    code_file = f"{KERNEL_SLUG}.py"
    (build / code_file).write_text(source)

    meta = {
        "id": f"{OWNER}/{KERNEL_SLUG}",
        "title": "Kronos TimesFM3 V10 Full Val and OOS Audit",
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DATASET_SOURCE],
        "kernel_sources": [KERNEL_SOURCE],
        "competition_sources": [],
        "model_sources": [],
    }
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"build": str(build), "meta": meta}, indent=2))

    if not args.dry_run:
        subprocess.check_call(["kaggle", "kernels", "push", "-p", str(build)])


if __name__ == "__main__":
    main()
