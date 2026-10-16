"""打包并推送到 Kaggle 执行 TimesFM 3.0 V11 Clean C02 全量验证集 (12.3万条) 与 True OOS (17.5万条) 终极大考。

用法:
    python finetune/timesfm3_v11_trajectory/eval_oos/push_eval.py
"""

import argparse
import json
from pathlib import Path
import subprocess

def get_current_kaggle_user() -> str:
    try:
        out = subprocess.check_output(["kaggle", "config", "view"], text=True)
        for line in out.splitlines():
            if "username:" in line:
                return line.split("username:")[1].strip()
    except Exception:
        pass
    return "user281434"


HERE = Path(__file__).parent
DATASET_SOURCE = "smmt315/a-share-120d-temporal-symbol-holdout"
KERNEL_SLUG = "kronos-timesfm3-v11-clean-c07-oos-audit"


def main():
    current_user = get_current_kaggle_user()
    KERNEL_SOURCE = f"{current_user}/kronos-timesfm3-v11-clean-c07"

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    build = HERE / "build" / KERNEL_SLUG
    build.mkdir(parents=True, exist_ok=True)

    source = (HERE / "eval.py").read_text()
    code_file = f"{KERNEL_SLUG}.py"
    (build / code_file).write_text(source)

    meta = {
        "id": f"{current_user}/{KERNEL_SLUG}",
        "title": "Kronos TimesFM3 V11 Clean C07 OOS Audit",
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
