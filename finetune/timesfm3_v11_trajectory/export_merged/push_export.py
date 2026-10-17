"""打包并推送到 Kaggle 执行 TimesFM 3.0 原生大模型与 C07 SWA 权重的彻底烘焙导出。

用法:
    python finetune/timesfm3_v11_trajectory/export_merged/push_export.py
    python finetune/timesfm3_v11_trajectory/export_merged/push_export.py --dry-run
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
    return "ibcaoo"


HERE = Path(__file__).parent
DATASET_SOURCE = "smmt315/a-share-120d-temporal-symbol-holdout"
KERNEL_SLUG = "kronos-timesfm3-v11-c07-merged-export"


def main():
    current_user = get_current_kaggle_user()
    KERNEL_SOURCE = f"{current_user}/kronos-timesfm3-v11-clean-c07"

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="是否启用 GPU (默认关闭，纯 CPU 几十秒内即可搞定，0 配额消耗)")
    args = ap.parse_args()

    build = HERE / "build" / KERNEL_SLUG
    build.mkdir(parents=True, exist_ok=True)

    source = (HERE / "export_kernel.py").read_text()
    code_file = f"{KERNEL_SLUG}.py"
    (build / code_file).write_text(source)

    meta = {
        "id": f"{current_user}/{KERNEL_SLUG}",
        "title": "Kronos TimesFM3 V11 C07 Merged Model Export",
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": bool(args.gpu),
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4" if args.gpu else None,
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
