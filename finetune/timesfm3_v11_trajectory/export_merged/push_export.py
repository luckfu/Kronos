"""打包并推送到 Kaggle 执行 TimesFM 3.0 原生大模型烘焙导出。

默认导出 C07 (best+last 1:1 SWA)，也支持任意上游训练 kernel:

    # C07
    python finetune/timesfm3_v11_trajectory/export_merged/push_export.py

    # SweetSpot
    python finetune/timesfm3_v11_trajectory/export_merged/push_export.py \
        --upstream ibcaoo/kronos-timesfm3-v11-c07-sweetspot \
        --slug kronos-timesfm3-v11-sweetspot-merged-export \
        --tag sweetspot --label SweetSpot \
        --finetune-version "V11-SweetSpot-SWA (Step 10,000 + 14,104)"

导出为纯 CPU 任务，不消耗 GPU 配额。
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
KERNEL_SLUG = "kronos-timesfm3-v11-c07-merged-model-export"


def main():
    current_user = get_current_kaggle_user()

    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=str, default=None,
                    help="权重来源 kernel (默认当前用户的 kronos-timesfm3-v11-clean-c07)")
    ap.add_argument("--slug", type=str, default=KERNEL_SLUG,
                    help="导出 kernel slug")
    ap.add_argument("--tag", type=str, default="c07",
                    help="模型标识，用于输出文件名 timesfm3_v11_<tag>_native_merged.pt")
    ap.add_argument("--label", type=str, default="C07",
                    help="人类可读模型名，用于标题/日志")
    ap.add_argument("--finetune-version", type=str,
                    default="V11-Clean-C07-SWA (Step 157.5k + 159.5k)",
                    help="写入 metadata 的血缘描述")
    ap.add_argument("--gpu", action="store_true", help="是否启用 GPU (默认关闭，纯 CPU 几十秒内即可搞定)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    upstream = args.upstream or f"{current_user}/kronos-timesfm3-v11-clean-c07"

    build = HERE / "build" / args.slug
    build.mkdir(parents=True, exist_ok=True)

    source = (HERE / "export_kernel.py").read_text()
    # 用 Python repr 注入，避免 JSON 布尔值/字符串转义陷阱
    source = source.replace('MODEL_TAG = "c07"', f"MODEL_TAG = {args.tag!r}")
    source = source.replace('MODEL_LABEL = "C07"', f"MODEL_LABEL = {args.label!r}")
    source = source.replace(
        'FINETUNE_VERSION = "V11-Clean-C07-SWA (Step 157.5k + 159.5k)"',
        f"FINETUNE_VERSION = {args.finetune_version!r}",
    )

    code_file = f"{args.slug}.py"
    (build / code_file).write_text(source)

    meta = {
        "id": f"{current_user}/{args.slug}",
        "title": args.slug.replace("-", " ").title(),
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": bool(args.gpu),
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4" if args.gpu else None,
        "dataset_sources": [DATASET_SOURCE],
        "kernel_sources": [upstream],
        "competition_sources": [],
        "model_sources": [],
    }
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"build": str(build), "meta": meta}, indent=2))

    if not args.dry_run:
        subprocess.check_call(["kaggle", "kernels", "push", "-p", str(build)])


if __name__ == "__main__":
    main()
