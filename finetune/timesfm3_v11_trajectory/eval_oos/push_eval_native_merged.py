"""打包并推送到 Kaggle 执行原生烘焙合并大模型的全量 True OOS 盲测。

默认评估 C07 导出模型，也支持任意导出 kernel:

    # C07
    python finetune/timesfm3_v11_trajectory/eval_oos/push_eval_native_merged.py

    # SweetSpot
    python finetune/timesfm3_v11_trajectory/eval_oos/push_eval_native_merged.py \
        --upstream ibcaoo/kronos-timesfm3-v11-sweetspot-merged-model-export \
        --slug kronos-timesfm3-v11-sweetspot-merged-oos-audit \
        --tag sweetspot --label SweetSpot
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
KERNEL_SLUG = "kronos-timesfm3-v11-c07-merged-oos-audit"


def main():
    current_user = get_current_kaggle_user()

    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=str, default=None,
                    help="原生模型导出 kernel (默认当前用户的 kronos-timesfm3-v11-c07-merged-model-export)")
    ap.add_argument("--slug", type=str, default=KERNEL_SLUG,
                    help="评估 kernel slug")
    ap.add_argument("--tag", type=str, default="c07",
                    help="模型标识，用于输出文件名 eval_predictions_<tag>_native.npz")
    ap.add_argument("--label", type=str, default="C07",
                    help="人类可读模型名，用于标题/日志")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    upstream = args.upstream or f"{current_user}/kronos-timesfm3-v11-c07-merged-model-export"

    build = HERE / "build" / args.slug
    build.mkdir(parents=True, exist_ok=True)

    source = (HERE / "eval_native_merged.py").read_text()
    # 用 Python repr 注入，避免转义陷阱
    source = source.replace('MODEL_TAG = "c07"', f"MODEL_TAG = {args.tag!r}")
    source = source.replace('MODEL_LABEL = "C07"', f"MODEL_LABEL = {args.label!r}")

    code_file = f"{args.slug}.py"
    (build / code_file).write_text(source)

    meta = {
        "id": f"{current_user}/{args.slug}",
        "title": args.slug.replace("-", " ").title(),
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
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
