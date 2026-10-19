"""打包并推送到 Kaggle 执行 TimesFM 3.0 V11 Clean (C06/C07/C08) 全量 True OOS (17.5万条) 盲测与预测落盘。

用法:
    python finetune/timesfm3_v11_trajectory/eval_oos/push_eval.py --chunk c07
    python finetune/timesfm3_v11_trajectory/eval_oos/push_eval.py --chunk c08
    python finetune/timesfm3_v11_trajectory/eval_oos/push_eval.py --chunk c06 --dry-run
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


def main():
    current_user = get_current_kaggle_user()

    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", choices=["c06", "c07", "c08", "sweetspot"], default="c07", help="指定评估的实验 Chunk")
    ap.add_argument("--upstream", type=str, default=None, help="显式指定权重来源 kernel (如 ibcaoo/kronos-timesfm3-v11-c07-sweetspot)")
    ap.add_argument("--slug", type=str, default=None, help="显式指定评估 kernel slug")
    ap.add_argument("--dry-run", action="store_true", help="仅生成打包文件而不推送")
    ap.add_argument("--tags", type=str, default=None,
                    help='只评估指定权重 tag，逗号分隔（如 "best" / "best,best_acc"）。默认评估全部发现的权重')
    ap.add_argument("--max-seconds", type=int, default=None,
                    help="全局熔断秒数。1 个权重约 1400s，2 个约 2900s；GPU 配额紧张时按剩余额度下调")
    args = ap.parse_args()

    chunk = args.chunk.lower()
    if args.slug:
        kernel_slug = args.slug
    elif chunk == "sweetspot":
        kernel_slug = "kronos-timesfm3-v11-c07-sweetspot-oos"
    else:
        kernel_slug = f"kronos-timesfm3-v11-clean-{chunk}-oos-audit"

    title = kernel_slug.replace("-", " ").title()

    if args.upstream:
        kernel_source = args.upstream
    elif chunk == "sweetspot":
        kernel_source = f"{current_user}/kronos-timesfm3-v11-c07-sweetspot"
    elif chunk == "c06":
        kernel_source = "user281434/kronos-timesfm3-v11-clean-c06"
    else:
        kernel_source = f"{current_user}/kronos-timesfm3-v11-clean-{chunk}"

    build = HERE / "build" / kernel_slug
    build.mkdir(parents=True, exist_ok=True)

    source = (HERE / "eval.py").read_text()

    # 动态适配 SwanLab 标识与 Chunk 名
    source = source.replace('SWANLAB_RUN_ID = "tfm3v11-clean-c07-oos-audit"', f'SWANLAB_RUN_ID = "{kernel_slug}"')
    source = source.replace('EXPERIMENT_NAME = "timesfm3-v11-clean-c07-oos-audit"', f'EXPERIMENT_NAME = "{kernel_slug}"')

    # GPU 配额紧张时：只跑指定 tag + 下调熔断秒数（写死进 kernel，运行时无需环境变量）
    if args.tags is not None:
        source = source.replace('EVAL_TAGS = ""', f'EVAL_TAGS = "{args.tags}"', 1)
    if args.max_seconds is not None:
        source = source.replace('MAX_SAFE_SECONDS = 3000', f'MAX_SAFE_SECONDS = {args.max_seconds}', 1)
        source = source.replace('MAX_SAFE_SECONDS = 1980', f'MAX_SAFE_SECONDS = {args.max_seconds}', 1)

    code_file = f"{kernel_slug}.py"
    (build / code_file).write_text(source)

    meta = {
        "id": f"{current_user}/{kernel_slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DATASET_SOURCE],
        "kernel_sources": [kernel_source],
        "competition_sources": [],
        "model_sources": [],
    }
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"build": str(build), "meta": meta}, indent=2))

    if not args.dry_run:
        subprocess.check_call(["kaggle", "kernels", "push", "-p", str(build)])


if __name__ == "__main__":
    main()
