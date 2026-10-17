"""打包并推送 TimesFM 3.0 V11 Clean C07-Refine (甜点学习率 7.92e-5 恒定平台微调实验) 到 Kaggle。

设计背景:
- 基于 C07 best 与 last 1:1 SWA 融合初始化;
- 单开 5 小时独立微调分支 (预算 18,000 秒);
- 挂载上游: ibcaoo/kronos-timesfm3-v11-clean-c07
- 挂载数据: smmt315/a-share-120d-temporal-symbol-holdout
"""

import argparse
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).parent
DATASET_SOURCE = "smmt315/a-share-120d-temporal-symbol-holdout"
DEFAULT_UPSTREAM_KERNEL = "ibcaoo/kronos-timesfm3-v11-clean-c07"
SLUG = "kronos-timesfm3-v11-clean-c07-refine"


def get_current_kaggle_user() -> str:
    try:
        out = subprocess.check_output(["kaggle", "config", "view"], text=True)
        for line in out.splitlines():
            if "username:" in line:
                return line.split("username:")[1].strip()
    except Exception:
        pass
    return "ibcaoo"


def main() -> None:
    current_user = get_current_kaggle_user()
    ap = argparse.ArgumentParser(description="推送 C07-Refine 甜点学习率 5 小时精修实验到 Kaggle")
    ap.add_argument("--owner", type=str, default=current_user, help=f"Kaggle 用户名 (当前环境: {current_user})")
    ap.add_argument("--upstream", type=str, default=DEFAULT_UPSTREAM_KERNEL, help="上游 C07 Kernel")
    ap.add_argument("--budget", type=int, default=18000, help="运行秒数 (默认 18000 秒 / 5小时)")
    ap.add_argument("--reserve", type=int, default=900, help="收尾保护秒数 (默认 900 秒 / 15分钟)")
    ap.add_argument("--learning-rate", type=float, default=7.92e-5, help="甜点基准学习率 (默认 7.92e-5)")
    ap.add_argument("--min-lr", type=float, default=7.00e-5, help="平台底噪学习率 (默认 7.00e-5)")
    ap.add_argument("--batch-size", type=int, default=32, help="批大小 (默认 32)")
    ap.add_argument("--eval-batch-size", type=int, default=512, help="评估推断批大小 (默认 512)")
    ap.add_argument("--val-samples", type=int, default=8192, help="日常抽验样本数 (默认 8192)")
    ap.add_argument("--eval-every", type=int, default=2500, help="抽验评估步频 (默认 2500 步)")
    ap.add_argument("--seed", type=int, default=20261017, help="随机种子")
    ap.add_argument("--dry-run", action="store_true", help="仅生成文件不执行推送")
    args = ap.parse_args()

    build = HERE / "build" / SLUG
    build.mkdir(parents=True, exist_ok=True)

    config = {
        "budget_seconds": args.budget,
        "reserve_seconds": args.reserve,
        "learning_rate": args.learning_rate,
        "min_lr": args.min_lr,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "val_samples": args.val_samples,
        "eval_every": args.eval_every,
        "seed": args.seed,
    }

    source = (HERE / "train_refine.py").read_text()
    source, n = re.subn(
        r"^REFINE_CONFIG: dict = \{\}.*$",
        f"REFINE_CONFIG: dict = {json.dumps(config)}",
        source,
        flags=re.M,
    )
    if n != 1:
        raise RuntimeError("train_refine.py 中未找到 REFINE_CONFIG 占位行")

    code_file = f"{SLUG}.py"
    (build / code_file).write_text(source)

    dataset_sources = [DATASET_SOURCE]
    kernel_sources = [args.upstream]

    meta = {
        "id": f"{args.owner}/{SLUG}",
        "title": "Kronos TimesFM3 V11 Clean C07 Refine SweetSpot",
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": dataset_sources,
        "kernel_sources": kernel_sources,
        "competition_sources": [],
        "model_sources": [],
    }

    meta_file = build / "kernel-metadata.json"
    meta_file.write_text(json.dumps(meta, indent=2))

    print(json.dumps({
        "status": "ready",
        "build_dir": str(build),
        "kernel_id": meta["id"],
        "config": config,
        "kernel_sources": kernel_sources,
        "dataset_sources": dataset_sources,
    }, indent=2))

    if not args.dry_run:
        print(f"\n[Pushing] 正在将 {meta['id']} 推送到 Kaggle...", flush=True)
        subprocess.check_call(["kaggle", "kernels", "push", "-p", str(build)])
        print(f"[Success] 推送完成！Kernel ID: {meta['id']}")


if __name__ == "__main__":
    main()
