"""打包并 push TimesFM-3 V10 Alpha 收益率驱动版接力 chunk 到 Kaggle。

当前账号: smmt315
数据集: luckfu/a-share-120d-temporal-symbol-holdout (原生数据集直连)

用法:
  python finetune/timesfm3_v10_alpha/push_chunk.py 1                 # 第 1 段: 30 分钟冒烟 (预算 1800 秒)
  python finetune/timesfm3_v10_alpha/push_chunk.py 2 --budget 43200  # 第 2 段: 12 小时大 Chunk 深度训练 (预算 43200 秒)
"""

import argparse
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).parent
OWNER = "wynstonliu"
DATASET_SOURCE = "smmt315/a-share-120d-temporal-symbol-holdout"


def slug(n: int) -> str:
    return f"kronos-timesfm3-v11-clean-c{n:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chunk", type=int)
    ap.add_argument("--budget", type=int, default=3600, help="本段运行秒数 (默认 3600 秒 / 1小时)")
    ap.add_argument("--max-steps", type=int, default=500000)
    ap.add_argument("--batch-size", type=int, default=32, help="训练批大小 (默认升级为 32: 双卡每卡 16, 抑制梯度噪声)")
    ap.add_argument("--eval-batch-size", type=int, default=512, help="评估推断批大小 (默认 512)")
    ap.add_argument("--val-samples", type=int, default=0, help="验证集采样数 (0 为 100% 全量 123,839 条)")
    ap.add_argument("--power-law-ratio", type=float, default=0.6, help="幂律大波段采样比例 (默认 0.6)")
    ap.add_argument("--amp-loss-weight", type=float, default=0.1, help="振幅防暴走软约束权重 (默认 0.1)")
    ap.add_argument("--eval-every", type=int, default=500, help="每多少步做一次全量大考验证 (默认 500 步)")
    ap.add_argument("--seed", type=int, default=20261009, help="随机种子 (训练中会自动按 seed + chunk 递增)")
    ap.add_argument("--learning-rate", type=float, default=None, help="覆盖初始学习率")
    ap.add_argument("--reserve", type=int, default=300, help="保留缓冲秒数 (默认 300 秒 / 5分钟)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    build = HERE / "build" / slug(args.chunk)
    build.mkdir(parents=True, exist_ok=True)
    config = {
        "budget_seconds": args.budget,
        "reserve_seconds": args.reserve,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "val_samples": args.val_samples,
        "power_law_ratio": args.power_law_ratio,
        "amp_loss_weight": args.amp_loss_weight,
        "eval_every": args.eval_every,
        "seed": args.seed,
    }
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    source = (HERE / "train.py").read_text()
    source, n = re.subn(r"^CHUNK_CONFIG: dict = \{\}.*$",
                        f"CHUNK_CONFIG: dict = {json.dumps(config)}", source, flags=re.M)
    if n != 1:
        raise RuntimeError("train.py 中找不到 CHUNK_CONFIG 占位行")
    code_file = f"{slug(args.chunk)}.py"
    (build / code_file).write_text(source)

    kernel_sources = [f"{OWNER}/{slug(args.chunk - 1)}"] if args.chunk > 1 else []
    meta = {
        "id": f"{OWNER}/{slug(args.chunk)}",
        "title": f"Kronos TimesFM3 V11 Clean C{args.chunk:02d}",
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DATASET_SOURCE],
        "kernel_sources": kernel_sources,
        "competition_sources": [],
        "model_sources": [],
    }
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"build": str(build), "config": config, "dataset_sources": [DATASET_SOURCE], "kernel_sources": kernel_sources}, indent=2))
    if not args.dry_run:
        subprocess.check_call(["kaggle", "kernels", "push", "-p", str(build)])


if __name__ == "__main__":
    main()
