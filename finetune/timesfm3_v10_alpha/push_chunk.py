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
OWNER = "smmt315"
DATASET_SOURCE = "smmt315/a-share-120d-temporal-symbol-holdout"


def slug(n: int) -> str:
    return f"kronos-timesfm3-v10-alpha-c{n:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chunk", type=int)
    ap.add_argument("--budget", type=int, default=1800, help="本段运行秒数 (默认 1800 秒 / 30分钟)")
    ap.add_argument("--max-steps", type=int, default=500000)
    ap.add_argument("--val-samples", type=int, default=4096, help="验证集采样数 (默认 4096 条，0 为全量 12.3 万条)")
    ap.add_argument("--eval-every", type=int, default=200, help="每多少步做一次分层验证")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    build = HERE / "build" / slug(args.chunk)
    build.mkdir(parents=True, exist_ok=True)
    config = {
        "budget_seconds": args.budget,
        "max_steps": args.max_steps,
        "val_samples": args.val_samples,
        "eval_every": args.eval_every,
    }
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
        "title": f"Kronos TimesFM3 V10 Alpha C{args.chunk:02d}",
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
