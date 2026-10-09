"""打包并 push 一个接力 chunk 到 Kaggle。

用法:
  python finetune/timesfm3_v8_relay/push_chunk.py 1                 # 第 1 段 (冒烟, 默认 30 分钟)
  python finetune/timesfm3_v8_relay/push_chunk.py 2 --budget 10800  # 第 2 段, 接力第 1 段
  python finetune/timesfm3_v8_relay/push_chunk.py 2 --dry-run       # 只生成, 不 push

chunk N 的 kernel_sources = [正式数据集 kernel, chunk N-1]。
训练脚本启动时会自动在 /kaggle/input 里找到上一段的 relay/state.pt 并续训。
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
OWNER = "wynstonliu"
FORMAL_KERNEL = f"{OWNER}/timesfm3-formal-split-dataset-v1-cpu"


def slug(n: int) -> str:
    return f"kronos-timesfm3-v8-relay-c{n:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chunk", type=int)
    ap.add_argument("--budget", type=int, default=1800, help="本段总秒数(含安装/下载模型)")
    ap.add_argument("--max-steps", type=int, default=200000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    build = HERE / "build" / slug(args.chunk)
    build.mkdir(parents=True, exist_ok=True)
    config = {"budget_seconds": args.budget, "max_steps": args.max_steps}
    source = (HERE / "train.py").read_text()
    source, n = re.subn(r"^CHUNK_CONFIG: dict = \{\}.*$",
                        f"CHUNK_CONFIG: dict = {json.dumps(config)}", source, flags=re.M)
    if n != 1:
        raise RuntimeError("train.py 中找不到 CHUNK_CONFIG 占位行")
    code_file = f"{slug(args.chunk)}.py"
    (build / code_file).write_text(source)

    sources = [FORMAL_KERNEL] + ([f"{OWNER}/{slug(args.chunk - 1)}"] if args.chunk > 1 else [])
    meta = {
        "id": f"{OWNER}/{slug(args.chunk)}",
        "title": f"Kronos TimesFM3 V8 Relay C{args.chunk:02d}",
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "kernel_sources": sources,
        "competition_sources": [],
        "model_sources": [],
    }
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"build": str(build), "config": config, "kernel_sources": sources}, indent=2))
    if not args.dry_run:
        subprocess.check_call(["kaggle", "kernels", "push", "-p", str(build)])


if __name__ == "__main__":
    main()
