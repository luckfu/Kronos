"""TimesFM 3.0 V11 Clean: 权重空间无缝融合 (Model Soups / SWA) 工具。

将多个 Checkpoint 权重（例如 C07 best_weights.pt 与 last_weights.pt）在参数空间线性融合，
生成单一的 merged_weights.pt。
在实盘或测试中，仅需加载该单一权重，单次推理即可获得集成增益，零额外计算开销。

用法示例:
    python finetune/timesfm3_v11_trajectory/merge_weights.py \
        --weights finetune/timesfm3_v11_trajectory/scratch_c07/relay/best_weights.pt:0.5 \
                  finetune/timesfm3_v11_trajectory/scratch_c07/relay/last_weights.pt:0.5 \
        --output finetune/timesfm3_v11_trajectory/scratch_c07/relay/c07_swa_weights.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch


def merge_checkpoints(weight_specs: list[tuple[Path, float]], output_path: Path) -> Path:
    total_weight = sum(w for _, w in weight_specs)
    if abs(total_weight - 1.0) > 1e-4:
        print(f"[Warning] 权重系数和为 {total_weight:.4f}，将自动归一化到 1.0")
        weight_specs = [(p, w / total_weight) for p, w in weight_specs]

    print(f"\n[Weight Merging] 正在准备融合 {len(weight_specs)} 个权重文件:")
    loaded_dicts = []
    for path, weight in weight_specs:
        print(f"  - 权重: {weight:.2f} | 路径: {path}")
        d = torch.load(path, map_location="cpu", weights_only=False)
        loaded_dicts.append((d, weight))

    ref_dict, _ = loaded_dicts[0]
    ref_keys = set(ref_dict.keys())

    # 严格检验键名对齐
    for idx, (d, _) in enumerate(loaded_dicts[1:], 1):
        d_keys = set(d.keys())
        if d_keys != ref_keys:
            missing = ref_keys - d_keys
            extra = d_keys - ref_keys
            raise ValueError(f"权重 {weight_specs[idx][0]} 与首个权重的参数键不匹配! 缺失: {missing}, 多余: {extra}")

    merged_dict = {}
    print(f"\n[Merging] 开始线性融合 {len(ref_keys)} 个参数张量...")

    for k in ref_keys:
        merged_tensor = torch.zeros_like(ref_dict[k], dtype=torch.float32)
        for d, weight in loaded_dicts:
            t = d[k].to(torch.float32)
            merged_tensor.add_(t, alpha=weight)

        if not torch.isfinite(merged_tensor).all():
            raise ValueError(f"参数 {k} 融合后出现 NaN 或 Inf！")

        merged_dict[k] = merged_tensor.to(ref_dict[k].dtype)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_dict, output_path)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[Success] 合并权重已成功落盘: {output_path} ({file_size_mb:.2f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="TimesFM 3.0 V11 模型权重空间融合 (SWA)")
    parser.add_argument("--weights", nargs="+", required=True,
                        help="待融合权重列表，格式: path/to/weights.pt:weight_ratio (例如 weights.pt:0.5)")
    parser.add_argument("--output", type=str, required=True, help="融合后的输出权重路径")
    args = parser.parse_args()

    weight_specs = []
    for item in args.weights:
        if ":" in item:
            path_str, ratio_str = item.rsplit(":", 1)
            weight_specs.append((Path(path_str), float(ratio_str)))
        else:
            weight_specs.append((Path(item), 1.0))

    if len(weight_specs) < 2:
        print("[Error] 融合至少需要提供 2 个权重文件！", file=sys.stderr)
        sys.exit(1)

    merge_checkpoints(weight_specs, Path(args.output))


if __name__ == "__main__":
    main()
