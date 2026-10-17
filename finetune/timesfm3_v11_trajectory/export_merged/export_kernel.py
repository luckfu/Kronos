"""TimesFM 3.0 V11 Clean: 原生基础大模型与微调 LoRA SWA 权重彻底烘焙导出套件。

执行逻辑:
1. 从 HuggingFace 下载/加载官方原生 TimesFM 3.0 PyTorch 基础模型;
2. 从 C07 产物中加载 best_weights.pt 与 last_weights.pt 并执行 SWA 参数空间平滑融合;
3. 按照矩阵分配律: W_final = W_base + (lora_b @ lora_a) * (alpha / rank)，将微调增量永久加回原版对应层;
4. 消除所有 LoRA 旁路结构，恢复成最纯净原生的稠密 Transformer 权重;
5. 执行快速前向自检，验证数值健康度与输出形状;
6. 保存导出的全量原生模型: timesfm3_v11_c07_merged.pt。
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.nn as nn

INPUT_ROOT = Path(os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"))
OUTPUT = Path(os.environ.get("KAGGLE_OUTPUT", "/kaggle/working/export_results"))
RANK = 16
ALPHA = 32
SCALE = float(ALPHA / RANK)  # 2.0


def log(event: str, **fields) -> None:
    line = json.dumps({"time": datetime.now(timezone.utc).isoformat(), "event": event, **fields},
                      ensure_ascii=False, default=float)
    print(line, flush=True)


def discover_c07_weights(root: Path):
    best_weights = sorted(root.glob("**/best_weights.pt"))
    last_weights = sorted(root.glob("**/last_weights.pt"))
    splits = sorted(root.glob("**/symbol_split.csv"))

    if not best_weights:
        raise FileNotFoundError(f"未在 {root} 找到 best_weights.pt")
    if not splits:
        raise FileNotFoundError(f"未在 {root} 找到 symbol_split.csv")

    p_best = best_weights[-1]
    p_last = last_weights[-1] if last_weights else p_best
    return p_best, p_last, splits[0]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("export_start")

    # 1. 发现 C07 权重
    p_best, p_last, split_csv = discover_c07_weights(INPUT_ROOT)
    print(f"[Input Check] 找到最佳权重: {p_best}")
    print(f"[Input Check] 找到收官权重: {p_last}")

    import pandas as pd
    sectors = sorted(pd.read_csv(split_csv)["sector"].dropna().unique().tolist())
    num_sectors = len(sectors) + 5
    print(f"[Sector Check] 发现行业数量: {len(sectors)} (总嵌入空间: {num_sectors})")

    # 2. 执行 SWA 权重融合
    print("\n[SWA Merging] 正在对 C07 best 与 last 进行 1:1 权重空间线性融合...")
    w_best = torch.load(p_best, map_location="cpu", weights_only=False)
    w_last = torch.load(p_last, map_location="cpu", weights_only=False)

    w_swa = {}
    for k in w_best.keys():
        w_swa[k] = 0.5 * w_best[k].to(torch.float32) + 0.5 * w_last[k].to(torch.float32)

    lora_weights = {k.replace("lora.", ""): v for k, v in w_swa.items() if k.startswith("lora.")}
    sec_weights = {k.replace("sector.", ""): v for k, v in w_swa.items() if k.startswith("sector.")}
    print(f"[SWA Done] 融合完成: LoRA 矩阵数={len(lora_weights)}, 行业嵌入数={len(sec_weights)}")

    # 3. 加载官方原生基础大模型
    print("\n[Model Loading] 正在从 HuggingFace 加载官方原生 TimesFM 3.0 PyTorch 基础模型...")
    from timesfm3 import TimesFM3Forecaster
    forecaster = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", per_core_batch_size=512)
    base_model = forecaster.model.cpu()
    patch_len = int(base_model.input_patch_len)

    sector_embedding = nn.Embedding(num_sectors, patch_len)
    sector_embedding.load_state_dict(sec_weights, strict=True)
    print(f"[Sector Embedding] 行业嵌入权重匹配成功!")

    # 4. 彻底将 LoRA 权重烘焙 (Bake) 进基础模型的对应 Linear 层
    print(f"\n[LoRA Baking] 开始执行永久合并: W_new = W_base + (lora_b @ lora_a) * {SCALE}...")
    targets = {"query_proj", "key_proj", "value_proj", "out_proj", "ff0", "ff1"}
    merged_linear_count = 0

    for mod_name, module in base_model.named_modules():
        for child_name, child in module.named_children():
            if child_name in targets and isinstance(child, nn.Linear):
                # 寻找该层对应的 lora_a 与 lora_b 键名
                full_prefix = f"{mod_name}.{child_name}" if mod_name else child_name
                a_key = f"{full_prefix}.lora_a"
                b_key = f"{full_prefix}.lora_b"

                if a_key in lora_weights and b_key in lora_weights:
                    la = lora_weights[a_key]  # shape: (rank, in_features)
                    lb = lora_weights[b_key]  # shape: (out_features, rank)

                    # 计算矩阵增量: shape (out_features, in_features)
                    delta_w = (lb @ la) * SCALE

                    # 永久加回 base_linear.weight
                    orig_dtype = child.weight.dtype
                    child.weight.data = (child.weight.data.to(torch.float32) + delta_w).to(orig_dtype)

                    assert torch.isfinite(child.weight.data).all(), f"层 {full_prefix} 合并后出现非有限值！"
                    merged_linear_count += 1

    print(f"[LoRA Baking Success] 成功烘焙融合 {merged_linear_count} 个核心 Linear 投影层！")
    assert merged_linear_count == 200, f"融合层数不匹配! 预期 200 个 Linear (400 个 lora 张量), 实测 {merged_linear_count}"

    # 5. 前向自检：原生基础模型单次无 LoRA 推断测试
    print("\n[Self Check] 执行原生合并模型零 LoRA 前向自检...")
    base_model.eval()
    with torch.no_grad():
        dummy_batch = 2
        dummy_ctx_len = 120
        dummy_horizon = 10
        ctx_padding = (patch_len - dummy_ctx_len % patch_len) % patch_len
        num_ctx_patches = (dummy_ctx_len + ctx_padding) // patch_len
        out_len = int(base_model.output_patch_len)
        padded_horizon = (dummy_horizon + out_len - 1) // out_len * out_len
        num_horizon_patches = padded_horizon // patch_len
        total_patches = num_ctx_patches + num_horizon_patches

        # 构建 8 通道输入
        dummy_continuous = torch.randn(dummy_batch, 7, total_patches, patch_len)
        dummy_sec_ids = torch.tensor([0, 1], dtype=torch.long)
        dummy_sec = sector_embedding(dummy_sec_ids)[:, None, None, :].expand(dummy_batch, 1, total_patches, patch_len)
        dummy_values = torch.cat([dummy_continuous, dummy_sec], dim=1)

        dummy_masks = torch.zeros(dummy_batch, dummy_ctx_len + ctx_padding + padded_horizon, dtype=torch.bool)
        dummy_masks = dummy_masks[:, None, :].expand(dummy_batch, 8, -1).reshape(dummy_batch, 8, -1, patch_len)
        dummy_target = torch.zeros(dummy_batch, 8, total_patches, dtype=torch.bool)
        dummy_target[:, 0, :] = True
        dummy_cpm = torch.zeros(dummy_batch, total_patches, dtype=torch.bool)
        dummy_cpm[:, num_ctx_patches:] = True

        out = base_model(
            {"values": dummy_values, "masks": dummy_masks, "patch_is_target": dummy_target},
            patch_cpm_mask=dummy_cpm
        )
        logits = out["logits"]
        pred_sample = logits[:, 0, num_ctx_patches - 1, :, 4][:, :dummy_horizon]
        print(f"[Self Check OK] 原生前向推断通过! 预测输出形状: {pred_sample.shape}, 数值均值: {pred_sample.mean().item():.4f}")

    # 6. 保存烘焙合并后的纯原生大模型
    output_model_file = OUTPUT / "timesfm3_v11_c07_native_merged.pt"
    print(f"\n[Saving] 正在保存原生合并完整大模型至: {output_model_file}...")
    torch.save({
        "model_state_dict": base_model.state_dict(),
        "sector_embedding_state_dict": sector_embedding.state_dict(),
        "metadata": {
            "source_architecture": "TimesFM 3.0 200M Native Dense",
            "finetune_version": "V11-Clean-C07-SWA (Step 157.5k + 167.8k)",
            "merged_at": datetime.now(timezone.utc).isoformat(),
            "num_sectors": num_sectors,
            "patch_len": patch_len,
            "horizon": 10,
            "input_patch_len": base_model.input_patch_len,
            "output_patch_len": base_model.output_patch_len,
        }
    }, output_model_file)

    size_mb = output_model_file.stat().st_size / (1024 * 1024)
    print(f"[Complete] 原生完整模型落盘完成! 文件大小: {size_mb:.2f} MB, 总耗时: {time.time() - t0:.2f} 秒")
    log("export_success", file=str(output_model_file), size_mb=size_mb)


if __name__ == "__main__":
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "git+https://github.com/google-research/timesfm.git"])
    main()
