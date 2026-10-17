#!/usr/bin/env python3
"""将 TimesFM 3.0 V11 Clean C03 LoRA 适配器数学合并至主模型权重并发布至魔搭社区 (ModelScope)。

核心步骤:
1. 加载官方底座 model.safetensors 与 C03 最优 LoRA 权重 best_weights.pt (Step 92,500);
2. 数学合并: W_merged = W_base + (alpha / rank) * (lora_b @ lora_a), 其中 rank=16, alpha=32, scale=2.0;
3. 严格校验全部 200 个目标 Linear 投影层 100% 精确完成权重融合;
4. 导出完全合并的单一主模型权重 model.safetensors (~1.32 GB);
5. 导出独立行业风格特征嵌入权重 sector_embedding.pt;
6. 自动组装模型卡片 README.md、架构与审计配置;
7. 上传至魔搭社区: luckfu/Kronos-TimesFM3-V11-Clean-C03。
"""

import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_SAFETENSORS = Path("/Users/fupengcheng/Documents/Kronos/artifacts/models/timesfm3_v11_clean_c02/publish/model.safetensors")
BASE_CONFIG = Path("/Users/fupengcheng/Documents/Kronos/artifacts/models/timesfm3_v11_clean_c02/publish/config.json")
C03_ROOT = Path("/Users/fupengcheng/Documents/Kronos/artifacts/models/timesfm3_v11_clean_c03")
C03_WEIGHTS = C03_ROOT / "relay" / "best_weights.pt"
C03_AUDIT = C03_ROOT / "oos_audit" / "eval_results" / "audit_report.json"
C03_METRICS = C03_ROOT / "relay" / "best_metrics.json"
C03_HISTORY = C03_ROOT / "relay" / "history.json"

PUBLISH_DIR = C03_ROOT / "publish"
REPO_ID = "luckfu/Kronos-TimesFM3-V11-Clean-C03"


def main():
    print(f"=== [1/5] 加载底模与 C03 权重 ===", flush=True)
    if not BASE_SAFETENSORS.exists():
        raise FileNotFoundError(f"未找到底模: {BASE_SAFETENSORS}")
    if not C03_WEIGHTS.exists():
        raise FileNotFoundError(f"未找到 C03 权重: {C03_WEIGHTS}")

    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)

    c03_state = torch.load(C03_WEIGHTS, map_location="cpu", weights_only=False)
    lora_keys = [k for k in c03_state if k.startswith("lora.")]
    sec_keys = [k for k in c03_state if k.startswith("sector.")]
    print(f"✓ C03 权重加载成功: LoRA 参数 {len(lora_keys)} 个, Sector 参数 {len(sec_keys)} 个", flush=True)
    assert len(lora_keys) == 400, f"LoRA 参数数量不为 400, 实际为 {len(lora_keys)}"
    assert len(sec_keys) == 1, f"Sector 参数数量不为 1, 实际为 {len(sec_keys)}"

    # 读取底模所有张量到内存
    print("正在载入底模 safetensors 全部张量...", flush=True)
    base_tensors = {}
    with safe_open(BASE_SAFETENSORS, framework="pt", device="cpu") as f:
        for k in f.keys():
            base_tensors[k] = f.get_tensor(k)
    print(f"✓ 底模包含 {len(base_tensors)} 个张量", flush=True)

    print(f"\n=== [2/5] 执行数学融合: W_merged = W_base + 2.0 * (lora_b @ lora_a) ===", flush=True)
    rank, alpha = 16, 32
    scale = alpha / rank  # 2.0
    merged_count = 0
    merged_tensors = dict(base_tensors)

    # 寻找所有的 lora_a 与 lora_b 配对
    for lk_a in lora_keys:
        if not lk_a.endswith(".lora_a"):
            continue
        lk_b = lk_a[:-7] + ".lora_b"
        if lk_b not in c03_state:
            raise KeyError(f"未找到对应的 lora_b: {lk_b}")

        # 对应的 base_key
        # lora.transformer_stack.layers.0.seq_attn.query_proj.lora_a -> transformer_stack.layers.0.seq_attn.query_proj.weight
        base_key = lk_a.replace("lora.", "")[:-7] + ".weight"
        if base_key not in merged_tensors:
            raise KeyError(f"底模中未找到对应权重: {base_key}")

        lora_a = c03_state[lk_a].float()  # (rank, in_features)
        lora_b = c03_state[lk_b].float()  # (out_features, rank)
        w_base = merged_tensors[base_key].float()  # (out_features, in_features)

        # 增量矩阵
        delta_w = (lora_b @ lora_a) * scale
        w_merged = w_base + delta_w

        # 保持原始张量 dtype 与连续性
        merged_tensors[base_key] = w_merged.to(dtype=base_tensors[base_key].dtype).contiguous()
        merged_count += 1

    print(f"✓ 成功融合 {merged_count} 个核心投影层权重 (400 个 LoRA 参数全部注入完成)!", flush=True)
    assert merged_count == 200, f"预期的融合层数应为 200, 实际为 {merged_count}"

    print(f"\n=== [3/5] 保存合并后的权重文件 ===", flush=True)
    merged_safetensors_path = PUBLISH_DIR / "model.safetensors"
    print(f"正在保存已合并的主模型: {merged_safetensors_path}...", flush=True)
    t0 = time.time()
    save_file(merged_tensors, merged_safetensors_path)
    file_mb = merged_safetensors_path.stat().st_size / (1024 * 1024)
    print(f"✓ model.safetensors 写入完成! 大小: {file_mb:.2f} MB, 耗时: {time.time() - t0:.2f} 秒", flush=True)

    # 保存行业嵌入权重
    sector_path = PUBLISH_DIR / "sector_embedding.pt"
    sec_weight = {k.replace("sector.", ""): v for k, v in c03_state.items() if k.startswith("sector.")}
    torch.save(sec_weight, sector_path)
    print(f"✓ 行业嵌入权重已保存: {sector_path} ({sector_path.stat().st_size / 1024:.2f} KB)", flush=True)

    # 保存底座 config.json
    shutil.copy2(BASE_CONFIG, PUBLISH_DIR / "config.json")
    print(f"✓ 复制底座架构配置: {PUBLISH_DIR / 'config.json'}", flush=True)

    # 保存相关元数据与审计报告
    if C03_AUDIT.exists():
        shutil.copy2(C03_AUDIT, PUBLISH_DIR / "audit_report.json")
    if C03_METRICS.exists():
        shutil.copy2(C03_METRICS, PUBLISH_DIR / "best_metrics.json")
    if C03_HISTORY.exists():
        shutil.copy2(C03_HISTORY, PUBLISH_DIR / "history.json")

    # 生成 ModelScope configuration.json
    ms_config = {
        "framework": "pytorch",
        "task": "time-series-forecasting",
        "model_type": "timesfm3_merged",
        "pipeline": {
            "type": "time-series-forecasting"
        },
        "base_model": "google/timesfm-3.0-pytorch",
        "merged_checkpoint": "model.safetensors",
        "auxiliary_weights": "sector_embedding.pt",
        "step": 92500,
        "input_channels": 8,
        "metrics": {
            "oos_direction_acc": 0.5864,
            "oos_alpha_edge": 0.1183,
            "oos_trajectory_corr": 0.1146,
            "val_direction_acc": 0.6226,
            "val_trajectory_corr": 0.2253
        }
    }
    with open(PUBLISH_DIR / "configuration.json", "w", encoding="utf-8") as f:
        json.dump(ms_config, f, indent=2, ensure_ascii=False)
    print(f"✓ 生成 ModelScope configuration.json", flush=True)

    print(f"\n=== [4/5] 生成模型卡片 README.md ===", flush=True)
    write_readme()

    print(f"\n=== [5/5] 上传推送到魔搭社区 (ModelScope: {REPO_ID}) ===", flush=True)
    from modelscope.hub.api import HubApi
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        raise ValueError("未检测到 MODELSCOPE_API_TOKEN 环境变量，请在环境中配置新 Token！")
    api.login(token)

    # 尝试创建仓库 (如果不存在)
    try:
        api.create_model(
            model_id=REPO_ID,
            visibility=1,  # 1 为公开
            chinese_name="Kronos TimesFM 3.0 V11 Clean C03 (全量权重合并版)",
            original_model_id="google/timesfm-3.0-pytorch"
        )
        print(f"✓ 成功创建魔搭社区公开仓库: {REPO_ID}", flush=True)
    except Exception as e:
        print(f"ℹ 仓库已存在或创建提示: {e}", flush=True)

    print(f"开始推送整个发布目录至 {REPO_ID}...", flush=True)
    t0 = time.time()
    api.upload_folder(
        repo_id=REPO_ID,
        folder_path=str(PUBLISH_DIR),
        commit_message="Publish Kronos TimesFM 3.0 V11 Clean C03 with fully merged LoRA weights into base model",
    )
    print(f"✓ 全部资产已成功发布至魔搭社区! 总耗时: {time.time() - t0:.2f} 秒", flush=True)
    print(f"👉 仓库主页: https://modelscope.cn/models/{REPO_ID}")


def write_readme():
    content = """---
license: Apache-2.0
tags:
  - time-series
  - finance
  - stock-prediction
  - timesfm
  - merged-weights
  - quantitative-finance
pipeline_tag: time-series-forecasting
---

# Kronos TimesFM3 V11 Clean C03 (全量主模型权重合并版)

> [!IMPORTANT]
> **全量权重合并 (Fully Merged Base Model)**：
> 本模型已将经历 **92,500 步深水区微调的 LoRA (Rank 16, Alpha 32) 适配器数学物理合并至 Google TimesFM 3.0 底座主模型**！
> **核心优势**：
> 1. **零 LoRA 包装负担**：无需调用 `attach_lora`，下游用户无需维护 LoRA 动态注入代码；
> 2. **原生开箱即用**：标准 `TimesFM3Forecaster.from_pretrained(model_dir)` 即可直接加载完整的走势大模型；
> 3. **100% 离线自治**：自带完整 `model.safetensors` 与 `config.json`，无需访问海外 HuggingFace。

---

## 1. 终极大考权威评测报告 (175,428 条真实未见交易日)

本模型在 **175,428 条绝对封箱、从未曝光的 True OOS 盲测大考** 中，未来 10 天方向胜率达到 **58.64%**，超越市场自然上涨底噪 **+11.83%**，刷新历史最高战报！

| 评测维度与指标 | Holdout 全量验证集<br>(123,839 条样本) | 全量 True OOS 盲测集<br>(**175,428 条真实未见交易日**) | 上一代 C02 基准 | 市场底噪 / 随机漫步 |
| :--- | :---: | :---: | :---: | :---: |
| **未来10天走势胜率 (DirAcc)** | **62.26%** | **58.64%** | 58.51% | 46.80% |
| **超越市场底噪净胜率 (Alpha Edge)**| **+14.01%** | **+11.83%** | +11.71% | 0.00% |
| **10天波形轨迹形态同频度 (TrajCorr)**| **+0.2253** | **+0.1146** | +0.1050 | 0.0000 |
| **全样本 Pooled IC (10天收益率)**| -- | **+0.1548** | +0.1600 | 0.0000 |
| **走势综合评分 (Trajectory Score)**| **0.7352** | **0.6638** | 0.6376 | 0.4825 |
| **OOS 正 IC 交易日占比** | -- | **70.59%** | 85.29% | 50.00% |

### True OOS 逐日胜率展开（Day 1 $\to$ Day 10 全周期单调递增）

```
Day 1:  52.55%  (超底噪 +5.7%)
Day 2:  54.34%  ↗
Day 3:  55.06%  ↗ (突破 55%)
Day 4:  55.68%  ↗
Day 5:  56.54%  ↗ (超底噪 +9.7%)
Day 6:  57.04%  ↗ (突破 57%)
Day 7:  57.41%  ↗
Day 8:  57.83%  ↗
Day 9:  58.16%  ↗ (突破 58%)
Day 10: 58.64%  ★ (峰值胜率 58.64%, 超底噪 +11.83%)
```

---

## 2. 极简加载与使用示例 (免 LoRA 原生推断)

由于 LoRA 权重已被数学物理合并入主模型，推断代码极度简洁：

```python
import torch
from torch import nn
from modelscope import snapshot_download
from timesfm3 import TimesFM3Forecaster

# 1. 一键下载完整合并模型 (包含已融合权重 model.safetensors)
model_dir = snapshot_download("luckfu/Kronos-TimesFM3-V11-Clean-C03")

# 2. 原生直接加载底模 (完全离线运行，零 LoRA 依赖)
device = "cuda" if torch.cuda.is_available() else "cpu"
forecaster = TimesFM3Forecaster.from_pretrained(model_dir, per_core_batch_size=512)
model = forecaster.model.to(device)

# 3. 加载 8 维输入专属的申万行业风格嵌入
patch_len = int(model.input_patch_len)
sector_embedding = nn.Embedding(89 + 5, patch_len).to(device)
sector_w = torch.load(f"{model_dir}/sector_embedding.pt", map_location=device, weights_only=False)
sector_embedding.load_state_dict(sector_weight if "weight" in sector_w else {"weight": sector_w})

print("Kronos TimesFM 3.0 V11 Clean C03 合并模型加载完毕，随时可执行前向走势推断！")
```

---

## 3. 许可证与免责声明

- 本模型基于 Google TimesFM 3.0，遵循 **Apache-2.0** 许可证；
- 本模型输出仅供学术研究与量化时序建模探索，不构成任何投资建议或收益承诺。
"""
    readme_path = PUBLISH_DIR / "README.md"
    readme_path.write_text(content, encoding="utf-8")
    print(f"✓ README.md 编写完毕: {readme_path}", flush=True)


if __name__ == "__main__":
    main()
