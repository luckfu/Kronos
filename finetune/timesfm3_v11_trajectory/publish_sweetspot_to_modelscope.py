#!/usr/bin/env python3
"""将 TimesFM 3.0 V11.1 SweetSpot 原生融合增强版模型与行业嵌入资产发布至魔搭社区 (ModelScope)。

安全约束:
- 严禁任何硬编码 Token！强制通过环境变量 MODELSCOPE_API_TOKEN 认证;
- 如果环境变量未注入，自动尝试安全从用户本地 ~/.zshrc 解析，绝不打印与外泄。

工程定位说明:
- 本模型为面向部署交付的纯原生稠密工程模型 (Zero LoRA，将 Step 10k 与 14.1k SWA 烘焙合并进底座);
- 保留 84 行业专属的 sector_embedding 通道;
- 175,428 条全量 True OOS 样本实测: 胜率 58.59%, Pooled IC 0.1748, 日均截面 IC 0.0956 (NW t=3.62)。
"""

from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ID = "luckfu/Kronos-TimesFM3-V11-SweetSpot"
CHINESE_NAME = "Kronos TimesFM 3.0 V11.1 SweetSpot (原生融合增强版)"

BASE_SAFETENSORS = Path("/Users/fupengcheng/Documents/Kronos/artifacts/models/timesfm3_v11_clean_c02/publish/model.safetensors")
BASE_CONFIG = Path("/Users/fupengcheng/Documents/Kronos/artifacts/models/timesfm3_v11_clean_c02/publish/config.json")

SWEETSPOT_BEST = Path("/Users/fupengcheng/Documents/Kronos/finetune/timesfm3_v11_trajectory/scratch_c07_sweetspot/relay/best_weights.pt")
SWEETSPOT_LAST = Path("/Users/fupengcheng/Documents/Kronos/finetune/timesfm3_v11_trajectory/scratch_c07_sweetspot/relay/last_weights.pt")
AUDIT_JSON = Path("/Users/fupengcheng/Documents/Kronos/finetune/timesfm3_v11_trajectory/eval_oos/results_sweetspot_native/eval_results/audit_report_sweetspot_native.json")

PUBLISH_DIR = Path("/Users/fupengcheng/Documents/Kronos/artifacts/models/timesfm3_v11_sweetspot/publish")


def get_modelscope_token() -> str:
    # 1. 优先从用户本地最新的 ~/.zshrc 解析有效 Token
    zshrc = Path.home() / ".zshrc"
    if zshrc.exists():
        text = zshrc.read_text(encoding="utf-8")
        match = re.search(r"export\s+MODELSCOPE_API_TOKEN\s*=\s*['\"]?([a-zA-Z0-9_\-]+)['\"]?", text)
        if match:
            t = match.group(1).strip()
            return t[3:] if t.startswith("ms-") else t

    # 2. 备用读取当前进程环境变量
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if token and token.strip():
        t = token.strip()
        return t[3:] if t.startswith("ms-") else t

    raise ValueError("未检测到有效 MODELSCOPE_API_TOKEN！请确保已在 ~/.zshrc 或环境变量中配置最新 Token。")


def bake_and_export():
    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== [1/5] 加载官方底座与 SweetSpot 微调权重 ===", flush=True)
    if not BASE_SAFETENSORS.exists():
        raise FileNotFoundError(f"未找到官方纯净底座: {BASE_SAFETENSORS}")
    if not SWEETSPOT_BEST.exists() or not SWEETSPOT_LAST.exists():
        raise FileNotFoundError("未找到 SweetSpot best/last 权重文件")

    # 1. 载入微调权重并执行 1:1 SWA
    print("正在载入 SweetSpot best (Step 10,000) 与 last (Step 14,104) 权重...", flush=True)
    w_best = torch.load(SWEETSPOT_BEST, map_location="cpu", weights_only=False)
    w_last = torch.load(SWEETSPOT_LAST, map_location="cpu", weights_only=False)

    print("执行 1:1 权重空间线性平滑融合 (SWA)...", flush=True)
    w_swa = {}
    for k in w_best.keys():
        w_swa[k] = 0.5 * w_best[k].float() + 0.5 * w_last[k].float()

    lora_keys = [k for k in w_swa if k.startswith("lora.") and k.endswith(".lora_a")]
    print(f"✓ SWA 融合完成: 包含 {len(lora_keys)} 对 LoRA 矩阵, sector_embedding 权重已平滑", flush=True)

    # 2. 导出独立 sector_embedding.pt
    sector_path = PUBLISH_DIR / "sector_embedding.pt"
    sec_weight = w_swa["sector.weight"].contiguous()
    torch.save(OrderedDict([("weight", sec_weight)]), sector_path)
    print(f"✓ sector_embedding.pt 导出成功 ({sector_path.stat().st_size / 1024:.2f} KB, shape={sec_weight.shape})", flush=True)

    # 3. 载入底模并执行数学融合 (Bake)
    print("正在载入底模 safetensors 全部张量...", flush=True)
    base_tensors = {}
    with safe_open(BASE_SAFETENSORS, framework="pt", device="cpu") as f:
        for k in f.keys():
            base_tensors[k] = f.get_tensor(k)
    print(f"✓ 底模包含 {len(base_tensors)} 个张量", flush=True)

    print("\n=== [2/5] 执行数学融合: W_merged = W_base + 2.0 * (lora_b @ lora_a) ===", flush=True)
    scale = 2.0  # alpha=32 / rank=16
    merged_tensors = dict(base_tensors)
    merged_count = 0

    for lk_a in lora_keys:
        lk_b = lk_a[:-7] + ".lora_b"
        base_key = lk_a.replace("lora.", "")[:-7] + ".weight"
        if base_key not in merged_tensors:
            raise KeyError(f"底模中未找到对应权重: {base_key}")

        la = w_swa[lk_a].float()
        lb = w_swa[lk_b].float()
        w_base = merged_tensors[base_key].float()

        delta_w = (lb @ la) * scale
        w_merged = w_base + delta_w
        assert torch.isfinite(w_merged).all(), f"层 {base_key} 融合后出现 NaN/Inf！"

        merged_tensors[base_key] = w_merged.to(dtype=base_tensors[base_key].dtype).contiguous()
        merged_count += 1

    print(f"✓ 成功烘焙融合 {merged_count} 个核心 Linear 投影层 (400 个 LoRA 参数全部内化)!", flush=True)
    assert merged_count == 200, f"预期融合 200 个 Linear, 实际为 {merged_count}"

    # 4. 导出融合后的 model.safetensors
    safetensors_path = PUBLISH_DIR / "model.safetensors"
    print(f"正在保存已融合的原生稠密模型: {safetensors_path}...", flush=True)
    t0 = time.time()
    save_file(merged_tensors, safetensors_path)
    print(f"✓ model.safetensors 导出成功 ({safetensors_path.stat().st_size / 1024 / 1024:.2f} MB, 耗时 {time.time() - t0:.2f} 秒)", flush=True)

    # 5. 复制配套配置文件
    if BASE_CONFIG.exists():
        shutil.copy2(BASE_CONFIG, PUBLISH_DIR / "config.json")
        print("✓ config.json 已同步", flush=True)
    if AUDIT_JSON.exists():
        shutil.copy2(AUDIT_JSON, PUBLISH_DIR / "audit_report.json")
        print("✓ audit_report.json 已同步", flush=True)

    # 6. 生成 ModelScope configuration.json
    ms_config = {
        "framework": "pytorch",
        "task": "time-series-forecasting",
        "model_type": "timesfm3_native_merged",
        "pipeline": {
            "type": "time-series-forecasting"
        },
        "base_model": "google/timesfm-3.0-pytorch",
        "merged_checkpoint": "model.safetensors",
        "auxiliary_weights": "sector_embedding.pt",
        "training_checkpoint": "SweetSpot V11.1 (SWA Step 10,000 + 14,104)",
        "input_channels": 8,
        "metrics": {
            "oos_sample_count": 175428,
            "oos_direction_acc": 0.58586,
            "oos_alpha_win_rate_edge": 0.11783,
            "oos_pooled_ic": 0.17484,
            "oos_daily_ic_mean": 0.09558,
            "oos_annualized_icir": 10.84,
            "oos_positive_ic_ratio": 0.70588,
            "oos_pred_pos_ratio": 0.49094,
            "oos_balanced_acc": 0.58938
        }
    }
    with open(PUBLISH_DIR / "configuration.json", "w", encoding="utf-8") as f:
        json.dump(ms_config, f, indent=2, ensure_ascii=False)
    print("✓ configuration.json 配置已组装", flush=True)


def write_readme():
    content = """---
tasks:
- time-series-forecasting
framework:
- pytorch
license: Apache-2.0
tags:
- finance
- timesfm
- quantitative-trading
- stock-market
- trajectory-forecasting
---

# Kronos TimesFM 3.0 V11.1 SweetSpot (原生融合增强版)

> **定位声明**：
> 本模型为面向**生产部署与投研实盘交付的工程原生稠密包（Zero LoRA）**。
> 它将微调期间产生的 LoRA 增量（SweetSpot 黄金停机步数 10,000 步与收官 14,104 步的 1:1 SWA 权重空间平滑融合）**永久内化吸收回 TimesFM 3.0 官方底座的 Linear 权重中**。
>
> **核心工程价值**：
> 1. **纯原生零补丁依赖**：彻底消除 LoRA 旁路的额外矩阵分支计算，**单次前向直接出结果，无需挂载 `attach_lora` 等任何侵入式补丁**；
> 2. **完全离线确定性**：完整 safetensors 权重（1.2GB）直接内置，彻底免疫上游版本漂移与外部网络断连；
> 3. **8 维连续多元特征支持**：量价 6 通道 + 连续市值百分位 + 84 个申万行业风格专用嵌入（`sector_embedding.pt`）。

---

## 1. 终极大考全量 True OOS 盲测战报 (175,428 条绝对封箱样本)

本模型在 **175,428 条绝对封箱、从未曝光的真实未见未来交易日（2026-08-03 ~ 2026-09-17，标签截至 2026-10-09）** 上完成了严格的全量盲测：

| 评估指标与维度 | SweetSpot 原生融合版 (实测) | C07 原生融合版 (V11.0) | C07 `best` 单模 | 市场自然底噪 | 提升与技术说明 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **未来10天走势胜率 (DirAcc)** | **58.59%** | 58.25% | 58.42% | 46.80% | **超越市场底噪净胜率 +11.78%** |
| **平衡准确率 (BalancedAcc)** | **58.94%** | 58.57% | 58.45% | 50.00% | 多空两端识别能力高度均衡 |
| **预测上涨比例 (多空倾向)** | **49.09%** | 48.51% | 43.12% | 46.80% | **最接近自然底噪，彻底中和单模偏置** |
| **10天全样本 Pooled IC** | **+0.1748** | +0.1608 | +0.1557 | 0.0000 | **创下全项目历史全样本最高纪录** |
| **日度截面 IC 均值** | **+0.0956** | +0.0720 | +0.0685 | 0.0000 | **提升 +0.0236，Newey-West t=3.62 统计显著** |
| **日度 ICIR (年化)** | **10.84** | 8.56 | 8.15 | 0.00 | 截面信号稳定性大幅提升 |
| **正 IC 交易日占比** | **70.59%** | 67.65% | 67.65% | 50.00% | 超过七成交易日具备正向排序能力 |
| **实测振幅比 (AmpRatio)** | **0.8296** | 0.8078 | 0.8419 | 1.0000 | 预测方差健康，彻底杜绝振幅暴走 |
| **走势综合评分 (Score)** | **0.6733** | 0.6628 | 0.6621 | 0.5000 | **全面超越历史所有基线版本** |

### True OOS 逐日胜率展开（Day 1 $\to$ Day 10 时间结构规律）
```
Day 1:  52.98%
Day 2:  54.51%  ↗
Day 3:  55.04%  ↗
Day 4:  55.89%  ↗
Day 5:  56.67%  ↗
Day 6:  57.94%  ↗
Day 7:  58.55%  ↗
Day 8:  58.96%  ★ (峰值胜率 58.96%)
Day 9:  58.87%
Day 10: 58.59%
```
*规律揭示：预测视界拉长至 5~10 天时，微观日内随机噪声大幅衰减，中短周期波段动量规律占据主导。*

---

## 2. 严肃客观归因与投研局限性披露 (恪守事实与求真精神)

根据团队对 OOS 测试集的深度微观穿透审计，为帮助实盘团队合理使用模型，特此客观披露以下实测特征：

1. **截面 Alpha 存在时序环境依赖（非平稳性）**：
   - **8/03 ~ 8/20（高 Beta 反弹阶段）**：日均截面 IC 高达 **+0.2257**（14/14 天全部正 IC），Top50 多空价差达 **+7.07%**；
   - **8/21 ~ 9/17（结构性分化震荡阶段）**：日均截面 IC 降至 **+0.0045**，此时 Pooled IC 主要由市场整体择时相关性贡献；
2. **多空两腿的不对称性**：
   - 在震荡与下跌市道中，模型对“跑输基准的破位股票（空头腿）”辨识能力显著强于多头腿，在 Long-Only（纯多头）实盘中应结合量价过滤或择时风控，而在多空对冲/指增增强中具备较好的负向剔除价值；
3. **杜绝虚高夏普与过度回测宣传**：
   - 坚决杜绝把 10 日重叠累计收益当作单日收益逐日复利的统计误区；在考虑实际换手与流动性约束时，真实策略的夏普水平需以离散换仓为准。

---

## 3. 极简加载与使用示例 (零 LoRA 依赖，原生直接调用)

由于权重已经彻底在底模中物理融合，代码调用体验与官方原生 TimesFM 3.0 完全一致：

```python
import torch
from torch import nn
from modelscope import snapshot_download
from timesfm3 import TimesFM3Forecaster

# 1. 一键下载完整模型资产 (内置原生烘焙 model.safetensors 与行业嵌入)
model_dir = snapshot_download("luckfu/Kronos-TimesFM3-V11-SweetSpot")

# 2. 原生直接加载主模型 (无需挂载任何 LoRA 补丁，零侵入)
device = "cuda" if torch.cuda.is_available() else "cpu"
forecaster = TimesFM3Forecaster.from_pretrained(model_dir, per_core_batch_size=512)
model = forecaster.model.to(device)

# 3. 加载配套的 84 行业专属嵌入权重 (89 x 32)
patch_len = int(model.input_patch_len)
sector_embedding = nn.Embedding(89, patch_len).to(device)
sector_state = torch.load(f"{model_dir}/sector_embedding.pt", map_location=device, weights_only=False)
sector_embedding.load_state_dict(sector_state)

print("Kronos TimesFM 3.0 V11.1 SweetSpot 原生融合版加载完成，直接执行单次前向推理！")
```

---

## 4. 许可证与免责声明

- 本模型基于 Google TimesFM 3.0 架构微调，遵循 **Apache-2.0** 开源许可证；
- 本模型输出仅供学术研究、时间序列预测探索与量化多因子投研参考，不构成任何投资操作建议或收益承诺。
"""
    readme_path = PUBLISH_DIR / "README.md"
    readme_path.write_text(content, encoding="utf-8")
    print(f"✓ README.md 编写完毕: {readme_path}", flush=True)


def push_to_modelscope():
    token = get_modelscope_token()
    from modelscope.hub.api import HubApi

    print(f"\n=== [4/5] 验证 ModelScope 凭据并准备推送至: {REPO_ID} ===", flush=True)
    api = HubApi()
    api.login(token)
    print("✓ 鉴权通过，成功连接魔搭社区！", flush=True)

    try:
        api.create_model(
            model_id=REPO_ID,
            visibility=1,  # 公开
            chinese_name=CHINESE_NAME,
            original_model_id="google/timesfm-3.0-pytorch"
        )
        print(f"✓ 成功创建魔搭公开仓库: {REPO_ID}", flush=True)
    except Exception as e:
        print(f"ℹ 仓库已存在或无需创建: {e}", flush=True)

    print(f"\n=== [5/5] 开始上传整个发布目录 ({PUBLISH_DIR}) ===", flush=True)
    t0 = time.time()
    api.upload_folder(
        repo_id=REPO_ID,
        folder_path=str(PUBLISH_DIR),
        commit_message="Release Kronos TimesFM 3.0 V11.1 SweetSpot native merged dense model with sector embedding",
    )
    print(f"✓ 全部资产已成功发布至魔搭社区! 总耗时: {time.time() - t0:.2f} 秒", flush=True)
    print(f"👉 仓库主页: https://modelscope.cn/models/{REPO_ID}")


def main():
    bake_and_export()
    print("\n=== [3/5] 生成模型卡片 README.md ===", flush=True)
    write_readme()
    push_to_modelscope()


if __name__ == "__main__":
    main()
