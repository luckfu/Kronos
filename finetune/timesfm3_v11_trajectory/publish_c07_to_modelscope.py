#!/usr/bin/env python3
"""将 TimesFM 3.0 V11 Clean C07 原生烘焙合并模型与行业嵌入资产发布至魔搭社区 (ModelScope)。

安全约束:
- 严禁任何硬编码 Token！强制通过环境变量 MODELSCOPE_API_TOKEN 认证;
- 如果环境变量未注入，自动尝试安全从用户本地 shell profile 读取，绝不在脚本中留存明文。

工程定位说明:
- 本模型为面向部署交付的稠密工程包 (LoRA 增量已 bake 进底座，推理单次跑、不挂 patch);
- 保留 84 行业专属的 sector_embedding 通道;
- 17.5 万条 True OOS 样本实测: 胜率 58.25%, Pooled IC 0.1608, 年化 ICIR 8.56。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

import torch
from safetensors.torch import save_file

REPO_ID = "luckfu/Kronos-TimesFM3-V11-Clean-C07"

BASE_CONFIG = Path("/Users/fupengcheng/Documents/Kronos/artifacts/models/timesfm3_v11_clean_c02/publish/config.json")
MERGED_PT = Path("/Users/fupengcheng/Documents/Kronos/finetune/timesfm3_v11_trajectory/export_merged/results/export_results/timesfm3_v11_c07_native_merged.pt")
AUDIT_JSON = Path("/Users/fupengcheng/Documents/Kronos/finetune/timesfm3_v11_trajectory/eval_oos/results_native_merged/eval_results/audit_report_native_merged.json")

PUBLISH_DIR = Path("/Users/fupengcheng/Documents/Kronos/artifacts/models/timesfm3_v11_clean_c07/publish")


def get_modelscope_token() -> str:
    # 1. 优先从当前进程环境变量读取
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if token and token.strip():
        return token.strip()

    # 2. 尝试从用户本地 ~/.zshrc 安全解析 (仅在本地执行时起效，绝不打印)
    zshrc = Path.home() / ".zshrc"
    if zshrc.exists():
        text = zshrc.read_text(encoding="utf-8")
        match = re.search(r"export\s+MODELSCOPE_API_TOKEN\s*=\s*['\"]?([a-zA-Z0-9_\-]+)['\"]?", text)
        if match:
            return match.group(1).strip()

    raise ValueError("未检测到有效 MODELSCOPE_API_TOKEN！请确保已在环境变量中配置最新 Token。")


def main():
    print(f"=== [1/5] 准备发布资产目录: {PUBLISH_DIR} ===", flush=True)
    if not MERGED_PT.exists():
        raise FileNotFoundError(f"未找到原生烘焙合并模型文件: {MERGED_PT}")

    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)

    print("正在加载原生合并大模型权重字典...", flush=True)
    ckpt = torch.load(MERGED_PT, map_location="cpu", weights_only=False)
    model_state = ckpt["model_state_dict"]
    sector_state = ckpt["sector_embedding_state_dict"]
    metadata = ckpt.get("metadata", {})

    # 2. 导出标准的 model.safetensors 格式
    safetensors_path = PUBLISH_DIR / "model.safetensors"
    print(f"=== [2/5] 导出原生稠密权重至标准 safetensors 格式 ({safetensors_path.name}) ===", flush=True)
    clean_state = {k: v.contiguous() for k, v in model_state.items()}
    save_file(clean_state, safetensors_path)
    print(f"✓ model.safetensors 导出成功 ({safetensors_path.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)

    # 3. 导出独立 sector_embedding.pt
    sector_path = PUBLISH_DIR / "sector_embedding.pt"
    torch.save(sector_state, sector_path)
    print(f"✓ sector_embedding.pt 导出成功 ({sector_path.stat().st_size / 1024:.2f} KB)", flush=True)

    # 4. 同步 config.json 与 audit_report.json
    if BASE_CONFIG.exists():
        shutil.copy2(BASE_CONFIG, PUBLISH_DIR / "config.json")
    if AUDIT_JSON.exists():
        shutil.copy2(AUDIT_JSON, PUBLISH_DIR / "audit_report.json")

    # 5. 生成 ModelScope configuration.json
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
        "training_checkpoint": "C07 (SWA Step 157.5k + 167.8k)",
        "input_channels": 8,
        "metrics": {
            "oos_direction_acc": 0.5825,
            "oos_pooled_ic": 0.1608,
            "oos_daily_ic_mean": 0.0720,
            "oos_daily_icir_annualized": 8.56,
            "oos_positive_ic_ratio": 0.676
        }
    }
    with open(PUBLISH_DIR / "configuration.json", "w", encoding="utf-8") as f:
        json.dump(ms_config, f, indent=2, ensure_ascii=False)
    print(f"✓ configuration.json 配置已组装", flush=True)

    # 6. 生成模型卡片 README.md
    print(f"=== [3/5] 生成模型卡片 README.md ===", flush=True)
    write_readme()

    # 7. 登录并推送到魔搭社区
    print(f"\n=== [4/5] 验证 ModelScope 凭据并准备推送至: {REPO_ID} ===", flush=True)
    token = get_modelscope_token()
    from modelscope.hub.api import HubApi
    api = HubApi()
    api.login(token)
    print("✓ 鉴权通过，成功连接魔搭社区！", flush=True)

    # 尝试创建仓库 (如果不存在)
    try:
        api.create_model(
            model_id=REPO_ID,
            visibility=1,  # 公开
            chinese_name="Kronos TimesFM 3.0 V11 Clean C07 (可交付原生稠密合并版)",
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
        commit_message="Release Kronos TimesFM 3.0 V11 Clean C07 native merged dense model with sector embedding",
    )
    print(f"✓ 全部资产已成功发布至魔搭社区! 总耗时: {time.time() - t0:.2f} 秒", flush=True)
    print(f"👉 仓库主页: https://modelscope.cn/models/{REPO_ID}")


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
---

# Kronos TimesFM 3.0 V11 Clean C07 (可交付原生稠密合并版)

> **定位声明**：
> 本模型为面向**生产部署侧交付的工程优化稠密包**。它将微调期间产生的 LoRA 增量（C07 步数 15.7w 与 16.7w 的 SWA 权重空间融合）**永久内化吸收回底座 Linear 权重中**。
> **核心工程价值**：推理端彻底消除了 LoRA 旁路的额外矩阵乘法与加法开销，**单次推理即可跑完，无需注入 `attach_lora` 等任何补丁类**，一只股票只算一次。
> **特别说明**：本模型不是通用官方裸模型，输入通道**依然强依赖 84 个申万分类的 Sector Embedding 专用通道**。

---

## 1. 终极大考全量 True OOS 盲测成绩 (175,428 条绝对封箱样本)

本模型在 **175,428 条绝对封箱、从未曝光的真实交易日** 上的全量盲测实测指标如下：

| 评估指标 | 原生烘焙合并模型 (实测) | C07 `best` 单模 (15.7w) | C07 `last` 单模 (16.7w) | 状态说明 |
| :--- | :---: | :---: | :---: | :--- |
| **未来10天走势胜率 (DirAcc)** | **58.25%** | 58.42% | 58.08% | 处于两单模黄金分割中枢 |
| **平衡准确率 (BalancedAcc)** | **58.57%** | 58.45% | 58.70% | 多空两端高度对称均衡 |
| **预测上涨比例 (真实底噪 46.8%)** | **48.51%** | 43.12% (偏空) | 53.88% (偏多) | **彻底中和单模方向偏置** |
| **10天全样本 Pooled IC** | **+0.1608** | +0.1557 | +0.1600 | **突破 0.160 关口** |
| **10天形态相关度 (Corr)** | **+0.1081** | +0.1196 | +0.0939 | 兼顾形态与关键转折点 |
| **日度 IC 均值** | **+0.0720** | +0.0685 | +0.0720 | 维持全场最高日度 IC 水平 |
| **日度 ICIR (年化)** | **8.56** | 8.15 | 8.91 | 显著超越 C07 best (8.15) |
| **正 IC 交易日胜率** | **67.6%** | 67.6% | 67.6% | 保持高位胜率 |
| **实测振幅比 (AmpRatio)** | **0.8078** | 0.8419 | 0.8067 | 振幅健康收敛 |
| **skill_nrmse** | **-0.2791** | -0.3068 | -0.2751 | 相对 C07 best 明显改善 |
| **走势综合评分 (Score)** | **0.6628** | 0.6621 | 0.6608 | **超越两个单模** |

> **数据边界说明**：
> 1. 权重空间 SWA 平均在数学上并不严格等于预测输出平均，实测胜率 58.25% 与 Pooled IC 0.1608 为经验近似结果；
> 2. 本模型不是全部指标单调超越单模（DirAcc 与形态相关度略低于 C07 best，截面 ICIR 略低于 C07 last），其实质是兼顾了全局胜率与截面稳健性的平衡折中态；
> 3. 封箱盲测周期共覆盖 34 个交易日，所测指标用于客观刻画泛化分布，不可直接等同于实盘年化验收。

---

## 2. 极简加载与使用示例 (完全零 LoRA 依赖)

由于权重已彻底物理融合，加载代码极其纯净，就像调用原生 TimesFM 3.0 一样：

```python
import torch
from torch import nn
from modelscope import snapshot_download
from timesfm3 import TimesFM3Forecaster

# 1. 一键下载完整模型资产 (包含原生稠密权重 model.safetensors)
model_dir = snapshot_download("luckfu/Kronos-TimesFM3-V11-Clean-C07")

# 2. 原生直接加载底模 (完全离线运行，零 LoRA 补丁依赖)
device = "cuda" if torch.cuda.is_available() else "cpu"
forecaster = TimesFM3Forecaster.from_pretrained(model_dir, per_core_batch_size=512)
model = forecaster.model.to(device)

# 3. 载入配套的 8 维输入申万行业风格嵌入
patch_len = int(model.input_patch_len)
sector_embedding = nn.Embedding(84 + 5, patch_len).to(device)
sector_state = torch.load(f"{model_dir}/sector_embedding.pt", map_location=device, weights_only=False)
sector_embedding.load_state_dict(sector_state)

print("Kronos TimesFM 3.0 V11 Clean C07 可交付模型就绪，直接执行原生单次推理！")
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
