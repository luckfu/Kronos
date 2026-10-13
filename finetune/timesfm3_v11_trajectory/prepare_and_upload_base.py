#!/usr/bin/env python3
"""整理 TimesFM 3.0 官方底座固化版本至发布目录并增量推送至魔搭社区 (ModelScope)。

目的:
1. 固化官方 google/timesfm-3.0-pytorch 权重 (model.safetensors + config.json)，防范上游版本漂移破损 LoRA 对齐;
2. 保障内网 / 离线环境一站式加载 (无需外网访问 HuggingFace);
3. 规范目录结构:
   - config.json: TimesFM 3.0 官方底座架构定义
   - model.safetensors: 官方底座权重 (~1.32 GB)
   - best_weights.pt: V11 C02 最优 LoRA 适配器 (Step 45,000, 32.9 MB)
   - last_weights.pt: V11 C02 最终接力权重 (Step 45,310, 32.9 MB)
   - lora_config.json: V11 训练超参、特征通道与大考评测指标
   - configuration.json: ModelScope 模型元配置
   - audit_report.json: True OOS 盲测权威战报
   - history.json / summary.json: 全周期大考打卡
   - README.md: 详尽模型卡片与离线加载示例
"""

import json
import os
from pathlib import Path
import shutil
import sys
import time

PUBLISH_DIR = Path("/Users/fupengcheng/Documents/Kronos/artifacts/models/timesfm3_v11_clean_c02/publish")
REPO_ID = "luckfu/Kronos-TimesFM3-V11-Clean-C02"
BASE_COMMIT = "43046b85ec22d584a13f8098c2ed39c889e129c2"


def main():
    print(f"=== [1/4] 检查底座权重文件 ===", flush=True)
    safetensors_path = PUBLISH_DIR / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"未找到 {safetensors_path}，下载可能尚未完成")
    size_mb = safetensors_path.stat().st_size / (1024 * 1024)
    print(f"✓ 发现底座权重 model.safetensors: {size_mb:.2f} MB", flush=True)
    if size_mb < 1200:
        raise ValueError(f"model.safetensors 文件大小异常 ({size_mb:.2f} MB < 1200 MB)")

    print(f"=== [2/4] 规范化配置分离 (LoRA 配置 vs 底座配置) ===", flush=True)
    curr_config = PUBLISH_DIR / "config.json"
    lora_config = PUBLISH_DIR / "lora_config.json"
    
    # 检查 curr_config 是否是原本的 LoRA 配置
    if curr_config.exists():
        with open(curr_config, "r", encoding="utf-8") as f:
            c_data = json.load(f)
        if "model_type" in c_data and c_data.get("model_type") == "timesfm3_lora":
            print("正在将当前 LoRA 配置备份为 lora_config.json...", flush=True)
            with open(lora_config, "w", encoding="utf-8") as f:
                json.dump(c_data, f, indent=2, ensure_ascii=False)
            curr_config.unlink()

    # 写入官方底座的标准 config.json (固定 commit 43046b8)
    base_config = {
        "input_patch_len": 32,
        "input_transform": "identity",
        "linear_detrending_threshold": 0.5,
        "output_patch_len": 64,
        "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "residual_block_config": {
            "activation": "relu",
            "dropout": 0.0,
            "hidden_dims": 1280,
            "identity_skip": False,
            "output_dims": 1280,
            "prenorm": "none",
            "use_bias": False
        },
        "transformer_config": {
            "num_layers": 20,
            "transformer": {
                "attention_norm": "rms",
                "causal_attention": True,
                "debug_no_masking": False,
                "deterministic": True,
                "feedforward_norm": "rms",
                "ff_activation": "relu",
                "hidden_dims": 1280,
                "max_variates": 32,
                "model_dims": 1280,
                "num_heads": 16,
                "paired_token_skip_second": False,
                "qk_norm": "rms",
                "training": True,
                "use_bias": False,
                "use_memory_efficient_attention": True,
                "use_rope_seq": True,
                "use_rope_var": False,
                "use_sdpa": True,
                "v_norm": "none"
            },
            "use_remat": True
        },
        "use_frozen_running_stats": False,
        "use_iterative_cpm_revin": True,
        "use_linear_detrending": True,
        "use_stitching": True,
        "use_variate_attention": True,
        "value_clip": 1e+20,
        "_base_model_source": "google/timesfm-3.0-pytorch",
        "_base_model_commit": BASE_COMMIT
    }
    with open(PUBLISH_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(base_config, f, indent=2)
    print("✓ 官方底座 config.json 配置已固化到位", flush=True)

    print(f"=== [3/4] 更新 README.md 加载示例与版本固化说明 ===", flush=True)
    update_readme()

    print(f"=== [4/4] 增量推送到魔搭社区 (ModelScope: {REPO_ID}) ===", flush=True)
    from modelscope.hub.api import HubApi
    api = HubApi()
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if token:
        api.login(token)
    
    print(f"开始上传整个目录 {PUBLISH_DIR} 到 {REPO_ID} (包含 ~1.32GB 底模权重)...", flush=True)
    t0 = time.time()
    api.upload_folder(
        repo_id=REPO_ID,
        folder_path=str(PUBLISH_DIR),
        commit_message=f"Include frozen Google TimesFM 3.0 base model weights (commit: {BASE_COMMIT[:7]}) for full offline reproducibility",
    )
    print(f"✓ 上传完成! 耗时: {time.time() - t0:.2f} 秒", flush=True)


def update_readme():
    readme_path = PUBLISH_DIR / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    
    # 替换加载示例
    old_call = 'forecaster = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", per_core_batch_size=512)'
    new_call = '# 直接从本地下载目录加载官方底模 (已内置固化 model.safetensors 与 config.json，完全离线运行)\nforecaster = TimesFM3Forecaster.from_pretrained(model_dir, per_core_batch_size=512)'
    if old_call in content:
        content = content.replace(old_call, new_call)
    
    # 在简介中强调底模版本不可变性
    base_note = (
        "> [!IMPORTANT]\n"
        "> **底座版本不可变固化保证 (Deterministic Base Snapshot)**：\n"
        "> 本仓库已将微调所依据的官方底模 `google/timesfm-3.0-pytorch`（Commit: `43046b8`）全量权重 `model.safetensors`（1.32GB）直接内置在模型根目录下！\n"
        "> **核心优势**：\n"
        "> 1. **免疫上游版本漂移**：彻底杜绝上游官方未来更新代码或重构权重导致 LoRA 投影矩阵不匹配的问题；\n"
        "> 2. **100% 离线开箱即用**：下载即可直接前向推断，无须任何海外 HuggingFace 网络连接。\n\n"
    )
    
    if "底座版本不可变固化保证" not in content:
        content = content.replace("## 1. 模型简介\n\n", f"## 1. 模型简介\n\n{base_note}")
        
    readme_path.write_text(content, encoding="utf-8")
    print("✓ README.md 已补充版本固化保证与离线调用示例")


if __name__ == "__main__":
    main()
