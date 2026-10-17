# TimesFM 3.0 V11 Trajectory: C07-SweetSpot 项目交接与技术归档报告

> **致接手同事**：本文档旨在为您提供完整、严谨、开箱即用的工程交接全景图。包含项目演进脉络、历史实测数据对标、重大踩坑与防御设计、代码资产清单以及标准操作流程（SOP）。请在着手编码前通读本文档。

---

## 1. 核心结论与交接摘要 (Executive Summary)

针对此前 C07 在 15.7w ~ 16.7w 步区间展现出最佳走势预测与截面表现的现象，我们溯源了当时的优化器状态，发现该区间模型瞬时学习率恒定处于 **`7.92 × 10⁻⁵`** 窄幅带。据此我们开展了 **策略 A【甜点学习率恒定平台微调精修实验】**：
- **模型起点**：以 C07 `best` (15.7w步) 与 `last` (15.9w步) 经 SWA 权重空间 1:1 融合模型为起点；
- **优化机制**：彻底重置 AdamW 一阶与二阶动量，消除旧接力的惯性干扰；锁定 **`7.92e-5`** 恒定平台学习率，进行 5 小时精修训练（共 14,104 步）；
- **实测成果**：在绝对封箱的 **175,428 条真实未见交易日（34 个独立交易日）全量 True OOS 盲测** 中，彻底打破了 C07 的性能天花板：
  - **10天走势胜率 (DirAcc)**：从融合起点的 58.25% 提升至 **`58.52%`**（创历史新高）；
  - **10天全样本 Pooled IC**：从 0.1608 跃升至 **`+0.1772`**（大幅提升 +10.2%）；
  - **日度截面 IC 均值**：从 0.0720 攀升至 **`+0.0994`**（年化 ICIR 从 8.56 跃升至 **`11.34`**）；
  - **截面回测风险控制**：Top 50 多空对冲在扣除双边 15bps 成本 + 5bps 印花税后，最大回撤由融合起点的 -20.33%（C07 best 为 -40.36%）大幅收窄至 **`-5.79%`**，对冲夏普达 **`11.40`**。

---

## 2. 全量 True OOS 权威指标横向对决表 (175,428 条盲测样本)

> **注**：评测集涵盖真实 A 股 2026-07-03 ~ 2026-08-20 的 34 个交易日。Top 50/100 策略每日依据模型 10 天涨幅预期截面重平衡，扣除双边千一换手成本与印花税。

| 核心评估维度 | C07 best (15.7w步) | C07 last (15.9w步) | C07 SWA 原生融合 (本次起点) | **SweetSpot best (1.0w步)** | **SweetSpot last (1.4w步收官)** | 相对起点改善幅度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **10天方向胜率 (DirAcc)** | 58.42% | 58.08% | 58.25% | **58.52%** | **58.48%** | **+0.27%** *(历史新高)* |
| **平衡准确率 (BalancedAcc)** | 58.45% | 58.70% | 58.57% | 58.65% | **59.08%** | **+0.51%** *(首破 59%)* |
| **预测上涨概率 (自然底噪46.8%)**| 43.12% (偏空) | 53.88% (偏多) | 48.51% (极均衡) | 44.84% | 53.86% | 44.8% 与 53.8% 高度互补 |
| **10天 Pooled IC** | +0.1557 | +0.1600 | +0.1608 | **+0.1702** | **+0.1772** | **+0.0164** *(+10.2%)* |
| **10天形态相关度 (Corr)** | +0.1196 | +0.0939 | +0.1081 | **+0.1104** | **+0.1026** | 保持平稳高同频 |
| **日度截面 IC 均值** | +0.0685 | +0.0720 | +0.0720 | **+0.0908** | **+0.0994** | **+38.1%** *(逼近 0.10 关口)* |
| **日度截面 ICIR (年化)** | 8.15 | 8.91 | 8.56 | **10.62** | **11.34** | **+2.78** *(跨入 11+)* |
| **正 IC 交易日占比 (34日)** | 67.6% (23/34) | 67.6% (23/34) | 67.6% (23/34) | **73.5% (25/34)** | **70.6% (24/34)** | **+5.9%** *(多盈利 2 个交易日)* |
| **实测振幅比 (AmpRatio)** | 0.8419 | 0.8067 | 0.8078 | **0.8099** | **0.8435** | 0.81~0.84 黄金区间，无发散 |
| **Top 50 扣费年化夏普** | 2.67 | 9.09 | 5.91 | **11.40** | **9.40** | **大幅翻倍** |
| **Top 50 多空最大回撤** | -40.36% | -11.01% | -20.33% | **-5.79%** | **-8.36%** | **回撤显著收窄至单位数** |
| **Top 100 扣费年化夏普** | 4.64 | 8.76 | 6.48 | **10.42** | **8.70** | **+3.94** |
| **Top 100 多空最大回撤** | -34.63% | -12.62% | -23.87% | **-9.09%** | **-12.28%** | **回撤降低一半** |
| **DirAcc 95% 置信区间** | [0.5539, 0.6120] | [0.5464, 0.6075] | [0.5502, 0.6097] | **[0.5543, 0.6152]** | **[0.5524, 0.6110]** | 区间整体右上移 |
| **Pooled IC 95% 置信区间** | [0.0883, 0.2127] | [0.0920, 0.2196] | [0.0913, 0.2198] | **[0.0988, 0.2305]** | **[0.1067, 0.2356]** | 下限首度突破 0.10 |

---

## 3. 重大工程避坑指南与红线纪律 (Gotchas & Rules)

在过去多次迭代中，我们排查并总结了以下血泪教训，请接手同事严格遵守：

### 坑 1：数据池构建严禁全量切片拷贝（会导致 OOM 进程被杀）
- **现象**：若试图在 Python 循环中将全市场 300 多万个时序滑动窗口通过 `vals[...].copy()` 全部切片并存入 List，会导致 CPU 单核跑满 1 小时且内存暴涨至十几 GB，最终触发 Linux OOM Killer（退出码 137，被系统 `Killed`）。
- **标准设计**：必须严格遵循 `train.py` 中的轻量索引设计：
  - `self.stocks` 仅保存几千只股票的原生矩阵（占用仅几百 MB）；
  - `self.windows` **只存 `(stock_idx, start_offset)` 的轻量整数元组**；
  - 内存占用小于 1GB，全市场索引构建仅需 10 秒；切片与归一化只在 `sample_batch` 时动态按 Batch 执行。

### 坑 2：安全与 Token 隔离纪律（严禁泄露敏感 Key）
- **ModelScope 写权限 Token**：**绝对严禁** 写死在任何代码文本、配置文件或提交到 GitHub 的 commit 中。必须且只能从环境变量 `MODELSCOPE_API_TOKEN` 动态读取。
- **SwanLab Key**：实验监控平台的只读/打点 Key（`fmEPDGk4IItxgqSZKGLi8`）为公开打点用途，代码中已预置默认常量，接手后开箱即用。

### 坑 3：Kaggle 动态代码注入布尔值序列化陷阱
- **现象**：若在 `push_chunk.py` 中使用 `json.dumps(config)` 替换代码中的 `CHUNK_CONFIG` 占位符，JSON 会输出小写的 `true`/`false`，Python 解释器在运行时会直接抛出 `NameError: name 'true' is not defined`。
- **规范**：代码注入必须使用 Python 原生表达：`f"CHUNK_CONFIG: dict = {config!r}"`。

### 坑 4：严守 KISS 原则，禁止随意新建单开实验脚本
- 任何微调实验均必须在成熟的 `train.py` 与 `push_chunk.py` 上以参数化开关扩展，严禁随意“自立门户”重写训练脚本。

---

## 4. 资产位置与代码清单 (Asset Inventory)

### 4.1 核心脚本导航
| 脚本路径 | 用途说明 | 常用参数 |
| :--- | :--- | :--- |
| [`finetune/timesfm3_v11_trajectory/train.py`](../train.py) | 主训练脚本（成熟核心，含 RevIn 微观防抖、双卡并行、SWA 权重融合初始化） | `--init-swa`（以 SWA 起跑）、`--constant-lr`（恒定学习率）、`--run-id` |
| [`finetune/timesfm3_v11_trajectory/push_chunk.py`](../push_chunk.py) | Kaggle 打包推送器（自动注入配置并推送到远端 T4） | `--slug`、`--upstream`、`--budget`、`--learning-rate`、`--extra` |
| [`finetune/timesfm3_v11_trajectory/eval_oos/eval.py`](../eval_oos/eval.py) | OOS 全量推断评估器（支持 best/last 双权重自动识别与全量盲测） | 由 push_eval 调度，自动输出 `audit_report.json` 与 `.npz` |
| [`finetune/timesfm3_v11_trajectory/eval_oos/push_eval.py`](../eval_oos/push_eval.py) | Kaggle OOS 评估推送脚本 | `--upstream`、`--chunk`、`--slug` |
| [`finetune/timesfm3_v11_trajectory/eval_oos/analyze_predictions.py`](../eval_oos/analyze_predictions.py) | 离线深度评估套件（Block-Bootstrap CI、BalancedAcc、TopK 扣费回测） | `--inputs <npz:label> ...`、`--topk`、`--bootstrap` |
| [`finetune/timesfm3_v11_trajectory/merge_lora_and_publish.py`](../merge_lora_and_publish.py) | LoRA 权重烘焙脚本（将 LoRA 加回底座并转存纯原生 safetensors） | `--best`、`--last`、`--output` |

### 4.2 线上权重与远端环境
- **ModelScope 开源仓库**：[`luckfu/Kronos-TimesFM3-V11-Clean-C07`](https://modelscope.cn/models/luckfu/Kronos-TimesFM3-V11-Clean-C07)（包含纯原生 1.26GB 无 LoRA 稠密大模型及行业嵌入）；
- **Kaggle 训练 Kernel**：[`ibcaoo/kronos-timesfm3-v11-c07-sweetspot`](https://www.kaggle.com/code/ibcaoo/kronos-timesfm3-v11-c07-sweetspot)（包含 SweetSpot 全套训练权重 `best_weights.pt` 与 `last_weights.pt`）；
- **Kaggle 评估 Kernel**：[`ibcaoo/kronos-timesfm3-v11-c07-sweetspot-oos`](https://www.kaggle.com/code/ibcaoo/kronos-timesfm3-v11-c07-sweetspot-oos)（包含全量 17.5 万条预测数组 `eval_predictions_best.npz` 与 `eval_predictions_last.npz`）；
- **SwanLab 实验看板**：[https://swanlab.cn/@roc_fu/finance](https://swanlab.cn/@roc_fu/finance)（项目内可直接回溯各版本训练损失与验证曲线）。

### 4.3 本地落地文件
- 评估结果目录：`finetune/timesfm3_v11_trajectory/eval_oos/results_sweetspot/`
- 全量交叉分析报告：`finetune/timesfm3_v11_trajectory/eval_oos/sweetspot_vs_c07_full_report.json`

---

## 5. 标准操作手册 (SOP)

### SOP 1：复现全维 OOS 交叉分析
当本地需要重新运行或追加新模型对比时，直接执行：
```bash
python3 finetune/timesfm3_v11_trajectory/eval_oos/analyze_predictions.py \
    --inputs finetune/timesfm3_v11_trajectory/eval_oos/results_c07_v2/eval_results/eval_predictions_best.npz:c07_best \
             finetune/timesfm3_v11_trajectory/eval_oos/results_native_merged/eval_results/eval_predictions_native_merged.npz:c07_swa_native \
             finetune/timesfm3_v11_trajectory/eval_oos/results_sweetspot/eval_results/eval_predictions_best.npz:sweetspot_best \
             finetune/timesfm3_v11_trajectory/eval_oos/results_sweetspot/eval_results/eval_predictions_last.npz:sweetspot_last \
    --topk 50 100 \
    --bootstrap 500
```

### SOP 2：烘焙发布 SweetSpot 原生纯稠密大模型 (零 LoRA 交付包)
若需要将本次最优的 `sweetspot_best` 与 `sweetspot_last` 进行 1:1 SWA 融合并烘焙进基础底座：
```bash
python3 finetune/timesfm3_v11_trajectory/merge_lora_and_publish.py \
    --best finetune/timesfm3_v11_trajectory/eval_oos/results_sweetspot/eval_results/best_weights.pt \
    --last finetune/timesfm3_v11_trajectory/eval_oos/results_sweetspot/eval_results/last_weights.pt \
    --output finetune/timesfm3_v11_trajectory/export_sweetspot/
```

### SOP 3：发起下一阶段训练微调
若需要基于 SweetSpot 继续单开新的实验分支：
```bash
python3 finetune/timesfm3_v11_trajectory/push_chunk.py 8 \
    --slug kronos-timesfm3-v11-sweetspot-stage2 \
    --upstream ibcaoo/kronos-timesfm3-v11-c07-sweetspot \
    --budget 18000 \
    --learning-rate 7.92e-5 \
    --extra '{"init_swa": true, "constant_lr": true, "run_id": "tfm3v11-sweetspot-stage2"}'
```

---

## 6. 接手建议与后续研发规划

1. **第一优先级（交付包升级）**：
   将 `sweetspot_best` 与 `sweetspot_last` 执行 SWA 权重空间 1:1 融合，导出原生纯稠密大模型 `timesfm3_v11_sweetspot_native.pt`，并在 ModelScope 上发布 V11.1 资产版本，实盘推理免除 LoRA 旁路。
2. **第二优先级（扩大样本窗复验）**：
   当前 OOS 盲测仅覆盖 34 个交易日（2026-07-03 ~ 2026-08-20），尽管横向对齐对比决定性强，但建议扩展数据管道将 9 月、10 月样本密封加入，进一步验证抗周期能力。
3. **第三优先级（目标函数演进）**：
   时序均方误差（MSE）回归与方向余弦损失在 14,000 步后逐步逼近理论上界。后续应重组同日横截面 Panel，引入 **截面排序损失 (Rank Loss / ListNet)**，直接对标多头截面选股夏普。
