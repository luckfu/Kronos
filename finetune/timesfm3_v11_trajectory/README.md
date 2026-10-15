# TimesFM 3.0 V11 Trajectory (全市场走势大模型微调体系)

## 体系概述

`timesfm3_v11_trajectory` 是 Kronos 走势大模型在 TimesFM-3.0 基础上的最新一代轨迹预测微调体系。  
通过引入 **8 通道特征输入（OHLCVA + 市值百分位 + 行业 Embedding）**、**Rank-16 / Alpha-32 LoRA 扩容适配**、**有上限幂律加权采样** 与 **双向宽带振幅正则**，实现了对全市场股票未来 10 天起伏形态与方向趋势的高维表征。

---

## 核心技术文档索引

- 📄 **C06 阶段性全盘复盘与脏梯度深度技术专报（必读）**：  
  [`docs/c06_training_and_gradient_anomaly_technical_report_cn.md`](docs/c06_training_and_gradient_anomaly_technical_report_cn.md)  
  *记录 C01~C06 训练演进、17.5 万条 True OOS 读数、工程事实（全量验证约 15.9 分钟、eval-every 2500、Kaggle 路径平铺、配额预留），以及 Step 111,338 的 Hook：`forward_clean: true`，`output_head` 是第一个接到脏 `backward_grad_output` 的模块。RevIN 开方为零、`1e-7` FP16 下溢和 `tanh(20x)` 仍是待 `autograd.grad(loss, pred)` 检验的候选假说。*
- 📄 **架构与评测规范**：  
  [`docs/architecture_and_eval_spec.md`](docs/architecture_and_eval_spec.md)
- 📊 **C06 True OOS 盲测报告数据**：  
  [`../../reports/timesfm3_v11_clean_c06_true_oos_report.json`](../reports/timesfm3_v11_clean_c06_true_oos_report.json)

---

## C06 评测读数（Step 122,500 权重）

数字来自 [`../../reports/timesfm3_v11_clean_c06_true_oos_report.json`](../reports/timesfm3_v11_clean_c06_true_oos_report.json)。日度方向准确率是符号一致率，这次评测没有交易成本、涨跌停、标的筛选或仓位。

- **全量验证集（123,839 条样本，step 122500）**：  
  未来 10 天方向准确率 **63.01%**，轨迹相关 **+0.2460**，综合分 **0.7532**。验证集 `skill_nrmse` 仍为负（约 -0.08～-0.13）。
- **全量 True OOS（175,428 条封箱样本）**：  
  - Day 1 **54.28%**，Day 2 **57.33%**，Day 5 **57.58%**，Day 10 **57.03%**。Day 2 至 Day 10 落在 57.03%～57.58%。  
  - 相对自然上涨率（约 46.8%）的方向差 `alpha_win_rate_edge` 约 **+10.23** 个百分点。  
  - 正 IC 交易日占比 **70.59%**。  
  - `skill_nrmse` 约 **-0.307**（`nrmse` 0.090，`naive_nrmse` 0.069）。平方误差缺口还在。
