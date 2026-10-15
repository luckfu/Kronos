# TimesFM 3.0 V11 Trajectory (全市场走势大模型微调体系)

## 体系概述

`timesfm3_v11_trajectory` 是 Kronos 走势大模型在 TimesFM-3.0 基础上的最新一代轨迹预测微调体系。  
通过引入 **8 通道特征输入（OHLCVA + 市值百分位 + 行业 Embedding）**、**Rank-16 / Alpha-32 LoRA 扩容适配**、**有上限幂律加权采样** 与 **双向宽带振幅正则**，实现了对全市场股票未来 10 天起伏形态与方向趋势的高维表征。

---

## 核心技术文档索引

- 📄 **C06 阶段性全盘复盘与脏梯度深度技术专报（必读）**：  
  [`docs/c06_training_and_gradient_anomaly_technical_report_cn.md`](docs/c06_training_and_gradient_anomaly_technical_report_cn.md)  
  *涵盖 C01~C06 训练演进、17.5 万条 True OOS 战果复盘、四大工程避坑（15.6分钟大考瓶颈/振幅方差萎缩/Kaggle路径平铺/配额倒计时），以及基于 Step 111,338 真实 Hook 捕获对“脏梯度（非有限梯度）”发源地的第一性原理深度剖析。*
- 📄 **架构与评测规范**：  
  [`docs/architecture_and_eval_spec.md`](docs/architecture_and_eval_spec.md)
- 📊 **C06 True OOS 盲测报告数据**：  
  [`../../reports/timesfm3_v11_clean_c06_true_oos_report.json`](../reports/timesfm3_v11_clean_c06_true_oos_report.json)

---

## 核心战报里程碑（C06 最终收官）

- **全量验证集（123,839 条样本）**：  
  未来 10 天方向胜率达到 **63.01%**，轨迹形态相关度达到 **+0.2460**，走势综合评分 **0.7532**。
- **全量 True OOS 盲测（175,428 条真实未见交易日封箱样本）**：  
  - **Day 2 胜率直接飙升至 57.33%**；  
  - **Day 5 达到全周期顶峰 57.58%**；  
  - 10 天全路径平滑稳定在 57.0%~57.6% 高胜率区间，超越市场弱市底噪（46.8%）达 **+10.23%**；  
  - 正 IC 真实交易日胜率达到 **70.59%**。
