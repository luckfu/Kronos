# TimesFM 3.0 V11 Trajectory (全市场走势大模型微调体系)

## 体系概述

`timesfm3_v11_trajectory` 是 Kronos 走势大模型在 TimesFM-3.0 基础上的轨迹预测微调体系。  
主要组成：**8 通道输入（close + OHLV/成交额 + 市值百分位 + 行业 Embedding）**、**Rank-16 / Alpha-32 LoRA**、**有上限幂律加权采样**、**双向宽带振幅正则**，预测全市场股票未来 10 天的价格路径与方向。

---

## 文档索引

- 📄 **C06 训练复盘与非有限梯度专题报告**：  
  [`docs/c06_training_and_gradient_anomaly_technical_report_cn.md`](docs/c06_training_and_gradient_anomaly_technical_report_cn.md)  
  *包含 C02~C06 指标对比、C06 True OOS 分日胜率、四项工程问题（评估耗时 / 振幅收缩 / Kaggle 路径平铺 / 时间预算存档），以及基于 Step 111,338 Hook 记录的非有限梯度来源分析。*
- 📄 **架构与评测规范**：  
  [`docs/architecture_and_eval_spec.md`](docs/architecture_and_eval_spec.md)
- 📊 **C06 True OOS 评测数据**：  
  [`../reports/timesfm3_v11_clean_c06_true_oos_report.json`](../reports/timesfm3_v11_clean_c06_true_oos_report.json)

---

## C06 结果摘要（最佳 checkpoint：Step 122,500）

- **验证集（123,839 条）**：10 天方向胜率 **63.01%**，轨迹相关度 **+0.2460**，综合分 **0.7532**。
- **True OOS（175,428 条）**：
  - 10 天方向胜率 57.03%；OOS 期间市场自然上涨率 46.80%，相对"恒猜下跌"基线（53.20%）高 **3.8 个百分点**；
  - Day 2 胜率 57.33%，Day 5 达到峰值 57.58%，Day 2 ~ Day 10 维持在 57.0% ~ 57.6%；
  - 日度 IC 均值 0.0685，正 IC 交易日占比 70.59%，Pooled IC 0.1383，均弱于 C02；
    - 振幅比 `amp_ratio = 0.73`；`skill_nrmse = -0.307`（主动接受可交易振幅导致的方差惩罚，且叠加局部过冲残差）；
- **进展**：非有限梯度（跳批）根因已彻底证实（超大盘股市值排位走平撞上 TimesFM RevIN 除零反传）并在 `train.py` 实施双重阻断修复；待补齐含交易成本的收益回测与平衡准确率。
