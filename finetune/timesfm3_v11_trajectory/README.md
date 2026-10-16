# TimesFM 3.0 V11 Trajectory (全市场走势大模型微调体系)

## 体系概述

`timesfm3_v11_trajectory` 是 Kronos 走势大模型在 TimesFM-3.0 基础上的轨迹预测微调体系。  
主要组成：**8 通道输入（close + OHLV/成交额 + 市值百分位 + 行业 Embedding）**、**Rank-16 / Alpha-32 LoRA**、**有上限幂律加权采样**、**双向宽带振幅正则**，预测全市场股票未来 10 天的价格路径与方向。

---

## 文档索引

- 📄 **C08 训练收官与 True OOS 盲测技术复盘报告**：  
  [`docs/c08_training_and_oos_technical_report_cn.md`](docs/c08_training_and_oos_technical_report_cn.md)  
  *包含 C01~C08 全阶段指标演进、同分布验证集(64.99%)与 True OOS(56.84%)脱钩深度复盘、实测振幅比(0.85)与负 Skill 恶化机理解析、横截面排序瓶颈定性，以及 NaN 完全根除实证。*
- 📄 **C06 训练复盘与非有限梯度专题报告**：  
  [`docs/c06_training_and_gradient_anomaly_technical_report_cn.md`](docs/c06_training_and_gradient_anomaly_technical_report_cn.md)  
- 📄 **架构与评测规范**：  
  [`docs/architecture_and_eval_spec.md`](docs/architecture_and_eval_spec.md)
- 📊 **C08 True OOS 评测数据**：  
  [`../reports/timesfm3_v11_clean_c08_true_oos_report.json`](../reports/timesfm3_v11_clean_c08_true_oos_report.json)
- 📊 **C06 True OOS 评测数据**：  
  [`../reports/timesfm3_v11_clean_c06_true_oos_report.json`](../reports/timesfm3_v11_clean_c06_true_oos_report.json)

---

## C08 收官与 True OOS 结果摘要（全局收官步数：Step 192,865）

- **验证集（123,839 条）**：10 天方向胜率 **64.99%**，轨迹形态相关度 **+0.2862**，走势综合分 **0.7930**（历史高点）。
- **True OOS（175,428 条绝对封箱样本）**：
  - 10 天方向胜率 **56.84%**（超越 46.80% 自然上涨率 **+10.04%** 净胜率，较 C06 的 57.03% 持平微降 -0.19%）；
  - 分日胜率：Day 1 为 53.76%，Day 7~10 稳定在 56.8% ~ 57.0%；
  - 日度 IC 均值 **+0.0705**，年化 ICIR **7.38**，正 IC 交易日占比 67.65%，Pooled IC **+0.1503**；
  - 实测振幅比 `amp_ratio = 0.8496`，`skill_nrmse = -0.3341`（因低相关下振幅放大导致方差惩罚加剧）；
- **主线判定**：时序纯监督回归（SFT）长训步数红利榨干完毕（验证涨点 ≠ 未达上限），后续必须重组同日截面 panel 转向同日截面排序目标（Rank Loss）。
