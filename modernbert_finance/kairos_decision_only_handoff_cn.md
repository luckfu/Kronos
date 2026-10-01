# Kairos 决策系统交接（Decision-Only Charter）

日期：2026-10-02（北京时间）。  
状态：**产品路径 = 决策标签 only**；表格特征路径已在时序切分上 **过闸**（Phase S / **T2 Kaggle 确认**）；排序探针已停作产品。

## 一句话

一切实验围绕 **`y = 1{mfe10 ≥ 0.10}`**，闸门 = `Δ logloss vs 常数先验 ≤ −0.04`。  
**当前赢家**：时序切分上 `Logistic⊕浅MLP` 融合 / elastic-net（Base+xsection）；**序列 ModernBERT 短烟（I/J/K/P）已证伪**。  
**Kaggle 确认**：Phase T2 COMPLETE，blend Δ≈**−0.041793**（= Phase S）。

## 标签契约

| 项 | 值 |
| --- | --- |
| mfe10 | `max(high[T+1:T+10]) / close[T] - 1` |
| 标签 | `y = 1{mfe10 ≥ 0.10}` |
| 闸门 | `Δ ≤ −0.04` |
| 判读切分 | **时序**优先于随机 75/25；下一步补 **train→val** |

## 诊断摘要

| 路径 | 结果 | 产品含义 |
| --- | ---: | --- |
| Phase I/J/K/P 序列 sidecar | Δ≈0 ~ −0.003 | 短预算序列路径失败 |
| G2 / Q 随机 Logistic | Δ≈−0.034 | 表格基线 |
| Phase R 时序 Logistic C=0.01 | Δ≈**−0.0398** | 近闸 |
| **Phase S 时序 enet/blend** | Δ≈**−0.0418** | **过闸** |
| **Phase T2 Kaggle 确认** | Δ≈**−0.041793** | **过闸复现**；train→val 因对齐 SKIP |
| Phase Q 随机 mlp_3x | Δ≈−0.052（时序崩溃） | 随机切分不可作过闸证据 |

## 当前下一步

- **Phase U（RUNNING，2026-10-02 00:48:45 CST）**：信号窗 targets 对齐已合入；主协议 **train 2023–2024 → 全 val**；次协议 val_temporal 仍须过闸。  
- Kernel：`user281434/kairos-mfe10-decision-tabular-phase-u`  
- URL：https://www.kaggle.com/code/user281434/kairos-mfe10-decision-tabular-phase-u  
- SwanLab：`kairos-mfe10-decision-tabular-phase-u-20261002`  
- **禁止**：Ranking 产品；22 层长训；TPU WIP；重启失败序列配方。

## 明确不做

1. 不重启 ranking Phase。  
2. 不重启 `kairos-r2-r1-restart-lr-3e-5` / 旧 8 头。  
3. 不提交同事 TPU WIP（`beta_v21_c1*` 等 dirty）。  
4. 不开全量 R2。
