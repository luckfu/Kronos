# Kairos 决策系统交接（Decision-Only Charter）

日期：2026-10-02（北京时间，交接更新 2026-10-02 08:14:16 CST）。  
状态：**产品路径 = 决策标签 only**；时序切分仍过闸（S/T2/U/V/W secondary）；**主协议 train→val 在 U/V/W（含加长）均 FAIL 闸** → **HARD-CONCLUDE**；排序探针已停作产品。

## 一句话

一切实验围绕 **`y = 1{mfe10 ≥ 0.10}`**，闸门 = `Δ logloss vs 常数先验 ≤ −0.04`。  
**当前赢家（仅时序协议）**：时序切分上 `Logistic⊕浅MLP` 融合 / elastic-net（Base+xsection）；**序列 ModernBERT 短烟（I/J/K/P）已证伪**；**train→val 主协议未过闸（U/V/W）**。  
**Kaggle 确认**：Phase T2/U/V/W secondary blend Δ≈**−0.041793**（= Phase S）；Phase W 加长后主协议 best Δ≈**−0.028500**（仍 FAIL）。

## 标签契约

| 项 | 值 |
| --- | --- |
| mfe10 | `max(high[T+1:T+10]) / close[T] - 1` |
| 标签 | `y = 1{mfe10 ≥ 0.10}` |
| 闸门 | `Δ ≤ −0.04` |
| 判读切分 | **时序**可过闸；**train→val 主协议** U/V/W 均 FAIL |

## 诊断摘要

| 路径 | 结果 | 产品含义 |
| --- | ---: | --- |
| Phase I/J/K/P 序列 sidecar | Δ≈0 ~ −0.003 | 短预算序列路径失败 |
| G2 / Q 随机 Logistic | Δ≈−0.034 | 表格基线 |
| Phase R 时序 Logistic C=0.01 | Δ≈**−0.0398** | 近闸 |
| **Phase S 时序 enet/blend** | Δ≈**−0.0418** | **过闸** |
| **Phase T2 Kaggle 确认** | Δ≈**−0.041793** | **过闸复现**；train→val 因对齐 SKIP |
| **Phase U train→val** | Δ≈**−0.027598** | **主协议 FAIL 闸**；temporal 仍 −0.041793 |
| **Phase V train2024+hist_gbm** | Δ≈**−0.022722** | **主协议 FAIL 且劣于 U**；temporal 仍 −0.041793 |
| **Phase W 加长赢家** | Δ≈**−0.028500** | **主协议仍 FAIL**（略好于 U）；预算否证 |
| Phase Q 随机 mlp_3x | Δ≈−0.052（时序崩溃） | 随机切分不可作过闸证据 |

## 当前状态（Phase W COMPLETE → HARD-CONCLUDE）

- **Phase W COMPLETE（2026-10-02 08:14:16 CST）**：主协议 train→val **FAIL 闸**（blend Δ≈−0.028500；cap 1.2M / MLP 160）；次协议 val_temporal 仍过闸（Δ≈−0.041793）。  
- **硬结论**：样本量与 MLP 迭代不是 train→val 瓶颈；日线 Base+xsection 表格轴（logistic/enet/blend/hist_gbm；短烟与加长）无法清主协议闸。  
- Kernel W：`user281434/kairos-mfe10-decision-tabular-phase-w` → COMPLETE  
- URL：https://www.kaggle.com/code/user281434/kairos-mfe10-decision-tabular-phase-w  
- SwanLab：`kairos-mfe10-decision-tabular-phase-w-20261002`  
- **禁止**：Ranking 产品；22 层长训；TPU WIP；同轴再烟测/加长；重启失败序列配方。  
- **若重启**：需用户选定新证据轴（标签/特征/协议），非本轮已测预算轴。

## 明确不做

1. 不重启 ranking Phase。  
2. 不重启 `kairos-r2-r1-restart-lr-3e-5` / 旧 8 头。  
3. 不提交同事 TPU WIP（`beta_v21_c1*` 等 dirty）。  
4. 不开全量 R2。  
5. Phase W 主协议已 FAIL：硬结论成立，不再沿同一表格轴空转。
