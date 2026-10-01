# Kairos 决策系统交接（Decision-Only Charter）

日期：2026-10-02（北京时间，交接更新 2026-10-02 07:25:27 CST）。  
状态：**产品路径 = 决策标签 only**；时序切分仍过闸（S/T2/U/V secondary）；**Phase U/V 主协议 train→val 均 FAIL 闸**；**Phase W 加长赢家 confirm 中**；排序探针已停作产品。

## 一句话

一切实验围绕 **`y = 1{mfe10 ≥ 0.10}`**，闸门 = `Δ logloss vs 常数先验 ≤ −0.04`。  
**当前赢家（仅时序协议）**：时序切分上 `Logistic⊕浅MLP` 融合 / elastic-net（Base+xsection）；**序列 ModernBERT 短烟（I/J/K/P）已证伪**；**train→val 主协议未过闸（U/V）**。  
**Kaggle 确认**：Phase T2/U/V secondary blend Δ≈**−0.041793**（= Phase S）。

## 标签契约

| 项 | 值 |
| --- | --- |
| mfe10 | `max(high[T+1:T+10]) / close[T] - 1` |
| 标签 | `y = 1{mfe10 ≥ 0.10}` |
| 闸门 | `Δ ≤ −0.04` |
| 判读切分 | **时序**可过闸；**train→val 主协议** U/V 均 FAIL |

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
| Phase Q 随机 mlp_3x | Δ≈−0.052（时序崩溃） | 随机切分不可作过闸证据 |

## 当前状态（Phase W 加长 confirm）

- **Phase V COMPLETE（2026-10-02 02:12:48 CST）**：主协议 train2024→val **FAIL 闸**（hist_gbm Δ≈−0.022722，劣于 U）；次协议 val_temporal 仍过闸（Δ≈−0.041793）。  
- **用户方向（覆盖 V HARD-STOP）**：已过闸赢家可做 **适度加长** —— 不为赢家永远停在短烟测。  
- **Phase W（2026-10-02 07:25:27 CST）**：对 blend/enet 赢家加长 train→val（cap **1.2M**、MLP **160** epoch；回 U 窗 2023–2024；**去掉** hist_gbm）。若仍 FAIL → **硬结论**。  
- Kernel W：`user281434/kairos-mfe10-decision-tabular-phase-w`  
- URL：https://www.kaggle.com/code/user281434/kairos-mfe10-decision-tabular-phase-w  
- SwanLab：`kairos-mfe10-decision-tabular-phase-w-20261002`  
- **禁止**：Ranking 产品；22 层长训；TPU WIP；重启失败序列配方。

## 明确不做

1. 不重启 ranking Phase。  
2. 不重启 `kairos-r2-r1-restart-lr-3e-5` / 旧 8 头。  
3. 不提交同事 TPU WIP（`beta_v21_c1*` 等 dirty）。  
4. 不开全量 R2。  
5. 若 Phase W 主协议仍 FAIL：硬结论，不再沿同一短烟/加长表格轴空转。
