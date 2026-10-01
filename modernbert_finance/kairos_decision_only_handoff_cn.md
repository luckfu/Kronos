# Kairos 决策系统交接（Decision-Only Charter）

日期：2026-10-02（北京时间，交接更新 2026-10-02 01:33:21 CST）。  
状态：**产品路径 = 决策标签 only**；时序切分仍过闸（S/T2/U secondary）；**Phase U 主协议 train→val FAIL 闸**；排序探针已停作产品。

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
| **Phase U train→val** | Δ≈**−0.027598** | **主协议 FAIL 闸**；temporal 仍 −0.041793 |
| Phase Q 随机 mlp_3x | Δ≈−0.052（时序崩溃） | 随机切分不可作过闸证据 |

## 当前下一步

- **Phase U COMPLETE（2026-10-02 01:33:21 CST）**：主协议 train2023–2024→val **FAIL 闸**（blend Δ≈−0.027598）；次协议 val_temporal 仍过闸（Δ≈−0.041793）。协议脆弱。  
- **Phase V（下一步 / 启动中）**：主协议 **train 2024 → 全 val**（近期性）+ `hist_gbm`；次协议 val_temporal；slug `kairos-mfe10-decision-tabular-phase-v`。  
- Kernel U：`user281434/kairos-mfe10-decision-tabular-phase-u` COMPLETE  
- **禁止**：Ranking 产品；22 层长训；TPU WIP；重启失败序列配方。

## 明确不做

1. 不重启 ranking Phase。  
2. 不重启 `kairos-r2-r1-restart-lr-3e-5` / 旧 8 头。  
3. 不提交同事 TPU WIP（`beta_v21_c1*` 等 dirty）。  
4. 不开全量 R2。
