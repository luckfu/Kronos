# Kairos Phase W 结果（赢家加长 train→val confirm）

日期：2026-10-02 08:14:16 CST（北京时间）。  
Kernel：`user281434/kairos-mfe10-decision-tabular-phase-w` → **COMPLETE**  
URL：https://www.kaggle.com/code/user281434/kairos-mfe10-decision-tabular-phase-w  
SwanLab：https://swanlab.cn/@roc_fu/finance/runs/kairos-mfe10-decision-tabular-phase-w-20261002  
前置：T2/S val_temporal 过闸（Δ≈−0.041793）；U/V 主协议 FAIL；用户方向允许对已过闸赢家做适度加长（覆盖 V HARD-STOP 的「同轴不再发」）。

## 一句话结论（硬结论）

**主协议 train→val 仍未过闸**（best=`blend_lr0.5_mlp0.5` Δ≈**−0.028500**，需 ≤ −0.04）。  
相对 Phase U（−0.027598）**略好**（约 −0.0009），但距离闸门仍远。  
**次协议 val_temporal 仍过闸**（blend Δ≈**−0.041793**，= S/T2/U/V）。  
→ 把 train_cap 提到 **1.2M**、MLP 提到 **160** iter **不是** train→val 瓶颈 → **HARD-CONCLUDE**（同表格轴不再空转烟测/加长）。

## 硬指标（主协议 = train 2023–2024 → 全 val）

| 项 | 值 |
| --- | ---: |
| best_model | `blend_lr0.5_mlp0.5` |
| best_delta_vs_prior | **−0.028500119076822594** |
| gate (≤ −0.04) | **false** |
| vs Phase U best (−0.027598) | **略好**（Δ_W − Δ_U ≈ −0.00090） |
| vs Phase V best (−0.022722) | 更好（预算/窗不同；V 含 hist_gbm） |
| logistic_C0.01 | −0.022397 |
| enet_C0.01_l1_0.7 | −0.021981 |
| enet_C0.01_l1_0.5 | −0.022198 |
| blend_lr0.5_mlp0.5 | **−0.028500**（mlp_n_iter=71 / max 160） |
| blend_lr0.7_mlp0.3 | −0.027665 |
| blend_lr0.8_mlp0.2 | −0.026605 |
| blend_lr0.9_mlp0.1 | −0.025028 |
| n_train / n_test | **1200000** / 123836 |
| n_train_raw（截断前） | 2347269 |
| train_prior | 0.25（平衡抽样） |
| feature_dim | 26 |
| train_cap / MLP | **1.2M** / **160**（相对 U：500k / 80） |

## 次协议 val_temporal（仍过闸）

| 项 | 值 |
| --- | ---: |
| best_model | `blend_lr0.5_mlp0.5` |
| best_delta_vs_prior | **−0.04179349770224927** |
| gate | **true**（复现 S/T2/U/V） |
| n_train / n_test | 63268 / 60568 |

耗时 ≈2104 s。

## 诊断含义

1. **预算否证**：×2.4 样本 + ×2 MLP 迭代仅把主协议 Δ 从 −0.0276 挪到 −0.0285，**远未到 −0.04**。  
2. 信号仍稳存在于 **近端时序 OOS**（−0.0418），但 **远端 train→全 val** 下日线 Base+xsection 表格赢家族（logistic/enet/blend）过不了主闸。  
3. U（短烟）→ W（加长）方向正确但幅度不够；V（近期窗+树）更差。三者合起来：同轴再烟测无增量。  
4. **HARD-CONCLUDE**：不再沿同一表格轴发下一 kernel；不复活 Ranking 作产品；不开 22 层 R2；不碰 TPU WIP。若重启需 **新证据轴**（标签/特征/协议重定义，而非同一配方加长）。

## 产物

- 结果 JSON：`modernbert_finance/ablations/kairos_phase_w_tabular_decision_results.json`
- Launch JSON：已标 COMPLETE
- 日志/报告：`scratch/kairos_phase_w_outputs/`（`report.json` + kernel log）
- 宪章：`modernbert_finance/kairos_decision_only_handoff_cn.md`（已更新 HARD-CONCLUDE）
