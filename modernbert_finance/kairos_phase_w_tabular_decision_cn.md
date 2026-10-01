# Kairos 决策表格赢家加长确认（Phase W）启动

日期：2026-10-02 07:25:13 CST（北京时间）。  
前置：Phase T2/S val_temporal **过闸**（blend Δ≈−0.041793）；Phase U train→val **FAIL**（Δ≈−0.027598）；Phase V train2024+hist_gbm **FAIL 且更差**（Δ≈−0.022722）并曾 HARD-STOP。  
**用户方向**：已过闸配方可做 **适度加长** 训练 —— 不为赢家永远停在短烟测。本 Phase 覆盖 V 的「同轴不再发」禁令，仅做一次赢家加长 confirm。  
Kernel：`user281434/kairos-mfe10-decision-tabular-phase-w` → **待推送**  
URL：https://www.kaggle.com/code/user281434/kairos-mfe10-decision-tabular-phase-w  
SwanLab：`roc_fu/finance` / `kairos-mfe10-decision-tabular-phase-w-20261002`  
URL：https://swanlab.cn/@roc_fu/finance/runs/kairos-mfe10-decision-tabular-phase-w-20261002

## 假设

U/V 短烟测（cap 500k、MLP 80 epoch）可能不足；对 **已过闸赢家**（blend LR⊕MLP / enet）加长 n_train + MLP 迭代，主攻 **train→val**。若仍 FAIL → **硬结论**（预算不是瓶颈）。

## 相对 U/V 的变化

| 项 | Phase U（短烟） | Phase V | **Phase W（加长）** |
| --- | --- | --- | --- |
| 主协议窗 | 2023–2024 | **仅 2024** | **2023–2024**（回 U 窗） |
| train_cap | 500k 平衡 | 500k 平衡 | **1.2M** 平衡（×2.4） |
| MLP max_iter | 80 | 80 | **160**（×2） |
| n_iter_no_change | 8 | 8 | **12** |
| hist_gbm | 无 | 有（失败） | **去掉**（省预算给赢家） |
| 配方 | logistic/enet/blend | +hist_gbm | logistic/enet/blend **赢家 only** |
| 闸门 | 主协议 Δ≤−0.04 | 同 | 同 |

## 设定

| 项 | 值 |
| --- | --- |
| 标签 | **`y=1{mfe10≥0.10}`** |
| 特征 | Base+xsection（**无** tokenizer 序列） |
| **主协议** | train 2023–2024（≤1.2M 平衡抽样）→ 全 val |
| 次协议 | val 时序 2025H2→2026H1（仍须报告） |
| 配方 | Logistic C=0.01；enet C=0.01 l1∈{0.5,0.7}；LR⊕mlp1 融合（max_iter=160） |
| 闸门 | 主协议 Δ ≤ −0.04 |
| 机器 | CPU；internet on（clone）；短到中等预算（非多小时深训） |

## 目的

对已过闸赢家做一次 **适度加长** train→val 确认。  
若仍 FAIL → 硬结论：预算/样本量不是 train→val 闸门瓶颈。  
**不做**：序列 ModernBERT；Ranking；22 层 R2；TPU WIP；hist_gbm 重跑。

## 产物预期

- Kernel logs + `/kaggle/working/kairos_mfe10_tabular_decision/report.json`
- Launch JSON：`modernbert_finance/ablations/kairos_phase_w_tabular_decision_launch.json`
