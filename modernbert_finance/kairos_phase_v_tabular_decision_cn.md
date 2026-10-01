# Kairos 决策表格 train2024→val + hist_gbm（Phase V）启动

日期：2026-10-02 01:34:03 CST（北京时间）。  
前置：Phase U COMPLETE — 主协议 train2023–2024→val **FAIL 闸**（blend Δ≈−0.027598）；val_temporal 仍过闸（Δ≈−0.041793）。  
Kernel：`user281434/kairos-mfe10-decision-tabular-phase-v` → **RUNNING**  
URL：https://www.kaggle.com/code/user281434/kairos-mfe10-decision-tabular-phase-v  
SwanLab：`roc_fu/finance` / `kairos-mfe10-decision-tabular-phase-v-20261002`  
URL：https://swanlab.cn/@roc_fu/finance/runs/kairos-mfe10-decision-tabular-phase-v-20261002

## 假设

U 主协议失败可能来自 **远端训练窗稀释**（2023–2024）。Phase V 测：
1. **近期性**：仅 train 2024 → 全 val  
2. **树非线性**：`HistGradientBoostingClassifier`（hist_gbm）

## 设定

| 项 | 值 |
| --- | --- |
| 标签 | **`y=1{mfe10≥0.10}`** |
| 特征 | Base+xsection（**无** tokenizer 序列） |
| **主协议** | train 2024-01-01..2024-12-31（≤500k 平衡抽样）→ 全 val |
| 次协议 | val 时序 2025H2→2026H1（仍须报告） |
| 配方 | Logistic / enet / LR⊕mlp 融合 + **hist_gbm**（max_iter=120, lr=0.08, max_depth=6, min_samples_leaf=80） |
| 闸门 | 主协议 Δ ≤ −0.04 |
| 机器 | CPU；internet on（clone） |

## 目的

在决策-only 宪章下，用短实验区分「远端窗」vs「线性配方」对 U 失败的贡献。  
**不做**：序列 ModernBERT；Ranking；22 层；TPU WIP；覆盖 T2/U slug。

## 产物预期

- Kernel logs + `/kaggle/working/kairos_mfe10_tabular_decision/report.json`
- Launch JSON：`modernbert_finance/ablations/kairos_phase_v_tabular_decision_launch.json`

启动：2026-10-02 01:34:49 CST
Kernel push：version 1
