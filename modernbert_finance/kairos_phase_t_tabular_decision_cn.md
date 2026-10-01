# Kairos 决策表格 enet/blend 确认（Phase T）启动

日期：2026-10-02（北京时间）。  
前置：Phase S 时序切分 **过闸**（blend Δ≈−0.0418；enet Δ≈−0.0404）。  
Kernel：`user281434/kairos-mfe10-decision-tabular-phase-t` → **ERROR**（见 results memo）  
URL：https://www.kaggle.com/code/user281434/kairos-mfe10-decision-tabular-phase-t  
SwanLab：`roc_fu/finance` / `kairos-mfe10-decision-tabular-phase-t-20261002`  
URL：https://swanlab.cn/@roc_fu/finance/runs/kairos-mfe10-decision-tabular-phase-t-20261002  
Commit：`943b40c`  
启动：2026-10-01 16:14:17 CST

## 设定

| 项 | 值 |
| --- | --- |
| 标签 | **`y=1{mfe10≥0.10}`** |
| 特征 | Base+xsection（**无** tokenizer 序列） |
| 主协议 | val 时序 2025H2→2026H1（复验 Phase S） |
| 可选 | train 2023–2024 → 全 val |
| 配方 | Logistic C=0.01；enet C=0.01 l1∈{0.5,0.7}；LR⊕mlp1 融合 |
| 闸门 | Δ ≤ −0.04 |
| 机器 | CPU（无 GPU） |

## 目的

在 Kaggle 官方 holdout 数据上复验 Phase S 过闸配方；若 train→val 也可完成则一并报告。  
**不做**：序列 ModernBERT；Ranking；22 层；TPU WIP。

## 本地对照（Phase S）

| 模型 | 时序 Δ | gate |
| --- | ---: | --- |
| blend_lr0.5_mlp0.5 | −0.041793 | true |
| enet_C0.01_l1_0.7 | −0.040419 | true |
| logistic C=0.01 | −0.039815 | false |

## 产物预期

- Kernel logs + `/kaggle/working/kairos_mfe10_tabular_decision/report.json`
- Launch JSON：`modernbert_finance/ablations/kairos_phase_t_tabular_decision_launch.json`

## 结果

见 `kairos_phase_t_tabular_decision_results_cn.md`（infra ERROR → Phase T2）。
