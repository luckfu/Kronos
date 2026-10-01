# Kairos 决策表格 train→val 确认（Phase U）启动

日期：2026-10-02 00:48:45 CST（北京时间）。  
前置：Phase T2 COMPLETE，val_temporal 过闸（blend Δ≈−0.041793）；train→val 因信号窗 targets 未对齐 SKIP。  
Kernel：`user281434/kairos-mfe10-decision-tabular-phase-u` → **COMPLETE**（主协议 FAIL 闸；见 results_cn）  
URL：https://www.kaggle.com/code/user281434/kairos-mfe10-decision-tabular-phase-u  
SwanLab：`roc_fu/finance` / `kairos-mfe10-decision-tabular-phase-u-20261002`  
URL：https://swanlab.cn/@roc_fu/finance/runs/kairos-mfe10-decision-tabular-phase-u-20261002  
Commit：`da32fb2`  
启动：2026-10-02 00:48:45 CST

## 相对 T2 的修复

`build_enriched_matrix`：在枚举特征前按 `asof_date ∈ [signal_start, signal_end]` **过滤 targets**（保序）。  
默认 val 窗（2025-07-03→2026-07-02）行为不变；train 2023–2024 特征与 targets 长度对齐（2,347,269）。

## 设定

| 项 | 值 |
| --- | --- |
| 标签 | **`y=1{mfe10≥0.10}`** |
| 特征 | Base+xsection（**无** tokenizer 序列） |
| **主协议** | train 2023–2024（≤500k 平衡抽样）→ 全 val |
| 次协议 | val 时序 2025H2→2026H1（仍须过闸） |
| 配方 | Logistic C=0.01；enet C=0.01 l1∈{0.5,0.7}；LR⊕mlp1 融合 |
| 闸门 | 主协议 Δ ≤ −0.04 |
| 机器 | CPU；internet on（clone） |

## 目的

补齐 T2 缺失的更强 OOS：train→val 决策确认。  
**不做**：序列 ModernBERT；Ranking；22 层；TPU WIP。

## 产物预期

- Kernel logs + `/kaggle/working/kairos_mfe10_tabular_decision/report.json`
- Launch JSON：`modernbert_finance/ablations/kairos_phase_u_tabular_decision_launch.json`

## 完成摘要（2026-10-02 01:33:21 CST）

- 主协议 train→val：**gate_passed=false**，best Δ≈−0.027598（blend）
- 次协议 val_temporal：**gate_passed=true**，Δ≈−0.041793
- 结果：`modernbert_finance/kairos_phase_u_tabular_decision_results_cn.md`
- 下一步：**Phase V**（train2024 + hist_gbm）
