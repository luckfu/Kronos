# Kairos 决策表格 enet/blend 确认（Phase T2）启动

日期：2026-10-02 00:27:50 CST（北京时间）。  
前置：Phase S 时序切分 **过闸**；Phase T Kaggle 确认因 **`modernbert_finance` 未上 sys.path** 报 ERROR（无科学指标）。  
Kernel：`user281434/kairos-mfe10-decision-tabular-phase-t2` → **RUNNING**  
URL：https://www.kaggle.com/code/user281434/kairos-mfe10-decision-tabular-phase-t2  
SwanLab：`roc_fu/finance` / `kairos-mfe10-decision-tabular-phase-t2-20261002`  
URL：https://swanlab.cn/@roc_fu/finance/runs/kairos-mfe10-decision-tabular-phase-t2-20261002
启动：2026-10-02 00:28:54 CST

## 相对 Phase T 的修复

Kaggle script kernel 不会把 `vendor/` 放到 `/kaggle/src/` 旁。T2 的 `setup_vendor_path()`：

1. 仍尝试本地 / vendor；
2. 失败则 **shallow clone** `https://github.com/luckfu/Kronos.git` → `/kaggle/working/Kronos`（与 sidecar 同模式）；
3. 校验 `import modernbert_finance` 后再建特征。

## 设定

| 项 | 值 |
| --- | --- |
| 标签 | **`y=1{mfe10≥0.10}`** |
| 特征 | Base+xsection（**无** tokenizer 序列） |
| 主协议 | val 时序 2025H2→2026H1（复验 Phase S） |
| 可选 | train 2023–2024 → 全 val |
| 配方 | Logistic C=0.01；enet C=0.01 l1∈{0.5,0.7}；LR⊕mlp1 融合 |
| 闸门 | Δ ≤ −0.04 |
| 机器 | CPU（无 GPU）；internet on（clone） |

## 目的

在 Kaggle 官方 holdout 上复验 Phase S 过闸配方。  
**不做**：序列 ModernBERT；Ranking；22 层；TPU WIP。

## 产物预期

- Kernel logs + `/kaggle/working/kairos_mfe10_tabular_decision/report.json`
- Launch JSON：`modernbert_finance/ablations/kairos_phase_t2_tabular_decision_launch.json`
