# Kairos Phase U 结果（表格 train→val 决策确认）

日期：2026-10-02 01:33:21 CST（北京时间）。  
Kernel：`user281434/kairos-mfe10-decision-tabular-phase-u` → **COMPLETE**  
URL：https://www.kaggle.com/code/user281434/kairos-mfe10-decision-tabular-phase-u  
SwanLab：https://swanlab.cn/@roc_fu/finance/runs/kairos-mfe10-decision-tabular-phase-u-20261002  
对齐修复：`build_enriched_matrix` 按信号窗过滤 targets — **生效**（train→val 已跑完，非 SKIP）。

## 一句话结论（硬）

**主协议 train→val 未过闸**（best Δ≈**−0.027598**，需 ≤ −0.04）。  
**次协议 val_temporal 仍过闸**（blend Δ≈**−0.041793**，= Phase S/T2）。  
→ 信号存在但 **协议脆弱**：近端时序 OOS ≠ 远端 train(2023–2024)→val。

## 硬指标（主协议 = train 2023–2024 → 全 val）

| 项 | 值 |
| --- | ---: |
| best_model | `blend_lr0.5_mlp0.5` |
| best_delta_vs_prior | **−0.027597938051234006** |
| gate (≤ −0.04) | **false** |
| logistic_C0.01 | −0.022698 |
| enet_C0.01_l1_0.7 | −0.022875 |
| blends | −0.02515 .. −0.02760 |
| n_train / n_test | 500000 / 123836 |
| train_prior | 0.25（平衡抽样） |

## 次协议 val_temporal（仍过闸）

| 项 | 值 |
| --- | ---: |
| best_model | `blend_lr0.5_mlp0.5` |
| best_delta_vs_prior | **−0.04179349770224927** |
| gate | **true**（复现 S/T2） |
| n_train / n_test | 63268 / 60568 |

耗时 ≈1424 s。

## 诊断含义

1. 对齐 bug 已排除；主协议失败是 **科学结果**，不是 infra。  
2. 远端训练窗可能稀释近期可迁移结构（或非线性未覆盖）。  
3. **下一步 Phase V**：train **仅 2024** → 全 val（近期性）+ 增加 `HistGradientBoostingClassifier`（树非线性）；闸门仍 Δ≤−0.04。  
4. **不做**：Ranking；序列 ModernBERT；22 层；碰 TPU WIP。

## 产物

- 结果 JSON：`modernbert_finance/ablations/kairos_phase_u_tabular_decision_results.json`
- Launch JSON：已标 COMPLETE
- 日志/报告：`scratch/kairos_phase_u_outputs/`（及 `/workspace/scratch/kairos_phase_u_outputs/`）
