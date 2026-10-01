# Kairos Phase T2 结果（表格 enet/blend 确认）

日期：2026-10-02 00:47:16 CST（北京时间）。  
Kernel：`user281434/kairos-mfe10-decision-tabular-phase-t2` → **COMPLETE**  
URL：https://www.kaggle.com/code/user281434/kairos-mfe10-decision-tabular-phase-t2  
SwanLab：https://swanlab.cn/@roc_fu/finance/runs/kairos-mfe10-decision-tabular-phase-t2-20261002  
包装：git clone `luckfu/Kronos` → sys.path（sidecar 模式）**成功**。

## 一句话结论

**val_temporal 闸门在 Kaggle 上确认通过**，复现 Phase S（blend Δ≈−0.041793）。  
**train→val 未跑成**：特征窗口与全量 train_targets 长度不对齐（2347269 ≠ 9010965）。属对齐 bug，不是信号失败。

## 硬指标（主协议 = val_temporal）

| 项 | 值 |
| --- | ---: |
| best_model | `blend_lr0.5_mlp0.5` |
| best_delta_vs_prior | **−0.04179349770224927** |
| gate (≤ −0.04) | **true** |
| Phase S 对照 | −0.04179349770224905（实质一致） |
| enet_C0.01_l1_0.7 | −0.040419（gate true） |
| logistic_C0.01 | −0.039815（gate false） |
| n_train / n_test | 63268 / 60568 |
| 耗时 | ≈304 s |

## train→val（跳过）

```
ValueError: feature windows 2347269 != targets 9010965; check signal date bounds / panel
```

根因：`build_enriched_matrix` 按 `signal_start/end` 过滤特征窗口，但未同步过滤 targets 的 `asof_date`。  
train 全量 targets=9,010,965；2023–2024 窗口恰好 2,347,269（与特征一致）。  
**Phase U**：在 `build_enriched_matrix` 内按信号窗过滤 targets（保留行序），主协议改为 train→val。

## 决策

1. T2 科学目标（Kaggle 复验 Phase S）**达成**。  
2. 下一步 Phase U = train→val 决策确认（短实验）。  
3. **不做**：Ranking 产品；序列 ModernBERT；22 层；碰 TPU WIP。

## 产物

- 结果 JSON：`modernbert_finance/ablations/kairos_phase_t2_tabular_decision_results.json`
- Launch JSON：已标 COMPLETE
- 日志/报告：`scratch/kairos_phase_t2_outputs/`（及 `/workspace/scratch/kairos_phase_t2_outputs/`）
