# Kairos Phase V 结果（train2024→val + hist_gbm）

日期：2026-10-02 02:12:48 CST（北京时间）。  
Kernel：`user281434/kairos-mfe10-decision-tabular-phase-v` → **COMPLETE**  
URL：https://www.kaggle.com/code/user281434/kairos-mfe10-decision-tabular-phase-v  
SwanLab：https://swanlab.cn/@roc_fu/finance/runs/kairos-mfe10-decision-tabular-phase-v-20261002  
前置：Phase U 主协议 train2023–2024→val FAIL（best Δ≈−0.027598）；假设「远端窗稀释 / 缺树非线性」。

## 一句话结论（硬停）

**主协议 train2024→val 未过闸**（best=`hist_gbm` Δ≈**−0.022722**，需 ≤ −0.04）。  
相对 Phase U（−0.027598）**更差**，非改进。  
**次协议 val_temporal 仍过闸**（blend Δ≈**−0.041793**，= S/T2/U）。  
→ 近期性与 hist_gbm **均未打开** train→val 闸；**日线表格特征在此绝对触达标签上无法清主协议闸** → **HARD-STOP**（不再发下一 kernel；不复活 Ranking；不开 22 层 R2）。

## 硬指标（主协议 = train 2024 → 全 val）

| 项 | 值 |
| --- | ---: |
| best_model | `hist_gbm` |
| best_delta_vs_prior | **−0.02272150397385786** |
| gate (≤ −0.04) | **false** |
| vs Phase U best (−0.027598) | **更差**（Δ_V − Δ_U ≈ +0.0049） |
| logistic_C0.01 | −0.018690 |
| enet_C0.01_l1_0.7 | −0.018750 |
| enet_C0.01_l1_0.5 | −0.018792 |
| hist_gbm | **−0.022722** |
| blends | −0.02036 .. −0.02182 |
| n_train / n_test | 500000 / 123836 |
| train_prior | 0.25（平衡抽样） |
| feature_dim / raw_n_train | 26 / 1199850（截断前） |

## 次协议 val_temporal（仍过闸）

| 项 | 值 |
| --- | ---: |
| best_model | `blend_lr0.5_mlp0.5` |
| best_delta_vs_prior | **−0.04179349770224927** |
| gate | **true**（复现 S/T2/U） |
| hist_gbm（次协议） | −0.031846（**未过闸**；树在时序上弱于线性融合） |
| n_train / n_test | 63268 / 60568 |

耗时 ≈1442 s。

## 诊断含义

1. **近期性否证**：仅 train 2024 并未优于 2023–2024；主协议 Δ 从 −0.0276 退到 −0.0227。  
2. **树非线性否证为闸门解药**：hist_gbm 在主协议上略优于线性/融合，但仍远未到 −0.04；在 val_temporal 上反而弱于 blend。  
3. 信号仍存在于 **近端时序 OOS**（−0.0418），但 **远端 train→全 val** 协议下日线 Base+xsection 表格配方族（logistic/enet/blend/hist_gbm）无法过闸。  
4. **HARD-STOP**：不再沿同一表格轴发 Phase W；不把 Ranking 当产品；不启 22 层 R2；不碰 TPU WIP。若未来重启，需 **新证据轴**（非本轮已测的 recency/tree）。

## 产物

- 结果 JSON：`modernbert_finance/ablations/kairos_phase_v_tabular_decision_results.json`
- Launch JSON：已标 COMPLETE
- 日志/报告：`scratch/kairos_phase_v_outputs/`（`report.json` + kernel log）
- 宪章：`modernbert_finance/kairos_decision_only_handoff_cn.md`（已更新 HARD-STOP）
