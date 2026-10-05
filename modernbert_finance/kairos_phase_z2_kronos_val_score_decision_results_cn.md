# Kairos Phase Z2 结果（C2 Seg@179 → Kairos val 决策头）

日期：2026-10-02 12:17:16 CST（北京时间）。  
范围：父决策 Phase Z2 — 物化 C2 Best@Seg179 生产分数到 Kairos val 日期，再测 `y=1{mfe10≥0.10}`；**仍是 buy/not 决策**（非排序）。

## 一句话结论（硬结论）

**HARD-STOP FAIL（永久停 binary mfe10≥10% 决策路径）。**  
Kernel 已完成：https://www.kaggle.com/code/user281434/kronos-c2-seg179-kairos-val-prod-scores  
Kairos val 全覆盖分数（242d / n=123836 / 100% join mfe10）上 **val_temporal 最佳 Δ≈−0.01986**（`mlp_score_plus_xsec`），**未过闸 ≤ −0.04**。  
``logistic_score_plus_xsec` Δ≈−0.00419；单分数 logistic/isotonic 更弱。  
主协议 **train→val 仍 BLOCKED**（train 面板未打分；全量 train 为周级 GPU，本枪未开）。  
→ 与 Phase Z 密封窗上界（Δ≈−0.012）同量级失败；**真实 path-touch 标签 + 真实 C2 分数仍远未过闸**。

## 分数物化

| 项 | 值 |
| --- | --- |
| checkpoint | cosine_c2_best **Seg@179** sha 4ee469d4…b5a |
| decode | prod T=0.65 top_p=0.8 N=5 seed=20260906 |
| 物化窗 | **2025-07-03..2026-07-02**（与 Kairos val 对齐） |
| n_rows / n_dates / n_sym | 123836 / 242 / 516 |
| join→mfe10 | **100%** |
| kernel elapsed | ≈2874 s（~48 min, 2×T4） |
| 污染注 | 日期落在 C2 训练窗内（latest target 2026-07-31）→ 仅决策头校准，非 Kronos 时间 OOS |

## 硬指标

### 主协议 train→val

| 项 | 值 |
| --- | ---: |
| status | **BLOCKED**（无 train 分数） |
| gate_passed | **false** |
| best_delta_vs_prior | null |

### 次协议 val_temporal（Kairos val 前半→后半）

| 项 | 值 |
| --- | ---: |
| best_model | `mlp_score_plus_xsec` |
| best_delta_vs_prior | **−0.019857** |
| gate (≤ −0.04) | **false** |
| logistic_score_plus_xsec | ≈−0.00419 |
| logistic_score | 更弱 |
| isotonic_score | 更弱 |
| n_train / n_test | 见 results JSON |
| pos_rate_test | 见 results JSON |

对照 Phase Z 密封 OOS 最佳 Δ≈−0.0124；Z2 略好但仍 ≫ −0.04。

## 诊断含义

1. **生产分数轴在真实 Kairos val + path-touch 标签上仍不过闸** → 排序 alpha ≠ 二分类决策 alpha（已用真标签复证）。  
2. train→val 未评不改变结论：val_temporal 上界已否证过闸可能（且 C2 在窗内更偏乐观）。  
3. Y/W 表格头与 Z/Z2 Kronos 分数轴均未过主闸 → **停止** binary `mfe10≥10%` 决策空转。

## 建议（默认 STOP）

1. **STOP** binary `mfe10≥10%` 决策产品路径（本枪为最后一轴，已否证）。  
2. 不追加 train 全量打分 / 不发 confirm Kaggle / 不碰同事 TPU WIP。  
3. 排序（Rank IC on return_10d）已知存活，但父已排除 IC/TopK 作成功标准 → 不作为本决策产品续作。

## 产物

- Kernel: https://www.kaggle.com/code/user281434/kronos-c2-seg179-kairos-val-prod-scores  
- 分数: `scratch/kairos_phase_z2_outputs/predictions_prod_t065_p80_n5.csv.gz`  
- 评测: `modernbert_finance/ablations/phase_z2_kronos_val_score_decision.py`  
- JSON: `modernbert_finance/ablations/kairos_phase_z2_kronos_val_score_decision_results.json`
