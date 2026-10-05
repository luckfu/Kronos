# Kairos Phase Z 结果（Kronos C2 Best@Seg179 分数 → 决策头烟测）

日期：2026-10-02 11:07:54 CST（北京时间）。  
范围：父决策 Phase Z — 用现有 Kronos 生产分数作特征/校准决策规则，目标 `y=1{mfe10≥0.10}`；**仍是 buy/not 决策**（非排序产品）。

## 一句话结论（硬结论）

**HARD-REPORT FAIL。**  
主协议 **train→val = BLOCKED**（C2 Seg@179 分数仅密封 18 日窗，与 Kairos train/val **无日期重叠**；箱内无 `model.safetensors`，不可本地重打分；未开 TPU/长训）。  
可跑表面（密封 OOS）上最佳 Δ≈**-0.012399**（`logistic_score_plus_xsec`），**远未过闸 ≤ −0.04**。  
→ **停止** binary mfe10 决策空转；标签族 churn（Y/Y-alt）+ Kronos 分数轴均未过主闸。

## 分数库存

| 项 | 值 |
| --- | --- |
| checkpoint | cosine_c2_best **Seg@179** |
| decode | prod T=0.65 top_p=0.8 N=5（locked） |
| 物化窗 | **2026-08-11..2026-09-03**（18d，n=92751） |
| Kairos val | 2025-07-03..2026-07-02 |
| date_overlap | **false** |
| 本地权重 | `scratch/kronos_small_0_1_cosine_c2_best/` 仅 README/config，**无 model.safetensors** |

## 硬指标

### 主协议 train→val

| 项 | 值 |
| --- | ---: |
| status | **BLOCKED** |
| gate_passed | **false** |
| best_delta_vs_prior | null |

### 次协议 val_temporal（密封 OOS 前半→后半，最接近可用时序切分）

| 项 | 值 |
| --- | ---: |
| split | sealed_oos_temporal_2026-08-11_to_2026-08-21__vs__2026-08-24_to_2026-09-03 |
| best_model | `logistic_score_plus_xsec` |
| best_delta_vs_prior | **-0.012398564058** |
| gate (≤ −0.04) | **false** |
| logistic_score | -0.003662 |
| isotonic_score | -0.000175 |
| threshold_calibrated | -0.001456 |
| logistic_cs_pctile | -0.006546 |
| mlp_score_plus_xsec | -0.009547 |
| n_train / n_test | 46439 / 46312 |
| pos_rate_test | 0.1526 |

### 密封 OOS 分层随机 75/25（对照 Phase H）

| 项 | 值 |
| --- | ---: |
| best_model | `logistic_score_plus_xsec` |
| best_delta_vs_prior | **-0.011681022713** |
| gate | **false** |
| logistic_score | -0.003178 |

标签注：OOS dump 无 high 路径，用 `max(actual_return_d1..d10)` 近似 path-touch。

耗时 ≈5.3 s。

## 诊断含义

1. **生产分数轴无法服务主闸**：分数不在 train/val 日期上 → train→val 不可评 = Phase Z 过闸失败。  
2. **密封窗上界也弱**：真实 C2 分数→path-touch≥10% 最佳仅 Δ≈−0.012，与 Phase H teacher logit（≈−0.0035）同量级，证明 **排序 alpha ≠ 二分类决策 alpha**。  
3. 与 Y（Δ≈−0.031）/ W（Δ≈−0.028）表格头相比，Kronos 单分数决策更弱。  
4. **不加长 confirm / 不发 Kaggle**（未过主闸）。

## 建议（二选一，默认 STOP）

1. **STOP** binary `mfe10≥10%` 决策产品路径（推荐）：Y 标签族 + Z Kronos 分数均已否证。  
2. **最后一轴（仅当父明确预算）**：把 C2 Seg@179 分数物化到 Kairos val 日期（全量推理，非同事 TPU WIP），再跑一次 logistic/isotonic；若仍 Δ≫−0.04 则永久停。  
3. 排序（Rank IC on return_10d）已知存活，但父已排除 IC/TopK 作成功标准 → 不作为本决策产品续作。

## 产物

- 脚本：`modernbert_finance/ablations/phase_z_kronos_score_decision_smoke.py`
- 结果 JSON：`modernbert_finance/ablations/kairos_phase_z_kronos_score_decision_results.json`
- 日志/报告：`scratch/kairos_phase_z_outputs/report.json`
