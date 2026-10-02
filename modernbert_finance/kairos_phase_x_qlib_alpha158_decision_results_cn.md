# Kairos Phase X 结果（Qlib 轴 Alpha158 + LGB binary 短烟）

日期：2026-10-02 10:38:47 CST（北京时间）。  
范围：短名单第 1 项 Microsoft Qlib 证据轴；标签 `y=1{mfe10≥0.10}`；闸门 `Δ logloss vs prior ≤ -0.04`。  
实现：复用 Kaggle 面板 `luckfu/a-share-120d-temporal-symbol-holdout` + 已有 `ashare120d-modernbert-targets`；**未**下载 `cn_data`。CPython 3.13 无 pyqlib 轮子，特征公式按 `Alpha158DL` 默认配置移植（158 维），模型 = LightGBM `objective=binary`（对齐 `LGBModel loss:binary`）。

## 一句话结论（硬结论）

**主协议 train→val FAIL 闸**（LGB Δ≈**−0.025819**，需 ≤ −0.04）。  
**次协议 val_temporal 也 FAIL**（Δ≈**−0.035231**，未达 −0.04；且劣于 Phase S/W Base+xsection 的 −0.041793）。  
→ Alpha158 + 树二分类相对失败的 26 维表格轴**未提供主闸增量**，时序次闸亦未过 → **HARD-CONCLUDE Qlib/Alpha158 本标签轴**；不进入加长 confirm。

## 硬指标

### 主协议 train→val（2023–2024 → 全 val）

| 项 | 值 |
| --- | ---: |
| model | `lgb_binary_alpha158` |
| delta_vs_prior | **−0.025819046770836573** |
| gate (≤ −0.04) | **false** |
| model_log_loss | 0.5397453274183326 |
| prior_log_loss | 0.5655643741891692 |
| train_prior | 0.25（平衡抽样） |
| n_train / n_test | **300000** / 123836 |
| n_train_raw_window | 2347269 |
| feature_dim | **158** |
| best_iteration | 105 |
| vs Phase W best (−0.028500) | **更差**（约 +0.0027） |
| vs Phase U best (−0.027598) | 更差 |

### 次协议 val_temporal（2025H2→2026H1）

| 项 | 值 |
| --- | ---: |
| delta_vs_prior | **−0.035230586799937846** |
| gate | **false** |
| n_train / n_test | 63268 / 60568 |
| best_iteration | 29 |
| vs Phase S/W temporal (−0.041793) | **更差**（未过闸） |

耗时 ≈94 s（本地）。

## 诊断含义

1. **新特征轴否证**：158 维 Alpha158 + LGB binary 在主协议上 Δ≈−0.026，**未优于**已 HARD-CONCLUDE 的 Base+xsection 赢家族（W −0.0285），更远未到 −0.04。  
2. **时序次闸也掉**：不同于 W（temporal 仍 −0.0418 过闸），本轴 temporal 仅 −0.035 → 不是「主 FAIL / 次 PASS」同构，而是**双协议未过**。  
3. 预算：短烟 cap=300k；与 W 预算否证一致——同决策标签下加长树/样本预期无法填补 ~0.014 的闸门缺口，故**不发 Phase X+ 加长 confirm**。  
4. **HARD-CONCLUDE**：关闭 Qlib/Alpha158 / LGB binary 对本 `mfe10≥10%` 标签的产品路径；禁止用 Rank IC / TopK 回救。

## 下一动作（短名单）

| 选项 | 建议 |
|---|---|
| **停** | 合理默认：表格轴（U/V/W）+ Qlib/Alpha158（X）均未清主闸；需**换标签定义**或用户点名新证据轴后再开 |
| 短名单 #2 Time-Series-Library | **高复刻序列失败风险**（I/J/K/P）；仅当窗口/校准与已死 ModernBERT 烟测明显不同才值得；不自动开 |
| #3 TFT / #4 Chronos | 间接决策、偏预测 FM；成本高，不作自动下一烟 |
| Ranking / ModernBERT / TPU `beta_v21_c1*` | **禁止** |

## 产物

- 结果 JSON：`modernbert_finance/ablations/kairos_phase_x_qlib_alpha158_decision_results.json`
- 脚本：`modernbert_finance/ablations/qlib_alpha158_mfe10_decision_smoke.py`
- 日志：`scratch/kairos_qlib_alpha158_smoke_outputs/`
- 短名单：`modernbert_finance/kairos_oss_decision_framework_shortlist_cn.md`
