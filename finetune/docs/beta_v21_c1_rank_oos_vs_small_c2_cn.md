# Beta C1 Ranking OOS vs Small C2（同密封包 · 决定性对比）

更新：2026-10-05 CST

## 密封包（两边相同）

- 包名：`kronos_beta_v2_time_oos_through_20260903`
- Signal：`2026-08-11` → `2026-09-03`（**18** 个交易日）
- 样本：**92,751**（C2↔Beta 按 `symbol,asof_date` **92751/92751** 对齐）
- `return_10d` / utility 标签来自同一密封包；utility 从 Beta OOS predictions 行 join（零 GPU）
- C2 预测：`/workspace/kaggle_c2_18d_alpha_oos/kronos_c2_18d_alpha_oos/predictions_*.csv.gz`
- Beta 预测：`/workspace/kronos_patrol_out/oos_20261005_2215/output/beta_v2_1_c1_rank_oos/results/*_predictions.csv`

**不要**拿 Beta 训练 val（score↔utility ≈ 0.28）或用户粘贴的 **19 日 audit（08-03→08-27）** 和本表 18 日密封包混比。

## 指标合同

| 列 | 定义 | 诚实说明 |
|---|---|---|
| **Return10d Rank IC** | 日均 Spearman(分数, 10 日收益) | C2=`predicted_return_d10` vs `actual_return_d10`（与已发表 metrics 一致）；Beta=`expected_utility_score` vs 包内 `return_10d` |
| **Utility Rank IC** | 日均 Spearman(分数, utility) | Beta=训练合同；**C2 无 utility 头** → 用 `predicted_return_d10` 当分数的 **ablation** |
| **Pairwise** | 同日 `\|Δutility\|≥0.005`，分数打平算错 | Beta=训练合同；C2 同上 ablation（score=`predicted_return_d10`） |
| **WFL** | C1 加权 teacher-forcing CE | 仅 Beta；C2 本表不报 |

## 决定性并列表（18 日密封包）

| 模型 | 分数种类 | Return10d IC（日均） | Return10d IC（pooled） | Utility IC（日均） | Utility IC（pooled） | Pairwise | WFL |
|---|---|---:|---:|---:|---:|---:|---:|
| **Small C2 best** rank decode T0.6/p0.9/N16 | `predicted_return_d10` | **0.1796** | **0.2428** | **0.1210**† | **0.1539**† | **56.49%**† | — |
| Small C2 best prod T0.65/p0.8/N5 | `predicted_return_d10` | 0.1769 | 0.2373 | 0.1208† | 0.1543† | 56.50%† | — |
| **Beta Seg19** rank-frozen | `expected_utility_score` | **-0.0073** | 0.0305 | **0.0300** | 0.0774 | **51.54%** | **2.43604** |
| Beta Seg8 rank-unfreeze | `expected_utility_score` | -0.0298 | 0.0065 | 0.0224 | 0.0729 | 51.09% | 2.44870 |
| Beta Seg155 forecast | （无 ranking score） | — | — | — | — | — | 2.43604 |

† C2 的 Utility IC / Pairwise 是 **return-score ablation**（把 `predicted_return_d10` 接到包内 utility 标签），**不是**训练过的 utility 头。Pairwise 合格 pair 数 ≈ **1.2487e8**（与 Beta 同量级）。

补充：

- C2 rank：utility ICIR **0.856**，IC+ 日 **77.8%**；return10d ICIR **1.668**，IC+ 日 **88.9%**。
- Seg19：utility ICIR **0.327**，IC+ 日 **66.7%**；return10d ICIR **-0.069**，IC+ 日仅 **33.3%**。
- C2 对包内 `return_10d` 日均 IC 为 0.1793（与 `actual_return_d10` 的 0.1796 几乎相同；~173 行涨跌停/重建异常）。

机器可读：`finetune/reports/beta_v21_c1_rank_oos_vs_small_c2_return10d.json`

## 这证明什么 / 不证明什么

1. **同包 Return10d Rank IC（真正 apples-to-apples）**：C2 **~0.18** ≫ Seg19 **-0.007**。Beta ranking 头在原始 10 日收益截面上基本无效。
2. **同包 Utility 合同（C2 为 ablation）**：即便只用收益预测当分数，C2 的 utility IC **0.121**、pairwise **56.5%** 仍高于 Seg19 训练后的 **0.030 / 51.5%**。说明当前 Beta aux ranking 在密封 OOS 上弱于「用 C2 收益预测硬套 utility」这一朴素基线。
3. **不证明**：C2 有更好的 utility 模型（它没有）；也不证明 val 上 0.28 的 utility IC 可外推。
4. **用户 19 日数字**（dir 50.69%、pooled 0.169、daily 0.162）来自 audit **08-03→08-27 combined_19**，不是本密封 18 日包。

## 以后 OOS

`finetune/evaluate_beta_v21_time_oos.py` 已同时记录 utility 与 return10d Rank IC。C2 侧复用本 JSON 的 join 公式即可，无需新 GPU kernel。

## 相关路径

- 本对比 JSON：`finetune/reports/beta_v21_c1_rank_oos_vs_small_c2_return10d.json`
- OOS 计划：`finetune/docs/beta_v21_c1_rank_oos_plan_cn.md`
- 评测脚本：`finetune/evaluate_beta_v21_time_oos.py`
