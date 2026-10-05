# Beta C1 Ranking OOS vs Small C2（同密封包对比）

更新：2026-10-05 CST

## 密封包（两边相同）

- 包名：`kronos_beta_v2_time_oos_through_20260903`
- Signal：`2026-08-11` → `2026-09-03`（**18** 个交易日）
- 样本：**92,751**
- Dataset：`luckfu/a-share-120d-temporal-symbol-holdout`
- `return_10d` 定义：`close_h / close_signal - 1`（与 Small D10 实际收益同一合同）

**不要**拿 Beta 训练 val（`val_data.pkl`，score↔utility ≈ 0.28）和 Small OOS ≈ 0.18 比。

## 指标合同（读表前先看）

| 列 | Beta 含义 | Small 含义 |
|---|---|---|
| **Utility Rank IC** | 日均 Spearman(`expected_utility_score`, `utility`) | 无（Small 无 ranking/utility 头）→ 表中为 — |
| **Return10d Rank IC** | 日均 Spearman(`score`, `return_10d`) | 日均 Spearman(`predicted_return_d10`, `actual_return_d10`) = D10 `daily_rank_ic_mean` |
| **Pairwise** | 同日、`|Δutility| ≥ 0.005`，分数打平算错 | 无（本表不报） |
| **WFL** | C1 加权 teacher-forcing CE | 本密封对比未重报（C2 有独立 forecast 轨迹） |

可比的核心列是 **Return10d Rank IC**：同一包、同一 `return_10d`、都是日内 Spearman 再对日等权平均。

## 并列表（主表：日均 Rank IC）

数据来源：

- Beta：`/workspace/kronos_patrol_out/oos_20261005_2215/output/beta_v2_1_c1_rank_oos/results/*_predictions.csv` 本地重算（无需新 kernel）
- Small：`finetune/reports/c2_18d_alpha_oos_metrics_rank.json` / `c2_18d_alpha_oos_summary.json`
- 机器可读汇总：`finetune/reports/beta_v21_c1_rank_oos_vs_small_c2_return10d.json`

| 模型 | Utility Rank IC（日均） | Return10d Rank IC（日均） | Return10d Rank IC（pooled） | Pairwise | WFL |
|---|---:|---:|---:|---:|---:|
| **Small C2 best seg179**（rank decode T0.6/p0.9/N16） | — | **0.1796** | **0.2428** | — | — |
| Small C2 best（prod T0.65/p0.8/N5） | — | 0.1769 | 0.2373 | — | — |
| **Beta Seg19** rank-frozen best | **0.0300** | **-0.0073** | 0.0305 | **51.54%** | **2.43604** |
| Beta Seg8 rank-unfreeze best | 0.0224 | -0.0298 | 0.0065 | 51.09% | 2.44870 |
| Beta Seg155 forecast best | —（无 ranking score） | — | — | — | 2.43604 |

补充：

- Seg19 utility ICIR **0.327**，IC+ 日占比 **66.7%**；return10d ICIR **-0.069**，正 IC 日仅 **33.3%**。
- Seg8 utility ICIR 0.249；return10d ICIR -0.289，正 IC 日 33.3%。
- Seg155：`use_beta_v21_auxiliary=False`，predictions 里 `score` 全空，不能报 Rank IC / pairwise；WFL 与 Seg19 相同（冻结 trunk 预报地板）。

## 结论

1. 用户记得的 Small「~0.2」≈ 同包 D10 **日均 Return Rank IC ~0.18**（pooled ~0.24）。不是 utility IC，也不是方向准确率。
2. Beta 报告的 **0.0300** 是 **Utility Rank IC**，合同不同，不能直接和 0.18 比。
3. **即便换成同一 Return10d 合同，Seg19 日均 IC 为 -0.007**，仍远弱于 Small ~0.18。因此：既是指标口径差异，也是 Beta ranking 头在收益截面上明显更弱。
4. Seg19 仍略优于 Seg8（utility IC / pairwise / WFL）；Seg155 只能当预报地板对照。

## 以后 OOS 怎么记

`finetune/evaluate_beta_v21_time_oos.py` 已同时写出：

- `rank_ic` / `pooled_rank_ic`（vs utility）
- `return10d_rank_ic` / `pooled_return10d_rank_ic`（vs return_10d）

下次跑 `luckfu/kronos-beta-v21-c1-rank-oos` 无需手工重算。

## 相关路径

- 评测脚本：`finetune/evaluate_beta_v21_time_oos.py`
- OOS 计划：`finetune/docs/beta_v21_c1_rank_oos_plan_cn.md`
- 本对比 JSON：`finetune/reports/beta_v21_c1_rank_oos_vs_small_c2_return10d.json`
- Small 18d：`finetune/reports/c2_18d_alpha_oos_summary.json`
