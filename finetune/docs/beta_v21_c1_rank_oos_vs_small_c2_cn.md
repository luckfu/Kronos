# Beta C1 Ranking OOS vs Small C2（同密封包 · 决定性对比）

更新：2026-10-06 CST（新增 Baseline 2：Seg155 生成式派生收益，18 日完整结果）

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
| **Beta Seg155 生成式** prod T0.65/p0.8/N5 | 生成式 `predicted_return_d10` | **0.1170** | **0.1631** | **0.0757**‡ | 0.1061‡ | **54.06%**‡ | 2.43604 |
| Beta Seg155 生成式 rank T0.6/p0.9/N16 | 生成式 `predicted_return_d10` | 0.1242 | 0.1764 | 0.0773‡ | 0.1110‡ | 54.16%‡ | 2.43604 |

‡ Seg155 生成式同样是 return-score ablation（无 utility 头）；Baseline 2 详见下节。

† C2 的 Utility IC / Pairwise 是 **return-score ablation**（把 `predicted_return_d10` 接到包内 utility 标签），**不是**训练过的 utility 头。Pairwise 合格 pair 数 ≈ **1.2487e8**（与 Beta 同量级）。

补充：

- C2 rank：utility ICIR **0.856**，IC+ 日 **77.8%**；return10d ICIR **1.668**，IC+ 日 **88.9%**。
- Seg19：utility ICIR **0.327**，IC+ 日 **66.7%**；return10d ICIR **-0.069**，IC+ 日仅 **33.3%**。
- C2 对包内 `return_10d` 日均 IC 为 0.1793（与 `actual_return_d10` 的 0.1796 几乎相同；~173 行涨跌停/重建异常）。

机器可读：`finetune/reports/beta_v21_c1_rank_oos_vs_small_c2_return10d.json`

## Baseline 2：同 trunk 生成式派生收益（2026-10-06，零训练）

**问题**：C2 的 ~0.18 是否来自生成式 OHLC 路径？我们的 `return_head` 是直接回归、被 ranking 污染？
**做法**：冻结 trunk，不训练。用 **Seg155 forecast-best**（SHA `8b11a759e72d…`，无 aux 头；Seg19 trunk 由它冻结而来，OOS WFL 完全相同 2.43604 → 生成式路径相同）在同一密封包上走 C2 同配方 AR 采样：
`score = mean_N( denorm_close[d10] / last_close − 1 )`，真 `auto_regressive_inference`（非 teacher-forcing），seed 20260906，top_k=0，clip=5，sector + size_percentile 条件。
Kernel：`luckfu/kronos-beta-v21-c1-gen-return-oos` v1 **COMPLETE**（42,221 s ≈ 11.73 h，~10-06 10:55 CST 结束，距 12h 硬杀 ~16 min）。代码 `78b3bda`，runner 修复 `cbd4b0a`。

### 18 日完整表（主指标 = return10d Rank IC）

| 模型 / 分数 | ret10d IC 日均 | pooled | ICIR | IC+ 日 | utility IC 日均 | Pairwise |
|---|---:|---:|---:|---:|---:|---:|
| **Seg155 生成式 prod** T0.65/p0.8/N5 | **0.1170** | 0.1631 | 1.25 | 15/18（83.3%） | 0.0757 | 54.06% |
| **Seg155 生成式 rank** T0.6/p0.9/N16 | **0.1242** | 0.1764 | 1.19 | 15/18（83.3%） | 0.0773 | 54.16% |
| Small C2 rank T0.6/p0.9/N16 | 0.1796 | 0.2428 | 1.67 | 16/18（88.9%） | 0.1210 | 56.49% |
| Small C2 prod T0.65/p0.8/N5 | 0.1769 | 0.2373 | 1.69 | 16/18（88.9%） | 0.1208 | 56.50% |
| Seg19 `return_head`（`expected_utility_score`） | −0.0073 | 0.0305 | −0.07 | 6/18（33.3%） | 0.0300 | 51.54% |

差值：生成式 vs Seg19 return_head **+0.124（prod）/ +0.131（rank）**；vs C2 同臂 **−0.060 / −0.056**。两臂只差 ~0.007。

### 逐日 return10d Rank IC

| asof | Seg155 prod | Seg155 rank | C2 prod | C2 rank |
|---|---:|---:|---:|---:|
| 08-11 | 0.171 | 0.179 | 0.191 | 0.177 |
| 08-12 | 0.219 | 0.231 | 0.263 | 0.270 |
| 08-13 | 0.113 | 0.115 | 0.181 | 0.171 |
| 08-14 | 0.199 | 0.203 | 0.242 | 0.243 |
| 08-17 | 0.212 | 0.236 | 0.288 | 0.296 |
| 08-18 | 0.260 | 0.284 | 0.353 | 0.355 |
| 08-19 | 0.132 | 0.168 | 0.174 | 0.180 |
| 08-20 | 0.126 | 0.150 | 0.127 | 0.136 |
| 08-21 | 0.183 | 0.200 | 0.264 | 0.278 |
| 08-24 | 0.119 | 0.134 | 0.133 | 0.130 |
| 08-25 | 0.071 | 0.074 | 0.130 | 0.142 |
| 08-26 | 0.030 | 0.027 | 0.080 | 0.082 |
| 08-27 | 0.168 | 0.178 | 0.264 | 0.266 |
| 08-28 | 0.104 | 0.102 | 0.187 | 0.200 |
| 08-31 | 0.134 | 0.131 | 0.267 | 0.276 |
| **09-01** | **−0.003** | **−0.024** | 0.097 | 0.102 |
| **09-02** | **−0.075** | **−0.079** | −0.034 | −0.041 |
| **09-03** | **−0.058** | **−0.073** | −0.025 | −0.029 |

**尾部转负**：最后 3 日（09-01..09-03）两臂均为负。C2 在 09-02/09-03 也转负（但幅度更小，且 09-01 仍为 +0.10），说明尾部一部分是市场状态，但我们的尾部更差。前 6 日 prod 均值 0.196、rank 0.208。

### 结论

1. **方向没死，头死了。** 同一 trunk，生成式派生分数比 `return_head` 高 ~0.12–0.13 return10d IC。之后的 ranking 工作从 **生成式派生收益** 出发：**不再新建回归头，trunk 冻结**。
2. **评分合同**：纯收益；主指标 = return10d Rank IC（日均 Spearman vs 包内 `return_10d`）。utility / pairwise 只作为 ablation 旁证。
3. **更正**：早先 6 日读数（~0.196，「达到 C2 水平」）为时过早。完整 18 日只有 C2 的 **~2/3**（0.117 / 0.124 vs 0.180），ICIR 更弱（1.19–1.25 vs 1.67），且尾部转负。**因此不取消从 C2 蒸馏**：若做，蒸馏 C2 的 **排序（pairwise 次序）而不是数值**，并从第一个 segment 起就做 out-of-time 验证。
4. **脚注（值得挖）**：同样用纯收益分数去排 utility 标签，C2 **0.121** vs 我们 **0.076**。说明 C2 的收益预测在不同标签定义之间更稳定（可能尾部更紧 / 校准更好）。若纯收益 IC 遇到平台期，从这里挖。
5. **流程教训**：
   - 每个臂 decode 完 **立即打分并落盘**（runner 已修，`cbd4b0a`；原版只在两臂都完后才打分，差 16 min 就会被 12h 硬杀全丢）。
   - **先跑最便宜的决定性臂**：prod N5 ~2.7 h vs rank N16 ~8.7 h（2×T4），两臂只差 ~0.007 IC。

机器可读：`finetune/reports/beta_v21_c1_baseline2_gen_return_oos_18d.json`（逐日 return/utility IC + pairwise、两臂全指标、参考值、结论）；汇总条目也写进 `beta_v21_c1_rank_oos_vs_small_c2_return10d.json` 的 `beta_seg155_generative_return`。
配方 / kernel：`finetune/docs/beta_v21_c1_baseline2_gen_return_oos_cn.md`。

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
- Baseline 2 生成式收益：`finetune/evaluate_beta_v21_generative_return_oos.py`；结果 JSON `finetune/reports/beta_v21_c1_baseline2_gen_return_oos_18d.json`
