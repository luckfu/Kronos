# Beta C1 Ranking OOS vs Small C2（同密封包 · 决定性对比）

更新：2026-10-07 CST（新增 Best@475 / Seg9 密封 OOS 生成式收益最终评估；此前：Baseline 2 Seg155 生成式派生收益、val 生成式 IC 重选、Best@475 余弦 pilot val 结果）

> **最新头条（2026-10-07）：Beta 线 OOS 最强的模型现在是发布版 Best@475（生成式收益，未退火）：密封 18d return10d IC 日均 0.155 vs C2 0.180（rank）/ 0.177（prod）。**
> 差距从 Seg155 的 ~0.060 缩小到 ~0.02–0.025（Seg0 只在 5/18 天胜过 C2）；余弦退火 Seg9（0.146）没有超过 Best@475。
> 详见下方「Best@475 / Seg9 密封 OOS 最终评估」与 [`beta_v21_c1_forecast_cosine_pilot_cn.md`](beta_v21_c1_forecast_cosine_pilot_cn.md)。
> 注意：这个密封包已读过 Seg155、Seg9、Seg0，**已部分用于选择**。

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

**尾部转负**：最后 3 日（09-01..09-03）两臂均为负。C2 在 09-02/09-03 也转负（但幅度更小，且 09-01 仍为 +0.10），说明尾部主要是市场状态；我们的尾部 IC 绝对值更低，但与 C2 的差距（−0.058）与全期差距（−0.060）相当，没有放大（见下方「Top-Bottom 价差与尾部刻画」）。前 6 日 prod 均值 0.196、rank 0.208（前 6 天早期读数，已纠正；同期 C2 rank 为 0.252）。

### Top-Bottom 价差与尾部刻画（2026-10-06，零 GPU）

脚本 / 原始输出：`/workspace/kronos_patrol_out/genret_followup_1006/analyze.py` → `analysis.json`。
**这里的收益和 IC 都对齐包内 `return_10d`**（C2 IC 0.177/0.179 与上表 `actual_return_d10` 口径 0.1769/0.1796 只差约 0.0003）。Top-Bottom = 每日按分数分组，最高组减最低组的平均 return10d，再取日均。**不含换手和成本**。

| 模型 | IC 18d | TB 十分位 18d | TB 五分位 18d | IC 前15日 | TB 十分位 前15日 | IC 尾3日 | TB 十分位 尾3日 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seg155 生成式 prod N5 | 0.117 | +2.07% | +1.82% | 0.149 | +3.12% | -0.045 | -3.17% |
| Seg155 生成式 rank N16 | 0.124 | +2.37% | +1.99% | 0.161 | +3.52% | -0.059 | -3.41% |
| C2 prod N5 | 0.177 | +2.83% | +2.65% | 0.209 | +3.98% | 0.013 | -2.87% |
| C2 rank N16 | 0.179 | +2.85% | +2.63% | 0.213 | +3.97% | 0.010 | -2.77% |

全 18 日：我们的十分位价差 **+2.07%（prod）/ +2.37%（rank）**，C2 是 +2.83% / +2.85%。前 15 日价差为 +3.1% / +3.5%，C2 为 +4.0%。尾 3 日四个模型的十分位价差**全部为负**，在 −2.8% 到 −3.4% 之间。

**尾 3 日（09-01..09-03）逐日**（十分位单调性 = 十组均值与组号的 Spearman，+1 表示完全单调）：

| 日期 | 全市场 return10d 均值 | 截面 std | 上涨占比 | Seg155 prod IC / TB | C2 prod IC / TB | Seg155 prod 十分位单调性 | C2 prod 十分位单调性 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 09-01 | -4.26% | 7.50% | 17.4% | -0.003 / -2.09% | 0.097 / -1.24% | -0.33 | +0.30 |
| 09-02 | -2.05% | 7.83% | 25.5% | -0.075 / -3.99% | -0.034 / -4.59% | -0.71 | -0.26 |
| 09-03 | -1.62% | 7.55% | 26.7% | -0.058 / -3.42% | -0.025 / -2.78% | -0.84 | -0.56 |
| 前15日均值 | +0.68% | 8.22% | 50.8% | 0.149 / +3.12% | 0.209 / +3.98% | +0.55 | +0.84 |
| 尾3日均值 | -2.64% | 7.63% | 23.2% | -0.045 / -3.17% | 0.013 / -2.87% | — | — |

十分位收益分布（单位 %）：

| 日期 | 模型 | D1（最低预测）… D10（最高预测）return10d 均值 |
|---|---|---|
| 09-01 | Seg155 生成式 prod N5 | -2.7 -4.5 -4.6 -4.2 -4.2 -4.2 -4.5 -4.3 -4.5 -4.8 |
| 09-01 | C2 prod N5 | -2.3 -4.7 -5.3 -4.9 -5.0 -5.0 -4.3 -4.1 -3.6 -3.5 |
| 09-02 | Seg155 生成式 prod N5 | 1.4 -0.9 -2.2 -2.8 -2.5 -2.4 -2.2 -3.2 -3.1 -2.6 |
| 09-02 | C2 prod N5 | 2.2 -1.0 -2.4 -3.1 -3.2 -3.0 -2.8 -2.4 -2.5 -2.4 |
| 09-03 | Seg155 生成式 prod N5 | 1.5 -1.3 -1.9 -1.8 -1.9 -1.8 -2.0 -2.3 -2.5 -2.0 |
| 09-03 | C2 prod N5 | 1.0 -0.3 -2.1 -2.2 -1.9 -2.1 -2.2 -2.2 -2.4 -1.7 |

**判断：以市场状态为主，不是我们独有的衰减。**
- **大盘异常**：尾 3 日全市场 10 日收益均值 **−2.64%**（前 15 日 +0.68%），上涨占比只有 **23.2%**（前 15 日 50.8%），截面离散度略低（7.63% vs 8.22%）。08-28 和 08-31 已经开始下跌（上涨占比 27% / 25%），但两边价差仍为正，只是变小。
- **两个模型同样翻转**：尾 3 日 C2 的十分位价差也全为负（−1.2%、−4.6%、−2.8%）。翻转的形态一致：**预测最低的 D1 组反而跌得最少**（例如 09-02，D1 为 +1.4%/+2.2%，其余组都在 −1% 到 −3%），属于普跌行情里的「最差预测反弹」。C2 在 09-01 的 IC 仍有 +0.10，是因为它 D2–D10 的排序还算单调；但它的极端组价差同样是负的。
- **我们没有额外的衰减**：尾 3 日 IC 差（我们 −0.045 vs C2 +0.013，差 −0.058）≈ 全 18 日 IC 差（−0.060）。也就是说，尾部只是把我们和 C2 之间一直存在的约 0.06 差距照常延续下来，并没有放大。
- **样本提醒**：09-01..09-03 的 10 日标签窗口几乎完全重叠，实际只相当于**约 1 个独立观测**。不能据此判断「新的市场状态是否已经开始」。

**IC 差距的显著性**（我们 − C2，同一臂逐日配对）：prod **−0.0595**，iid SE 0.0081，t ≈ −7.4；NW(9) SE 0.0065。rank **−0.0552**，iid SE 0.0102，t ≈ −5.4；NW(9) SE 0.0122，t ≈ −4.5。18 天里有 17 天（prod）/ 15 天（rank）我们低于 C2。
如果把两边当成相互独立（不配对），各自日均 IC 的 SE 约 0.022–0.025，差距约为 **2.4–2.7 个 SE**。配对后之所以看起来更显著，是因为两个模型每天的 IC 同涨同跌。但 10 日标签有重叠，18 个信号日只相当于约 2 个独立的 10 日窗口，NW 在这么少的点上并不可靠。结论应该保守地写成「差距大致在 2–3 SE、方向稳定」。

**可交易性（换手 / 成本）暂缓**：先把尾部刻画清楚（需要 09-03 之后的扩展密封窗口），再做换手和成本回测。

### 核心数（主指标 return10d Rank IC 日均）

| 读数 | Seg155 prod N5 | Seg155 rank N16 | C2 rank |
|---|---:|---:|---:|
| **完整 18 日（正式数）** | **0.117** | **0.124** | 0.180 |
| 前 6 天早期读数，已纠正 | 0.196 | 0.208 | 0.252 |

> 早期 0.196 只是前 6 天（08-11..08-18）的读数，当时被解读为「达到 C2 水平」，这个解读已纠正。同样前 6 天 C2 是 0.252，所以即使在那 6 天里我们也没有达到 C2 水平。正式数以完整 18 日为准。

> **prod 和 rank 是同一个 checkpoint（Seg155），只是 decode 不同（N5 vs N16，T/top_p 也不同）**。两臂结果一致（差 ~0.007）只能说明结果**对 decode 设置不敏感**，**不能**说明换 checkpoint 也稳定。

### 结论

1. **方向没死，头死了。** 同一 trunk，生成式派生分数比 `return_head` 高 ~0.12–0.13 return10d IC。之后的 ranking 工作从 **生成式派生收益** 出发：**不再新建回归头，trunk 冻结**。
2. **评分合同**：纯收益；主指标 = return10d Rank IC（日均 Spearman vs 包内 `return_10d`）。utility / pairwise 只作为 ablation 旁证。
3. **更正**：前 6 天早期读数 0.196（当时称「达到 C2 水平」）已纠正，见上方「核心数」表。完整 18 日只有 C2 的 **~2/3**（0.117 / 0.124 vs 0.180），ICIR 更弱（1.19–1.25 vs 1.67），且尾部转负。**因此不取消从 C2 蒸馏**：若做，蒸馏 C2 的 **排序（pairwise 次序）而不是数值**，并从第一个 segment 起就做 out-of-time 验证。
4. **脚注（值得挖）**：同样用纯收益分数去排 utility 标签，C2 **0.121** vs 我们 **0.076**。说明 C2 的收益预测在不同标签定义之间更稳定（可能尾部更紧 / 校准更好）。若纯收益 IC 遇到平台期，从这里挖。
5. **流程教训**：
   - 每个臂 decode 完 **立即打分并落盘**（runner 已修，`cbd4b0a`；原版只在两臂都完后才打分，差 16 min 就会被 12h 硬杀全丢）。
   - **先跑最便宜的决定性臂**：prod N5 ~2.7 h vs rank N16 ~8.7 h（2×T4），两臂只差 ~0.007 IC。

### 待解问题

- **(a) 与 C2 的约 0.06 差距**：为什么 C2 的收益预测更好？可能的方向：尾部更紧、校准更好、换一种标签定义也稳（同样的纯收益分数，C2 排 utility 0.121 vs 我们 0.076）。另一个线索：前 15 日 C2 的十分位单调性约 +0.83，我们只有约 +0.55（prod）/ +0.63（rank）。待做：比较预测分布的形状和尾部，按分数分位校准，以及 C2 / Seg155 分数之间的相关性。
- **(b) 09-01..09-03 尾部翻转**：是噪声还是新的市场状态开始？两个模型同时翻转，说明市场状态是主因。但这 3 天的标签窗口重叠，只相当于约 1 个独立观测。**需要 09-03 之后的扩展密封窗口**才能判断。
- **(c) 18 日样本太小**：差距大致在 2–3 SE（不配对），配对 t 偏乐观，原因是标签重叠导致有效独立窗口约 2 个。结论只能是「差距方向稳定、量级约 0.06」。
- **可交易性（换手 / 成本）**：暂缓，等 (b) 有结论之后再做。

机器可读：`finetune/reports/beta_v21_c1_baseline2_gen_return_oos_18d.json`（逐日 return/utility IC + pairwise、两臂全指标、参考值、结论）；汇总条目也写进 `beta_v21_c1_rank_oos_vs_small_c2_return10d.json` 的 `beta_seg155_generative_return`。
配方 / kernel：`finetune/docs/beta_v21_c1_baseline2_gen_return_oos_cn.md`。

## Val 生成式 IC 重选 checkpoint（2026-10-06，零训练，未读密封 OOS）

Kernel `luckfu/kronos-beta-v21-c1-val-gen-ic`（COMPLETE，4514 s）。验证集 24 日 linspace 子样本 / 12,256 窗口，
Baseline 2 prod 配方（T0.65/p0.8/N5，seed 20260906），主指标 = 生成式 return10d Rank IC 日均。

| checkpoint | val ret10d IC 日均 | pooled | ICIR | TB 十分位 | WFL（子样本） |
|---|---:|---:|---:|---:|---:|
| **Best@475（Beta v2.1 发布版，C1 之前）** | **0.3145** | 0.4574 | 2.544 | +7.17% | 2.3205 |
| Seg8 rank-unfreeze | 0.2989 | 0.4375 | 2.344 | +6.03% | 2.3221 |
| Seg155 forecast | 0.2954 | 0.4312 | 2.399 | +6.24% | **2.3091** |

配对：Best@475 − Seg155 **+0.0191**（t ≈ 3.64，19/24 天）；Best@475 − Seg8 **+0.0155**（t ≈ 3.56，20/24 天）。
WFL 排序与 IC 排序相反：C1 forecast 训练降低了 WFL，但让 val 生成式 IC 小幅下降，**WFL 不能用来选生成式 checkpoint**。
Seg155 父本的余弦 pilot 因此暂缓。

注意：Seg155 的 val IC（0.295）远高于它的密封 OOS IC（0.117 / 0.124），val 偏乐观；**Best@475 还没有跑过密封 OOS**，
它的 OOS 生成式 IC 未知，不能据此认为它能接近 C2 的 0.18。

详见：`finetune/docs/beta_v21_c1_val_gen_ic_reselection_cn.md`；JSON `finetune/reports/beta_v21_c1_val_gen_ic_reselection.json`。

### 余弦 pilot（Best@475 父本）val 结果（2026-10-06，未读密封 OOS）

12 段 forecast-only uniform_cosine 1e-5→1e-6（`luckfu/kronos-beta-v21-c1-forecast-cosine-pilot-best475`，COMPLETE）。同上 val 合同：
Seg0（Best@475）0.3141 → Seg6 0.3245 / **Seg9 0.3254** / Seg12 0.3234（Seg9 − Seg0 配对 +0.011，t 1.80）；WFL 最好的 Seg10 0.3196。
退火在 val 上只多带来约 +0.01，**补不上**密封 OOS 上 C2 − Seg155 ≈ 0.060 的差距。随后的一次性密封 OOS 最终评估（`wynstonliu/kronos-beta-v21-c1-gen-return-oos-pilot-seg9` v1，COMPLETE）：
Seg0（Best@475）**0.1546** > Seg9 0.1455，val 上的退火收益没有保住，见下节。
详见 `finetune/docs/beta_v21_c1_forecast_cosine_pilot_cn.md`；JSON `finetune/reports/beta_v21_c1_forecast_cosine_pilot_best475.json`。

### Best@475 / Seg9 密封 OOS 最终评估（2026-10-07，生成式收益，prod N5）

Kernel `wynstonliu/kronos-beta-v21-c1-gen-return-oos-pilot-seg9` v1 **COMPLETE**（2026-10-07 03:56 CST，18,477 s）。同包、同 Baseline 2 合同（prod T0.65/p0.8/N5，seed 20260906，
分数 = 生成式 `predicted_return_d10`），事先登记为**一次性最终评估**。下表全部按包内 `return_10d` 在 box 上从 shard 重新计算（与 kernel 输出一致）。

| 模型 | ret10d IC 日均 | pooled | ICIR | IC+ 日 | utility IC 日均 | Pairwise | TB 十分位 | IC 尾3日 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Beta Best@475（Seg0，发布版，未退火）** | **0.1546** | 0.2189 | 1.47 | 16/18 | 0.0956 | 55.13% | +2.68% | -0.025 |
| Beta 余弦 pilot Seg9 | 0.1455 | 0.1887 | 1.28 | 16/18 | 0.0981 | 55.24% | +2.69% | -0.050 |
| Beta Seg155 forecast-best | 0.1170 | 0.1631 | 1.25 | 15/18 | 0.0757 | 54.06% | +2.06% | -0.045 |
| Small C2 prod T0.65/p0.8/N5 | 0.1766 | 0.2369 | 1.69 | 16/18 | 0.1208 | 56.50% | +2.83% | +0.013 |
| Small C2 rank T0.6/p0.9/N16 | 0.1793 | 0.2425 | 1.67 | 16/18 | 0.1210 | 56.49% | +2.84% | +0.010 |

逐日配对（n = 18，t 偏乐观，原因是 10 日标签窗口重叠）：
Seg0 − Seg9 **+0.0091**（Seg0 赢 12/18，t +2.25）；
Seg0 − Seg155 **+0.0376**（18/18）；
Seg0 − C2 prod **-0.0220**（5/18）、− C2 rank **-0.0247**（5/18）；
Seg9 − C2 prod -0.0311（1/18）、− C2 rank -0.0338（1/18）。

要点：

1. **Best@475 是目前 Beta 线 OOS 最强的单模型**，与 C2 的差距约 0.02–0.025（Seg155 时约 0.060）；TB 十分位已接近 C2（+2.68% vs +2.83% / +2.84%）。
2. **C1 forecast 训练让 OOS 生成式 IC 掉了约 0.038**（Seg0 − Seg155，18/18 天）。
3. **余弦退火（Seg9）没有帮助**：val 上 +0.011，OOS 上 −0.009，属于选择噪声。
4. **尾部 09-01..09-03 所有模型（含 C2）一起失效**，属市场状态；Seg0 尾 3 日 IC −0.025，好于 Seg155（−0.045）与 Seg9（−0.050），但仍差于 C2（+0.01）。
5. 密封包已读过 Seg155 / Seg9 / Seg0：用这些数字选 Best@475 本身就是一次选择，0.155 作为前瞻估计略偏乐观；无偏读数要等 09-03 之后的新窗口。

逐日 IC / TB 表、尾部表、完整结论与注意事项：[`beta_v21_c1_forecast_cosine_pilot_cn.md`](beta_v21_c1_forecast_cosine_pilot_cn.md)「密封 OOS 最终评估结果」；
JSON：[`finetune/reports/beta_v21_c1_gen_return_oos_pilot_seg9_vs_best475.json`](../reports/beta_v21_c1_gen_return_oos_pilot_seg9_vs_best475.json)。

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
- Val 生成式 IC 重选：`finetune/docs/beta_v21_c1_val_gen_ic_reselection_cn.md`；JSON `finetune/reports/beta_v21_c1_val_gen_ic_reselection.json`
- 余弦 pilot（Best@475）：`finetune/docs/beta_v21_c1_forecast_cosine_pilot_cn.md`；JSON `finetune/reports/beta_v21_c1_forecast_cosine_pilot_best475.json`
- Best@475 / Seg9 密封 OOS 最终评估：JSON `finetune/reports/beta_v21_c1_gen_return_oos_pilot_seg9_vs_best475.json`
