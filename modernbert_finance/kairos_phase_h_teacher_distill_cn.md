# Kairos Phase H：C2 教师蒸馏 + 路径触及 MFE≥10%（廉价消融）

日期：2026-10-01 18:19 CST。

## 一句话结论

**student_distills_proxy_teacher__ranking_alpha_present__no_binary_train**；train_yes_no=`no`。

- 建议：Path-touch MFE≥10% logistic did not clear Δ≤-0.04; C2 teacher on sealed OOS shows Rank IC on continuous returns but weak TopK/binary for +10% path-touch — alpha lives in ranking, not this binary. Do not open Kairos binary train. If pursuing ranking, need C2 scores on val dates or a true ranking objective — not more threshold knobs.

## 标签澄清（相对 Phase G）

- **主标签（本 Phase）**：`y=1{mfe10>=0.10} where mfe10=max(high[T+1:T+10])/close[T]-1 (path-touch)`
- **Phase G**：`y=1{fwd_ret_10>=0.10} close-to-close (Phase G)`
- 正类率 path-touch MFE≥10% = `0.2529` (`31323` / `123836`)
- 正类率 close-to-close ≥10% = `0.1104`
- 二者同时为正率 = `0.1104`；P(ctc|mfe)=`0.4365`；P(mfe|ctc)=`1.0000`

## C2 Best Seg@179 可用性

- checkpoint=`cosine_c2_best` segment=`179`
- OOS asof=`2026-08-11`..`2026-09-03`
- Val asof=`2025-07-03`..`2026-07-02`
- date_overlap=`False`
- **blocker**：C2 Best Seg@179 OOS scores only cover sealed window 2026-08-11..2026-09-03; Kairos val panel 2025-07-03..2026-07-02. No date×symbol join without full C2 inference (skipped: no TPU / long train).
- **fallback**：relative_strength_pairing_proxy_on_val

## 1) C2 教师 → 标签上界（密封 18 日 OOS，真实分数）

- 协议=`rank_t060_p90_n16` n=`92751`
- 路径触及近似：Approx MFE = max(actual_return_d1..d10); C2 dumps use cumulative close returns from entry — not high-price MFE.
- Daily Rank IC vs return_10d mean=`0.1793179326743008` (n_days=`18`)
- Daily Rank IC vs path-touch approx mean=`-0.016549481696334332`
- Top20% hit vs path-touch mean=`0.1589631532288894` chance=`0.2` lift=`-0.04103684677111061`
- Teacher→path-touch≥10% logistic Δ=`-0.003542422776723897`
- buy_profit ctc pos teacher pctile=`0.4411741980535561` vs neg=`0.5057031668742924`
- path-touch pos teacher pctile=`0.4303553579312847` vs neg=`0.5127181656996951`

## 2) Val 面板：路径触及 logistic（特征 → 标签）

| Head | prior LL | model LL | Δ |
| --- | ---: | ---: | ---: |
| mfe_touch_10pct_base | 0.565550 | 0.545441 | -0.020109 |
| mfe_touch_10pct_comb | 0.565550 | 0.531420 | -0.034130 |
| ctc_10pct_base | 0.347427 | 0.338827 | -0.008599 |
| ctc_10pct_comb | 0.347427 | 0.332067 | -0.015359 |
| fwd_sign_up_comb | 0.692555 | 0.689248 | -0.003308 |

- Phase G comb Δ（ctc）=`-0.015359321769126855`
- 本 Phase 重算 ctc comb Δ=`-0.015359321769126855`
- 本 Phase mfe-touch comb Δ=`-0.034130436131804`
- 过绝对门槛 Δ≤-0.04：`False`；明显优于 G：`False`

## 3) RS/配对代理教师（因 C2 无法对齐 val）

- 定义：`held-out RS: 0.35*cs_rank(ret10)+0.35*cs_rank(ret60)+0.15*cs_rank(ind_rel_ret10)+0.15*cs_rank(ind_rel_ret60)`
- Rank IC vs fwd_ret_10=`-0.04822241654501768`
- Rank IC vs mfe10=`0.16430765421773486`
- Top20 hit lift vs mfe=`0.13898440920342753`
- Teacher→mfe-touch logistic Δ=`-0.014095075009395774`

## 4) Student 蒸馏（Base / Base+xsection → 代理教师）

- Ridge `teacher_rs_base`: R²=`0.4185180151833724` spearman=`0.667469815638623`
- Ridge `teacher_rs_comb`: R²=`0.6395072833050761` spearman=`0.7880902619166877`
- Logistic `teacher_top20_base`: Δ=`-0.1991274259412648`
- Logistic `teacher_top20_comb`: Δ=`-0.25224293907790774`
- Student pred daily IC vs teacher (comb)=`0.7858326130759751`
- Student pred daily IC vs mfe (comb)=`0.1774923275770929`
- student_above_chance=`True`

## 5) 决策

- verdict = `student_distills_proxy_teacher__ranking_alpha_present__no_binary_train`
- train_yes_no = `no`
- teacher_shows_ranking_alpha = `True`
- alpha_in_ranking_not_binary = `True`
- C2 Rank IC (return_10d)=`0.1793179326743008`；path-touch=`-0.016549481696334332`

### 建议下一步

1. **不要**基于 path-touch MFE≥10% 或 Phase G ctc 开 Kairos 二分类长训。
2. C2 教师在密封窗证明**排序**有 alpha；+10% 路径/收盘二分类不是同一回事。
3. 若要继续蒸馏，需把 C2 Seg@179 分数物化到 val 日期（推理成本高）或改用排序目标。
4. **不要**重启同配置 R2 / 不要动同事 TPU WIP。

## 用户要点（中文）

- **定义**：持有至多 10 日，路径最高价相对买入价涨幅 ≥10% 记正类（mfe10）。
- **正类率**：MFE=`0.2529`；ctc=`0.1104`
- **特征→MFE logistic Δ**：`-0.034130436131804`（G 的 ctc Δ=`-0.015359321769126855`）
- **C2 上界 Rank IC**：`0.1793179326743008`（排序有用，二分类弱）
- **开训**：`否`

