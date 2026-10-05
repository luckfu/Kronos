# Kairos 买入盈利(+10%/10日)二分类消融（Phase G）

日期：2026-10-01 18:06 CST。

## 一句话结论

**买入盈利(+10%/10日)未达开训门槛。** verdict=`buy_profit_10pct__no_lift__do_not_train`；train_yes_no=`no`。

- 建议：买入盈利(+10%/10日)未过 Δ≤-0.04，也未相对 Phase E/F 二分类明显更好；不要据此开训。可结束廉价消融，或换质变标签，而非再拧阈值。

## 设定 / 定义

- n_samples=`123836`，seed=`20261001`，base_dim=`20`，xsec_dim=`6`
- **主定义**：`y = 1{fwd_ret_10 >= 0.10} where fwd_ret_10 = close[T+10]/close[T]-1`
- 可选行业中性：`y_ind = 1{ind_resid_fwd_10_mean >= 0.10} (Phase F mean demean; fallback market when industry-day n < 2)`
- 阈值：`0.1`；horizon=`10`
- 特征：Phase D/E/F 同款 Base + 截面（看 comb）
- **未**重启 R2 / **未** Kaggle 长训；**未**动同事 TPU WIP

## 门槛

- 绝对：Δ logloss ≤ `-0.04` vs 常数先验
- 相对 Phase E/F 二分类：Δ 再负 `0.02` （对照 best of E sign / E CS-top20 / F resid-sign / F resid CS-top20）
- Phase E fwd_sign_up_comb Δ=`-0.0033075507187767528`，fwd_cs_top20_comb Δ=`-0.012270859304657045`
- Phase F ind_resid_sign_up_mean_comb Δ=`-0.002215324511504857`，ind_resid_cs_top20_comb Δ=`-0.010187678560042601`
- best Phase E/F binary Δ=`-0.012270859304657045`

## 标签分布

- **正类率 (buy_profit_10pct)** = `0.1104` (`13674` / `123836`)
- class_too_rare=`False` (flag < `0.05`)；class_imbalanced=`False` (warn < `0.1`)
- fwd_ret_10 mean/std=`0.0050`/`0.0977`
- fwd_ret_10 p90/p95/p99=`0.1073`/`0.1659`/`0.3437`
- fwd_sign_up rate=`0.4828`
- ind_resid ≥ threshold 正类率=`0.0777` (n_pos=`9621`)

## 1) Logistic（买入盈利 vs 先验）

| Head | prior LL | model LL | Δ | pos_rate_test |
| --- | ---: | ---: | ---: | ---: |
| buy_profit_10pct_base | 0.347427 | 0.338827 | -0.008599 | 0.1104 |
| buy_profit_10pct_comb | 0.347427 | 0.332067 | -0.015359 | 0.1104 |
| fwd_sign_up_base | 0.692555 | 0.690043 | -0.002512 | 0.4828 |
| fwd_sign_up_comb | 0.692555 | 0.689248 | -0.003308 | 0.4828 |
| buy_profit_10pct_ind_resid_base | 0.273075 | 0.265391 | -0.007684 | 0.0777 |
| buy_profit_10pct_ind_resid_comb | 0.273075 | 0.258104 | -0.014970 | 0.0777 |

## 2) 决策

| 指标 | 值 | 过关 |
| --- | ---: | :---: |
| 主 Δ (buy_profit comb) | -0.015359321769126855 | N |
| vs Phase E/F clearly better | — | N |
| 行业中性 Δ (optional) | -0.01497032949139887 | N |
| 行业中性 vs E/F | — | N |
| 正类过稀 | False | warn |

- lifts_enough = `False`
- verdict = `buy_profit_10pct__no_lift__do_not_train`
- train_yes_no = `no`
- 判据：Primary: y=1{fwd_ret_10>=0.1} logistic comb. Pass if Δ≤-0.04 or clearly better than best Phase E/F binary (Δ more negative by 0.02). Industry-neutral same-threshold resid is optional diagnostic/alternate.

### 诊断备注

- Primary positive rate=0.1104 (n_pos=13674/123836); threshold=0.10 on fwd_ret_10.
- Industry-neutral resid >= 0.10 positive rate=0.0777.
- Train panel not on disk locally; ablation uses validation panel only (n=123836).
- Primary comb Δ=-0.015359 vs bar -0.04; best Phase E/F binary Δ=-0.012270859304657045.

### 建议下一步

1. **不要**基于买入盈利(+10%/10日)开标签重建长训。
2. 廉价标签消融路径可收束；若再试需质变，非再拧阈值。
3. **不要**重启同配置 R2。

## 证据强度

| 主张 | 强度 | 依据 |
| --- | --- | --- |
| 买入盈利可开下一步 | 弱/否 | Δ 见表 |
| 应维持停 R2 | 强 | 与 Phase C–F 一致 |

## 用户要点（中文）

- **定义**：未来 10 个交易日收益 `close[T+10]/close[T]-1 ≥ 0.1` 记为买入盈利正类。
- **正类率**：`0.1104`；行业中性残差同阈值正类率=`0.0777`
- **指标**：Base Δ=`-0.008599478298647378`；Base+xsection Δ=`-0.015359321769126855`；行业中性 comb Δ=`-0.01497032949139887`
- **过关**：`否`（门槛 Δ≤-0.04 或明显优于 Phase E/F 二分类）
- **开训**：`否`

