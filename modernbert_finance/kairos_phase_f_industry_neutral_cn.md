# Kairos 行业中性前向收益消融（Phase F）

日期：2026-10-01 18:01 CST。

## 一句话结论

**行业中性 fwd 未达开训门槛。** verdict=`industry_neutral__no_lift__do_not_train`；train_yes_no=`no`。

- 建议：行业中性 fwd 未过 Phase E 同门槛，也未相对 Phase E 原 fwd 明显抬升；不要据此开训。可结束廉价消融，或换成本感知/更长 horizon，而非再拧 demean。

## 设定 / 定义

- n_samples=`123836`，seed=`20261001`，base_dim=`20`，xsec_dim=`6`
- **主定义**：`ind_resid_fwd_10 = close[T+10]/close[T]-1 - same-day industry mean(fwd); fallback to market mean when industry-day n < 2`
- 敏感性：`same with industry/market median`；vol_scaled=`resid_mean / (vol20 * sqrt(H))`
- 行业字段：`sector` — Industry id = panel column `sector` (CSRC industry string, e.g. 'G56航空运输业'). No separate industry_id field in panel; sector_vocabulary.json maps the same labels. Train panel not present locally — Phase F uses validation panel only (same as Phase D/E).
- 行业日样本：n_sectors=`73`，industry-day n mean/median=`23.07`/`18.0`，market-fallback rate=`0.0247`
- 特征：Phase D/E 同款 Base + 截面（看 comb）
- **未**重启 R2 / **未** Kaggle 长训；**未**动同事 TPU WIP

## 门槛

- 绝对：方向 R²≥`0.15` 或 sign Δ≤`-0.04`（与 Phase E 相同）
- 相对 Phase E Protocol A fwd：R²+`0.04` 或 Δ 再负`0.02`
- Phase E fwd_ret_10_comb R²=`0.00983496291297703`，fwd_sign_up_comb Δ=`-0.0033075507187767528`

## 标签分布

- raw fwd_ret_10 mean/std=`0.0050`/`0.0977`
- ind_resid_mean mean/std=`-0.0002`/`0.0832`
- sign_up rate mean/median demean=`0.4376`/`0.4779`

## 1) Ridge（连续残差）

| Target | naive MAE | ridge MAE | ridge R² | beats |
| --- | ---: | ---: | ---: | :---: |
| fwd_ret_10_base | 0.066813 | 0.066589 | 0.008385 | Y |
| fwd_ret_10_comb | 0.066813 | 0.066565 | 0.009835 | Y |
| ind_resid_fwd_10_mean_base | 0.053548 | 0.053506 | 0.007822 | Y |
| ind_resid_fwd_10_mean_comb | 0.053548 | 0.053480 | 0.010795 | Y |
| ind_resid_fwd_10_median_base | 0.052387 | 0.052208 | 0.010700 | Y |
| ind_resid_fwd_10_median_comb | 0.052387 | 0.052110 | 0.014591 | Y |
| ind_resid_fwd_5_mean_base | 0.037439 | 0.037415 | 0.005358 | Y |
| ind_resid_fwd_5_mean_comb | 0.037439 | 0.037382 | 0.007012 | Y |
| ind_resid_fwd_10_vol_scaled_base | 0.724498 | 0.724135 | 0.002917 | Y |
| ind_resid_fwd_10_vol_scaled_comb | 0.724498 | 0.723778 | 0.004139 | Y |

## 2) Logistic（符号）

| Head | prior LL | model LL | Δ |
| --- | ---: | ---: | ---: |
| fwd_sign_up_base | 0.692555 | 0.690043 | -0.002512 |
| fwd_sign_up_comb | 0.692555 | 0.689248 | -0.003308 |
| ind_resid_sign_up_mean_base | 0.685334 | 0.684071 | -0.001263 |
| ind_resid_sign_up_mean_comb | 0.685334 | 0.683119 | -0.002215 |
| ind_resid_sign_up_median_base | 0.692169 | 0.690680 | -0.001490 |
| ind_resid_sign_up_median_comb | 0.692169 | 0.689085 | -0.003084 |
| ind_resid_cs_top20_comb | 0.501930 | 0.491742 | -0.010188 |
| ind_resid_cs_bot20_comb | 0.499200 | 0.474170 | -0.025031 |

## 3) 决策

| 指标 | 值 | 过关 |
| --- | ---: | :---: |
| 主方向 R² (mean demean comb) | 0.01079463032172745 | N |
| 主方向 Δ (sign mean comb) | -0.002215324511504857 | N |
| vs Phase E R² lift | — | N |
| vs Phase E Δ lift | — | N |
| median demean R² / Δ | 0.014591104055576065 / -0.0030839517557547103 | diag |
| vol-scaled resid R² | 0.004139485559858991 | diag |
| raw fwd R² / Δ (echo) | 0.00983496291297703 / -0.0033075507187767528 | — |

- lifts_enough = `False`
- verdict = `industry_neutral__no_lift__do_not_train`
- train_yes_no = `no`
- 判据：Primary: industry-mean-demeaned fwd_ret_10 (comb). Pass if R²≥0.15 or sign Δ≤-0.04, or clear lift vs Phase E Protocol A fwd (+0.04 R² or Δ more negative by 0.02). Median demean / vol-scaled / CS top-bot are diagnostics.

### 诊断备注

- Industry field=`sector` (CSRC); n_sectors=73; industry-day n mean/median=23.07/18.0; market-fallback rate=0.0247.
- Train panel not on disk locally; ablation uses validation panel only (n=123836).
- Mean-demean resid R²=0.010795 vs raw fwd R²=0.009835 (ΔR²=+0.000960).
- Industry-neutral sign Δ≈0 — demeaning does not unlock linear direction.

### 建议下一步

1. **不要**基于行业中性 fwd 开标签重建长训。
2. 廉价标签消融路径可收束；若再试需质变（成本/更长 horizon），非 demean 变体。
3. **不要**重启同配置 R2。

## 证据强度

| 主张 | 强度 | 依据 |
| --- | --- | --- |
| 行业中性可开下一步 | 弱/否 | R²/Δ 见表 |
| 应维持停 R2 | 强 | 与 Phase C–E 一致 |

