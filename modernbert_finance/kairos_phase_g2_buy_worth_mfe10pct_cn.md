# Kairos 买入值得(MFE≥+10%/10日)二分类消融（Phase G2）

日期：2026-10-01 18:18 CST。

## 一句话结论

**绝对门槛 Δ≤-0.04 未过（主 Δ=`-0.034130436131804`），但相对 Phase E/F 二分类明显更好 → 仅短 sidecar。** verdict=`buy_worth_mfe10pct__lifts_enough_for_next_step`；train_yes_no=`yes_short_sidecar_only`。

- 建议：MFE买入值得(+10%/10日路径触及)二分类过关：可做短 sidecar 标签重建 + 线性确认；仍不要全量 R2 八头重启 / Kaggle 长训。注意 MFE≠可成交高点（成本未建模）。

## 设定 / 定义（纠正 Phase G）

- n_samples=`123836`，seed=`20261001`，base_dim=`20`，xsec_dim=`6`
- **主定义（决策系统）**：`y = 1{mfe10 >= 0.10} where mfe10 = max(high[T+1:T+10]) / close[T] - 1`
- **mfe10 公式**：`max(high[T+1:T+10]) / close[T] - 1` —— 与 Phase D / `build_targets` 一致：入场后未来 10 个交易日 **最大有利偏移 (MFE)**，非收盘对收盘。
- Phase G 错误目标：`Phase G used y = 1{fwd_ret_10 >= 0.10} where fwd_ret_10 = close[T+10]/close[T]-1 (close-to-close; not path MFE)`
- 决策语义：`worth-it if path reaches +10% within 10 sessions (max favorable excursion from entry ≥ 10%)`
- 阈值：`0.1`；horizon=`10`
- 特征：Phase D/E/F 同款 Base + 截面（看 comb）
- **未**重启 R2 / **未** Kaggle 长训；**未**动同事 TPU WIP

### 成本 / 滑点（仅备注，未建模）

- MFE≥10% 表示路径**触及**高点，不等于能在高点成交离场；买卖价差、冲击与滑点会降低经济命中率。本消融**未**扣成本，仅作语义提醒。

## 门槛

- 绝对：Δ logloss ≤ `-0.04` vs 常数先验
- 相对 Phase E/F 二分类：Δ 再负 `0.02` （对照 best of E sign / E CS-top20 / F resid-sign / F resid CS-top20）
- Phase E fwd_sign_up_comb Δ=`-0.0033075507187767528`，fwd_cs_top20_comb Δ=`-0.012270859304657045`
- Phase F ind_resid_sign_up_mean_comb Δ=`-0.002215324511504857`，ind_resid_cs_top20_comb Δ=`-0.010187678560042601`
- best Phase E/F binary Δ=`-0.012270859304657045`
- Phase G（close-to-close）comb Δ=`-0.015359321769126855`；base Δ=`-0.008599478298647378`

## 标签分布

- **正类率 (buy_worth_mfe10pct)** = `0.2529` (`31323` / `123836`)
- Phase G close-to-close 正类率 = `0.11042023321166704`
- class_too_rare=`False` (flag < `0.05`)；class_imbalanced=`False` (warn < `0.1`)
- mfe10 mean/std=`0.0784`/`0.0937`
- mfe10 p50/p75/p90/p95/p99=`0.0498`/`0.1005`/`0.1822`/`0.2528`/`0.4579`
- mfe_sign_up rate=`0.9592`

## 1) Logistic（MFE 买入值得 vs 先验）

| Head | prior LL | model LL | Δ | pos_rate_test |
| --- | ---: | ---: | ---: | ---: |
| buy_worth_mfe10pct_base | 0.565550 | 0.545441 | -0.020109 | 0.2529 |
| buy_worth_mfe10pct_comb | 0.565550 | 0.531420 | -0.034130 | 0.2529 |
| mfe_sign_up_base | 0.170363 | 0.168614 | -0.001749 | 0.9592 |
| mfe_sign_up_comb | 0.170363 | 0.168216 | -0.002147 | 0.9592 |

## 2) 对照 Phase G（错误 close-to-close）

| 设定 | 正类率 | Base Δ | Comb Δ |
| --- | ---: | ---: | ---: |
| Phase G close-to-close | 0.11042023321166704 | -0.008599478298647378 | -0.015359321769126855 |
| Phase G2 MFE path | 0.2529 | -0.020109348008008898 | -0.034130436131804 |
| Δ(G2−G) comb | — | — | -0.018771114362677144 |
- G2 相对 G 更好 = `True`

## 3) 决策

| 指标 | 值 | 过关 |
| --- | ---: | :---: |
| 主 Δ (MFE buy_worth comb) | -0.034130436131804 | N |
| vs Phase E/F clearly better | — | Y |
| 正类过稀 | False | warn |

- lifts_enough = `True`
- verdict = `buy_worth_mfe10pct__lifts_enough_for_next_step`
- train_yes_no = `yes_short_sidecar_only`
- 判据：Primary: y=1{mfe10>=0.1} with mfe10=max(high[T+1:T+10]) / close[T] - 1 (path MFE / decision-system worth-it). Pass if Δ≤-0.04 or clearly better than best Phase E/F binary (Δ more negative by 0.02). Phase G close-to-close was the wrong target; reported for comparison only.

### 诊断备注

- Primary positive rate=0.2529 (n_pos=31323/123836); threshold=0.10 on mfe10 = max(high[T+1:T+10]) / close[T] - 1.
- Definition matches max favorable excursion from entry over next 10 sessions (Phase D mfe10 field / build_targets).
- Phase G close-to-close pos_rate=0.1104 vs G2 MFE pos_rate=0.2529 (MFE≥thr ⊇ path-touch; typically higher than close-to-close).
- Phase G comb Δ=-0.015359; G2 comb Δ=-0.034130; Δ(G2−G)=-0.018771 (G2 better).
- Train panel not on disk locally; ablation uses validation panel only (n=123836).
- Costs/slippage not modeled: MFE is path high-touch, not a guaranteed exit fill at the high; round-trip cost would lower economic hit rate (memo note only).
- Primary comb Δ=-0.034130 vs bar -0.04; best Phase E/F binary Δ=-0.012270859304657045.

### 建议下一步

1. **可**按 MFE≥+10%/10日（路径触及）做短 sidecar 标签 + 线性确认。
2. **不要**重启同配置 R2 八头 BCE 长训。
3. 若上线需另加成本/可成交约束；本消融未建模滑点。

## 证据强度

| 主张 | 强度 | 依据 |
| --- | --- | --- |
| MFE 买入值得可开下一步 | 中 | Δ 见表 |
| Phase G close-to-close 非决策目标 | 强 | 用户：路径 MFE≥10% |
| 应维持停 R2 | 强 | 与 Phase C–G 一致 |

## 用户要点（中文）

- **定义**：`mfe10 = max(high[T+1:T+10]) / close[T] - 1`；`y=1{mfe10≥0.1}`（路径触及 +10%，非收盘对收盘）。
- **正类率**：`0.2529`（Phase G close-to-close=`0.11042023321166704`）
- **指标**：Base Δ=`-0.020109348008008898`；Base+xsection Δ=`-0.034130436131804`；Phase G comb Δ=`-0.015359321769126855`；Δ(G2−G)=`-0.018771114362677144`
- **过关**：`是`（门槛 Δ≤-0.04）
- **开训**：`是（仅短 sidecar）`

