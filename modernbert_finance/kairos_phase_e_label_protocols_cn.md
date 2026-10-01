# Kairos 替代标签协议消融（Phase E）

日期：2026-10-01 17:53 CST。

## 一句话结论

**三种替代标签均未相对 Phase D 抬升到可开训门槛。** verdict=`none_lift_enough__do_not_justify_relabel_train_yet`。

- 建议：三种主方向协议均未过关：不要基于这些定义开标签重建长训；要么结束 Kairos 廉价消融路径，要么换质变标签（更长 horizon / 行业中性 fwd+成本），而非再拧 10D MFE/MAE。

## 设定

- n_samples=`123836`，seed=`20261001`，base_dim=`20`，xsec_dim=`6`
- 特征：Phase D 同款 Base OHLCVA 汇总 + 截面（默认看 comb）
- 协议 A：前向收益 fwd_ret_5/10；符号 / 截面顶底 20% 二分类
- 协议 B：MFE/MAE 对 vol20 标准化与 train-fold OLS 残差化
- 协议 C：同日双边触达→neither；5%/8% first-touch；decisive 子样本方向
- **未**重启 R2 / **未** Kaggle 长训；**未**动同事 TPU WIP

## 对照基线（Phase B/D）

| 指标 | 值 |
| --- | ---: |
| Phase B 八头 macro Δ | -0.019569535139329325 |
| Phase D 截面八头 macro Δ | -0.029923786884898024 |
| Phase D 最好方向 Ridge R² | 0.11423838817314091 |
| Phase D FT upside_first comb Δ | -0.008981954753079613 |

门槛（仅主方向头）：R²≥`0.15` 或 logistic Δ≤`-0.04`；或相对 Phase D 方向 R²+`0.04`。CS 分位 / FT-day 不作过关条件。

## 标签分布

- fwd_sign_up_rate=`0.4828`，fwd_ret_10 mean/std=`0.0050`/`0.0977`
- FT clean 5%: up=`0.4294` dn=`0.3878` neither=`0.1828` decisive_n=`101194`
- FT clean 8%: up=`0.3126` dn=`0.2573` neither=`0.4302` decisive_n=`70563`

## 1) 协议 A — 前向收益

| Target | naive MAE | ridge MAE | ridge R² | beats |
| --- | ---: | ---: | ---: | :---: |
| fwd_ret_5_base | 0.047093 | 0.046931 | 0.006497 | Y |
| fwd_ret_5_comb | 0.047093 | 0.046866 | 0.007661 | Y |
| fwd_ret_10_base | 0.066813 | 0.066589 | 0.008385 | Y |
| fwd_ret_10_comb | 0.066813 | 0.066565 | 0.009835 | Y |

| Head | prior LL | model LL | Δ |
| --- | ---: | ---: | ---: |
| fwd_sign_up_base | 0.692555 | 0.690043 | -0.002512 |
| fwd_sign_up_comb | 0.692555 | 0.689248 | -0.003308 |
| fwd_cs_top20_base | 0.501930 | 0.494346 | -0.007584 |
| fwd_cs_top20_comb | 0.501930 | 0.489659 | -0.012271 |
| fwd_cs_bot20_base | 0.499200 | 0.482867 | -0.016333 |
| fwd_cs_bot20_comb | 0.499200 | 0.470456 | -0.028744 |

## 2) 协议 B — 波动残差化 / 标准化 MFE·MAE

### Vol-scaled

| Target | naive MAE | ridge MAE | ridge R² | beats |
| --- | ---: | ---: | ---: | :---: |
| mfe_vol_scaled_base | 0.843123 | 0.839713 | 0.005494 | Y |
| mfe_vol_scaled_comb | 0.843123 | 0.822073 | 0.035175 | Y |
| mae_vol_scaled_base | 0.526872 | 0.522281 | 0.018482 | Y |
| mae_vol_scaled_comb | 0.526872 | 0.510351 | 0.067268 | Y |
| net_vol_scaled_base | 1.186982 | 1.184233 | 0.004143 | Y |
| net_vol_scaled_comb | 1.186982 | 1.181976 | 0.006262 | Y |

### Train-fold residual vs vol20

| Target | naive MAE | ridge MAE | ridge R² | beats |
| --- | ---: | ---: | ---: | :---: |
| mfe_resid_vol_base | 0.060024 | 0.059205 | 0.021537 | Y |
| mfe_resid_vol_comb | 0.060024 | 0.058930 | 0.030873 | Y |
| mae_resid_vol_base | 0.037729 | 0.037253 | 0.020290 | Y |
| mae_resid_vol_comb | 0.037729 | 0.037109 | 0.031653 | Y |
| net_resid_vol_base | 0.085988 | 0.085658 | 0.009216 | Y |
| net_resid_vol_comb | 0.085988 | 0.085615 | 0.010525 | Y |

| Head | prior LL | model LL | Δ |
| --- | ---: | ---: | ---: |
| mfe_vs_gt1_base | 0.664936 | 0.662146 | -0.002791 |
| mfe_vs_gt1_comb | 0.664936 | 0.650667 | -0.014269 |
| net_vs_pos_base | 0.692531 | 0.691031 | -0.001500 |
| net_vs_pos_comb | 0.692531 | 0.690502 | -0.002029 |

## 3) 协议 C — 更干净 first-touch

| Target | naive MAE | ridge MAE | ridge R² | beats |
| --- | ---: | ---: | ---: | :---: |
| ft_signed_8_base | 3.627255 | 3.598729 | 0.002660 | Y |
| ft_signed_8_comb | 3.627255 | 3.582124 | 0.003296 | Y |
| ft_up_day_8_base | 2.770439 | 2.623141 | 0.064911 | Y |
| ft_up_day_8_comb | 2.770439 | 2.548360 | 0.093756 | Y |
| ft_dn_day_8_base | 2.275090 | 2.139114 | 0.071791 | Y |
| ft_dn_day_8_comb | 2.275090 | 2.082146 | 0.100369 | Y |

| Head | prior LL | model LL | Δ |
| --- | ---: | ---: | ---: |
| ft_clean5_up_all_base | 0.683138 | 0.678541 | -0.004597 |
| ft_clean5_up_all_comb | 0.683138 | 0.674156 | -0.008982 |
| ft_clean5_up_decisive_base | 0.691853 | 0.691082 | -0.000771 |
| ft_clean5_up_decisive_comb | 0.691853 | 0.690880 | -0.000972 |
| ft_clean8_up_all_base | 0.621120 | 0.606906 | -0.014214 |
| ft_clean8_up_all_comb | 0.621120 | 0.596223 | -0.024897 |
| ft_clean8_up_decisive_base | 0.688436 | 0.686345 | -0.002091 |
| ft_clean8_up_decisive_comb | 0.688436 | 0.686270 | -0.002166 |

## 4) 与 Phase D 对比 + 决策

| 协议 | 主方向 R² | 主方向 Δ | lifts_enough |
| --- | ---: | ---: | :---: |
| A 前向收益 | 0.009835 | -0.003308 | N |
| B 波动残差 | 0.030873 | -0.002029 | N |
| C 干净 FT | 0.003296 | -0.000972 | N |

- best_protocol = `None`
- winners = `[]`
- verdict = `none_lift_enough__do_not_justify_relabel_train_yet`
- 判据：Primary directional metrics only (fwd_ret_10 / fwd_sign; vol-resid net; decisive FT). Absolute: R²≥0.15 or Δ≤-0.04; relative: +0.04 R² vs Phase D directional. CS-quintile and FT-day magnitude heads are diagnostics, not win conditions.

### 诊断备注

- A CS top/bot20 logistic Δ=-0.0287 is modest but cross-section label+features; not treated as directional lift alone.
- B vol-residualizing collapses Phase D mfe/mae R² — confirms much of Phase D continuous signal was volatility/magnitude, not direction.
- C decisive first-touch direction Δ≈0 — cleaner FT removes the weak all-sample edge; direction still not linearly available.

### 建议下一步（每条一句）

1. **不要**基于这三种定义立刻开标签重建长训。
2. 若继续研究：换更长 horizon / 行业中性 fwd / 成本感知标签，而非微调 10D MFE 阈值。
3. **不要**重启同配置 R2；短神经探针仍非主路径。

## 证据强度

| 主张 | 强度 | 依据 |
| --- | --- | --- |
| A 前向收益可开下一步 | 弱/否 | R²/Δ 见表 |
| B 残差化可开下一步 | 弱/否 | R²/Δ 见表 |
| C 干净 FT 可开下一步 | 弱/否 | R²/Δ 见表 |
| 应维持停 R2 | 强 | 与 Phase C/D 一致 |

