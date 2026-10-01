# Kairos 连续目标 / 截面特征消融（Phase D）

日期：2026-10-01 17:19 CST。

## 一句话结论

**优先改 10D MFE/MAE 标签定义；不把短神经探针当主路径。** 截面把八头 macro Δ 从 Phase B `-0.01957` 抬到 `-0.02992`（相对改善约 53%），但**方向性**连续目标最好 R² 仅 `0.1142`（mfe10），而 R²=`0.2384` 的 `range_excursion`/`ft_*_day` 主要是波动率/幅度可预测性，不是涨跌方向。 仅可做可选的极短 sanity 探针确认深度模型能否吃到线性已有信号。

## 设定

- n_samples=`123836`，base_dim=`20`，xsec_dim=`6`，seed=`20261001`
- 截面列：`cs_rank_ret5, cs_rank_ret20, cs_rank_vol20, ind_rel_ret5, ind_rel_ret20, ind_rel_vol20`（同日横截面分位 + 行业相对）
- 连续目标：mfe10 / mae10 / |mae| / net / range / first-touch day & signed
- 模型：StandardScaler + Ridge(α=1) 或 Logistic；同一 seed 的 75/25 切分
- **未**重启 R2 / **未** Kaggle 长训；**未**动同事 TPU WIP

## 1) 连续回归（Ridge vs 训练均值 naive）

### 解读约定

- **方向性**：`mfe10`、`mae10`、`net_excursion`、`ft_signed` —— 与涨跌/偏斜相关
- **幅度性**：`range_excursion`、`ft_up_day`、`ft_dn_day` —— 易被波动率解释，不可当作方向 α

### Base OHLCVA 汇总

| Target | naive MAE | ridge MAE | ridge R² | MAE lift | beats |
| --- | ---: | ---: | ---: | ---: | :---: |
| mfe10 | 0.062244 | 0.059537 | 0.067952 | +0.002708 | Y |
| mae10 | 0.040526 | 0.038734 | 0.080271 | +0.001792 | Y |
| mae10_abs | 0.040526 | 0.038734 | 0.080271 | +0.001792 | Y |
| net_excursion | 0.085959 | 0.085556 | 0.011349 | +0.000402 | Y |
| range_excursion | 0.062650 | 0.055395 | 0.175646 | +0.007255 | Y |
| ft_up_day | 3.653655 | 3.499734 | 0.053594 | +0.153921 | Y |
| ft_dn_day | 3.470991 | 3.303044 | 0.065405 | +0.167947 | Y |
| ft_signed | 5.919910 | 5.908291 | 0.002170 | +0.011619 | Y |

### Base + 截面

| Target | naive MAE | ridge MAE | ridge R² | MAE lift | beats |
| --- | ---: | ---: | ---: | ---: | :---: |
| mfe10 | 0.062244 | 0.058627 | 0.088492 | +0.003617 | Y |
| mae10 | 0.040526 | 0.037984 | 0.114238 | +0.002542 | Y |
| mae10_abs | 0.040526 | 0.037984 | 0.114238 | +0.002542 | Y |
| net_excursion | 0.085959 | 0.085518 | 0.012915 | +0.000440 | Y |
| range_excursion | 0.062650 | 0.052565 | 0.238361 | +0.010085 | Y |
| ft_up_day | 3.653655 | 3.405281 | 0.084987 | +0.248374 | Y |
| ft_dn_day | 3.470991 | 3.209821 | 0.096672 | +0.261171 | Y |
| ft_signed | 5.919910 | 5.903488 | 0.002119 | +0.016423 | Y |

## 2) First-touch 二分类 logistic

| Head | prior LL | model LL | Δ |
| --- | ---: | ---: | ---: |
| ft_upside_first_base | 0.683138 | 0.678541 | -0.004597 |
| ft_upside_first_comb | 0.683138 | 0.674156 | -0.008982 |
| ft_downside_first_base | 0.670002 | 0.664578 | -0.005425 |
| ft_downside_first_comb | 0.670002 | 0.659871 | -0.010132 |

## 3) 八头 logistic：Base vs 仅截面 vs Base+截面

| 设定 | macro Δ(model−prior) |
| --- | ---: |
| Phase B 独立八头（对照） | -0.019570 |
| Base（复现） | -0.019570 |
| 仅截面 | -0.025882 |
| Base+截面 | -0.029924 |
| Base+截面 − Phase B | -0.010354 |

### 各头 Δ（Base+截面）

| Head | prior | model | Δ | beats |
| --- | ---: | ---: | ---: | :---: |
| up_003 | 0.640109 | 0.623740 | -0.016369 | Y |
| up_005 | 0.693142 | 0.668105 | -0.025037 | Y |
| up_008 | 0.633594 | 0.598627 | -0.034967 | Y |
| up_012 | 0.496355 | 0.464244 | -0.032111 | Y |
| down_003 | 0.648012 | 0.621522 | -0.026490 | Y |
| down_005 | 0.691862 | 0.656612 | -0.035251 | Y |
| down_008 | 0.595542 | 0.562425 | -0.033117 | Y |
| down_012 | 0.372672 | 0.336623 | -0.036049 | Y |

## 4) 决策

- verdict = `xsection_helps_modestly_but_direction_still_weak__prefer_relabel`
- justify_short_neural_probe = `False`
- borderline_sanity_neural_probe_ok = `True`
- recommend_change_10d_mfe_mae_label = `True`
- best **directional** ridge R² = `0.114238`
- best **magnitude** ridge R² = `0.238361`（勿与方向混淆）
- 8-head macro Δ: Phase B `-0.01957` → comb `-0.02992` (lift `-0.01035`)
- 判据：Separate magnitude (range/FT-day, partly vol) from direction (mfe/net/ft_signed). Justify short neural only if directional R²>0.10 with clear logistic lift, or macro Δ<-0.04. Otherwise change 10D MFE/MAE label definition; optional tiny probe only as linear-harvest sanity.

### 建议下一步（每条一句）

1. **改标签定义（主推）**：用前向收益 / 残差化 MFE（去波动）/ 更长或更稀 first-touch，替代嵌套 8 阈值 exceedance。
2. **截面特征可保留**：同日分位 + 行业相对把 Δ 抬了约 0.01，成本低，后续任何线性/神经探针应默认带上。
3. **短神经探针（可选 sanity）**：仅在确认「深度模型能否吃到 Δ≈0.03 线性信号」时做极短训；不作为产品主线。
4. **不要**重启同配置 R2 八头 BCE 长训。

## 证据强度

| 主张 | 强度 | 依据 |
| --- | --- | --- |
| 连续 mfe/mae 有弱–中线性可预测性 | 中 | dir R²≈0.114 |
| range/FT-day 高 R² 主要是波动幅度 | 强 | mag R²≈0.238；ft_signed≈0 |
| 截面抬升八头 Δ | 中 | -0.0196→-0.0299 |
| 值得把短神经当主路径 | 弱/否 | justify=False |
| 应改 10D 标签 | 强 | 方向 R² 弱且 Δ 仍 <0.04 |

