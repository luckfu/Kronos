# Kairos 排序目标消融（Phase L）

日期：2026-10-01 22:38 CST（北京时间）。用户对齐诊断：α 在排序不在二分类。

## 一句话

**ranking_target__clear_lift_vs_binary__prefer_rank_path**；train_yes_no=`yes_short_ranking_probe_only`。
最佳臂=`ridge_mfe_cs_comb`，Rank IC mean=`0.27620045257688736`，TopK lift=`0.17164223154982167`。
相对二分类清晰抬升=`True`。

- 建议：截面排序目标（fwd_ret/mfe 连续或分位）在 Base+xsection 上给出相对二分类路径更清晰的 Rank IC / TopK 抬升；下一步可做极短 listwise/pairwise 探针，不要再开 mfe≥10% 二分类长训 / 22 层 sidecar。

## 设定

- n=`123836`，base_dim=`20`，xsec_dim=`6`，seed=`20261001`
- 特征：Base OHLCVA 汇总 + 截面（同 Phase D）
- 目标：连续 `fwd_ret_10` / `mfe10`；截面分位；日内 pairwise logistic
- 门槛：Rank IC≥`0.05` 或 TopK lift≥`0.05`；二分类闸 Δ≤`-0.04`
- **未** 22 层长训；**未** 动 TPU WIP

## 二分类路径对照

| 项 | 值 |
| --- | ---: |
| Phase G2 Logistic Base+xsection Δ | -0.034130436131804 |
| Phase K identity best Δ | -0.002970473307763233 |
| Phase K gate_passed | False |
| 二分类绝对闸失败 | True |

## Ridge 排序（预测 → 日均 Rank IC / TopK / pairwise）

| 臂 | Rank IC mean | TopK hit (lift) | Pairwise acc (lift) |
| --- | ---: | ---: | ---: |
| fwd_ret_10_base | 0.0530 (n=242) | 0.2603 (lift=+0.0603) | 0.5189 (lift=+0.0189) |
| fwd_ret_10_comb | 0.0569 (n=242) | 0.2666 (lift=+0.0666) | 0.5219 (lift=+0.0219) |
| mfe10_base | 0.2590 (n=242) | 0.3675 (lift=+0.1675) | 0.5899 (lift=+0.0899) |
| mfe10_comb | 0.2734 (n=242) | 0.3744 (lift=+0.1744) | 0.5967 (lift=+0.0967) |
| fwd_cs_rank_comb | 0.1011 (n=242) | 0.1855 (lift=-0.0145) | 0.5324 (lift=+0.0324) |
| mfe_cs_rank_comb | 0.2762 (n=242) | 0.3716 (lift=+0.1716) | 0.6003 (lift=+0.1003) |

## Pairwise logistic（特征差 → 分数）

| 臂 | Rank IC mean | TopK hit (lift) | Pairwise acc (lift) |
| --- | ---: | ---: | ---: |
| fwd_ret_10_comb | 0.1047 (n=61) | 0.1897 (lift=-0.0103) | 0.5357 (lift=+0.0357) |
| mfe10_comb | 0.2528 (n=61) | 0.3629 (lift=+0.1629) | 0.5830 (lift=+0.0830) |

## 决策

- verdict = `ranking_target__clear_lift_vs_binary__prefer_rank_path`
- ranking_signal = `True`
- clear_lift_vs_binary = `True`
- best_arm = `ridge_mfe_cs_comb`
- best Rank IC = `0.27620045257688736`
- best TopK lift = `0.17164223154982167`
- train_yes_no = `yes_short_ranking_probe_only`

## 主读口径

- **诚实主臂**：`mfe10_comb` Ridge（连续 MFE，非分位标签）Rank IC≈**0.273**，TopK lift≈**+0.174**；`fwd_ret_10_comb` Rank IC≈**0.057**，TopK lift≈**+0.067**（刚过门槛）。
- 分位标签臂（`*_cs_rank_*`）与连续臂接近，不作单独过闸依据（同日分位在全截面上构造，存在弱标签构造泄漏）。
- **幅度通道提醒（Phase D）**：MFE 排序部分可能来自波动/幅度可预测性，不等于涨跌方向 α；`fwd_ret` IC 弱得多。但仍显著优于二分类闸路径（G2 Δ≈−0.034 未过 −0.04；K Δ≈−0.003）。

### 诊断笔记

- n_samples=123836; seed=20261001; TopK frac=0.2.
- Binary refs: G2 comb Δ=-0.034130436131804; Phase K best Δ=-0.002970473307763233; gate=-0.04.
- Best ranking arm=ridge_mfe_cs_comb; Rank IC mean=0.27620045257688736; TopK lift=0.17164223154982167.
- Train panel not on disk; val panel only (same as Phase D–H).
- No 22-layer train; no TPU WIP.

## 明确不做

1. 不再开 mfe≥10% 二分类 22 层 sidecar / 全量 R2。
2. 不碰同事 TPU WIP。
3. 若过闸，下一步仅极短 ranking probe（listwise/pairwise），非长训。

