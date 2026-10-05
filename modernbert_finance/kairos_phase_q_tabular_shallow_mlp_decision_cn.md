# Kairos 决策表格浅 MLP（Phase Q）结果

日期：2026-10-02（北京时间）。
设定：**Base+xsection 表格特征 → 浅 MLP / Logistic / 校准**；**无** tokenizer 序列。
状态：**COMPLETE**（本地 CPU）

## 一句话

**过闸。** best=`mlp_3x_comb` Δ=`-0.052449`；gate_passed=`True`。对照：Phase P≈0；K≈−0.003；G2 Logistic≈−0.034；闸门≤−0.04。

## 目的

Phase P（冻 embeds + 浅序列头）失败后换角度：不依赖 tokenizer 序列，
直接用已验证有效的 Base+xsection 表格特征训浅 MLP，看能否逼近/超过 Logistic 并冲闸。

## 数据与切分

| 项 | 值 |
| --- | --- |
| n_total / train / test | 123836 / 92877 / 30959 |
| 正类率（全） | 0.2529 |
| base / comb 维 | 20 / 26 |
| 切分 | 75/25 stratify seed=20261001 |
| 墙钟 | 74.6s |

## 主结果

| 模型 | Δ vs prior | gate |
| --- | ---: | --- |
| mlp_3x_comb | -0.052449 | True |
| mlp_2x_comb | -0.040488 | True |
| mlp_1x_comb | -0.038665 | False |
| mlp_2x_base | -0.038166 | False |
| mlp_3x_base | -0.035674 | False |
| logistic_comb | -0.034130 | False |
| logistic_calibrated_comb | -0.034061 | False |
| mlp_1x_base | -0.029562 | False |

## 对照

| 跑次 | best Δ |
| --- | ---: |
| Phase P 浅冻 embeds | 1.38038e-07 |
| Phase K identity | -0.002970 |
| 线性 BCE 冻结特征 | -0.031446 |
| G2 Logistic comb | -0.034130 |
| **Phase Q best** | **-0.052449** |
| 闸门 | ≤ -0.04 |

## 判读

- best_model=`mlp_3x_comb`；clears_logistic_ref=`True`。
- next=`consider_mild_feature_expand_or_calibrated_deploy`。
- **不做**：tokenizer 序列短烟；Ranking 产品；22 层；TPU WIP。



## 重要修正（同日 Phase R 时序切分）

Phase Q 的 **随机 75/25** 过闸（mlp_3x Δ≈−0.052）在 **时序切分**（2025H2→2026H1）上 **不成立**：
无强正则时 mlp_3x 时序 Δ 可到 **正值（过拟合）**；时序上真正接近闸门的是 **Logistic / 强正则浅 MLP**（见 Phase R/S）。
**产品判读以时序切分为准**；随机切分仅作与 G2 Logistic 对照。

## 产物

- JSON：`modernbert_finance/ablations/kairos_phase_q_tabular_shallow_mlp_decision_results.json`
- 本备忘：`modernbert_finance/kairos_phase_q_tabular_shallow_mlp_decision_cn.md`

