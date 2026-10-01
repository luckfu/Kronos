# Kairos 决策冲闸时序表格（Phase S）结果

日期：2026-10-02（北京时间）。
设定：时序 2025H2→2026H1；C 网格 / elastic-net / Logistic⊕mlp1 融合 / 多项式扩维。
状态：**COMPLETE**（本地 CPU）

## 一句话

**过闸。** best=`blend_lr0.5_mlp0.5` Δ=`-0.041793`；gate_passed=`True`。
elastic-net（C=0.01,l1≈0.5–0.7）与多个 LR⊕MLP 融合均 Δ≤−0.04。

## 主结果（时序）

| 模型 | Δ vs prior | gate |
| --- | ---: | --- |
| blend_lr0.5_mlp0.5 | -0.041793 | True |
| blend_lr0.7_mlp0.3 | -0.041495 | True |
| blend_lr0.8_mlp0.2 | -0.041103 | True |
| blend_lr0.9_mlp0.1 | -0.040545 | True |
| enet_C0.01_l10.7 | -0.040419 | True |
| enet_C0.01_l10.5 | -0.040304 | True |
| log_C0.003 | -0.039856 | False |
| enet_C0.02_l10.5 | -0.039856 | False |

## 对照

| 跑次 | best Δ |
| --- | ---: |
| Phase P 浅冻 embeds（序列） | ≈0 |
| Phase K identity | −0.003 |
| Phase R 时序 logistic C=0.01 | −0.0398 |
| **Phase S best（时序）** | **−0.0418** |
| 闸门 | ≤ −0.04 |

## 判读

- 在 **时序切分** 上首次稳定过闸：融合与 elastic-net 均可。
- 下一步 Phase T：Kaggle 上用官方 holdout（train→val）复验 winning recipe（enet / blend），短烟测。
- **不做**：tokenizer 序列重启；Ranking 产品；22 层；TPU WIP。

## 产物

- JSON：`modernbert_finance/ablations/kairos_phase_s_close_gate_temporal_results.json`
- 本备忘：`modernbert_finance/kairos_phase_s_close_gate_temporal_cn.md`

