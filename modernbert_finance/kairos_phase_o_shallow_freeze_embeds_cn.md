# Kairos 短浅层冻结 embeds 排序探针（Phase O）启动

日期：2026-10-01（北京时间）。  
前置：Phase N pairwise 过闸 Rank IC≈**0.082** ≈ Phase M MSE **0.085**；TopK≈**0.064**。  
Kernel：`user281434/kairos-ranking-probe-short-phase-o` → **PENDING**  
SwanLab：`roc_fu/finance` / `kairos-ranking-probe-short-phase-o-20261001`  

## 设定

| 项 | 值 |
| --- | --- |
| backbone | **`shallow`**（`SHALLOW_LAYERS=2`，**非** 22 层） |
| freeze | **`FREEZE_TOKENIZER_EMBEDS=True`**（冻 s1/s2）；bb 可训 |
| loss | **MSE**→连续 `mfe10`（相对 M identity 对照） |
| 指标 | Rank IC mean / TopK lift（闸门 ≥0.05） |
| 预算 | 3×20 000 / 60min；BS 64；LR `1e-4` |
| 对照 | M identity MSE IC≈0.085；N pairwise≈0.082；L Ridge≈0.273 |

## 目的

检验冻结 tokenizer embeds + 浅非 identity 头能否突破 identity 短预算天花板 ~0.08。  
**不做**：22 层二分类 / 全量 R2；不碰 TPU WIP。

## 产物预期

- Kernel logs + `kairos_ranking_probe/report.json`
- SwanLab Rank IC 曲线
- 本地 launch JSON：`modernbert_finance/ablations/kairos_phase_o_shallow_freeze_embeds_launch.json`
