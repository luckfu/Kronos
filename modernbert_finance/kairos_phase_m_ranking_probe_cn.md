# Kairos 短排序探针（Phase M）

日期：2026-10-01 22:45 CST（北京时间）。Phase L 过闸 → 用户继续 → 极短 ranking probe。

## 一句话

**轻量 identity + rank head**，预测连续路径 `mfe10`；主指标 **Rank IC / TopK**；短 Kaggle 预算（3×20k / 60min）；**不是** mfe≥10% 二分类 / 22 层 R2；**不**碰 TPU WIP。

## 设定

| 项 | 值 |
| --- | --- |
| 目标 | 连续 `mfe10 = max(high[T+1:T+10])/close[T]-1` |
| 可选训练标签 | `mfe_cs_rank`（日内分位） |
| 损失 | MSE |
| Backbone | `identity`（mean-pool embeds，无 22 层） |
| 预算 | 3 segments × 20 000；GPU 60min |
| 闸门 | Rank IC mean ≥ 0.05 **或** TopK lift ≥ 0.05 |
| Phase L 对照 | ridge_mfe_comb Rank IC ≈ 0.273 |

## Kernel

| 项 | 值 |
| --- | --- |
| slug | `user281434/kairos-ranking-probe-short-phase-m` |
| SwanLab | `kairos-ranking-probe-short-phase-m-20261001` |
| 训练脚本 | `finetune/kaggle_kairos_ranking_probe/train_ranking_probe.py` |
| 助手 | `modernbert_finance/ranking_probe.py` |

## 明确不做

1. 不开 mfe≥10% 二分类 22 层 sidecar / 全量 R2。
2. 不碰同事 TPU WIP。
3. 本轮仅短探针，非长训。
