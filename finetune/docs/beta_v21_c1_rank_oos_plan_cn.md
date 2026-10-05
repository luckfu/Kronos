# Beta v2.1 C1 Ranking True Time-OOS（Seg155 / Seg19 / Seg8）

更新：2026-10-05 CST

## 目的

v20 `rank_unfreeze` 已停。不在训练 val 上挑模型，而是用密封 **true time-OOS**
对三本 checkpoint 做同合同对比，再按 OOS 结果选最优。

## True OOS 合同（禁止用 val_data.pkl）

- Dataset：`luckfu/a-share-120d-temporal-symbol-holdout`
- 包名：`kronos_beta_v2_time_oos_through_20260903`
- 文件：`evaluation_manifest.json` / `evaluation_panel.pkl` / `evaluation_samples.jsonl`
- Signal：`2026-08-11` → `2026-09-03`（18 个交易日，约 92,751 样本）
- `purpose=evaluation_only_never_train_or_tune`

训练/选模用的 temporal-symbol val（2025-07-01→2026-07-02，520 只 holdout）**不是** OOS。

## 三本 checkpoint

| 标签 | Dataset | Val 备注（仅对照，非 OOS） |
|---|---|---|
| `seg155_forecast_best` | `luckfu/kronos-beta-v21-c1-seg155-forecast-best` | WFL 地板 2.31236787；包内 segment=26；**`use_beta_v21_auxiliary=False`**（无 return/barrier heads）→ OOS 只报 WFL/forecast，pairwise/rank IC 为 null |
| `seg19_rank_frozen_best` | `luckfu/kronos-beta-v21-c1-rank-frozen-seg19-best` | pairwise 0.65735，WFL 2.31236782 |
| `seg8_rank_unfreeze_best` | `luckfu/kronos-beta-v21-c1-rank-unfreeze-seg8-best` | pairwise 0.65897，WFL 2.32550 |

## 评分合同（与训练一致）

- 模型：`use_beta_v21_auxiliary=True`，return/barrier heads → `expected_utility_score`
- Pairwise：同日、`|utility gap| >= 0.005`，分数平局算错
- Rank IC：日内 Spearman(score, utility)；再报 ICIR、正 IC 日占比
- Forecast：teacher-forcing CE；WFL 使用 C1 权重 `1.364,1.364,1.364,1.136,1.136,0.909,0.909,0.682,0.682,0.455`
- 附加：top/bottom 20% utility / return_10d spread（便宜）

## 代码入口

- 评测核心：`finetune/evaluate_beta_v21_time_oos.py`
- Kaggle runner：`finetune/kaggle_beta_v21_c1_rank_oos.py`
- Builder：`finetune/build_kaggle_beta_v21_c1_rank_oos_kernel.py`
- Staging：`finetune/kaggle_beta_v21_c1_rank_oos_kernel/`
- Kernel slug：`luckfu/kronos-beta-v21-c1-rank-oos`（**不**覆盖训练 notebook）

重建并推送：

```bash
python3 finetune/build_kaggle_beta_v21_c1_rank_oos_kernel.py
kaggle kernels push -p finetune/kaggle_beta_v21_c1_rank_oos_kernel
```

## 日志里会出现的指标

每个 checkpoint：`pairwise_accuracy` / `pairwise_pairs` / `rank_ic` / `rank_icir` /
`rank_ic_positive_rate` / `weighted_forecast_loss` / `forecast_loss` /
`mean_top_bottom_utility_spread` / `mean_top_bottom_return10d_spread`

总表：`comparison.json` + `BEST=<label>`（规则：pairwise ↓ 优先，其次 rank IC，再次更低 WFL）。

## 密封提醒

此 OOS 一旦用于改 LR / 配方 / 提前停训规则，即消耗 sealed status。本 kernel 只报告、不训练。

## Seg155 注意（2026-10-05 修）

首跑 ERROR：`RuntimeError: seg155_forecast_best missing use_beta_v21_auxiliary=True`。
Seg155 是 WC forecast-only 地板，config 明确 `use_beta_v21_auxiliary=False`，权重里也没有 heads。
评测改为：aux 模型走 `expected_utility`；forecast-only 只报 CE/WFL，不参与 pairwise 夺冠。


## 与 Small C2 同包对比（2026-10-05）

密封 18 日包上已补 **Return10d Rank IC**，并与 Small C2 并列表见
`finetune/docs/beta_v21_c1_rank_oos_vs_small_c2_cn.md` 与
`finetune/reports/beta_v21_c1_rank_oos_vs_small_c2_return10d.json`。

要点：Beta Seg19 utility rank IC **0.0300**；同包 return10d rank IC **-0.0073**；
Small C2 D10 日均 **~0.18**。口径不同 + 收益截面上 Beta 更弱。
