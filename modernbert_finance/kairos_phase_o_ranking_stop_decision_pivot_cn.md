# Phase O 排序探针已结束 → 决策系统枢轴

日期：2026-10-01（北京时间）。

## 状态

| 项 | 值 |
| --- | --- |
| Kernel | `user281434/kairos-ranking-probe-short-phase-o` |
| 查询时状态 | **COMPLETE**（非 RUNNING） |
| 取消 | **不适用**（已自然结束；无需/无法 cancel） |
| URL | https://www.kaggle.com/code/user281434/kairos-ranking-probe-short-phase-o |

## 产品判读（硬）

- **排序探针 = 诊断死胡同，不是产品路径。**
- Phase L Ridge Rank IC≈0.27 ≫ neural M/N≈0.08；排序损失/浅头未改变「表格特征远强于短预算 identity/浅 neural」的结论。
- 产品目标回到 **决策标签** `y=1{mfe10≥0.10}`（路径触及 MFE），闸门 = `Δ logloss vs 常数先验 ≤ −0.04`。
- **停止** 以 Rank IC / TopK 作为产品主指标的新实验；后续仅决策头 / 校准 / 帮助 Δ 的特征。

## 不做

- 不再开 ranking Phase（listwise / 更深 rank 头 / 更长 ranking IC 训）。
- 不碰同事 TPU WIP；不开全量 R2 / 22 层二分类长训。
