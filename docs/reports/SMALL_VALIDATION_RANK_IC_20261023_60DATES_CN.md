# 验证集 Rank IC 选模诊断（60 日 × 16 路径，2026-10-23）

**结论先说**：把日期从 20 加到 **60**（每天 200 只、16 条路径 + P50 中位数不变）后，排序仍是 **C2 > C03 > epoch2 ≈ e03b**。交接单 §5.1 的验收问的是 **C2−C03、C2−e03b 的配对 CI 是否仍然跨 0**——答案是 **不再跨 0**（block=5 移动块 bootstrap，与 20 日报告同口径）。因此 **不能按「仍跨 0 → 放弃验证集 Rank IC 选模」那条关线**；可以考虑把 `validation_rank_ic` 以小规模 pilot 接进训练选模，但仍 **不要开新训练、不要替换 C2**，先写完 §5.2 再决定。

20 日对照见 [`SMALL_VALIDATION_RANK_IC_20261023_CN.md`](SMALL_VALIDATION_RANK_IC_20261023_CN.md)。

---

## 1. 实验设置

| 项目 | 值 |
|---|---|
| 数据 | `val_data.pkl`：520 只验证股票，2025-07-03 → 2026-07-02 |
| 子集 | **60** 个 signal dates × 每天 200 只 = **12,000** 窗口，固定 seed 20261023 |
| 模型 | C2、C03、epoch2 best@501、e03b best@457 |
| 生成 | 每窗口 **16 条路径**，取 D10 收益中位数；T=0.65，top_p=0.8，seed 20260906 |
| 指标 | 每个日期内部预测 D10 vs 实际 D10 的 Spearman，60 天平均 |
| 配对 CI | 同一天相减；**moving block bootstrap，block=5，2000 次，seed 20261023**（与 20 日报告 / 交接单 §2.2 一致） |
| Kernel | [`wynstonliu/kronos-small-validation-rank-ic-20261023`](https://www.kaggle.com/code/wynstonliu/kronos-small-validation-rank-ic-20261023) **COMPLETE** |
| 耗时 | 约 **71 分钟**（双 T4；wall ~4270s） |

代码：`N_DATES=60` 由 [#25](https://github.com/luckfu/Kronos/pull/25) / `4e23a4c` 合入。

---

## 2. 结果：60 日 vs 20 日（均为 16 条路径）

| 模型 | 60 日平均 IC | 60 日中位数 | 正 IC 天数 | 20 日平均 IC |
|---|---:|---:|---:|---:|
| C2 | **0.2423** | 0.2364 | 59/60 | 0.2535 |
| C03 | 0.2263 | 0.1958 | 60/60 | 0.2434 |
| epoch2 | 0.1891 | 0.1658 | 55/60 | 0.2028 |
| e03b | 0.1869 | 0.1747 | 55/60 | 0.2112 |

排序（按平均 IC）：**C2 > C03 > epoch2 ≈ e03b**。

- 60 日均值整体略低于 20 日子集（样本更密、含更多弱日），但 **相对排序不变**；
- C2 与 C03 的平均差距从 +0.0101 扩到 **+0.0159**；
- e03b / epoch2 仍明显落后；二者几乎打平（e03b 略低）。

---

## 3. 配对差值（60 日，同一天相减，block=5）

| 对比 | 平均差 | 95% CI | 前者胜天数 | CI 跨 0？ |
|---|---:|---|---:|---|
| C2 − C03 | +0.0159 | [+0.0037, +0.0329] | 36/60 | **否** |
| C2 − e03b | +0.0554 | [+0.0212, +0.0945] | 40/60 | **否** |
| C2 − epoch2 | +0.0532 | [+0.0231, +0.0831] | 44/60 | **否** |
| C03 − e03b | +0.0394 | [+0.0123, +0.0676] | 43/60 | **否** |
| C03 − epoch2 | +0.0373 | [+0.0149, +0.0553] | 46/60 | **否** |
| e03b − epoch2 | −0.0022 | [−0.0152, +0.0063] | 29/60 | 是 |

**交接门禁（§5.1）**：C2−C03 与 C2−e03b 的 CI **均不再跨 0**。

敏感性（block=10，同 seed/reps）：C2−C03 [+0.0005, +0.0369]、C2−e03b [+0.0125, +0.1097]，仍不跨 0。

和 20 日对比：

| 对比 | 20 日平均差 / CI | 60 日平均差 / CI |
|---|---|---|
| C2 − C03 | +0.0101 / [−0.0113, +0.0313]（跨 0） | **+0.0159 / [+0.0037, +0.0329]（不跨）** |
| C2 − e03b | +0.0422 / [−0.0069, +0.0963]（跨 0） | **+0.0554 / [+0.0212, +0.0945]（不跨）** |

---

## 4. 和 OOS 的对照

| 对比 | OOS 结论 | 60 日验证集（16 条） |
|---|---|---|
| C2 vs C03 | C2 更好（34 日 0.1424 vs 0.1200） | C2 更高且配对 CI 不跨 0 |
| C2 vs e03b | C2 更好（sealed 0.0652 vs 0.0525） | C2 更高且配对 CI 不跨 0 |
| C2 vs epoch2 | C2 更好（34 日 0.1424 vs 0.0843） | C2 更高且配对 CI 不跨 0 |

排序方向与 OOS / 20 日 16 条一致；60 日后 **统计显著性也跟上了**（在验证集 symbol-holdout 口径下）。

---

## 5. 判断（对照交接单 §5.1）

- 16 条路径 + 中位数仍是正确的生产口径；
- **60 日后，C2 对 C03 / e03b 的配对 CI 不再跨 0** → 交接单「若仍跨 0 则放弃验证集 Rank IC 选模」**不触发**；
- 可考虑把 `validation_rank_ic` **以小规模 pilot 接进训练选模**（先别改主训练循环、别替换线上 C2）；
- 验证集是 symbol holdout，不是时间切分；与 sealed / 未来 OOS 仍可能错位——pilot 时务必保留 OOS / sealed 纪律，**不要用 sealed 8 天调阈值**（已消费）；
- e03b vs epoch2 仍分不开（CI 跨 0），二者作为「已关线」的结论不变。

---

## 6. 下一步

1. **§5.2**：C2 的 8b 执行约束回测（10/09–10/23），不碰训练、不碰 sealed；
2. 若要做 validation Rank IC 选模 pilot：单独开小任务，写清门槛与回滚，**先不改训练**；
3. 仍可考虑时间切分验证（例如 2026-04 → 2026-07）作为补充，比 symbol holdout 更接近 OOS；
4. 在 pilot 设计完成前，不改训练循环，也不替换 C2。

---

## 7. 产物

- 60 日报告 JSON：[`finetune/reports/small_validation_rank_ic_20261023_60dates_16path.json`](../../finetune/reports/small_validation_rank_ic_20261023_60dates_16path.json)
- 20 日对照：[`finetune/reports/small_validation_rank_ic_20261023_16path.json`](../../finetune/reports/small_validation_rank_ic_20261023_16path.json)
- 评估器：[`finetune/small_validation_rank_ic/evaluate_validation_rank_ic.py`](../../finetune/small_validation_rank_ic/evaluate_validation_rank_ic.py)
- Kernel：[`wynstonliu/kronos-small-validation-rank-ic-20261023`](https://www.kaggle.com/code/wynstonliu/kronos-small-validation-rank-ic-20261023)（COMPLETE；本地 `artifacts/` 不入库）

不要再 `kaggle kernels push` 这个 slug。
