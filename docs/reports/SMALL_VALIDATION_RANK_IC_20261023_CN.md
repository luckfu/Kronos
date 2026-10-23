# 验证集 Rank IC 选模诊断（16 路径，2026-10-23）

**结论先说**：改成生产口径的 **16 条路径 + P50 中位数**后，排序仍然是 **C2 > C03 > e03b > epoch2**，和 sealed OOS 的大方向一致；而且 C2 对 C03/e03b/epoch2 的差距比 1 条路径时更大。但 **20 个日期下，配对 CI 仍然全部跨 0**，所以它还不能单独作为训练选模指标。下面第 2–6 节是这次已完成的 20 日结果，不是 60 日跑分。

**§5.1 代码状态**：评估器已改为 `N_DATES = 60`（`N_SYMBOLS = 200`，16 条路径 + P50 中位数不变）。Kernel [`wynstonliu/kronos-small-validation-rank-ic-20261023`](https://www.kaggle.com/code/wynstonliu/kronos-small-validation-rank-ic-20261023) 已由另一代理推送，状态 **RUNNING**。60 日数字出来之前不要改写下面的表，也不要改训练，不要再 `kaggle kernels push`。

---

## 1. 为什么改成 16 条

生产（Web UI / Serverless）默认每个窗口 **16 条路径**，按 **P50 中位数**排序；之前的诊断用了 1 条路径 + 均值，和线上不一致。

本次修改：

- `sample_count=16`；
- 每个窗口取 16 条路径 D10 收益的**中位数**；
- batch 12（和线上内存口径一致）。

代码提交：`fc32425`。

---

## 2. 实验设置

| 项目 | 值 |
|---|---|
| 数据 | `val_data.pkl`：520 只验证股票，2025-07-03 → 2026-07-02 |
| 子集 | 20 个 signal dates × 每天 200 只 = **4,000 窗口**，固定 seed 20261023 |
| 模型 | C2、C03、epoch2 best@501、e03b best@457 |
| 生成 | 每窗口 **16 条路径**，取 D10 收益中位数；T=0.65，top_p=0.8，seed 20260906 |
| 指标 | 每个日期内部预测 D10 vs 实际 D10 的 Spearman，20 天平均 |
| 耗时 | 4 个 checkpoint 约 **30 分钟**（双 T4 并行） |

---

## 3. 结果：16 条 vs 1 条

| 模型 | 16 条平均 IC | 16 条中位数 | 正 IC 天数 | 1 条平均 IC |
|---|---:|---:|---:|---:|
| C2 | **0.2535** | 0.2372 | 20/20 | 0.2204 |
| C03 | 0.2434 | 0.2300 | 20/20 | 0.2184 |
| e03b | 0.2112 | 0.2137 | 17/20 | 0.2174 |
| epoch2 | 0.2028 | 0.2362 | 16/20 | 0.1914 |

排序（按平均 IC）：**C2 > C03 > e03b > epoch2**。

- 16 条把 C2/C03 抬高、把 e03b/epoch2 压低，**差距变大了**；
- 但按**逐日中位数**，epoch2 的 0.2362 反而高于 e03b 的 0.2137，顺序变成 C2 > epoch2 > C03 > e03b——说明“最后一名”在均值/中位数下并不稳定；
- C2 仍然是唯一在两种口径下都排第一的。

---

## 4. 配对差值（16 条，同一天相减）

| 对比 | 平均差 | 95% CI | 前者胜天数 |
|---|---:|---|---:|
| C2 − C03 | +0.0101 | [-0.0113, +0.0313] | 13/20 |
| C2 − e03b | +0.0422 | [-0.0069, +0.0963] | 12/20 |
| C2 − epoch2 | +0.0506 | [-0.0101, +0.1089] | 13/20 |
| C03 − e03b | +0.0322 | [-0.0006, +0.0625] | 12/20 |
| C03 − epoch2 | +0.0406 | [-0.0011, +0.0822] | 14/20 |
| e03b − epoch2 | +0.0084 | [-0.0068, +0.0193] | 12/20 |

**所有 CI 都跨 0。** C03−e03b 最接近显著（下界 -0.0006），但还不够。

和 1 条路径相比：

- C2−C03：+0.0020 → **+0.0101**
- C2−e03b：+0.0030 → **+0.0422**
- C2−epoch2：+0.0290 → **+0.0506**

方向更清楚，但 20 个日期仍然不够。

---

## 5. 和 OOS 的对照

| 对比 | OOS 结论 | 16 条验证集 |
|---|---|---|
| C2 vs C03 | C2 更好（34 日 0.1424 vs 0.1200） | C2 更高（0.2535 vs 0.2434） |
| C2 vs e03b | C2 更好（sealed 0.0652 vs 0.0525） | C2 更高（0.2535 vs 0.2112） |
| C2 vs epoch2 | C2 更好（34 日 0.1424 vs 0.0843） | C2 更高（0.2535 vs 0.2028） |

**排序方向三次都对上了**，这是 16 条路径实验最有价值的地方。

---

## 6. 判断

- 16 条路径 + 中位数是**正确的生产口径**，比 1 条路径更可信；
- 它能把 C2 排第一、把 e03b/epoch2 压在后面，和 OOS 一致；
- 但 **20 个日期下，C2 与 C03/e03b/epoch2 的差距仍不显著**（CI 全跨 0），不能单独作为训练选模指标；
- 更适合当**筛除指标**：验证 IC 明显低的 checkpoint 可以否决；但不宜用它精确选 C2 vs C03。

---

## 7. 下一步

1. **日期从 20 加到 60 的代码已就绪**（每天 200 只，1.2 万窗口；16 条路径中位数不变）。Kernel `wynstonliu/kronos-small-validation-rank-ic-20261023` 已另行推送，状态 **RUNNING**；预计约 1.5 小时。结果未出，不要重复 push。
2. 如果 60 天后 C2 对 C03/e03b 的 CI 仍跨 0，就**放弃验证集 Rank IC 选模**；
3. 同时考虑**时间切分验证**（例如 2026-04 → 2026-07 作为验证），比 symbol holdout 更接近 OOS；
4. 在指标通过之前，不改训练循环，也不替换 C2。

---

## 8. 产物

- 16 条报告：[`finetune/reports/small_validation_rank_ic_20261023_16path.json`](../finetune/reports/small_validation_rank_ic_20261023_16path.json)
- 1 条对照：[`finetune/reports/small_validation_rank_ic_20261023.json`](../finetune/reports/small_validation_rank_ic_20261023.json)
- 评估器：[`finetune/small_validation_rank_ic/evaluate_validation_rank_ic.py`](../finetune/small_validation_rank_ic/evaluate_validation_rank_ic.py)
- Kernel：[`wynstonliu/kronos-small-validation-rank-ic-20261023`](https://www.kaggle.com/code/wynstonliu/kronos-small-validation-rank-ic-20261023)

本地重建（`build/` 不入库）。60 日 kernel 已在跑，**不要再 push**：

```bash
python3 finetune/small_validation_rank_ic/build_kernel.py
# kaggle kernels push 已由另一代理完成：
# wynstonliu/kronos-small-validation-rank-ic-20261023 状态 RUNNING
```
