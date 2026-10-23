# C2 第 8b 步：执行约束回测

代码在 [`finetune/small_clean_oos/execution_constrained.py`](../finetune/small_clean_oos/execution_constrained.py)，驱动在 [`finetune/small_clean_oos/run_c2_execution_backtest.py`](../finetune/small_clean_oos/run_c2_execution_backtest.py)。规则按 [`NEXT_STEP_SEALED_WINDOW_PLAN_20261022_CN.md`](NEXT_STEP_SEALED_WINDOW_PLAN_20261022_CN.md) §4.3 和 [`HANDOVER_20261023_NEXT_MACHINE_CN.md`](HANDOVER_20261023_NEXT_MACHINE_CN.md) §5.2 写死，没有扫描参数。

§3.1 落在 0.05–0.10，所以只做 `|predicted_return_d10| ≥ 5%`。不做全样本 TopK，不接 e03b / epoch2 / C03。

## 冻结规则

- 信号用收盘。下一交易日开盘成交。
- 一字涨停不能买（多头开仓、空头回补都取消或顺延）。一字跌停不能卖（空头开仓取消，多头平仓顺延到下一根能卖的开盘）。
- 成交日停牌顺延到复牌；信号日当天已经停牌的名字直接丢掉。
- 持有到信号日之后第 10 个市场交易日的收盘（和 D10 标签同一天，名字每个交易日都有成交时）。平仓日若卖不掉，改到下一根可成交开盘。
- 每个信号日一个袖套，资金固定 `1/10`，不因为当天袖套少就把仓位放大到满仓。
- 成本 **0.3% 双边一轮**（一笔来回扣一次 0.003，不是每边 0.003）。
- 滑点按信号日已知的 20 日成交额中位数分档，两边用同一档，不随出场日的成交额变：

| 20 日中位成交额 | 单边滑点 |
|---|---:|
| ≥ 5 亿 | 5 bp |
| ≥ 2 亿 | 10 bp |
| ≥ 1 亿 | 15 bp |
| ≥ 5 千万 | 20 bp |
| < 5 千万 | 不交易 |

5 千万这档沿用 [`backtest_august_fixed_top5.py`](../finetune/backtest_august_fixed_top5.py) 里已有的成交额地板，改成信号日及之前 20 根已成交 bar 的中位数。历史不足 20 根也不交易。分档没有在任何 OOS 窗口上拟合。

事前过滤：

- `panel.halt_gap_mask`（标签日期相对截面中位数滞后超过 3 个自然日）。这是 panel 里写明的事前子集，用的是 shard 上的 `label_date_d*`，不用已实现收益。
- 信号日 bar 必须存在且未停牌。
- 流动性只用信号日及之前的成交额。后一日放量不会把名字放进来。

涨跌停价按板块：主板 10%，创业板 / 科创板 20%，北交所 30%，ST 5%。一字的判定是开高低收在一个最小价位内，并且收在涨跌停价上。行情文件里如果已经有 `halted`、`one_word_limit_up`、`one_word_limit_down`，以这三列为准，不再用价格重算。

收益从 bar 的开收盘标，不用预测 shard 里的 `actual_return_*`。未成交的权重留成现金，不把没买到的额度分给同日其他名字。

## 现在没有 §5.2 的数字

目标信号窗是 **2026-10-09 → 2026-10-23**。最后一天的计划平仓是该日之后第 10 个交易日，工作日日历上为 **2026-11-06**。今天这条线还跑不出这个结果：

- 本仓库没有这份 C2 shard，也没有覆盖到 2026-11-06 的行情。`data/`、`artifacts/` 里没有对应面板。
- 已有 Kaggle 面板不能当这个结果：
  - `smmt315/a-share-120d-temporal-symbol-holdout`（34 日，信号到 2026-09-17）
  - `wynstonliu/a-share-120d-oos-20260803-20260930`（密封 42 日，含已消费的 2026-09-18 → 09-30）
- 2026-09-18 → 09-30 已消费，不能拿来选型或调这套规则。34 日数字同样不能写成 §5.2。
- 驱动会在 JSON 里写 `publishable_as_section_5_2`。只有信号日落在 2026-10-09..10-23、bar 覆盖计划平仓且不早于 2026-11-06、涨跌停旗标来自显式列或 `--unadjusted-bars`、并且 `label_date_d1..d10` 齐全时，这个字段才为 true。其它运行的净值、夏普都不要引用。

C2 权重在 `wynstonliu/kronos-small-c2-best-checkpoint`。本步不训练、不推 kernel。

## 怎么跑

预测文件是 `c2_best_YYYY-MM-DD.csv.gz` 目录，或一张含 `symbol, asof_date, predicted_return_d10, label_date_d1..d10` 的 CSV。行情是 `symbol, date, open, high, low, close, amount`，外加停牌 / 一字列。成交用的价格可以是前复权（和标签同一口径）；一字和停牌要用**未复权**打印，因为除权日上前复权高开低收对不齐涨跌停价。

```bash
python -m finetune.small_clean_oos.run_c2_execution_backtest \
  --predictions /path/to/c2_shards \
  --bars /path/to/front_adjusted.csv \
  --unadjusted-bars /path/to/raw.csv \
  --output finetune/reports/c2_execution_constrained_8b.json
```

未复权文件只提供 `halted` / `one_word_limit_up` / `one_word_limit_down`。BaoStock 前复权是 `adjustflag=2`（现有 [`download_a_share_baostock.py`](../finetune/download_a_share_baostock.py)），未复权用 `adjustflag=3` 另下一份，不要改下载脚本的训练口径。

缺文件时驱动写一份没有净值的 blocker，退出码 0。接口冒烟（2024 年的假面板，会故意挡掉一笔一字涨停）：

```bash
python -m finetune.small_clean_oos.run_c2_execution_backtest \
  --smoke \
  --output /tmp/c2_execution_smoke.json
```

冒烟输出路径的文件名里必须有 `smoke`。单测：

```bash
python -m pytest tests/test_c2_execution_constrained_backtest.py -q
```

输出 JSON 先看 `publishable_as_section_5_2` 和 `publication_blockers`。为 false 时，同文件里的收益只说明加载和约束走通了。
