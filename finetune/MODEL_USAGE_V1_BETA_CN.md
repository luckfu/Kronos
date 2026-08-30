# A-share Full-Market Beta v1.2 模型使用说明

当前 Beta 版本为 `Beta v1.2`，包含两个模型产出物：
`models/a_share_v1_beta/releases/beta_v1.2/best_model`（Best@871）和
`models/a_share_v1_beta/releases/beta_v1.2/last_model`（Last@1056）。配套 tokenizer 入口为
`models/a_share_v1_beta/releases/beta_v1.2/tokenizer`。本文档只适用于该全市场模型系列。它与当前 V6 生产模型的
离散市值桶条件不同：v1-beta 使用行业 embedding 和连续 `size_percentile`，
不使用离散市值桶。

Beta v1.2 当前仅作为研究候选：它尚未在全新严格未来时间段完成评估，不能继承 Beta v1.1
在旧 6 个信号日上的多头或空头成绩。发布依据和限制见
[`BETA_V1_2_RELEASE_CN.md`](BETA_V1_2_RELEASE_CN.md)。日志、评估和预测结果必须同时记录
`Best@871` 或 `Last@1056`，避免两个产出物混淆。

## 推理输入

每次预测需要提供：

- 目标股票最近 120 个交易日的前复权 OHLCVA 行情；
- 信号日对应的行业 ID；
- 信号日对应的连续市值百分位 `size_percentile`。

模型输出未来 10 个交易日的自回归预测路径。

## 市值百分位如何取得

`size_percentile` 不是目标股票独立计算的指标，而是目标股票在同一交易日
全市场股票中的横截面位置。计算口径必须与训练一致：

```text
market_cap_proxy = amount / (turnover_pct / 100)
```

其中 `amount` 和 `turn` 来自同一交易日的 BaoStock 日线数据。对当日全市场
股票的 `market_cap_proxy` 排序，再用 `rank(method="first", pct=True)` 得到
目标股票的 `[0, 1]` 百分位。

因此，预测端原则上需要“目标股票当日市值代理值 + 同日全市场市值排序参考”。
只获取目标股票、再用它自己的历史分布计算百分位是不正确的。

## 线上每日流程

生产环境不应在每次单股请求时下载完整历史市场。推荐在每个交易日收盘后执行
一次串行的全市场更新任务：

1. 获取当日股票池内全部股票的 `amount`、`turn`，按训练口径计算市值代理；
2. 生成该交易日的排序参考文件，至少包含股票代码和 `market_cap_proxy`；
3. 请求到达时刷新目标股票的最新 120 日行情；
4. 使用不晚于信号日的最近一份全市场参考文件计算 `size_percentile`；
5. 将行业 ID 和百分位同行情窗口一起送入 v1-beta 模型。

当日全市场参考不可用时，必须明确标记数据过期或拒绝预测，不能静默使用
一个未知日期的旧参考。参考文件应记录 `reference_date`、股票数量、数据源和
生成时间，便于审计。

## 回测与时间一致性

历史回测必须按每个信号日加载对应的历史横截面参考，或加载不晚于该信号日的
最近参考；禁止使用 2026-07-31 或其他未来横截面为历史样本计算百分位。
训练样本同样只使用观察窗口最后一个交易日的百分位，不读取预测窗口中的未来
市值信息。

## 部署检查清单

- 确认模型配置启用 `use_size_percentile=True`；
- 确认 `num_size_buckets=0`，不传入离散桶条件；
- 确认行情、`amount`、`turn` 和行业标签使用同一个信号日；
- 确认百分位来自全市场横截面，而不是个股历史百分位；
- 确认参考日期不晚于信号日；
- 记录 `reference_date` 和 `size_percentile` 到预测结果。

训练数据的完整定义见 [`data/a_share_full_market_v1_beta/DATA_DESIGN_CN.md`](../data/a_share_full_market_v1_beta/DATA_DESIGN_CN.md)，
训练与接力规则见 [`FULL_MARKET_V1_BETA_PLAN_CN.md`](FULL_MARKET_V1_BETA_PLAN_CN.md)。
