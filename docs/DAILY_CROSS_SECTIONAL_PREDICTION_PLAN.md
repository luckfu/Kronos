# Kronos 每日横截面预测系统开发方案

## 1. 目标与产品定位

将当前以“输入一只股票、即时生成一条预测”为中心的 WebUI，重构为以“每日固定股票池、统一批量推理、横截面排序、结果固化与持续验证”为中心的生产系统。

模型的主要可用能力是未来 10 个交易日收益的横截面排序，而不是单只股票涨跌方向判断。当前 C2 Best 在 13 个 OOS 信号日、66,986 个样本上的 D10 每日平均 Rank IC 约为 0.174，13/13 日为正；同期方向准确率约为 48%。因此第一版产品的主决策指标固定为 D10 横截面排名，D1-D9 作为预测路径和辅助信息。

系统不负责自动下单。它提供可审计的每日候选池、数据质量与风险标记，并积累新的、未参与模型选择的线上 OOS。

## 2. 当前基础与缺口

### 2.1 已有能力

- Supabase `ashare_daily_k` 已保存日频 OHLC、volume、amount、trade_status、is_st。
- 数据覆盖 2025-12-22 至 2026-09-17，现有长度足够提供模型所需的 120 个交易日上下文。
- 最新交易日约 4,559 只股票，OHLC、volume、amount 无缺失。
- Oracle 每晚 20:30 运行行情同步，K 线步骤通常在 20:45 前完成。
- Supabase `stock_factors` 有流通市值，可构造当日横截面 `size_percentile`。
- Modal 已部署 `luckfu/Kronos-small-0.1-Cosine-C2-Best`，检查点 Segment 179。
- 仓库已有模型固定行业词表：86 个行业 ID，unknown ID 为 86。
- 仓库已有 BaoStock 股票行业快照，覆盖 5,544 个证券，其中 334 个为 unknown。

### 2.2 必须补齐的缺口

1. **行业主数据缺口**：Supabase 没有规范化、可追溯的股票行业表；当前 JSON 只是 2026-08-24 静态快照。
2. **行业覆盖缺口**：334 个证券为 unknown，需区分退市证券、数据源缺失和新上市证券。
3. **市值条件输入已修复、仍需持续监控**：2026-09-17 的 `ashare_daily_k.turnover` 已完成回填，4,559/4,559 行均为正；以后每次正式预测前仍需把覆盖率、正值率和市值代理分布作为发布门槛。
4. **复权输入已补齐**：当前行情主要来自 Tushare daily 未复权数据，现已由 `public.ashare_adj_factor` 完整覆盖最近120个交易日。按 `raw * factor / signal_date_factor` 转换后，与训练使用的 BaoStock前复权 OHLC 近乎逐点一致；每日仍需执行覆盖率和来源一致性门槛。
5. **股票池差异**：生产行情约 4,559 只，OOS 每日约 5,150 只。当前采集范围主要是沪市 60、深市 00/30，不应对外称为“全 A 股”。
6. **批量推理缺口**：现有接口支持批量请求，但没有每日任务编排、幂等恢复、分片状态和数据库落库。
7. **结果与验证缺口**：没有每日不可变预测快照，也没有 D1-D10 实际收益回填和线上 Rank IC 统计。

## 3. 不可变的模型输入合同

每日正式推理必须使用与 OOS 相同的合同：

- `lookback = 120` 个有效交易日；
- 特征顺序固定为 `open, high, low, close, volume, amount`；
- `pred_len = 10`；
- 行业 ID 使用模型发布时冻结的 86 类词表；
- 未识别行业在模型合同中仍保留 unknown ID 86 以兼容模型，但生产正式横截面禁止使用 unknown；行业缺失或无法映射的证券必须排除；
- 市值条件为当日股票池内的连续 `size_percentile`，范围 `[0, 1]`；
- 预测主分数为 `predicted_return_d10`；
- 正式批次固定随机种子、采样参数、模型 revision 和 tokenizer revision；
- 同一个 `model_release + signal_date + universe_version + symbol` 只能有一条正式结果。

任何输入合同变化都必须创建新的 `model_release` 或 `pipeline_version`，不能覆盖历史结果。

## 4. 行业编码方案

### 4.1 为什么不能直接保存任意行业名称

模型的行业 embedding 已按训练时的排序固定。即使上游仍返回 86 个名称，只要重新按字母排序或加入一个新类别，所有 ID 都可能错位。因此：

- `sector_vocabulary.json` 是模型合同，而不是可自动更新的业务配置；
- 业务行业名称必须先映射到固定 `vocabulary_id`；
- 新行业或无法匹配的名称进入 unknown，不得动态分配新 ID；
- 模型升级后如词表改变，新增 vocabulary 版本，历史版本永久保留。

### 4.2 建议数据库表

#### `kronos_sector_vocabularies`

- `vocabulary_id`：主键；
- `model_release`；
- `num_sectors`；
- `unknown_sector_id`；
- `labels_json`：有序数组；
- `sha256`；
- `created_at`。

#### `kronos_symbol_sector_history`

- `symbol`；
- `vocabulary_id`；
- `sector_id`；
- `sector_label`；
- `effective_from`；
- `effective_to`，当前版本为空；
- `source`，第一版为 `baostock.query_stock_industry`；
- `source_reference_date`；
- `is_unknown`；
- `quality_status`；
- `created_at`。

主键建议为 `(symbol, vocabulary_id, effective_from)`；增加按 `(symbol, effective_from desc)` 的索引，并用约束避免同一证券的有效期重叠。

### 4.3 更新规则

1. 每周获取一次 BaoStock 行业全量快照。
2. 将原始行业名称映射到固定 86 类词表。
3. 与上一有效记录比较；只有行业或 unknown 状态变化时才新增历史版本。
4. 新数据覆盖率低于上一版本 98%时拒绝发布。
5. unknown 数量显著上升时阻断每日预测并报警。
6. 每次正式预测把最终 `sector_id`、label、来源日期复制到预测事实表，避免以后映射变化导致历史结果无法复现。

第一阶段可以把现有 `symbol_sector_map.json` 导入数据库作为初始版本，但必须保留原 `vocabulary_id` 和 reference date。

## 5. 市值条件方案

每日优先使用与原始训练完全一致的 `amount / (turnover_pct / 100)` 作为市值代理。`stock_factors.float_market_cap` 用于质量交叉验证，不能与代理值逐股混用后共同排序；否则部分证券会采用不同市值定义，扭曲横截面。

1. 正常交易证券使用 signal_date 当日有效的 amount 和 turnover；停牌或当日 turnover 无效时，按训练逻辑使用该证券不晚于 signal_date 的最近一次有效市值代理，并保存实际 as-of date；
2. 按与训练一致的当日全市场参考池计算 percentile；该参考池包含市值有效的 ST、停牌和行业 unknown 证券，因为原始 `all_a` 训练清单没有按这些条件过滤；
3. 交易过滤后的排名股票池与 size reference universe 分开，不能因 ST、停牌、行业 unknown 或流动性过滤而改变其他股票的模型条件；
4. 使用稳定排序处理相同市值，并固定用 symbol 作为并列值的次排序键以保证复现；
5. 缺失市值代理时标记 `size_quality_status = missing`，既不能进入 size reference universe，也强制排除正式股票池；
6. 将 `float_market_cap`、`size_percentile`、市值数据日期、来源和 reference universe 版本固化到预测行。

长期应在 Supabase 保存 point-in-time 市值代理快照，不能仅依赖当前值，否则无法严格复现历史预测。

## 6. 复权与输入一致性验证

这是上线前的硬门槛，不通过就不能用真钱依据排行榜交易。

建立一个 30-50 只股票的校验集，覆盖：

- 主板、创业板；
- 大中小市值；
- 近期发生分红、送转、拆并股的证券；
- 停牌后复牌证券；
- 高低成交量证券。

对同一 signal_date 的 120 日窗口逐字段比较 Supabase 与 OOS/Qlib 数据：

- 日期序列完全一致；
- OHLC 相对变化与复权口径一致；
- volume/amount 单位一致；
- 除权日前后不存在非经济性跳变；
- 标准化后的六维输入差异处于预设阈值内。

当前审计已经确认原始行情不一致，但复权因子方案已通过 BaoStock样本数值验收。实施方案固定为保存 point-in-time 复权因子历史，按 `raw_price(t) * adj_factor(t) / adj_factor(signal_date)` 生成前复权 OHLC；同时将 Tushare volume 从手乘100转换为股、amount 从千元乘1000转换为元。不得绕过落库因子直接使用原始 OHLC。

## 7. 股票池定义

第一版明确命名为 `cn_main_growth_liquid_v1`，只覆盖当前可靠采集范围，不冒充全市场。

初始纳入条件：

- 当日在 `ashare_daily_k` 有记录；
- `trade_status = 1`；
- 非 ST；
- 至少有连续 120 个有效交易日；
- 六维特征均有效；
- 有有效且不为 unknown 的行业映射；
- 有 point-in-time 流通市值；
- 最近 20 日成交额达到最低阈值；
- 非已退市、退市整理或长期停牌证券。

正式横截面采用 complete-case 规则：120 日窗口、六维特征、行业、市值、交易状态和流动性条件必须全部有效。预测表同时保存 `eligible_for_ranking` 和所有排除原因。可以为被排除证券保留数据质量记录，但它们不得送入正式模型批次，不得进入 D10 排名分母、预测排名 percentile 或 Top-K。若其市值本身有效，仍可按训练口径参与独立的 size reference universe；这不会产生该证券的正式预测。

## 8. 数据库设计

### 8.1 `kronos_prediction_runs`

一行代表一个 signal_date 的正式任务：

- `run_id` UUID；
- `signal_date`；
- `status`：pending/preparing/inferencing/validating/published/failed；
- `model_repo`、`model_revision`、`model_release`、`checkpoint`；
- `tokenizer_repo`、`tokenizer_revision`；
- `pipeline_version`、`universe_version`、`vocabulary_id`；
- `input_data_max_date`；
- `candidate_count`、`eligible_count`、`predicted_count`、`failed_count`；
- `random_seed`、`sample_count`、`temperature`、`top_p`；
- `started_at`、`completed_at`、`published_at`；
- `input_manifest_sha256`、`result_sha256`；
- `error_summary`、`metrics_json`。

唯一约束：`(signal_date, model_release, pipeline_version, universe_version)`。

### 8.2 `kronos_daily_predictions`

每日每股一行：

- 身份：`run_id, signal_date, symbol, stock_name`；
- 输入快照：`last_close, sector_id, sector_label, sector_reference_date, float_market_cap, size_percentile`；
- 数据质量：`history_rows, data_max_date, feature_quality_status, exclusion_reasons`；
- 预测路径：`predicted_close_d1 ... d10`；
- 累计收益：`predicted_return_d1 ... d10`；
- 排名：`market_rank_d10, market_percentile_d10, sector_rank_d10, sector_percentile_d10`；
- 风险和交易状态：`is_st, trade_status, liquidity_20d, eligible_for_ranking`；
- 审计：`input_sha256, created_at`。

主键：`(run_id, symbol)`。正式发布后禁止 UPDATE 预测字段，只能通过新的 run 重算。

### 8.3 `kronos_prediction_outcomes`

- `run_id, symbol`；
- `actual_close_d1 ... d10`；
- `actual_return_d1 ... d10`；
- `actual_rank_d1 ... d10`；
- `filled_through_horizon`；
- `last_filled_at`；
- `outcome_quality_status`。

实际收益必须使用与预测输入一致的复权与交易日口径。

### 8.4 `kronos_daily_metrics`

按 run 和 horizon 保存：

- `rank_ic`；
- Top 10%、Top 20%实际收益；
- top-bottom 10% spread；
- 市场等权收益与超额收益；
- 覆盖数、正收益率；
- 行业中性指标；
- 模拟成本后的收益；
- turnover、最大回撤和数据质量统计。

### 8.5 对外只读视图

- `kronos_latest_published_run`；
- `kronos_latest_rankings`；
- `kronos_symbol_prediction_history`；
- `kronos_live_oos_metrics`。

anon 角色只允许读取这些视图，不允许直接读底层任务表或写入任何数据。

## 9. 每日任务流

```text
Oracle 20:30 行情同步
  -> 数据完整性检查
  -> 确认 signal_date 与行情最大日期一致
  -> 构建股票池和 120 日输入
  -> 解析 point-in-time 行业与市值
  -> 生成输入 manifest 和 run
  -> 分批调用 Modal GPU 推理
  -> 分片结果幂等写入 staging
  -> 全量校验和横截面排名
  -> 原子发布 run
  -> WebUI 自动读取新快照
  -> 后续 D1-D10 每日回填实际结果
```

### 9.1 编排位置

建议 Oracle 负责调度和数据准备，Modal 只负责 GPU 批量推理：

- Oracle 已有稳定的日任务和数据库访问；
- Modal 不应承担行情采集和业务数据库迁移；
- 输入按 64-128 只股票分片；
- 每个分片有稳定 `shard_id`，重试不产生重复结果；
- Modal 返回预测数据，不直接持有 PostgreSQL 超级用户密码；
- Oracle 使用最小权限数据库角色写入预测表。

### 9.2 发布门槛

只有以下检查全部通过才把 run 标记为 published：

- input_data_max_date 等于 signal_date；
- eligible 数量不低于近期中位数的 95%；
- 预测成功率不低于 99.5%；
- 无重复 symbol；
- D10 分数有效且非退化常数；
- sector unknown 比例未异常上升；
- 正式预测结果中 `sector_id = 86` 的行数必须为 0；
- size percentile 分布覆盖合理；
- 输出 SHA 和行数一致；
- 排名从 1 连续到 eligible_count。

失败时前端继续显示上一 published run，并明确标注数据日期，不得展示半成品。

## 10. 前端重构

### 10.1 首页：每日排行榜

- 数据日期、模型版本、股票池名称和更新时间；
- Top 20/50/100 切换；
- D10 全市场排名、百分位和预测累计收益；
- D1-D10 路径缩略图；
- 行业、市值、成交额、ST/停牌与风险标识；
- 行业过滤、市值过滤、最低流动性过滤；
- 明确文案：“相对强弱排序，不是上涨概率或收益承诺”。

### 10.2 个股详情

优先从 `kronos_daily_predictions` 读取：

- 当日正式排名；
- D1-D10 预测路径；
- 全市场和行业内百分位；
- 最近历次预测及到期实际结果；
- 数据与行业/市值条件来源；
- 不在股票池时展示具体原因。

按需推理只能作为诊断功能，必须标记 `on_demand`，不能混入正式排行榜或线上 OOS。

### 10.3 验证面板

- 每日 D1/D3/D5/D10 Rank IC；
- 滚动平均 IC 与 ICIR；
- Top 10%/20%等权组合相对全股票池收益；
- 模拟交易成本后的收益；
- 行业暴露、换手率和最大回撤；
- 未成熟 horizon 的样本明确标记 pending。

## 11. 风控边界

- 第一版只生成研究候选，不自动交易。
- 默认等权模拟，设置单票和行业权重上限。
- 回测与线上指标必须加入手续费、滑点、涨跌停和无法成交约束。
- 排名发布后不可因后续行情修订而静默覆盖。
- 所有页面展示 signal_date、数据新鲜度、模型版本和股票池覆盖。
- OOS 只有 13 个设计污染日期，线上结果必须继续作为 fresh-window OOS 累积，至少观察 40-60 个新信号日后再调整资金规模。

## 12. 安全与权限

- 立即轮换已经暴露的 PostgreSQL 密码。
- 删除代码、文档和 archive 中的硬编码连接串；统一使用 `DB_URL`/`DATABASE_URL` 或平台 Secret。
- 新建最小权限角色：行情只读、预测写入、Web 只读三类权限分离。
- 浏览器不接触数据库密码；anon key 仅能访问 RLS 保护的只读视图。
- Modal Secret 不能使用 postgres 超级用户。
- 日志不得输出连接串、JWT、请求完整输入或用户认证信息。

## 13. 分阶段实施

### Phase 0：数据合同审计（1-2 天）

- 完成复权/单位对照；
- 固化模型输入合同测试；
- 统计 120 日完整历史覆盖；
- 输出生产股票池差异报告。

验收：选定样本的 Supabase 输入与 OOS 输入语义一致，全部自动测试通过。

### Phase 1：行业与市值主数据（1-2 天）

- 建行业 vocabulary 和 symbol-sector history 表；
- 导入当前 86 类词表与 5,544 只映射；
- 建每周刷新、覆盖率保护和 unknown 告警；
- 建 point-in-time 市值快照和 percentile 计算。

验收：当日合格股票的行业/市值条件覆盖率达到约定阈值，并可按历史日期重放。

### Phase 2：预测表与批量任务（2-4 天）

- 执行数据库 migration；
- 构建 universe/input manifest；
- 扩展 Modal 批量推理；
- 实现分片、重试、断点续跑、原子发布；
- 先 shadow run，不向用户展示。

验收：连续 3 个交易日自动完成，成功率不低于 99.5%，同输入重跑结果可复现。

### Phase 3：结果回填和线上 OOS（1-2 天）

- 实现 D1-D10 成熟度判断；
- 回填实际收益；
- 计算每日 Rank IC、spread 和成本后组合指标；
- 增加异常报警。

验收：用历史预测样本回放，指标与离线 evaluator 在容差内一致。

### Phase 4：WebUI 重构（3-5 天）

- 首页改为排行榜；
- 个股页改为查正式快照；
- 增加 OOS/数据质量面板；
- 保留旧即时预测入口为高级诊断功能。

验收：页面请求不触发 GPU 推理；排行榜、个股页和数据库快照一致；旧链接有兼容跳转。

### Phase 5：小资金观察（至少 40-60 个新信号日）

- 只使用预先冻结的选股与组合规则；
- 不因短期结果频繁调模型或筛选阈值；
- 每周审查数据质量，每月审查表现；
- 扩大资金前另做一次 sealed fresh-window 评审。

## 14. 测试与验收清单

- 行业 vocabulary 顺序和 SHA 与模型发布合同一致；
- 行业映射不得产生越界 ID；
- unknown 行业、缺失市值、历史不足或六维特征不完整的证券均被强制排除，并保留明确原因；
- 股票池过滤不存在未来数据；
- 120 日窗口没有日期重复、乱序或未来行；
- 六维特征顺序和单位固定；
- 同一 run 并发重试不重复写入；
- run 发布是原子的；
- D10 排名和 percentile 边界正确；
- outcome 只在 horizon 成熟后回填；
- evaluator 与离线 OOS 公式一致；
- RLS 阻止 anon 写入；
- 页面只展示 published run；
- 前端断网或最新任务失败时安全退回上一正式批次。

## 15. 第一版明确不做

- 不自动连接券商下单；
- 不基于单日结果动态改 Top-K 或持仓规则；
- 不把方向准确率包装成上涨概率；
- 不动态修改模型行业词表；
- 不允许前端为每次页面访问重新推理；
- 不在未验证复权口径前发布真钱排行榜；
- 不将现有 4,559 只覆盖范围宣传为完整全 A 股。

## 16. 推荐的立即执行顺序

1. 轮换 Supabase 数据库密码并清理硬编码凭据。
2. 完成 Supabase 与 OOS/Qlib 的复权和单位审计。
3. 建行业历史表、市值快照表和预测四张核心表。
4. 导入固定行业 vocabulary 与当前 symbol 映射。
5. 完成单日全股票池 shadow batch，不改前端。
6. 校验单日结果、成本和 Modal 运行费用。
7. 连续 shadow 三个交易日后再切换排行榜前端。
