# A-share Full-Market v1-beta 训练方案

本文档定义全市场、行业条件模型的唯一实验口径。版本名称固定为
`A-share Full-Market v1-beta`，不使用 V7 或其他临时名称。

模型上线后的输入准备和市值百分位计算见
[`MODEL_USAGE_V1_BETA_CN.md`](MODEL_USAGE_V1_BETA_CN.md)。

统一 `1e-4` 基线停止后的 Warmup 与差分学习率正式训练安排见
[`V1_BETA_FUTURE_TRAINING_PLAN_CN.md`](V1_BETA_FUTURE_TRAINING_PLAN_CN.md)。
该计划从 V6 Segment 568 新开实验，不续旧 v1-beta State。

## 1. 实验目标

在保留当前生产模型 V6 Segment 568 的通用 OHLCVA 能力基础上，引入：

- 全市场沪深 A 股训练数据；
- 点时行业条件；
- 修正后的交易口径市值代理；
- 连续市值百分位条件。

该版本是候选模型，不自动替换 V6 生产版本。V6 Segment 568 始终作为回滚点保留。

## 2. 数据口径

数据目录：

```text
data/a_share_full_market_2026/
```

数据清单：

- `universe_20260731.csv`：2026-07-31 市场快照中的 5,201 只沪深 A 股；
- `panel.csv`：2014-01-02 至 2026-07-31，11,577,162 行；
- `asset_metadata.csv`：逐股票、逐日期的行业和市值条件；
- `processed_datasets/train_data.pkl`：5,198 只，11,525,220 行；
- `processed_datasets/val_data.pkl`：5,201 只，1,364,731 行（含 2025-07-01 起的 120 日历史前缀）；
- `size_reference.json`：最新市值横截面参考；
- `validation_report.json`：下载和数据完整性报告。

行情使用 BaoStock `adjustflag='2'` 前复权。前复权价格只用于 OHLC 序列，不用于市值计算。

市值代理固定为：

```text
market_cap_proxy = amount / (turnover_pct / 100)
```

`size_percentile` 的定义是**每日全市场横截面百分位**：在每个交易日内，
对当日纳入 v1-beta 股票池的所有股票按 `market_cap_proxy` 从小到大排序，
使用 `rank(method="first", pct=True)` 得到 `[0, 1]` 的百分位。它表示该股票
当天相对于全市场股票的市值位置，不是单只股票自身的历史百分位，也不是用当前
市值回填过去。训练样本只读取观察窗口最后一个交易日的该值，因此是点时条件，
不使用预测窗口中的未来市值信息。

数据来源为 BaoStock 每日字段 `amount`、`turn` 以及前复权行情；换手率为零或
缺失时，数据准备阶段在股票内做前向填充。该值是用于横截面排序的市值代理，
不是财务口径的精确总市值。原始结果保存在 `asset_metadata.csv` 的
`size_percentile` 列，最新截面参考保存在 `size_reference.json`。

换手率为零或缺失的少量停牌/边界行，在数据准备阶段按股票前向、后向填充。行业历史按年度快照点时映射，无法获得历史标签的行保留为 `unknown`，不使用今天的行业标签回填历史。

## 3. 数据切分

```text
训练：截至 2026-07-17（完整 10 日标签边界）
训练信号起点：2015-01-01
验证信号：2026-01-01 至 2026-07-17
验证上下文：`val_data.pkl` 保留至 2025-07-01 起的历史行；2026-07-18 至 2026-07-31 仅作为后续标签覆盖
测试：暂不生成 2027 test，待有完整未来标签后再做
```

2014 年行情只作为 2015 年初信号的历史上下文，不作为 2014 年监督信号。训练 runner 必须显式设置 `KRONOS_TRAIN_SIGNAL_START=2015-01-01` 和 `KRONOS_TRAIN_SIGNAL_END=2026-07-17`，不能依赖空的默认筛选范围。

2026 验证集只用于训练过程监控和最终候选比较；上线晋级必须另外执行统一的时间外和股票外评估。

## 4. 模型与 batch 接口

初始化基座为生产 V6 Segment 568 的 `model.safetensors`。不从更早的原始 Kronos 重新开始。

```python
model = Kronos.from_pretrained(
    v6_segment568_path,
    num_sectors=86,
    num_size_buckets=0,
    context_layer=10,
    use_size_percentile=True,
)
```

配置中的 `num_sectors=86` 表示 86 个已知行业；实现必须额外保留 1 个 unknown 行业桶，因此行业 embedding 实际应为 87 行，符合朋友权重的 `(87, 832)` 形状。

每个 batch 必须提供：

```text
x                 (B, 120, 6)
x_stamp           (B, 120, 5)
y_stamp           (B, 10, 5)
sector_ids        (B,)
size_percentiles  (B,)
```

输入窗口为 120 个交易日，预测未来 10 个交易日。损失只计算未来预测 token 的离散化交叉熵，不用历史上下文位置主导最佳模型选择。

## 5. 条件权重处理

从 V6 继承：

- tokenizer；
- Transformer 主体；
- 预测头及其它已学习的通用参数。

重新初始化：

- `sector_emb`；
- `size_mlp`；
- optimizer；
- scheduler。

原因是 V6 的市值条件使用了受前复权价格影响的旧代理；本实验不再使用离散市值桶，只使用连续市值百分位。新数据已经改为 `amount / turnover`，不能把 V6 的 optimizer 或市值条件状态直接恢复到本实验。

## 6. 优化器与覆盖计划

完全采用朋友文档的优化方案：

```text
optimizer：AdamW
学习率：1e-4 起始，余弦衰减至 1e-6
segment：每段 20,000 个窗口
coverage：全市场完整覆盖 2 遍
```

训练 pickle 中的 11,525,220 是保留的行情行数；按 120 日上下文、10 日预测、训练信号 2015-01-01 至 2026-07-17 实测生成 `10,564,072` 个监督窗口。因此实际覆盖段数为：

```text
ceil(10,564,072 / 20,000) = 529 段/遍
529 × 2 = 1,058 段/两遍
```

朋友文档中的 321 段不能直接照搬，因为全市场窗口数量不同；禁止通过减少窗口或股票数强行凑成 321 段。

朋友记录的 321 段耗时为 10 小时 36 分（一遍、90 日上下文）。我们的 120 日上下文会增加单步计算和显存压力；按当前实际 529 段/遍，两遍纯训练初步按约 40 小时估算，连同 Kaggle 启动、保存和接力按约 42-45 小时规划，最终以首个完整 chunk 的实测速度修正。

## 7. Kaggle 生命周期

开始前必须执行 `KAGGLE_RUNBOOK_CN.md` 的 preflight：账号、Kernel slug、GPU、PyTorch、数据集、基座权重 SHA-256、输出目录和唯一 active Kernel 全部确认。

数据集只上传一次，两个阶段共用同一份 Kaggle Dataset。禁止把大文件下载到本地再上传到下一个 Kernel。

### Kaggle 专用接力流程

本实验以 Kaggle 为主运行环境，采用“一个 active Kernel、分段正常结束、服务端 Dataset 接力”的方式：

1. 首个 Kernel 从 v1-beta 数据 Dataset 和 V6 Segment 568 模型 Dataset 读取输入，输出到 `/kaggle/working/kronos_a_share_full_market_v1_beta/outputs/models/<output_name>/`。
2. Kernel 只运行预先设定的 `max_segments_per_run`，到达上限后主动完成当前 batch/segment，写完 checkpoint、`run.log` 和结构化指标后正常退出。
3. 通过 Kaggle 服务端把完整 output 树发布为 continuation Dataset；不把 checkpoint 下载到本地，也不从本地重新上传。
4. 下一 Kernel 只挂载一个 continuation Dataset，runner 在 `/kaggle/input` 中查找唯一的 `last_state.pt`，校验 Best/Last/State 和配置后复制到新的 `/kaggle/working` output 路径，再以 `resume_training=1` 继续。
5. 任何一个必需文件缺失、出现多个候选 output 树或 segment/数据/损失配置不一致，都必须拒绝接力，不能用 Last 猜造 Best。

每次接力必须继承完整 output 树，而不是只复制 `checkpoints/`。`run.log` 和
`metrics.jsonl` 以追加模式保留全局 segment/step 记录，确保可以从 Segment 1
一直查看到 Segment 1058 的连续 loss 曲线。接力启动时必须打印历史 metrics 行数、
恢复 segment、历史 Best loss 和实际输入树路径；校验失败就停止，不得带病续训。

Kaggle 页面日志只用于实时观察；可接力的持久证据必须位于 output 树并随 continuation Dataset 发布。每个 Kernel 的 output name、数据 Dataset、模型 Dataset 和实验 manifest 必须固定，不能靠手工选择“看起来最新”的目录。

每个 Kernel 使用安全的 `max_segments_per_run`，正常结束后由下一个 Kernel 从 Kaggle Dataset 挂载完整输出并续训。首个 Kernel：

```text
resume_training = 0
```

后续接力仅在找到唯一、完整且通过校验的输出树时设置：

```text
resume_training = 1
```

每个 chunk 必须保留：

```text
run.log
checkpoints/last_state.pt
checkpoints/best_model/model.safetensors
checkpoints/best_model/config.json
checkpoints/last_model/model.safetensors
checkpoints/last_model/config.json
metrics.jsonl
progress.json
summary.json
```

`best_model` 始终是整个实验历史上的最佳验证模型，不能被 `last_model` 覆盖。中断时必须从 `last_state.pt` 恢复 optimizer、scheduler、随机数状态、segment 和 batch 位置。

训练 stdout/stderr 必须通过 tee 同时写入 Kaggle 任务日志和
`<output>/run.log`，并使用无缓冲输出。Kaggle 页面日志用于实时监控，
`run.log` 随 checkpoint 树发布到 continuation Dataset，作为接力后的持久审计记录。
只存在于 Notebook 页面、未写入 output 的日志不能作为 chunk 的完整日志。

## 8. 运行阶段

### 阶段 A：代码和输入 preflight

在提交 Kaggle 前完成：

- 验证 `num_sectors=86` 时实际创建 87 行行业 embedding；
- 验证 `use_size_percentile=True`；
- 验证 120→10 窗口和 batch 字段形状；
- 验证 V6 主体权重加载，行业/市值分支为新初始化；
- 验证 AdamW 和 Cosine `1e-4 → 1e-6` 已替换当前 OneCycleLR；
- 验证输出目录不存在旧实验文件；
- 验证 stdout/stderr 会实时 tee 到 `run.log`；
- 校验数据集文件 SHA-256。

### 阶段 B：首个 Kaggle chunk

只运行一个 Kernel，使用小的安全 segment 上限，确认 loss、显存、保存和恢复契约。该阶段不是模型效果结论，只是运行验收。

在首个正式 chunk 启动前，必须先准备好后续接力所需的 Kernel/Dataset 输入方案，并完成一次挂载路径验证。首个 chunk 完成后只允许把已验证的 continuation 输出交给新 Kernel 自动恢复；不得等训练结束后再把 checkpoint 下载到本地上传。

### 阶段 C：两遍完整覆盖

按全局计划完成 `1,058` 段（529 段/遍 × 2 遍）。第一遍结束后不重置模型、optimizer 或 scheduler，直接进入第二遍；第二遍仍属于同一实验历史和同一全局 Cosine 学习率计划。每段结束检查训练 loss、forecast loss、验证 loss、学习率、显存和 checkpoint 完整性。任何失败先停止接力，定位原因后再恢复同一输出树。

### 阶段 D：统一评估

与 V6 Segment 568 使用完全相同的股票池、日期、预测长度和成本假设比较：

- Rank IC、ICIR；
- 方向准确率和预测涨跌比例；
- 头尾收益差；
- 含交易成本收益和换手率；
- MAE、预测波动率；
- 行业分组和市值桶分组表现；
- 时间外和股票外泛化。

训练 loss 不能单独决定上线。

## 9. 晋级规则

v1-beta 只有在统一样本外评估中相对 V6 有稳定改善，且没有明显的行业偏差、size bucket 偏差、预测方向偏置或换手恶化时，才进入灰度候选。

在评估完成前：

- V6 Segment 568 继续作为生产版本；
- v1-beta 只作为候选模型保存；
- 不覆盖 ModelScope 生产权重；
- 不删除旧的 V6 回滚包。

## 10. 禁止事项

- 不把 v1-beta 称为 V7；
- 不使用 V6 的 optimizer/scheduler 恢复；
- 不把 321 段硬套到 5,201 只全市场数据；
- 不把 20,000 窗口误当成 GPU batch size；
- 不同时启动多个 Kaggle Kernel；
- 不在 Kaggle 会话之间依赖 `/kaggle/working` 或 Notebook 临时盘；
- 不用人工选取旧 output、重命名 checkpoint 或复制单个 `last_state.pt` 冒充完整接力树；
- 不通过本地电脑中转 checkpoint 或训练数据；
- 不把只存在于 Kaggle 页面日志、未写入 `run.log` 的输出当作持久训练日志；
- 不因训练 loss 下降就直接替换 V6 生产版本。
