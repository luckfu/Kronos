# A 股增训方案

## 目标

在不改变 `Kronos-Tokenizer-base` 六维 OHLCV 表示的前提下，让 `Kronos-base` 识别 A 股的规模风格差异。当前实验只启用市值分层，不依赖行业标签。

## 数据口径

- 股票池：以 2015–2025 年 CSI800 历史成分并集为核心，再补入微盘、小盘和中小盘股票，避免 CSI800 对大中盘风格的系统性偏置。
- 频率：日线。
- 价格：统一使用前复权或后复权口径，不能混用。
- 规模：流通市值，按交易日横截面分成 10 桶。
- 行业：当前实验关闭；以后拿到可靠的 point-in-time 行业数据再单独做消融。
- V3 训练期：2015-01-01 至 2025-12-31。2015 年用于覆盖快速上涨及随后大幅下跌的完整市场状态。
- 验证期：2026-01-01 至当前可用日期。
- 样本外评估：使用 2026-01-05 至 2026-07-17 的日频滚动信号做第一版样本外回测；该结果仍属于 pilot，不是完整年度测试。
- 输入窗口：90 个交易日；预测窗口：未来 10 个交易日。

## 滚动增训与历史回放

生产模型以适应当前市场为目标，后续按最新已完整出标签的数据持续增训，但不能只喂最近数据。默认训练窗口配比为 `80%` 近期数据和 `20%` 历史回放数据；允许根据行情变化与遗忘程度在 `70%-90%` 近期、`10%-30%` 历史之间调整。配比按窗口数执行，历史面板必须下采样，不能直接与近期面板拼接后让历史数据重新占据绝对多数。

历史回放池按市场阶段、波动状态和市值桶分层，至少覆盖牛市、熊市、震荡市、高低波动期及十个市值桶。每次增训从各层抽取窗口并与近期窗口统一打乱，保持无放回覆盖；若旧行情指标下降过快，优先提高回放比例，而不是继续增加纯近期训练轮数。

每次滚动增训同时保留两类验证：固定的整股票 holdout 用于检查跨股票泛化，最近至少 20 个已完整出标签的交易日用于检查跨时间泛化。由于预测目标需要未来 10 个交易日，训练和验证的信号截止日都必须预留完整标签区间。生产晋级以最近时间外窗口的 Rank IC、方向准确率、头尾收益差和含成本策略收益为主，MAE作为绝对价格校准指标；历史回放集用于监控灾难性遗忘和决定回放比例，不要求当前市场模型在每个旧年份都超过旧生产模型。所有晋级都保留上一生产 checkpoint，并从上线日起保存逐日预测用于回滚判断。

## 输入格式

长表至少包含：

```text
symbol,date,open,high,low,close,volume,market_cap
```

也可以直接提供 `size_bucket`（`0..9`）和 `size_percentile`（`0..1`）。只有离散桶时，数据管线会用桶中点近似连续百分位；正式混合实验应从同日横截面 `market_cap` 排名生成真实百分位。每个 `symbol,date` 都应有一行。

## 数据准备

### 全市场市值分层 V3

V3 使用 2025 年末可交易 A 股横截面构造补充股票池，并优先选择 2016 年前已上市的股票，以获得尽可能完整的 2015–2025 历史。训练股票包括 1,489 只 CSI800 历史成分，以及微盘、小盘和中小盘各 300 只；另从三个层级各留出 80 只整股票作为跨股票样本外集合。训练集 2,389 只，holdout 240 只，股票代码完全不重叠。

本地已经生成的数据口径为：

- 原始行情：2,629 只股票、6,279,803 行，2015-01-05 至 2026-07-31。
- 训练集：2,389 只股票、5,472,438 行、5,233,538 个 90→10 窗口。
- 2026 时间外验证：2,312 只股票、320,840 行、89,730 个窗口。
- 整股票 holdout：240 只股票、486,525 行。

训练窗口按市值桶生成确定性的均衡无放回顺序。完整覆盖序列中的每个窗口只出现一次；每个 20,000 样本分段优先从十个桶各取 2,000 个，缺失市值的窗口保留到覆盖序列末尾，不作为额外的第十一种风格参与均衡。

```bash
python finetune/build_a_share_v3_universe.py
python finetune/download_a_share_parallel.py
python finetune/prepare_a_share.py \
  --input ./data/a_share_v3/a_share_daily_parallel.csv \
  --output-dir ./data/a_share_v3/processed_datasets \
  --metadata-out ./data/a_share_v3/asset_metadata.csv \
  --train-end 2025-12-31 \
  --val-start 2026-01-01 \
  --val-end 2026-12-31 \
  --universe-manifest ./data/a_share_v3/universe_manifest.csv \
  --holdout-output ./data/a_share_v3/processed_datasets/symbol_holdout_data.pkl \
  --size-reference-out ./data/a_share_v3/size_reference.json

KRONOS_BATCH_SIZE=16 KRONOS_NUM_WORKERS=0 \
bash finetune/train_a_share_v3.sh
```

V3 从当前生产 checkpoint warm-start，但会将 `size_emb` 清零，因为 V3 的桶边界基于更广的全市场横截面，不能直接沿用 CSI800 相对桶的语义。底部十层继续冻结，只训练顶部两层、归一化层、依赖层、输出头和市值 Embedding。训练完成后必须同时比较 2026 时间外结果和 240 只整股票 holdout，才能决定是否替换生产模型。

```bash
python finetune/download_a_share_baostock.py \
  --universe csi800 \
  --start 2020-01-01 \
  --end 2026-07-31 \
  --output ./data/a_share/a_share_daily.csv

python finetune/prepare_a_share.py \
  --input ./data/a_share/a_share_daily.csv \
  --output-dir ./data/a_share/processed_datasets \
  --metadata-out ./data/a_share/asset_metadata.csv
```

然后在 `finetune/config.py` 中打开条件输入：

```python
self.dataset_path = "./data/a_share/processed_datasets"
self.asset_metadata_path = "./data/a_share/asset_metadata.csv"
self.use_context_features = True
self.use_sector_features = False
self.use_size_features = True
self.use_size_percentile = False
self.num_sectors = 0
self.num_size_buckets = 10
self.pretrained_predictor_path = "./Kronos-base"
```

## 增训顺序

1. 加载已经学习过通用市场规律的 `Kronos-base`，冻结 tokenizer 和底部 10 层。
2. 在第 10 层注入市值 embedding，只训练顶部 2 层、预测头和市值 embedding。
3. 以 2026 年验证损失早停，并分别统计各市值桶的误差。
4. 使用 2026 年样本外数据做滚动回测，加入固定交易成本，并记录换手、空仓和退出信号；涨跌停、冲击成本和 point-in-time 成分仍待补齐。

## 完整覆盖增训页面

运行 `python webui/finetune_app.py` 后访问 `http://127.0.0.1:7071/`。这是独立于 7070 预测服务的 App 和进程。页面可选择本地 `NeoQuasar/Kronos-base` 或已有完整 checkpoint 作为训练起点，模型磁盘路径不会发送到浏览器；设备按 MPS、CUDA、CPU 的顺序自动选择，启动或停止增训不会卸载预测服务中的模型。

默认训练不再把一次“轮次”理解为从 150 万窗口里随机抽 800 个样本。训练窗口先按固定种子生成一个排列，每段取其中连续的 20,000 个唯一窗口，下一段从上段末尾继续；当前 1,509,252 个窗口需要 76 段完成一遍覆盖。验证集固定取 2,000 个窗口，保证每段 loss 可比较。完成指定覆盖遍数之前不触发早停，之后连续 5 段没有改善才结束。

正常训练每段结束都会写入 `checkpoints/last_state.pt`。点击停止时不再等待整个分段，而是在当前 batch 完成后停止计算并保存模型、优化器、学习率调度器、当前分段、下一 batch 位置和随机数状态；之后可在 checkpoint 表中选择“继续”精确恢复。页面每 100 step 更新批次 Loss 与段内平均 Loss，每段结束追加验证 Loss 和历史最佳 Loss；结构化指标保存在 `metrics.jsonl`，旧日志也能还原曲线。按当前 Apple MPS 实测速度，默认 1 遍覆盖加 5 段观察约需 29 小时，页面会按配置重新估算。

Apple Silicon 直接运行：

```bash
python finetune/train_predictor.py
```

也可以不改配置文件，直接覆盖数据路径：

```bash
KRONOS_DATASET_PATH=./data/a_share/processed_datasets \
KRONOS_METADATA_PATH=./data/a_share/asset_metadata.csv \
KRONOS_PREDICTOR_PATH=./Kronos-base \
python finetune/train_predictor.py
```

脚本会自动选择 `mps`；默认最多运行 50 轮，连续 5 轮验证集没有改善才早停。可通过 `KRONOS_EPOCHS` 和 `KRONOS_EARLY_STOPPING_PATIENCE` 覆盖；CUDA 环境仍可使用 `torchrun` 多进程。

混合条件消融通过环境变量启用：

```bash
KRONOS_USE_SIZE_PERCENTILE=1 \
KRONOS_PREDICTOR_SAVE_FOLDER=a_share_size_hybrid_kronos_base_earlystop50 \
KRONOS_PREDICTOR_PATH=./Kronos-base \
python finetune/train_predictor.py
```

该模型保留十桶 Embedding，并用两层 MLP 将连续百分位和已知标志映射到 832 维，在第 10 层后与桶 Embedding 相加。连续分支从零输出初始化，不影响原始 checkpoint 的初始行为。

## 下一版实验计划：V5 上下文窗口

下一版增加一个独立的上下文窗口消融：输入最近 **120 个交易日**，仍然预测未来 **10 个交易日**。当前生产的 `90+10` 口径保持不变，先作为严格对照组；在 120 日候选通过时间外评估之前，不修改前端默认输入长度。

### 设计约束

- 预测目标不变：未来 10 个交易日的路径，生产信号仍使用预测收盘价均值。
- 数据切分不变：训练、验证和 holdout 使用相同股票隔离、时间隔离和历史回放比例；只改变上下文长度。
- 每个窗口需要 `120 + 10 + 1 = 131` 行连续行情。计算信号日期时必须保留信号日前至少 120 个交易日，不能再次先裁掉上下文。
- 90 日和 120 日实验必须使用相同的信号日期、股票、随机种子、batch 语义、学习率和覆盖遍数，不能用不同样本数量造成比较偏差。
- 现有模型参数形状不因窗口长度改变，可以 warm-start；tokenizer、Transformer 层数、市值条件结构暂不修改。

### 资源与训练安排

120 日会增加每个 batch 的 token 数，并提高注意力显存和计算开销；当序列长度主要由上下文决定时，注意力部分的理论开销约按 `(120/90)^2 = 1.78` 倍增长。首次试跑先用 batch size 16 与 32 各做一个 segment benchmark，再决定正式 batch；不能只按 90 日的耗时估算 Kaggle 配额。

正式候选仍使用一遍完整覆盖加第二遍确认，保存每遍结束的 checkpoint、`best` 和 `last`，并记录实际覆盖窗口数。早停只作为第二遍之后的观察机制，不能在第一遍中途依据同一验证集淘汰上下文方案。

### 评估顺序

1. 在相同市值条件和相同 V3 基础权重上，先比较 `90+10` 控制组与 `120+10` 候选组。
2. 在固定的 2025 历史时间窗口检查长期能力和遗忘，在 2026 开发窗口检查近期适应性。
3. 新增的未来标签积累到至少 20 个完整信号日后，作为最终时间外测试；此前的 2026-06-18 至 2026-07-16 只作为开发验证，不能反复调参后再当最终证据。
4. 统一报告 token loss、Rank IC、Rank IC 正日期比例、方向准确率、balanced accuracy、MAE、预测下跌比例、头尾 20% 收益差、换手和含成本收益。
5. 先完成窗口长度消融，再决定是否在 120 日上继续做“无市值条件 vs condition dropout”实验，避免一次扩展成无法解释的组合矩阵。

### 晋级条件

120 日候选只有在新时间外 Rank IC 和头尾收益差不低于 90 日 incumbent、没有明显增加预测下跌偏差、历史回放能力没有显著退化，且训练吞吐和显存成本可接受时，才进入前端灰度。否则保留 90 日生产模型；120 日模型的失败结果也要记录，不能仅凭训练 loss 较低晋级。

## 当前生产结果

### 2026 增量数据审计与 V4 修正

旧 2026 增量流水线存在已确认的时间上下文裁剪错误：先把每只股票裁到 `2026-01-01` 以后，再构造 90 日输入和 10 日预测窗口。结果 1 月至 5 月上旬只能充当观察上下文，不能成为监督信号；有效信号日仅为 `2026-05-22` 至 `2026-07-16`，共 39 日。89,730 个窗口虽然股票数量多，但共享同一小段偏空市场状态，训练集下跌标签比例 `65.83%`，验证集为 `72.56%`。

修正后的连续面板保留 2025 年至少 90 个交易日作为 2026 年初上下文，再按照信号日期筛选窗口。使用现有本机数据严格复核得到：

- 训练信号 `2026-01-05` 至 `2026-06-17`：108 日、249,280 个窗口，下跌比例 `56.64%`。
- 时间验证 `2026-06-18` 至 `2026-07-16`：20 日、46,129 个窗口，下跌比例 `60.45%`。
- A 组仅使用修正近期数据；B 组加入 62,320 个按年份和市值桶分层的历史窗口，历史占比精确为 20%。
- 两组使用同一个增训前 V3 Last、相同近期窗口、学习率、batch size 和单遍完整覆盖；只有历史回放不同。

该实验先验证数据修正和历史回放，不修改 Kronos 主结构。生产晋级必须同时检查时间外 Rank IC、方向准确率、预测涨跌比例偏差、MAE 和股票池头尾收益差，不能再仅依据同期 symbol holdout 的 token loss。

V4 A/B 已在 P100 完成。最后 20 个时间外信号日、240 只固定股票的结果为：V3 Last Rank IC `0.01124`、A 组 `-0.04796`、B 组 `-0.03966`；A/B 的方向准确率虽然分别为 `64.23%` 和 `62.90%`，但实际下跌比例为 `73.37%`，两者平衡准确率均低于 `0.50`。B 只将预测下跌比例从 `81.63%` 降到 `78.87%`，没有恢复排序能力。结论是：数据裁剪修正已经生效，20% 回放也按预期生效，但这一训练配置不能晋级生产；当前生产 checkpoint 保持不变，候选模型及其时间外预测保留在 Kaggle 输出用于后续分析。

- 生产 checkpoint：`outputs/models/a_share_size_full_coverage_colab_bs32_latest/checkpoints/best_model/model.safetensors`。
- 训练覆盖：1,509,252 个有效窗口，batch size 32，81 个覆盖分段，完整遍历一轮。
- 可训练参数：20,260,416 / 102,319,744（19.8%）。
- 训练过程最佳验证 loss：`2.877660`；生产使用完整遍历结束后的 `latest` 权重。
- 2025 年未见股票测试：Rank IC `0.16549`，前后 20% 平均十日收益差 `4.46%`。
- 同条件旧在线模型：Rank IC `0.09122`，前后 20% 平均十日收益差 `2.44%`。
- ModelScope 公开仓库：<https://modelscope.cn/models/luckfu/a-share-size-kronos-base-earlystop50>，仓库地址保持兼容，权重已更新为全覆盖版本。

## 早期实验结果

- 数据：1176 只 CSI800 历史成分并集，1,786,689 行。
- 训练：2020-01-02 至 2025-12-31，共 1,626,852 行。
- 验证：2026-01-01 至 2026-07-31，共 159,837 行。
- 原始 `Kronos-base` 分桶评估 loss：3.142226。
- 市值分层增训后分桶评估 loss：2.951122，改善 6.082%。
- 0–9 十个市值桶全部改善，单桶改善范围为 4.292%–8.165%。
- 早期 checkpoint：`outputs/models/a_share_size_kronos_base_earlystop50/checkpoints/best_model/model.safetensors`，已退出生产。
- ModelScope 公开仓库：<https://modelscope.cn/models/luckfu/a-share-size-kronos-base-earlystop50>。
- 训练配置使用最多 50 轮早停；最佳训练验证 loss：2.951193。

### 连续百分位混合条件消融

- checkpoint：`outputs/models/a_share_size_hybrid_kronos_base_earlystop50/checkpoints/best_model`。
- 训练连续运行 26 轮后早停，最佳训练验证 loss：`2.9518`。
- 固定随机序列评估：离散模型 `2.951176`，混合模型 `2.951893`。
- 混合模型比离散模型高 `0.000717`（约 `0.024%`），没有形成验证集优势，暂不替换生产模型。
- 129 日滚动回测：混合模型 Rank IC `0.04250`，比离散模型 `0.04152` 高 `0.00098`；方向准确率 `55.05%`，比离散模型 `55.31%` 低 `0.25` 个百分点。
- 混合模型毛收益 `-21.93%`、计成本净收益 `-43.41%`，均弱于离散模型的 `-7.35%` 和 `-33.30%`。
- 结论：连续百分位只带来极小排序提升，没有形成可交易或验证损失优势，不部署该 checkpoint。

### 早停耐心消融

- 设置：离散桶模型，最多 50 轮，连续 5 轮验证集无改善才早停。
- 实际运行 43 轮，最佳点在第 38 轮，训练日志最佳 loss `2.9548`。
- 固定随机序列评估 loss `2.954815`，现有生产模型为 `2.951176`。
- 候选模型高 `0.003639`（约 `0.123%`），十个市值桶全部略差，因此不替换生产 checkpoint。
- 训练配置默认 patience 仍调整为 5，用于降低未来实验因短期波动过早停止的风险。

## 样本外回测

### 日频持仓规则

`finetune/backtest_a_share_daily.py` 每个交易日收盘后重新预测未来 10 个交易日：

1. 使用 64 只最小市值分层股票作为共同股票池。
2. 只买预测收益为正的股票，最多持有 8 只。
3. 已有持仓若预测变为非正，下一交易日开盘卖出。
4. 预测仍为正的持仓继续保留，空位由正预测股票补入；没有正预测时允许持有现金。
5. 按下一交易日开盘成交、收盘估值，交易成本按全组合换手 25 bps 计。

### 结果

- 信号日：129 个，覆盖 2026-01-05 至 2026-07-20 的次日执行。
- 增训模型平均 Rank IC：0.04152，Rank IC 为正的信号日比例：63.57%。
- 增训模型 10 日方向准确率：55.31%。
- 增训模型组合毛收益：-7.35%；计入交易成本后：-33.30%。
- 同期 64 只小市值股票等权组合：-5.98%。
- 增训模型触发 481 次非正预测退出，平均每日组合换手 101.7%。

这说明增训模型的横截面排序和方向判断比原始模型更好，但单路径日频信号反转过多，交易成本吞噬了预测优势。当前结果不支持直接实盘；下一步应测试多路径中位数、买卖阈值或确认期，并严格补齐 point-in-time 成分、停牌和冲击成本。

详细文件：

- `outputs/models/a_share_size_kronos_base_earlystop50/size_eval.json`
- `outputs/backtest_results/a_share_2026_smallcap/summary.json`
- `outputs/backtest_results/a_share_2026_daily_smallcap/summary.json`
- `outputs/backtest_results/a_share_2026_daily_smallcap_hybrid/summary.json`
