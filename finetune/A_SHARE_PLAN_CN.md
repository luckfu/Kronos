# A 股增训方案

## 目标

在不改变 `Kronos-Tokenizer-base` 六维 OHLCV 表示的前提下，让 `Kronos-base` 识别 A 股的规模风格差异。当前实验只启用市值分层，不依赖行业标签。

## 数据口径

- 股票池：优先 CSI800；数据供应商支持点时成分时，再使用全 A 股。
- 频率：日线。
- 价格：统一使用前复权或后复权口径，不能混用。
- 规模：流通市值，按交易日横截面分成 10 桶。
- 行业：当前实验关闭；以后拿到可靠的 point-in-time 行业数据再单独做消融。
- 训练期：2020-01-01 至 2025-12-31。
- 验证期：2026-01-01 至当前可用日期。
- 样本外评估：使用 2026-01-05 至 2026-07-17 的日频滚动信号做第一版样本外回测；该结果仍属于 pilot，不是完整年度测试。
- 输入窗口：90 个交易日；预测窗口：未来 10 个交易日。

## 输入格式

长表至少包含：

```text
symbol,date,open,high,low,close,volume,market_cap
```

也可以直接提供 `size_bucket`（`0..9`）和 `size_percentile`（`0..1`）。只有离散桶时，数据管线会用桶中点近似连续百分位；正式混合实验应从同日横截面 `market_cap` 排名生成真实百分位。每个 `symbol,date` 都应有一行。

## 数据准备

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

## 本次结果

- 数据：1176 只 CSI800 历史成分并集，1,786,689 行。
- 训练：2020-01-02 至 2025-12-31，共 1,626,852 行。
- 验证：2026-01-01 至 2026-07-31，共 159,837 行。
- 原始 `Kronos-base` 分桶评估 loss：3.142226。
- 市值分层增训后分桶评估 loss：2.951122，改善 6.082%。
- 0–9 十个市值桶全部改善，单桶改善范围为 4.292%–8.165%。
- 最终 checkpoint：`outputs/models/a_share_size_kronos_base_earlystop50/checkpoints/best_model/model.safetensors`。
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
