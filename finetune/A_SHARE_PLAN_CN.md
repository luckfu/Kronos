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

也可以直接提供 `size_bucket`，取值范围为 `0..9`。每个 `symbol,date` 都应有一行。

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
self.num_sectors = 0
self.num_size_buckets = 10
self.pretrained_predictor_path = "./Kronos-base"
```

## 增训顺序

1. 加载已经学习过通用市场规律的 `Kronos-base`，冻结 tokenizer 和底部 10 层。
2. 在第 10 层注入市值 embedding，只训练顶部 2 层、预测头和市值 embedding。
3. 以 2026 年验证损失早停，并分别统计各市值桶的误差。
4. 使用 2026 年样本外数据做滚动回测，加入固定交易成本，并记录换手、空仓和退出信号；涨跌停、冲击成本和 point-in-time 成分仍待补齐。

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

脚本会自动选择 `mps`；CUDA 环境仍可使用 `torchrun` 多进程。

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
