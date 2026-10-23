# 时间切分 Rank IC 选型窗

`selection_window.json` 里的**日期和日历 SHA 已经冻结**（60 个上交所交易日，`2026-04-01` → `2026-06-30`）。`panel_sha256` 仍是空字符串：行情面板不进 git，构建脚本第一次写出 pkl 之后才会填上。不要手填一个假 SHA，也不要退回 symbol holdout。

启用 `temporal_rank_ic_v1` 时，缺这个文件、日历 SHA 对不上、`panel_sha256` 还不是 64 位十六进制、或面板指向 `val_data.pkl` / symbol-holdout，训练和 bigbatch 推送都会直接失败。val loss 继续打日志，**不**选 `best_model`。

## 启用

| 变量 | 含义 |
|---|---|
| `KRONOS_CHECKPOINT_SELECTION_PROTOCOL` | `temporal_rank_ic_v1` 才按 Rank IC 选模。默认空，旧 run 仍按 val loss |
| `KRONOS_TEMPORAL_RANK_IC_MANIFEST` | 本文件旁边的 `selection_window.json` |
| `KRONOS_TEMPORAL_RANK_IC_PANEL` | 选型窗行情面板（pickle：`symbol -> DataFrame`）。**不要**用训练集的 `val_data.pkl` |
| `KRONOS_TEMPORAL_RANK_IC_PANEL_DATASET` | 推送时挂上的面板 Kaggle dataset，`owner/slug`。不能是训练数据集 |
| `KRONOS_TEMPORAL_RANK_IC_INTERVAL_SEGMENTS` | 默认 **10**。与 `should_run_large_validation` 同一组触发点，另加强制本 run 第一段和墙钟/chunk 最后一段 |

bigbatch2 的 `finetune/small_clean_bigbatch_relay/launch.py` 已把协议设为 `temporal_rank_ic_v1`。`push_chunk.py` 在 manifest 缺失、不是合法 JSON、日期 SHA 不符、日期落进禁止窗口、或 `panel_sha256` 未钉住时拒绝打包。`kernel-metadata.json` 的 `dataset_sources` 同时挂上训练数据集和 `KRONOS_TEMPORAL_RANK_IC_PANEL_DATASET`。启动时在 `/kaggle/input` 里找唯一一份路径不含 symbol-holdout 的 `temporal_rank_ic_panel.pkl`，再核对面板 SHA。

## 构建面板并钉上 SHA

训练数据是 `smmt315/a-share-120d-temporal-symbol-holdout`（`launch.py` 的 `DATA_SHA` `b85ccc45…`）。`split.train_cutoff` 应为 `2024-12-31`；若挂载的 manifest 不是这个 cutoff，脚本会停。

在仓库根目录：

```bash
python finetune/build_temporal_rank_ic_panel.py \
  --dataset-root /path/to/a-share-120d-temporal-symbol-holdout \
  --output /path/to/temporal_rank_ic_panel.pkl
```

这一条命令会：

1. 核对 clean `data_manifest.json` 的 SHA 和 `train_cutoff`。
2. 读取 `train_data.pkl` 与 `val_data.pkl`（holdout 股票在 cutoff 之后的 bar 只在验证 pickle 里），按股票拼接后裁到冻结信号日所需的 lookback / D10 标签。
3. 把 `sector_id` 和 `size_percentile` 写进每一行。行业编号与 `finetune/asset_metadata.py` 相同：`asset_metadata.csv` 里的行业字符串排序后从 0 编号，未知为 86。评分时不再 join 这份 metadata。
4. 确认每个冻结日期至少 30 只有效股票。
5. 原子更新本目录 `selection_window.json` 的 `panel_sha256`。`signal_dates` 和日历 `sha256` 不动。

然后把 pkl 上传成**单独**的 Kaggle dataset（数据集 slug 不要含 `symbol-holdout`，里面只放这一份 `temporal_rank_ic_panel.pkl`），提交更新后的 `panel_sha256`，再：

```bash
export KRONOS_TEMPORAL_RANK_IC_PANEL_DATASET=<owner>/<panel-dataset>
python -u finetune/small_clean_bigbatch_relay/push_chunk.py 1
```

`panel_sha256` 为空时不要推。e01 已取消。

## manifest 字段

```json
{
  "protocol_id": "temporal_rank_ic_v1",
  "split_kind": "temporal",
  "horizon": "D10",
  "n_paths": 16,
  "path_aggregation": "median",
  "temperature": 0.65,
  "top_p": 0.8,
  "top_k": 0,
  "path_seed": 20260906,
  "lookback": 120,
  "predict": 10,
  "train_time_upper_bound": "2024-12-31",
  "signal_dates": ["YYYY-MM-DD"],
  "sha256": "<canonical date-list sha256>",
  "panel_sha256": "<empty until the builder runs>",
  "panel_filename": "temporal_rank_ic_panel.pkl"
}
```

- `signal_dates` 必须唯一且升序。`sha256` = 这些日期用 `\n` 连接、末尾再加一个 `\n`、UTF-8 的 SHA256。加载时重算，不一致就失败，不会悄悄改日期。当前冻结值是 `0721b3defca10d2dab4dcaf433f1efab12304340298e6304fca6d3c9a9f3ce0d`（60 日）。
- `train_time_upper_bound` 必须 **严格早于** 最早 signal date。这里钉的是 clean split 的 `train_cutoff`（`2024-12-31`），不是训练股票 `train_data.pkl` 的最后一根 bar。预清洗报告里那些股票的 bar 一直到 `2026-07-17`。这 60 天因此晚于 cutoff，但仍和训练股票的行情日历重叠。协议说明见 `docs/VALIDATION_RANK_IC_SELECTION_PROTOCOL_CN.md` §9。
- 禁止放进选型窗：34 日包 `2026-08-03→2026-09-17`、sealed 8 日 `2026-09-18→2026-09-30`、§5.2 `2026-10-09→2026-10-23`。
- 日历来自上证公告〔2025〕45号，实现在 `finetune/ashare_sse_calendar_2026.py`。不要用 pandas 工作日代替：那会把清明周一、劳动节和端午周五算进去。
- 面板每个 symbol 的列要自带 `open/high/low/close/volume/amount` 以及 `sector_id`、`size_percentile`。代码不会在评分时去 join symbol-holdout 的 `asset_metadata.csv`。
- 每个冻结日期至少 30 只有效股票，否则该次评测失败，不会丢掉日期再算平均。
- 生成超参冻结：T=0.65，top_p=0.8，top_k=0，seed `20260906`，16 条路径，D10 收益取路径中位数。推理 batch 固定为 4（改 batch 会改 RNG 对齐，不提供扫描开关）。评分只在这次前向里设 seed，结束时恢复训练 RNG 和 train/eval 模式，避免 rank 0 的下一段和其它 rank 分叉。

日期 SHA 用同一实现计算：

```python
from temporal_rank_ic_selection import manifest_sha256
from ashare_sse_calendar_2026 import selection_window_sessions
dates = selection_window_sessions()
print(len(dates), manifest_sha256(dates))
```

## 代码入口

- 选模规则与 manifest 加载：`finetune/temporal_rank_ic_selection.py`
- 交易日：`finetune/ashare_sse_calendar_2026.py`
- 面板构建：`finetune/build_temporal_rank_ic_panel.py`
- 训练钩子：`finetune/train_predictor.py`（`score_temporal_rank_ic_segment`）
- 配置：`finetune/config.py` 的 `KRONOS_CHECKPOINT_SELECTION_PROTOCOL`
- bigbatch2 启动默认打开：`finetune/small_clean_bigbatch_relay/launch.py`
- 测试：`tests/test_temporal_rank_ic_selection.py`、`tests/test_build_temporal_rank_ic_panel.py`

协议正文：`docs/VALIDATION_RANK_IC_SELECTION_PROTOCOL_CN.md`。
