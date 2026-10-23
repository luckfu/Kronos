# 时间切分 Rank IC 选型窗（占位）

`selection_window.json` **故意不在仓库里**。协议要求选型日历在接入训练前冻结为「日期列表 + SHA」；日期还没冻结时，不能编一份临时日期充数，也不能退回 symbol holdout。

启用 `temporal_rank_ic_v1` 时，缺这个文件、SHA 对不上、或面板指向 `val_data.pkl` / symbol-holdout，训练和 bigbatch 推送都会直接失败。val loss 继续打日志，**不**选 `best_model`。

## 启用

| 变量 | 含义 |
|---|---|
| `KRONOS_CHECKPOINT_SELECTION_PROTOCOL` | `temporal_rank_ic_v1` 才按 Rank IC 选模。默认空，旧 run 仍按 val loss |
| `KRONOS_TEMPORAL_RANK_IC_MANIFEST` | 本文件旁边的 `selection_window.json` |
| `KRONOS_TEMPORAL_RANK_IC_PANEL` | 选型窗行情面板（pickle：`symbol -> DataFrame`）。**不要**用训练集的 `val_data.pkl` |
| `KRONOS_TEMPORAL_RANK_IC_INTERVAL_SEGMENTS` | 默认 **10**。与 `should_run_large_validation` 同一组触发点，另加强制本 run 第一段和墙钟/chunk 最后一段 |

bigbatch2 的 `finetune/small_clean_bigbatch_relay/launch.py` 已把协议设为 `temporal_rank_ic_v1`。`push_chunk.py` 在 manifest 缺失、不是合法 JSON、日期 SHA 不符、或日期落进禁止窗口时拒绝打包。面板字节不在仓库里，面板 SHA 到启动时再对。

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
  "train_time_upper_bound": "YYYY-MM-DD",
  "signal_dates": ["YYYY-MM-DD"],
  "sha256": "<canonical date-list sha256>",
  "panel_sha256": "<panel file sha256>",
  "panel_filename": "temporal_rank_ic_panel.pkl"
}
```

- `signal_dates` 必须唯一且升序。`sha256` = 这些日期用 `\n` 连接、末尾再加一个 `\n`、UTF-8 的 SHA256。加载时重算，不一致就失败，不会悄悄改日期。
- `train_time_upper_bound` 必须 **严格早于** 最早 signal date。训练样本的时间上界也必须早于这个窗。
- 禁止放进选型窗：34 日包 `2026-08-03→2026-09-17`、sealed 8 日 `2026-09-18→2026-09-30`、§5.2 `2026-10-09→2026-10-23`。
- 面板每个 symbol 的列要自带 `open/high/low/close/volume/amount` 以及 `sector_id`、`size_percentile`。代码不会去 join symbol-holdout 的 `asset_metadata.csv`。
- 每个冻结日期至少 30 只有效股票，否则该次评测失败，不会丢掉日期再算平均。
- 生成超参冻结：T=0.65，top_p=0.8，top_k=0，seed `20260906`，16 条路径，D10 收益取路径中位数。推理 batch 固定为 4（改 batch 会改 RNG 对齐，不提供扫描开关）。评分只在这次前向里设 seed，结束时恢复训练 RNG 和 train/eval 模式，避免 rank 0 的下一段和其它 rank 分叉。

日期 SHA 用同一实现计算：

```python
from temporal_rank_ic_selection import manifest_sha256
print(manifest_sha256(["2026-07-06", "2026-07-07"]))
```

## 代码入口

- 选模规则与 manifest 加载：`finetune/temporal_rank_ic_selection.py`
- 训练钩子：`finetune/train_predictor.py`（`score_temporal_rank_ic_segment`）
- 配置：`finetune/config.py` 的 `KRONOS_CHECKPOINT_SELECTION_PROTOCOL`
- bigbatch2 启动默认打开：`finetune/small_clean_bigbatch_relay/launch.py`
- 测试：`tests/test_temporal_rank_ic_selection.py`

协议正文：`docs/VALIDATION_RANK_IC_SELECTION_PROTOCOL_CN.md`。
