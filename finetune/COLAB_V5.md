# Colab V5：120 日上下文训练

V5 只有一个候选口径：当前生产 `last_model` 增训，输入 120 个交易日，预测未来 10 个交易日。不运行单独的 `V5-90-control`。现有生产页面仍使用 90 日，直到 V5 完成时间外审核。

## 1. 准备数据包

2014 年行情只用于给 2015 年初的窗口补上下文，监督信号从 2015 年开始。下载必须使用独立的 chunk 目录，不能复用旧的 2015+ chunks：

```bash
python finetune/download_a_share_parallel.py \
  --symbols-file data/a_share_v3/universe_manifest.csv \
  --start 2014-01-01 --end 2014-12-31 \
  --output data/a_share_v5/a_share_2014_context.csv \
  --chunk-dir data/a_share_v5/chunks_2014_context \
  --workers 4
```

如果 BaoStock 出现网络错误，使用新的 `chunks_2014_retry` 对缺失股票重试，然后合并 2014 文件。合并前必须确认重复键和冲突值为零。之后生成 V5 面板：

```bash
python finetune/prepare_a_share_context.py \
  --raw-input data/a_share_v3/a_share_daily_parallel.csv \
  --raw-input data/a_share_v5/a_share_2014_context_merged.csv \
  --universe-manifest data/a_share_v3/universe_manifest.csv \
  --output-root data/a_share_v5 \
  --lookback 120 --predict 10 --min-2014-trading-days 120 \
  --train-start 2015-01-01 --train-end 2025-12-17 \
  --val-start 2026-01-05 --val-end 2026-07-16 \
  --holdout-start 2014-01-01 --holdout-end 2026-07-31

bash finetune/package_a_share_context.sh

# 另外打包当前本机前端实际使用的 V4 B last_model
bash finetune/package_a_share_v5_base.sh
```

`v5_context_summary.json`、`context_coverage_manifest.csv` 和包内 SHA-256 是训练前检查依据。包至少要包含 120 个 2014 交易日；每个样本窗口为 `120 + 10 + 1 = 131` 行。不要把 2014 原始 CSV 或旧 V3 处理结果混进 V5 包。

## 2. Kaggle 运行

本版本正式训练使用 Kaggle P100，不使用 Colab。请参阅 [`KAGGLE_V5.md`](KAGGLE_V5.md)；下面的旧 Colab 命令仅作为历史备份，不是当前入口。

把 `kronos_a_share_v5_context_120d.tar.gz`、它的 `.sha256`、`a_share_v4_production_last.tar.gz` 和它的 `.sha256` 放到 Drive 的运行目录，然后在仓库根目录执行。V5 默认要求这个 V4 B `last_model`，不会把当前 ModelScope 上仍为 90 日的旧基座误当成生产基座。第一步可以在 CPU runtime 完成解包和校验：

```bash
cd /content/drive/MyDrive/Kronos
KRONOS_COLAB_V5_ROOT=/content/drive/MyDrive/kronos_a_share_v5 \
  bash finetune/colab_v5_bootstrap.sh
```

切换到 GPU runtime 后，重新挂载 Drive，执行训练：

```bash
cd /content/drive/MyDrive/Kronos
KRONOS_COLAB_V5_ROOT=/content/drive/MyDrive/kronos_a_share_v5 \
  bash finetune/colab_v5_train.sh
```

默认训练信号为 2015-01-01 至 2025-12-17，2026-01-05 至 2026-07-16 为时间验证。默认两遍完整覆盖、batch size 16；显存足够时可在启动前设置 `KRONOS_BATCH_SIZE=32`。训练输出会保存 `best_model`、`last_state.pt` 和最终 `last_model` 导出所需的状态。

## 3. 中断后续跑

使用同一个输出目录再次执行同一条 `colab_v5_train.sh` 即可自动检测 `last_state.pt` 并恢复。脚本锁定上下文长度、信号范围、batch 和覆盖遍数；不要在恢复时修改这些参数。若要另开实验，换 `KRONOS_PREDICTOR_SAVE_FOLDER`，不要覆盖旧目录。

V5 的 2014 行不会成为训练目标：训练和验证脚本都设置了 `KRONOS_*_SIGNAL_START/END`，数据面板只保留必要的上下文和标签行。
