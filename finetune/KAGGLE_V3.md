# Kaggle V3 GPU Benchmark

## V4 连续上下文 A/B 实验

2026 增量数据审计确认旧流水线先把面板裁到 2026 年，再生成 `90 + 10 + 1` 行窗口。90 日观察期因此吞掉了年初行情，89,730 个训练窗口实际只覆盖 `2026-05-22` 至 `2026-07-16` 的 39 个信号日；训练集未来十日均价收益为负的比例是 `65.83%`，同期 symbol holdout 验证集为 `72.56%`。这解释了旧增量模型的系统性偏跌。

V4 不再按文件日期裁掉上下文。加载器合并 2015-2025 `train_data.pkl` 与 2026 `val_data.pkl`，保留连续行情，并根据窗口第 90 行的信号日期筛选：

- 近期训练：`2026-01-05` 至 `2026-06-17`，108 个信号日、249,280 个窗口。
- 时间验证：`2026-06-18` 至 `2026-07-16`，20 个信号日、46,129 个窗口。
- A 组：仅使用修正后的近期训练窗口。
- B 组：同一批近期窗口加 62,320 个历史回放窗口，总计 311,600；回放占 20%，按年份和市值桶分层无放回抽取。
- 两组都从增训前 V3 Last 开始，predictor 学习率 `1e-6`、市值层学习率 `1e-4`，完整覆盖一遍。

本机严格校验但不启动训练：

```bash
PYTHONPATH=. python finetune/verify_a_share_v4_data.py --strict
```

Kaggle 直接复用已有的 `luckfu/kronos-train-set-a` 完整数据和 `luckfu/kronos-a-share-2026-incremental` 中保存的增训前 V3 Last，不需要重新上传大文件：

```bash
bash finetune/prepare_kaggle_v4_ab_kernel.sh
kaggle kernels push -p artifacts/kaggle_v4_ab_kernel --accelerator P100
kaggle kernels status luckfu/kronos-a-share-v4-ab-training
kaggle kernels output luckfu/kronos-a-share-v4-ab-training \
  -p artifacts/kaggle_v4_ab_output
```

Kernel 会依次训练 `a_share_v4_corrected_2026_recent_only` 和 `a_share_v4_corrected_2026_replay20`，然后在固定股票 holdout 的最后 20 个时间外信号日上生成 A/B 预测、Rank IC、方向准确率和 MAE。输出同时保留两个 `best_model`、两个 `last_state.pt` 和 `outputs/evaluation/v4_recent_vs_replay20`。

## 2026 增量训练

完成 V3 全市场两轮覆盖后，2026 增量任务使用独立流水线：

- 起点是完整 V3 `last_state.pt` 中的模型权重，不加载旧 optimizer 或 scheduler。
- 训练集为 2026-01-05 至 2026-07-31 的 2,312 只非 holdout 股票，共 89,730 个窗口。
- 验证集为同期 240 只 symbol holdout 股票，共 9,307 个窗口；它们不参与梯度更新。
- 完整覆盖两轮，共 179,460 个窗口暴露，20,000 窗口一段，总计 10 段。
- predictor 学习率为 `2e-6`，市值条件层学习率为 `2e-4`。
- 保留 V3 已学到的市值桶权重，即 `KRONOS_RESET_SIZE_EMBEDDING=0`。
- 训练结束同时保留 `best_model` 和 `last_state.pt`。生产晋级以当前市场的 Rank IC、方向准确率和交易收益为主，同时检查历史回放集上的退化程度并保留旧模型回滚能力。

### 滚动增训数据配比

后续增训不能只使用最近数据。默认按窗口数使用 `80%` 近期样本和 `20%` 历史回放样本；根据行情切换速度和遗忘程度，可在 `70%-90%` 近期、`10%-30%` 历史的范围内调整。历史数据总量远大于近期数据，必须按目标比例下采样，不能直接拼接后让历史样本重新占据多数。

历史回放样本需要分层抽取，至少覆盖牛市、熊市、震荡市、高低波动阶段和十个市值桶，避免某一年、某种行情或某个市值层垄断回放集。近期样本与历史回放样本在每轮开始前统一打乱，并继续使用无放回覆盖顺序。

数据切分必须同时满足：

- 股票隔离：保留固定 symbol holdout，检查对未参与训练股票的横截面泛化。
- 时间隔离：保留最近至少 20 个已完整出标签的交易日，完全不参与训练，用于真正的未来日期验证。
- 标签隔离：预测窗口为 10 日，因此训练截止日必须给未来 10 个交易日标签留足空间，不能把尚未完整实现的收益混进训练或验证。
- 生产留痕：每天保存预测、实际未来十日均价收益和模型版本；每次晋级都保留上一生产 checkpoint 作为一键回滚版本。

滚动模型的主要目标是适应当前市场。晋级优先比较当前时间外窗口的 Rank IC、方向准确率、头尾收益差及含成本策略收益；MAE 作为价格校准辅助指标。历史回放集退化用于衡量灾难性遗忘、调整回放比例或触发回滚，不再要求候选模型在每一个旧年份都优于旧模型。

本机准备并上传私有数据集：

```bash
bash finetune/package_a_share_2026_incremental.sh
kaggle datasets create -p artifacts/kaggle_2026_incremental_dataset
```

若数据集已存在，使用 `kaggle datasets version`。提交 Kernel：

```bash
mkdir -p artifacts/kaggle_2026_incremental_kernel
cp finetune/kaggle_2026_incremental.py artifacts/kaggle_2026_incremental_kernel/
cp finetune/kaggle_2026_incremental-metadata.json \
  artifacts/kaggle_2026_incremental_kernel/kernel-metadata.json
kaggle kernels push -p artifacts/kaggle_2026_incremental_kernel
```

Kernel 使用新 optimizer 和 OneCycle 调度器从 V3 Last 权重开始，不应设置 `KRONOS_RESUME_TRAINING=1`。只有同一个 2026 增量任务被 Kaggle 中断后，才允许从该任务自己的 `last_state.pt` 恢复。

### 2026-08-06 实验结果

Kaggle Kernel `luckfu/kronos-a-share-2026-incremental-training` 在 P100 上完成 `10/10` 段，实际运行 23 分 21 秒：

- 两轮完整覆盖 179,460 个窗口，`next_epoch=10`、`resume_step=0`。
- 最低验证 loss 为 `2.638070556640625`，出现在最后一段，因此 Best 和 Last 的 193 个张量完全相同。
- Best `model.safetensors` SHA-256：`3ae659b1b5a117bc3f23987ad8977a8f5b7539a9e22331e82f9c483c47368ce0`。
- Last `last_state.pt` SHA-256：`ca77949b86fe592a1d537d4c19b7baa80b23be5b967d93806c1a45fd08cf0d41`。

使用 240 只从未参与训练的 symbol holdout 股票、24 个信号日、未来 10 日预测收盘均值信号，对比完整 V3 Last 与 2026 增量 Last：

| 窗口 | 模型 | Rank IC | 方向准确率 | MAE |
| --- | --- | ---: | ---: | ---: |
| 2025 | 完整 V3 Last | 0.1509 | 61.0% | 0.0385 |
| 2025 | 2026 增量 Last | 0.1047 | 54.7% | 0.0504 |
| 2026 | 完整 V3 Last | 0.0241 | 52.8% | 0.0519 |
| 2026 | 2026 增量 Last | 0.0705 | 57.1% | 0.0566 |

2026 增量模型在 2026 同期未见股票上的 Rank IC 和方向准确率显著改善，适合作为面向当前市场的生产排序模型；MAE 显著恶化，因此不应把它当作绝对价格预测升级。2025 三项指标退化说明纯近期数据增训造成了历史能力遗忘，也直接形成后续规则：滚动增训必须混入 `10%-30%` 的分层历史回放样本，并使用增训截止日之后的数据继续做时间外监控。这里的 2026 结果是同日期、跨股票验证，不等同于对 2026-08 以后日期的验证。

Kaggle 数据集已经上传：[`luckfu/kronos-train-set-a`](https://www.kaggle.com/datasets/luckfu/kronos-train-set-a)。Kaggle 可能保留 tar 文件，也可能自动展开为：

```text
kronos_a_share_v3_colab_data/data/a_share_v3/processed_datasets/train_data.pkl
kronos_a_share_v3_colab_data/data/a_share_v3/processed_datasets/val_data.pkl
kronos_a_share_v3_colab_data/data/a_share_v3/processed_datasets/symbol_holdout_data.pkl
```

bootstrap 会自动识别两种布局。数据只包含处理后的训练、验证、holdout 和元数据，不包含原始 BaoStock CSV。

## 三段 P100 基准

在 Kaggle Notebook 中添加 `luckfu/kronos-train-set-a` 数据集，打开 GPU 和 Internet，并在硬件设置中选择 P100（如果当前账号有该资源）。Kaggle 不保证每次都分配 P100，先检查输出的 GPU 名称。

Kaggle 当前预装的 PyTorch 可能是 `2.10+cu128`，该 wheel 不包含 P100 的 `sm_60` CUDA kernel。bootstrap 检测到 P100 且当前 wheel 不兼容时，会自动安装 `torch==2.5.1` 的 CUDA 12.1 wheel；日志中看到安装提示是正常的。

Notebook 中执行：

```bash
!git clone https://github.com/luckfu/Kronos.git /kaggle/working/Kronos
!cd /kaggle/working/Kronos && bash finetune/kaggle_v3_bootstrap.sh
!cd /kaggle/working/Kronos && bash finetune/kaggle_v3_train.sh
```

或者使用仓库里的 `kaggle_v3_benchmark.py` 作为 Kaggle Script Kernel。它会自动 clone 当前 GitHub `master`、校验数据、下载生产基础模型，并运行 3 个 coverage segment。

正确启动输出应包含：

```text
GPU: Tesla P100-PCIE-16GB
Found 5233538 possible samples
Coverage plan: ... 3 segments
```

如果显示 T4、CPU 或其他 GPU，先不要拿它与 Colab T4 的速度作结论。记录每段的 `Time This Epoch`，并比较同样的 batch size `32`、20,000 窗口/段和 4,000 验证样本。

## 2026-08-04 实测结果

私有 Kernel `luckfu/kronos-a-share-v3-p100-benchmark` 使用 `Tesla P100-PCIE-16GB` 和 `torch 2.5.1+cu121` 完成三段：

| Segment | 耗时 | Validation Loss |
|---:|---:|---:|
| 1 | 2:05 | 2.8433 |
| 2 | 2:04 | 2.8420 |
| 3 | 2:04 | 2.8418 |

平均每段约 124 秒；同口径 Colab T4 首段约 261 秒，因此 P100 吞吐约为 T4 的 2.1 倍，耗时减少约 52%。按 529 段线性估算，纯训练约 18.3 小时，另加每个 Kaggle 会话安装兼容 PyTorch、下载基础模型和保存输出的时间。

三段 benchmark 使用三段长度的 OneCycle 学习率调度，只用于比较吞吐，不用于判断模型质量，也不能与 529 段正式调度的早期 Loss 直接比较。

## Kaggle CLI

本机安装并登录 Kaggle CLI：

```bash
python -m pip install -U kaggle
export KAGGLE_API_TOKEN='不要把 token 写入仓库'
```

Kernel metadata 模板在 `finetune/kaggle_v3_kernel-metadata.example.json`。提交前把它复制到一个 Kernel staging 目录，把 `code_file` 和 `id` 改成自己的配置，然后运行：

```bash
bash finetune/prepare_kaggle_v3_kernel.sh
kaggle kernels push -p artifacts/kaggle_v3_kernel --accelerator P100
kaggle kernels status luckfu/kronos-a-share-v3-p100-benchmark
kaggle kernels output luckfu/kronos-a-share-v3-p100-benchmark -p ./kaggle-output
```

`--accelerator P100` 是显式硬件请求；若当前账号没有 P100 配额或该资源不可用，不能用其他 GPU 的结果冒充 P100 benchmark。

## 两轮正式长训

正式训练总计划固定为 529 段：每遍 262 段，两遍共 524 段，再保留 5 段早停观察。`KRONOS_MAX_SEGMENTS_PER_RUN=180` 只让单次 Kaggle 会话在完整 segment 验证并保存 `last_state.pt` 后安全退出，不会把 OneCycle 学习率计划缩短成 180 段。三次预计分别运行 180、180、169 段。

Kaggle 每次只保留本次 Kernel 输出，所以使用 A/B 两个私有 Kernel 交替承接 checkpoint：

1. A 第一次从基础模型训练，输出 segment 180 的 checkpoint。
2. B 把 A 的 Kernel 输出作为只读输入，复制 checkpoint 后续跑到 segment 360。
3. 再更新 A，使它读取 B 的输出，完成到 segment 529。

准备两个 Kernel 目录：

```bash
bash finetune/prepare_kaggle_v3_kernel.sh
```

首次提交 A：

```bash
kaggle kernels push -p artifacts/kaggle_v3_long_a --accelerator P100
```

A 完成后提交 B：

```bash
kaggle kernels push -p artifacts/kaggle_v3_long_b --accelerator P100
```

第三次运行前，把 `artifacts/kaggle_v3_long_a/kernel-metadata.json` 的 `kernel_sources` 改为：

```json
["luckfu/kronos-a-share-v3-p100-long-training-b"]
```

然后再次提交 A。入口会自动寻找属于正式输出名的 `last_state.pt`；没找到就从第 1 段开始，找到多个则直接报错，避免误续跑。每次日志必须确认 `Resumed training at coverage segment ...` 与预期一致。

正式长训参数应为：

```bash
export KRONOS_KAGGLE_ROOT=/kaggle/working/kronos_a_share_v3
export KRONOS_KAGGLE_DATASET_INPUT=/kaggle/input/kronos-train-set-a
export KRONOS_PREDICTOR_SAVE_FOLDER=a_share_size_full_market_v3_kaggle
export KRONOS_COVERAGE_PASSES=2
export KRONOS_REQUIRE_FULL_COVERAGE=1
export KRONOS_EARLY_STOPPING_PATIENCE=5
export KRONOS_MAX_SEGMENTS_PER_RUN=180
```

不要把 Kaggle benchmark 的输出目录与 Colab 主训练目录混用，也不要让 Kaggle 和 Colab 同时写同一个 checkpoint。
