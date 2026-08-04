# Kaggle V3 GPU Benchmark

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
["luckfu/kronos-a-share-v3-p100-long-b"]
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
