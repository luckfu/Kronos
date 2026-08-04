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

## 长训注意事项

三段 benchmark 的 checkpoint 放在 `/kaggle/working` 没问题；Kaggle 运行时结束后该目录会被清空。正式两轮训练前必须增加 checkpoint 持久化方式（Kaggle Dataset 新版本或外部对象存储），不能只依赖本地工作目录。

正式长训参数应为：

```bash
export KRONOS_KAGGLE_ROOT=/kaggle/working/kronos_a_share_v3
export KRONOS_KAGGLE_DATASET_INPUT=/kaggle/input/kronos-train-set-a
export KRONOS_PREDICTOR_SAVE_FOLDER=a_share_size_full_market_v3_kaggle
export KRONOS_COVERAGE_PASSES=2
export KRONOS_REQUIRE_FULL_COVERAGE=1
export KRONOS_EARLY_STOPPING_PATIENCE=5
```

不要把 Kaggle benchmark 的输出目录与 Colab 主训练目录混用，也不要让 Kaggle 和 Colab 同时写同一个 checkpoint。
