# Colab GPU 训练

本项目不把 178 MB 原始 CSV 或 112 MB `train_data.pkl` 放进 Git。Colab 会从 BaoStock 重建同一口径的真实 A 股面板，并把数据和 checkpoint 放在本地目录或 Google Drive。

## Colab 终端使用

以下命令均在 Colab Terminal 中执行，不使用 Notebook 的 `!` 或 `%` 语法。

在 GPU Runtime 中直接训练：

```shell
cd /content/drive/MyDrive
test -d Kronos/.git || git clone https://github.com/luckfu/Kronos.git Kronos
cd /content/drive/MyDrive/Kronos
export KRONOS_COLAB_ROOT=/content/drive/MyDrive/kronos_a_share
bash finetune/colab_bootstrap.sh
bash finetune/colab_train.sh
```

默认使用 `luckfu/a-share-size-kronos-base-earlystop50` 作为训练起点、CUDA、离散市值桶、20,000 窗口/段、1 遍完整覆盖和 patience 5。

## 先用 CPU 准备数据，再用 GPU 训练

可以先选择 CPU Runtime 下载和处理数据，再切换到 GPU Runtime。一定要把运行目录放在 Google Drive；切换 Runtime 会清空 `/content` 下的临时文件。

CPU Runtime 的终端中执行：

```shell
cd /content/drive/MyDrive
test -d Kronos/.git || git clone https://github.com/luckfu/Kronos.git Kronos
cd /content/drive/MyDrive/Kronos
export KRONOS_COLAB_ROOT=/content/drive/MyDrive/kronos_a_share
export KRONOS_PREPARE_ONLY=1
bash finetune/colab_bootstrap.sh
```

这一步只下载 CSI800 BaoStock 数据并生成处理后的数据集，不启动训练，也不要求 CUDA。完成后把 Runtime 切换为 GPU，并重新挂载 Drive、恢复仓库：

```shell
cd /content/drive/MyDrive/Kronos
export KRONOS_COLAB_ROOT=/content/drive/MyDrive/kronos_a_share
export KRONOS_PREPARE_ONLY=0
bash finetune/colab_bootstrap.sh
bash finetune/colab_train.sh
```

GPU 阶段会发现 Drive 中已有处理好的数据，直接复用，只下载基础模型并开始训练。

如果当前终端还没有挂载 Google Drive，需要先在 Colab 的 Python 运行环境中挂载；挂载完成后，终端里应能看到 `/content/drive/MyDrive`。如果不使用 Drive，则把 `KRONOS_COLAB_ROOT` 改成 `/content/kronos_runtime`，但切换 Runtime 后数据不会保留。

默认数据和输出在 `/content/kronos_runtime`。Colab Runtime 被回收后这些文件会消失；使用 Drive 时，把 `KRONOS_COLAB_ROOT` 指向 `/content/drive/MyDrive/kronos_a_share`。

训练中断后，在终端恢复：

```shell
cd /content/drive/MyDrive/Kronos
export KRONOS_COLAB_ROOT=/content/drive/MyDrive/kronos_a_share
export KRONOS_RESUME_TRAINING=1
bash finetune/colab_train.sh
```

## 使用 google-colab-cli

本地安装 [google-colab-cli](https://github.com/googlecolab/google-colab-cli) 并完成 OAuth 后：

```bash
colab new -s kronos-train --gpu L4
colab drivemount -s kronos-train /content/drive
colab exec -s kronos-train --timeout 1800 -f finetune/colab_remote.py
```

上述命令会在远端 clone 当前仓库、重建数据、下载生产基础模型，并在独立进程中启动训练。挂载 Drive 后，默认持久化目录是 `/content/drive/MyDrive/kronos_a_share`；未挂载时使用 `/content/kronos_runtime`。

查看训练日志：

```bash
echo "print(open('/content/drive/MyDrive/kronos_a_share/colab-training.log').read()[-12000:])" \
  | colab exec -s kronos-train
```

`colab status -s kronos-train` 查看硬件和会话状态。远端脚本检测到 `last_state.pt` 时会自动设置恢复模式；完成后可从 Drive 直接读取 `best_model`，或用 `colab download` 取回文件。

## 选用原始基础模型

默认从当前生产模型继续增训。如果要从未增训的 `NeoQuasar/Kronos-base` 开始：

```bash
export KRONOS_COLAB_BASE_MODEL=original
bash finetune/colab_bootstrap.sh
bash finetune/colab_train.sh
```

## 参数覆盖

```bash
KRONOS_BATCH_SIZE=8 \
KRONOS_TRAIN_SAMPLES_PER_SEGMENT=20000 \
KRONOS_COVERAGE_PASSES=1 \
KRONOS_PREDICTOR_SAVE_FOLDER=a_share_colab_experiment_v1 \
bash finetune/colab_train.sh
```

`verify_colab_setup.py` 会在训练前检查 CUDA、基础权重、训练/验证文件、行数和窗口数。默认要求训练集至少有 1,000,000 个窗口，避免把不完整下载误当成正式训练。
