# Colab GPU 训练

本项目不把 178 MB 原始 CSV 或 112 MB `train_data.pkl` 放进 Git。Colab 会从 BaoStock 重建同一口径的真实 A 股面板，并把数据和 checkpoint 放在本地目录或 Google Drive。

## 直接使用 Colab 网页

在 Colab GPU Runtime 中执行：

```bash
!git clone https://github.com/luckfu/Kronos.git
%cd Kronos
!bash finetune/colab_bootstrap.sh
!bash finetune/colab_train.sh
```

默认使用 `luckfu/a-share-size-kronos-base-earlystop50` 作为训练起点、CUDA、离散市值桶、20,000 窗口/段、1 遍完整覆盖和 patience 5。

默认数据和输出在 `/content/kronos_runtime`。Colab Runtime 被回收后这些文件会消失，长训应先挂载 Google Drive：

```python
from google.colab import drive
drive.mount('/content/drive')
```

然后设置：

```bash
%env KRONOS_COLAB_ROOT=/content/drive/MyDrive/kronos_a_share
!bash finetune/colab_bootstrap.sh
!bash finetune/colab_train.sh
```

中断后恢复：

```bash
%env KRONOS_RESUME_TRAINING=1
!bash finetune/colab_train.sh
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
