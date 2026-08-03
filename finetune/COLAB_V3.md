# 全市场 V3 Colab 长训

V3 使用 2015–2025 年训练数据，2026 年只做时间外验证，并保留 240 只从未进入训练的股票作为整股票 holdout。本机只负责数据准备与短跑验证，正式完整覆盖在 Colab CUDA 上运行。

## 固定训练口径

- 基础模型：当前完整覆盖生产模型。
- 市值条件：离散十桶；V3 开始时清零旧 CSI800 相对桶 Embedding。
- 训练窗口：5,233,538 个；20,000 个/段；262 段完成一遍覆盖，默认完整遍历两遍，共 524 个覆盖分段。
- 早停：两遍完整覆盖后最多再观察 5 段，因此最多 529 段。
- Colab batch size：32；验证样本：4,000；默认 2 个 DataLoader worker。
- 输出目录：`a_share_size_full_market_v3_colab_bs32`，与本机 MPS 和旧 CSI800 任务隔离。
- 跨账号共享目录写 checkpoint 时，Drive 可能在当前账号 `MyDrive` 根目录生成 `last_state (N).pt` 和 `model (N).safetensors`。挂载盘删除只会把文件移入回收站，因此 V3 默认通过 Google Drive API 在保存前后永久删除根目录中匹配 `last_state[ (N)].pt` 与 `model[ (N)].safetensors` 的冲突副本。启动前必须在 Colab Cell 执行 `from google.colab import auth; auth.authenticate_user()`；API 预检不通过时训练拒绝启动，避免静默耗尽配额。

本机 MPS 的 `last_state.pt` 是 batch 16 的优化器和学习率调度状态，不复制到 Colab 新输出目录。Colab 从第 1 段开始，后续断线只恢复 Colab 自己产生的 batch 32 断点。

## 本机打包

```bash
cd /Users/fupengcheng/Documents/Kronos
bash finetune/package_a_share_v3_colab.sh
```

生成 `artifacts/kronos_a_share_v3_colab_data.tar.gz` 和对应的 `.sha256` 文件。包内只有处理后的训练、验证、holdout、元数据和清单，不含 1.2 GB 的原始 CSV/chunks。

把压缩包和 `.sha256` 一起上传到 Google Drive：

```text
MyDrive/kronos_a_share_v3/kronos_a_share_v3_colab_data.tar.gz
MyDrive/kronos_a_share_v3/kronos_a_share_v3_colab_data.tar.gz.sha256
```

## Colab GPU 终端

挂载 Drive 后，在 Colab Terminal 执行：

```bash
cd /content/drive/MyDrive
test -d Kronos/.git || git clone https://github.com/luckfu/Kronos.git Kronos
cd /content/drive/MyDrive/Kronos
export KRONOS_COLAB_V3_ROOT=/content/drive/MyDrive/kronos_a_share_v3
bash finetune/colab_v3_bootstrap.sh
bash finetune/colab_v3_train.sh
```

`colab_v3_bootstrap.sh` 先验证压缩包和包内逐文件 SHA-256，再解包数据、下载生产基础模型，并强校验股票数、行数、窗口数、日期边界、训练/holdout 零重叠和 CUDA。任一口径不一致都会退出，不会启动长训。

## 断线恢复

重新挂载 Drive、进入仓库后执行相同命令：

```bash
cd /content/drive/MyDrive/Kronos
export KRONOS_COLAB_V3_ROOT=/content/drive/MyDrive/kronos_a_share_v3
bash finetune/colab_v3_train.sh
```

脚本检测到同一输出目录的 `last_state.pt` 后自动恢复。`v3_run_config.env` 锁定 batch size、分段样本数、验证样本数和早停参数；恢复时配置不一致会直接报错，避免静默从错误 step 继续。
