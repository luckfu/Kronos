# Kaggle V5：120 日输入、10 日预测

本版本使用 Kaggle P100，不使用 Colab，也不训练 `V5-90-control`。训练基于当前 V4 B `last_model`，训练信号为 2015-01-01 至 2025-12-17，2026-01-05 至 2026-07-16 仅作时间验证；默认完整覆盖两遍。

## 上传输入

把以下两个文件作为同一个 Kaggle Dataset 的输入文件上传：

- `artifacts/kronos_a_share_v5_context_120d.tar.gz`
- `artifacts/a_share_v4_production_last.tar.gz`

也可以用 Kaggle CLI 创建/更新 Dataset。上传完成后，在 Kaggle Notebook 添加该 Dataset，文件会出现在 `/kaggle/input/<dataset-slug>/`。

## 运行

在 Notebook 的一个代码单元执行：

```python
!git clone https://github.com/luckfu/Kronos.git /kaggle/working/Kronos
%cd /kaggle/working/Kronos
!bash finetune/kaggle_v5_bootstrap.sh
!bash finetune/kaggle_v5_train.sh
```

P100 显存不足时，将 batch 调为 8；足够时可调为 32：

```python
import os
os.environ['KRONOS_BATCH_SIZE'] = '16'
```

## 中断和续跑

同一会话重新运行 `kaggle_v5_train.sh` 会从 `outputs/models/a_share_v5_context120_2pass/checkpoints/last_state.pt` 自动恢复。Kaggle 会话销毁后，先把该输出目录打包并发布为 Kaggle Dataset，再在新 Notebook 添加它；将旧输出复制到同一路径后再次运行，脚本会继续恢复。不要改变上下文长度、日期边界或输出名。

训练结束后，结果位于：

`/kaggle/working/kronos_a_share_v5_120d/outputs/models/a_share_v5_context120_2pass/`

其中 `checkpoints/best_model` 是验证集最佳权重，`checkpoints/last_model` 是第二遍完整覆盖后的生产候选，`checkpoints/last_state.pt` 用于续跑。训练进程正常结束后脚本会自动导出 `last_model`；中断时先保留 `last_state.pt`，下次续跑完成后再导出。
