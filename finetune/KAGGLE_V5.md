# Kaggle V5：120 日输入、10 日预测

> 所有 Kaggle 任务必须先遵守 [`KAGGLE_RUNBOOK_CN.md`](KAGGLE_RUNBOOK_CN.md) 的统一 preflight、单任务并发、输出完整性和服务端接力规则。

本版本使用 Kaggle P100，不使用 Colab，也不训练 `V5-90-control`。训练基于当前 V4 B `last_model`，训练信号覆盖 2015-01-01 至 2026-07-16（原始行情保留到 2026-07-31 以提供完整的 10 日标签）。验证池与训练池相同，由训练器随机抽样；真正的时间外和股票外评估在训练完成后单独执行。默认完整覆盖两遍。

## 上传输入

把以下两个文件作为同一个 Kaggle Dataset 的输入文件上传：

- `artifacts/kronos_a_share_v5_context120_train2015_2026.tar.gz`
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

队友可以接管训练：把包含 `checkpoints/last_state.pt`、`progress.json` 和 `config.json` 的输出目录发布为共享 Kaggle Dataset，并让队友把该 Dataset 添加到同一个 Kernel。`kaggle_v5_bootstrap.sh` 会自动寻找输入中的 `checkpoints/last_state.pt`，导入到标准输出目录；随后 `kaggle_v5_train.sh` 会自动设置恢复并继续训练。队友必须使用同一份 V5 数据集、同一基座和相同的 Kernel 输出名。

训练结束后，结果位于：

`/kaggle/working/kronos_a_share_v5_120d/outputs/models/a_share_v5_context120_2pass/`

其中 `checkpoints/best_model` 是训练池随机验证 loss 最佳权重，`checkpoints/last_model` 是完整覆盖后的生产候选，`checkpoints/last_state.pt` 用于续跑。训练进程正常结束后脚本会自动导出 `last_model`；中断时先保留 `last_state.pt`，下次续跑完成后再导出。

V5 使用原版全序列 next-token Loss，运行期间不改变。后续 V6 才切换为未来 10 日 `forecast_loss` 主目标；具体约束见 `A_SHARE_PLAN_CN.md`。

V6 还会先做 batch=32 的单段显存短测，依据 P100 的实际 allocated/reserved 峰值选择 batch 32、24 或 16，不在未测显存的情况下直接长训。
