# Kaggle V6/V6.01 训练补充说明

本文件只记录 V6/V6.01 的版本专属参数。所有 Kaggle 训练、续训、调参和验证任务必须先遵守 [`KAGGLE_RUNBOOK_CN.md`](KAGGLE_RUNBOOK_CN.md)；本文件不能覆盖总手册的并发、输出、接力和网络规则。

## 一、先定义运行目标

- V6 基线：`lookback=120`、`predict=10`。
- V6.01：`lookback=120`、`predict=5`，从生产 V6 Segment 568 的 `model.safetensors` 开始。
- V6.01 改变了监督目标，第一次启动必须使用新的 optimizer/scheduler；不能导入 V6 120→10 的 `last_state.pt`。
- V6.01 输出名固定为 `a_share_v6_01_context120_predict5_2pass`，不可复用 V6 输出目录。

## 二、Kaggle 并发规则

1. 同一个目标只能有一个 active Kernel。提交新版本前先查询 status。
2. 旧版本仍为 `RUNNING` 时，禁止 push 新版本；否则 Kaggle 会同时启动多个版本。
3. 需要替换运行版本时，先取消旧版本，确认状态变为 `CANCEL_ACKNOWLEDGED`、`COMPLETE` 或 `FAILED`，再提交新版本。
4. 页面显示多个版本时，只保留最新且配置正确的版本；旧版本即使正在下载或训练也必须停止。
5. `RUNNING` 只代表任务仍活着，不代表训练已完成；完成判定必须看输出文件和日志。

推荐检查命令：

```bash
kaggle kernels status luckfu/kronos-a-share-v6-01-context120-predict5
kaggle kernels logs luckfu/kronos-a-share-v6-01-context120-predict5
```

## 三、断点与 Best/Last 规则

每个可交接的训练 chunk 必须保留完整目录：

```text
outputs/models/a_share_v6_01_context120_predict5_2pass/
├── checkpoints/
│   ├── last_state.pt   # 模型、optimizer、scheduler、RNG、段/步位置
│   ├── best_model/     # 历史最佳验证模型，不能被 Last 覆盖
│   └── last_model/     # 最近完成状态的推理模型
├── metrics.jsonl
├── progress.json
└── summary.json
```

- 训练器每完成一个 segment 先保存 `last_state.pt`；验证集改善时才更新 `best_model`。
- 下一 chunk 必须同时恢复 `last_state.pt` 和已有 `best_model`。
- 如果发现 `last_state.pt` 存在但 `best_model/config.json` 或权重缺失，必须拒绝续训，不能把 Last 当成 Best。
- 训练脚本结束后必须显式运行 `finetune/export_last_model.py`，不能假设训练器会自动生成 `last_model`。
- 取消任务前应先等待当前安全 checkpoint；强制取消后，仍需检查输出中是否存在完整 Best/Last/State。

## 四、跨 Kernel 接力与网络

- Kaggle 的 `/kaggle/working` 不是永久盘；下一次 Kernel 不会自动看到上一次工作目录。
- 大模型、训练数据和 checkpoint 不得经本地电脑下载再上传，避免消耗本地网络。
- 优先使用 Kaggle 的 `kernel_sources` 挂载上一个 Kernel 的输出，由 Kaggle 服务端传递 checkpoint。
- 如果平台对 Kernel 输出依赖不提供所需目录结构，再使用 Kaggle Dataset 作为服务端中转；仍禁止本地下载/上传。
- 每次接力前必须在 metadata 中确认唯一的 `kernel_sources` 或 Dataset 输入，不能同时挂载多个候选 checkpoint。
- 接力 runner 必须从 `/kaggle/input` 找到完整 V6.01 输出树，并打印实际恢复路径；不能通过文件名模糊猜测。

## 五、启动前强制检查

- 当前没有同目标 active Kernel。
- Kernel title、id、code version 相互匹配。
- GPU 为 P100 时使用包含 `sm_60` 的 PyTorch；必要时安装 `torch==2.5.1` CUDA 12.1 wheel。
- 数据输入唯一，包含 `train_data.pkl` 和 `asset_metadata.csv`。
- 基础模型 SHA-256 为：

  ```text
  69999253b35afa641d001a5e77fd53be9b5c0beb8444abce1feb173b1f99d1e0
  ```

- 实际执行的训练脚本中必须是 `KRONOS_LOOKBACK_WINDOW=120 KRONOS_PREDICT_WINDOW=5`。
- 首次 V6.01 运行：`KRONOS_RESUME_TRAINING=0`；后续接力：只有发现完整 V6.01 checkpoint 时才设为 `1`。
- 每次最多运行的 segment 数必须事先记录；不要运行到 Kaggle 硬上限才发现没有可恢复输出。

## 六、监控与完成判定

- 日志顺序应为：环境准备 → 输入/模型准备 → `Coverage Segment x/568` → chunk limit 或 complete → export → 文件校验。
- 训练速度以已完成 segment 的实测平均值估算，不以页面显示的运行分钟数猜测。
- 一个 chunk 成功的必要条件是：`last_state.pt`、`best_model`、`last_model`、`progress.json`、`summary.json` 均存在。
- 只有最后一个 chunk 完成全部 `568` 段并通过文件校验，才称为 V6.01 训练完成。
- 训练完成后再做 July 2026 holdout 的 5 日/10 日评估；训练中途的 loss 不能替代样本外结论。

## 七、禁止事项

- 禁止同时启动两个同目标 Kernel 版本。
- 禁止在 active 版本上直接 push 新版本。
- 禁止只保存或只恢复 `last_state.pt`。
- 禁止用 V6 120→10 的 optimizer/scheduler 状态恢复 V6.01。
- 禁止让 Kaggle 运行时重复下载同一模型或 holdout。
- 禁止把 `RUNNING`、日志有输出或单个 loss 数字当成训练完成。
- 禁止为了修复脚本而取消一个已接近完成且 checkpoint 完整的 chunk；先判断是否能安全接力。
