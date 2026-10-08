# TimesFM-3 训练看板

更新时间：2026-10-09 CST

## 当前任务

| 项目 | 状态 |
|---|---|
| Kernel | `wynstonliu/kronos-timesfm3-lora-finetune-v7` |
| Version | `1` |
| Kaggle 状态 | `RUNNING`，不是最终完成 |
| GPU | 双 Tesla T4，实时日志已确认 |
| 单任务硬上限 | 16200 秒，即 4.5 小时 |
| 最大训练步数 | 21000；预算截止前 600 秒正常停止并验证、导出 |
| heartbeat | 每 10 steps |
| checkpoint | 每 50 steps |
| 最近日志 | 2026-10-09 00:58:50 CST，step 600，约 1.586 steps/s，snapshot 已保存 |
| SwanLab 在线看板 | https://swanlab.cn/@roc_fu/finance/runs/timesfm3-lora-finetune-v7-20261009 |
| SwanLab key | 已知为硬编码配置；不在看板记录明文 |
| 接力 source | 无；V7 从原始预训练模型重新开始；V6 仅挂载 train/val 数据 |
| 预算安排 | 保留用户 4.5 小时预算；取消短跑用的固定 30 分钟上限 |

## V7 重启计划

用户确认重新训练，且指出原 30 分钟上限不必要。V7 使用新的 run id，
不复用 V6 曲线或权重。模型、loss、batch 2、LR 1e-4、数据切分、seed、
LoRA rank/alpha/dropout 和双 T4 设置不变；验证仍为固定前 256 batches 子集，
不能宣称全量验证。新增初始验证以初始化 Best，并在最终停止边界验证。

- 新格式 `timesfm3_trainable_compact_v2` 保存所有实际可训练参数、persistent buffers、
  optimizer、RNG、数据 SHA-256、冻结 base SHA-256 和 epoch/batch cursor。
- 旧 V6 格式、缺失权重、不同 base/data/config、未完成验证的 snapshot 均拒绝接力。
- 每 50 steps 原子覆盖 `training_snapshot.pt`，它仅用于诊断，不能作为完整接力来源。
  验证完成后原子保存 `last_state.pt` 和 `latest.pt`；Best 另存 `best.pt`。
- 导出时在 Kaggle 内重读 `last_state.pt` 并逐 tensor 与验证模型比较，通过才发布成功标记。
- 入口立即输出 `phase=started`；所有长阶段每 30 秒心跳，日志同步到
  `run.log`、`metrics.jsonl`、`progress.json`；子进程逐行转发，无 quiet install。
- 本机不下载真实模型。测试仅使用小型模拟模型，覆盖完整训练、验证和恢复流程。
- 独立打包目录 `finetune/timesfm3_lora_finetune_v7_kernel/`，
 预算计划嵌入唯一上传的自包含 Python 入口，不依赖未上传的旁侧文件。

## V6 产物核验

已找回断线会话，且重新联网确认 `wynstonliu/kronos-timesfm3-lora-finetune-v6/1` 为 `COMPLETE`。
断线前下载已完成，本地目录为 `/tmp/timesfm3-v6-output`。

| 指标 | 值 |
|---|---|
| train loss | 1.5383958561 |
| val nMAE | 0.0257485849 |
| val nRMSE | 0.0300867940 |
| val path direction accuracy | 0.4620225760 |
| val endpoint bias | 0.0032487242 |

上述验证不是全量验证：入口默认 `val_max_batches=256`，batch size 为 2，
最多评估 512 个样本，不能把日志中的 `val_samples=87705` 当作实际验证数量。

- `latest.pt`：74,066,131 bytes；global step 1200；Best nRMSE 0.0300867940；history 1 行。
- `step_1200.pt`：74,079,234 bytes；global step 1200；验证前保存，Best 为 Infinity，history 为空。
- 两份 checkpoint 均为 `timesfm3_lora_compact_v1`，模型 tensor 有限值检查通过；optimizer step 均为 1200。
- **完整接力阻断**：optimizer 有 645 个已更新参数 tensor，共 7,126,976 个参数；
  checkpoint 模型仅有 400 个 LoRA tensor，共 4,096,000 个参数，漏存 3,030,976 个已训练参数。
  `attach_lora()` 只冻结被替换 Linear 的 base，而 `compact_model_state()` 只保存 LoRA A/B。
  因此不能从现有 compact checkpoint 精确恢复 V6 最终模型，也不能把对应验证指标视为恢复模型的指标。
- 不将 `latest.pt` 或 `step_1200.pt` 标为完整续训来源，不启动下一轮接力。

## 接力预算与监控约定

2026-10-09 用户指定剩余 GPU 预算 4.5 小时；本次 CLI quota 显示 4.90 小时，
按较小的用户预算 16,200 秒规划，提交前重新核验 quota。

- V6 日志中 step 200 之后的 heartbeat 差分：中位数 0.652 秒/step，
  均值 0.652 秒/step，最大区间均值 0.6844 秒/step。
- 在设备、batch、验证和 checkpoint 频率不变的前提下，保守按 0.685 秒/step，
  另预留 900 秒启动、验证、导出和发布，新增训练步数初估上限为 22,000。
  这不是已提交配置；若验证范围或实现改变，必须重新估算。
- `max_steps` 当前是全局累计上限，不能把新增步数直接当作全局上限；
  必须先验证来源 checkpoint，再以来源 global step 加新增预算计算。
- 同一实验普通接力保持 SwanLab project `finance`、workspace `roc_fu`、
  V7 run id `timesfm3-lora-finetune-v7-20261009`，日志 step、checkpoint step 与看板 step 必须一致。
  若因不可恢复而重新训练，必须新实验、新 run，不能接在 V6 曲线上。
- RUNNING 时必须使用 `kaggle kernels logs --follow`；
  SSE 异常时按统一 runbook 使用页面或 Python SSE 交叉验证，
  不根据无 `--follow` 的空结果判断任务状态。
- 本机只读取日志和轻量指标；模型、checkpoint 和接力数据保留在 Kaggle，
  下一 Kernel 通过挂载完整 Output 获取，不经本机下载或上传中转。

上述 22,000 steps 是初估；实际 V7 保守使用 21,000 上限，
并按入口以来真实耗时在剩余 600 秒时正常收尾。30 分钟旧限额已取消。

## 接力状态

V6 不完整 compact 禁止接力；V7 本地训练、验证、恢复测试通过，
实际启动预检确认 7,126,976 个可训练参数全部保存。最终 Output 尚未发布，
必须等待 COMPLETE 和导出验收才能确定真实接力边界。

- 正常 V7 接力挂载完整 Output，指定 `--continuation-slug`，恢复 `last_state.pt`。
- `training_snapshot.pt` 未完成验证，明确不可作为完整接力来源。
- 保持 V7 看板 identity 和全局 step；完整指标随 heartbeat 同步到 SwanLab。
- 本机只取日志/轻量指标；Kaggle 内验收模型，不下载模型。
- 当前单次上限为 4.5 小时，旧 30 分钟限制已取消。
- 详细交接见 `timesfm3_v7_handoff_cn.md`。

## 任务历史

| Kernel | 结果 | 原因 |
|---|---|---|
| `kronos-timesfm3-lora-fine-tune` v1 | ERROR | `peft` 与 Kaggle `torchao==0.10.0` 不兼容 |
| `kronos-timesfm3-lora-finetune-v2` v1 | 被用户停止 | 无硬超时、无完整接力状态 |
| `kronos-timesfm3-lora-finetune-v3` v1 | CANCEL_ACKNOWLEDGED | 旧短跑版本已结束 |
| `kronos-timesfm3-lora-finetune-v4` v1 | 待提交 | 完整 resume + SwanLab 看板 |
| `kronos-timesfm3-lora-finetune-v4` v4 | 手工停止 | 完成 200 step；生成 checkpoint 但任务取消后 output 不可挂载 |
| `kronos-timesfm3-lora-finetune-v5` v1 | 手工停止 | 不能把同一 kernel 作为 source，且 V4 output 不可用 |
| `kronos-timesfm3-lora-finetune-v6` v1 | COMPLETE；完整接力验收未通过 | 1200 steps；compact checkpoint 漏存已训练的非 LoRA 权重 |
| `kronos-timesfm3-lora-finetune-v7` v1 | RUNNING | 新干净起点，完整 trainable checkpoint，按剩余 4.5 小时预算训练 |

## 判定规则

- `COMPLETE`：只取日志和轻量 JSON，核对 Output 清单与 Kaggle 内 checkpoint 验收证据，才算完成；
- `ERROR`：先读取最后 100 行日志，修复后再提交；
- `RUNNING`：只记录为运行中，不宣称训练成功；
- `QUEUED`：不计入 GPU 实际训练时间。
