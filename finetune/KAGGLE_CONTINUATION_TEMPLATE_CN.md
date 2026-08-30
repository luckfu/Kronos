# Kaggle 训练接力模板

这份模板用于所有需要分多个 Kernel 完成的训练。复制本模板后，只替换实验名、
Kernel slug 和输入 Dataset，不得临时删减校验步骤。

## 首次提交前

```bash
KERNEL='luckfu/<experiment-slug>'
kaggle kernels status "$KERNEL"
kaggle kernels files "$KERNEL" --format json
```

确认没有同目标 `RUNNING`/`QUEUED` 版本，并逐项核对 metadata：

```json
{
  "id": "luckfu/<experiment-slug>",
  "code_file": "<runner>.py",
  "dataset_sources": ["luckfu/<training-data>"],
  "kernel_sources": []
}
```

首次运行必须明确 `resume_training=0`，并把 `max_segments_per_run` 设置为实测
12 小时以内的安全值。先完成一个正式 chunk 的运行验收，再进入长程训练；不能
先启动 GPU 再补接力设计。

## 每个 chunk 的完成判定

只有 Kaggle 状态为 `COMPLETE`、`kaggle kernels files <ref>` 非空，且以下文件全部存在，
chunk 才允许作为下一次输入：

```text
run.log
metrics.jsonl
progress.json
summary.json
experiment_manifest.json
checkpoints/last_state.pt
checkpoints/best_model/model.safetensors
checkpoints/best_model/config.json
checkpoints/best_model/best_metric.json
checkpoints/last_model/model.safetensors
checkpoints/last_model/config.json
```

`last_state.pt` 用于续训；`last_model` 只用于推理；`best_model` 用于历史最佳
评估。三者不能互相替代。

## 接力提交前

优先使用 Kaggle 服务端 `kernel_sources`。新 runner 必须从 `/kaggle/input` 找到
恰好一个 `checkpoints/last_state.pt`，并验证它所在 output 树的 Best、Last、State、
日志和指标全部存在。找到 0 个或超过 1 个时必须失败退出，不能用 `find -print -quit`
静默挑一个。

接力启动后日志必须出现：

```text
continuation_output=<历史 output 树>
resume_state=<当前恢复文件>
historical_metrics_lines=<历史指标行数>
resume=true
```

新 runner 复制完整 output 树，并以追加模式写入 `run.log`、`metrics.jsonl`。这样
跨 chunk 的 loss 才是同一条曲线。

## SwanLab（正式训练必须启用且统一）

v1-beta 的专用 runner 优先读取环境变量 `SWANLAB_API_KEY`，未设置时读取同名 Kaggle
Secret。密钥不得写入 runner、Dataset、`run.log` 或 `metrics.jsonl`；Kernel 必须在
提交前获得 Secret 访问权，否则 preflight 应立即失败。
所有接力 Kernel 复用固定 `SWANLAB_RUN_ID`（默认
`kronos-v1-beta-120d-to-10d`），runner 启动时先回填历史 `metrics.jsonl`，再上传
当前 chunk 的指标。正式训练启动前，登录或初始化失败必须直接退出；只有训练
开始后的短暂上传故障可由本地 `metrics.jsonl` 兜底并在下次接力回填。看板问题不能
改变 checkpoint、optimizer、scheduler 或接力流程；日志不得输出 API key。

## 停止和中断

- 计划停止：预先设置 `max_segments_per_run`，让训练程序在段末主动正常退出、导出 Last、
  校验输出契约，并确认 Kaggle 状态为 `COMPLETE`。不得用 `Stop Session` 代替正常结束。
- `CANCEL_ACKNOWLEDGED` 不是成功状态。若 `kaggle kernels files <ref>` 为空，本轮
  `/kaggle/working` 已丢失，不能作为 `kernel_source`，也不能按页面最后 Segment 续训。
- v1-beta 只使用 segment 级断点：首次训练前保存 Segment 0；每个 segment 的训练和
  验证全部完成后，立即原子替换 `last_state.pt`。State 的 `next_epoch=N` 表示前 N 个
  segment 已完整完成，续训从 Segment N+1 开始。这个保证只在当前 Worker 或已发布的
  COMPLETE Output 内成立。
- `SIGTERM`/`SIGINT`：训练器至多能在当前 batch 后停止，不能促使 Kaggle 发布
  `/kaggle/working`。只有外层任务随后正常结束并形成 COMPLETE Output 时，才可用其中
  的上一完整 segment State；`Stop Session` 不满足这个条件。
- `progress.json.current_step` 表示持久化位置，在段内固定为 0；实时页面位置只写入
  `observed_step`。step 日志用于看板和诊断，绝不作为续训起点。
- 强制 kill、OOM、机器崩溃：只能从上一份已发布的 COMPLETE Output 中的
  `last_state.pt.next_epoch` 推导下一 segment。本 chunk 内即使已完成多个 segment，
  没有独立远端发布就会随 Worker 一起丢失。禁止按页面 step/segment 猜造断点。
- 中断后若 Best/Last/State 或日志指标缺失，停止接力并保留现场，不能用 Last 猜造
  Best，也不能重新 push 多个试错 Kernel。

## 提交后监控

```bash
kaggle kernels status "$KERNEL"
kaggle kernels logs "$KERNEL"
kaggle kernels output "$KERNEL" -p /tmp/<output-check> --file-pattern 'run\\.log' -o
```

记录实际已完成 segment、每段耗时、最新 forecast loss、历史 Best loss、GPU 和预计
剩余时间。`RUNNING` 不是完成；必须以 output 契约判定。
