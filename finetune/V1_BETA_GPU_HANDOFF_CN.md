# Kronos v1-beta GPU 交接（2026-08-27）

## 结论先行

当前没有运行中的 Kaggle 训练任务。下一步**不要**立即再做一轮同配置训练。

应先以自然验证阶段的 `Best@21` 做一次只推理、不更新权重的横截面解偏/解码校准实验，验证“按每个信号日减去全市场预测均值（或中位数）”能否显著缓解预测偏空。该实验直接复用当前冻结的未来自然集与未来均衡集，成本远低于训练；只有它明确有效，才设计下一轮训练。

当前推荐推理候选：`natural Best@21`。

## 已完成任务与可用模型

三个远端任务都已确认 `COMPLETE`：

| 任务 | Kaggle Kernel | 用途 | 结论 |
|---|---|---|---|
| 双速率主训练 | `wynstonliu/kronos-v1-beta-best109-official-chunk-5` | 从 Best109 正式分支接力完成至 Segment 1058 | 保留 `Best@467` 与 `Last@1058`；主训练已经结束 |
| 自然验证统一 LR 阶段 | `wynstonliu/kronos-v1-beta-last1058-natural-validation-stage-1` | 从主训练 `Last@1058` 只载入权重，重建 optimizer，以统一 LR `1e-6` 训练 150 segments | 最优在 Segment 21；之后没有恢复，不能继续同配置 |
| 严格未来评估 | `wynstonliu/kronos-v1-beta-last1058-natural-evaluation` | 无训练，对比 `Last1058`、自然阶段 `Best@21`、`Last@150` | `Best@21` 是下一步主候选 |

### 模型选择依据

未来评估的信号日为 `2026-08-03` 至 `2026-08-10`，目标日为 `2026-08-17` 至 `2026-08-24`，严格晚于主训练的最晚目标日 `2026-07-31`。

| 指标 | Last@1058 | 自然 Best@21 | 自然 Last@150 |
|---|---:|---:|---:|
| 未来 teacher-forcing objective | 2.43777 | **2.42765** | 2.43118 |
| 未来均衡集 Rank IC | 0.07474 | **0.08476** | 0.07615 |
| 未来均衡集 return MAE | 0.08382 | 0.08067 | **0.08053** |
| 未来均衡集方向准确率 | 0.3630 | **0.3673** | 0.3643 |
| 未来均衡集 Top-Bottom spread | 0.01945 | **0.02111** | 0.01730 |
| 未来自然集 Rank IC | **0.08222** | 0.07816 | 0.08211 |
| 未来自然集 return MAE | 0.09713 | 0.09436 | **0.09406** |
| 未来自然集 Top-Bottom spread | 0.02002 | 0.02244 | **0.02261** |

结论不是“Best@21 在所有指标上无条件最好”，而是它在更能识别方向偏置的**未来均衡集**上有最好的 objective、Rank IC、方向准确率与头尾收益差，因此应作为下一步校准与训练设计的起点。`Last@150` 仅在 MAE 上略好，且未改善排序能力。

自然阶段本身：Best@21 的 validation objective 为 `2.4443049431`、forecast 为 `2.3927643299`；Segment 150 约为 `2.447927`。也就是说，150 段以后继续同一配置已经没有证据支持。

## 仍存在的问题

方向偏空尚未解决，只是 `Best@21` 略有缓解。未来均衡集的类别召回：

| 模型 | short recall | neutral recall | long recall |
|---|---:|---:|---:|
| Last@1058 | 87.4% | 11.1% | 10.4% |
| 自然 Best@21 | 84.4% | 13.4% | 12.4% |
| 自然 Last@150 | 83.0% | 13.5% | 12.8% |

条件层在未来样本上的诊断也不理想：真实条件相对打乱条件仍有约 `-0.0017` 的 forecast loss 优势，但 `full - none forecast` 为正；`Best@21` 约为 `+0.00734`。这表示条件包含少量语义信息，却可能在绝对预测误差上造成伤害。不能据此直接删除条件层，应该将它作为下一轮训练的明确消融项。

## 数据集清单

### 训练数据

本机目录：`data/a_share_full_market_v1_beta/`，约 `1.7 GB`。

- 主数据 manifest：`data/a_share_full_market_v1_beta/data_manifest.json`
- 训练/验证面板：`processed_datasets/train_data.pkl`、`processed_datasets/val_data.pkl`
- 数据内容 SHA-256：`214c375f47e9843b7d836e414199f444ac8cd139e2ef1bb51adf526bfdc6261c`
- 模型契约：120 个输入交易日预测 10 个交易日；86 个行业标签（含 `unknown`）；市值百分位条件；条件注入 Transformer Layer 10。
- 原训练候选池：10,564,072 个窗口；主训练共 1,058 segments。

### 固定自然验证集（训练监控，不用于最终晋级）

目录：`data/a_share_full_market_v1_beta/natural_validation_v1/`

- manifest SHA-256：`3db79fa5a5966f5d22f0c227b84c0e5a9293cdaaf3f19139ca85f7df427f28b6`
- 大集：12,000 窗口，116 个日期（`2026-01-05` 至 `2026-06-30`）
- 构成：short 6,316 / neutral 1,172 / long 4,512
- quick 集：3,000 窗口，short 1,576 / neutral 292 / long 1,132
- 按日期等额分配，日期内部保留真实方向比例；所有样本已从训练候选中排除。

### 固定均衡验证集（训练的 Best 选择与健康监控）

目录：`data/a_share_full_market_v1_beta/balanced_validation_v1/`

- manifest SHA-256：`14afe39e260a90e438051512167f50ee29c4d8561a27fe09896ec261c79a4ba6`
- 大集：12,000 窗口，short / neutral / long 各 4,000
- quick 集：3,000 窗口，各 1,000
- 同样按日期、行业和市值分层，并从训练候选中排除。

方向定义在两个验证集都固定为：第 10 个目标收盘价相对最后输入日收盘价的收益，小于 `-1%` 为 short，`[-1%, +1%]` 为 neutral，大于 `+1%` 为 long。

### 已有未来评估集（对当前模型严格样本外）

目录：`data/a_share_v1_beta_eval_20260826/package/`，约 `136 MB`。

- `evaluation_manifest.json`
- `evaluation_panel.pkl`
- `evaluation_samples.jsonl`
- 未来全量池：30,930，short 13,176 / neutral 3,758 / long 13,996，6 个信号日
- 未来均衡集：3,000，三类各 1,000
- 未来自然集：由评估脚本从上述 6 个日期按日期均衡抽取 3,000
- 不能把该集并回训练。
- 它对现有三个 checkpoint 是严格样本外；但 Task 0 会用它选择校准规则，因此 Task 0 后它将成为“校准调参集”，不再是未来训练方案的最终封存测试集。下一轮训练若要晋级，必须等新增、未使用的未来标签，再建立新的 final test。

## 远端断点与输出位置

不要下载模型到本机后再传给服务器。新 GPU 机器应使用 Kaggle Kernel Output / 数据集挂载，或从同一份受控对象存储直接取数。

父训练输出（`wynstonliu/kronos-v1-beta-best109-official-chunk-5`）：

```text
<output>/
  checkpoints/
    best_model/                  # Segment 467，仅用于保留和对照
      model.safetensors
      config.json
      best_metric.json
    last_model/                  # Segment 1058，曾作为自然阶段输入
      model.safetensors
      config.json
    last_state.pt                # 仅能恢复双速率主训练，不用于新阶段 optimizer
  metrics.jsonl
  progress.json
  summary.json
  experiment_manifest.json
  run.log
  parent_run.log
```

自然阶段输出（`wynstonliu/kronos-v1-beta-last1058-natural-validation-stage-1`）：

```text
<output>/
  checkpoints/
    best_model/                  # Segment 21：下一步主候选
      model.safetensors
      config.json
      best_metric.json
    last_model/                  # Segment 150：保留用于对照
      model.safetensors
      config.json
    last_state.pt
  metrics.jsonl
  progress.json
  summary.json
  experiment_manifest.json
  run.log
```

新阶段若只加载 `Best@21` 的模型权重，必须**重新建立 optimizer、scheduler 和 AMP scaler**；不能恢复自然阶段的 `last_state.pt`，否则会从 Segment 150 的参数继续。

## 下一步：GPU 任务单

### Task 0：先做校准评估（推荐，必须先于训练）

输入：自然阶段 `checkpoints/best_model`，加未来评估数据集。

不更新任何模型权重。按每个 `asof_date` 分组，对 autoregressive 的 `predicted_return_10d` 同时评估：

1. raw；
2. `raw - 当日预测均值`；
3. `raw - 当日预测中位数`；
4. 可选的小常数 offset 网格，仅作对照。

每个版本都要在未来自然集和未来均衡集给出：Rank IC、Top-Bottom spread、return MAE、方向准确率、三类 recall、预测均值/方差、预测方向占比。输出逐样本预测，确保可复核。

判断门槛：只有均衡集的 short/long 召回明显更接近、Rank IC 和头尾收益差不下降，且自然集结果不显著恶化，才可把 daily centering 作为推理后处理。该方案几乎不消耗训练 GPU 时间。注意：这会消耗现有未来集的“最终测试”资格；后续须用更晚日期的全新标签作独立确认。

现有 `finetune/calibrate_v1_beta_checkpoint.py` 和 `kaggle_v1_beta_last1058_calibration.py` 是早期校准草稿：它们仍以 `Last1058` 和历史抽样集为目标，**不可直接作为本 Task 0 的最终脚本**。应基于 `finetune/evaluate_last1058_natural_stage.py` 的数据读取和模型定位方式改写，并把模型切到自然阶段 `Best@21`、测试集切到未来自然/均衡集。

### Task 1：仅在 Task 0 后决定是否训练

若 Task 0 有效：先保留 `Best@21` 权重，推理使用被验证的校准；没有必要为“预测均值偏空”本身重训。

若 Task 0 无效：启动小型、有明确对照的训练实验，而不是续跑自然阶段。

- 起点：自然 `Best@21` 权重。
- optimizer/scheduler：新建；统一低 LR 仍可作为基线，但 LR、覆盖量和停止点应在 Task 0 后确定。
- 训练数据：保持原始训练分布，不能为追求均衡而直接重写训练样本分布。
- 监控/Best：固定均衡验证集作为每 segment quick 监控和 checkpoint 候选；自然验证集作为真实性诊断；当前未来集只能用于方案筛选，最终晋级必须由新收集的、从未参与校准的未来封存集决定。
- 必须新增一个条件层消融支路：至少比较保持条件与禁用条件/缩小条件注入强度，防止继续放大当前条件层的绝对误差问题。
- 先跑不超过 30-50 segments 的诊断块；若当前未来调参集没有改善，不继续扩展为长训。

`finetune/kaggle_v1_beta_uniform_balanced.py` 与 `finetune/kaggle_v1_beta_minimal/Kronos/finetune/kaggle_uniform_balanced_runner.py` 是“从主训练 Last1058 启动、统一 LR `1e-6`、均衡验证”的旧实现。它们固定校验父任务为 Segment 1058，因此**不可直接用于 Best@21 的后续实验**，除非先重写 lineage guard、输入模型定位、stage ID、输出名和 SwanLab run ID。

## 新 GPU 服务器启动前检查

1. 使用 Python 3.10+、CUDA 可用的 PyTorch；先运行一次 `nvidia-smi` 和最小 tokenizer/model 前向验证。
2. 复制代码仓库和上面两份数据目录，或使用等价的只读远端挂载；校验三个 manifest SHA-256。
3. 从 Kaggle 输出服务端挂载自然阶段与父训练输出，读取 `experiment_manifest.json`，不得猜测输出的运行时目录名。
4. 先执行 Task 0；保存 `summary.json`、`status.json`、`evaluation_manifest.json`、`run.log`、逐样本预测文件。
5. 若开始训练，输出契约必须包括 `best_model`、`last_model`、`last_state.pt`、`best_metric.json`、`metrics.jsonl`、`progress.json`、`summary.json`、`experiment_manifest.json`、`run.log`；每个 chunk 正常退出后才算可续训。
6. 训练过程不要手工 Stop；在脚本里设置明确 segment 上限并让程序主动退出、导出 Last、核对输出后再接力。

## 关键代码入口

- 未来评估与候选比较：`finetune/evaluate_last1058_natural_stage.py`
- 未来评估 Kaggle 包装：`finetune/kaggle_v1_beta_last1058_natural_evaluation.py`
- 自然验证阶段入口：`finetune/kaggle_v1_beta_last1058_natural.py`
- 固定自然验证集生成：`finetune/build_natural_validation_manifest.py`
- 固定均衡验证集生成：`finetune/build_balanced_validation_manifest.py`
- 训练 bundle 打包：`finetune/build_kaggle_v1_beta_bundle.py`
- 输出与断点规范：`finetune/KAGGLE_RUNBOOK_CN.md`

## 不应做的事

- 不要从 `Last@150` 继续同一自然验证统一 LR 配置。
- 不要再用双速率主训练的 `last_state.pt` 开启新的统一 LR 阶段。
- 不要以历史自然/均衡验证损失单独决定最终模型。
- 不要把未来评估样本混回训练或调参验证集。
- 不要覆盖或删除 `Best@467`、`Last@1058`、自然 `Best@21`、自然 `Last@150` 中任一模型。
