# v1-beta Best-11 双速率正式训练计划

登记日期：2026-08-23

本文件定义 warmup 实验停止后的双速率训练。新实验从 warmup run 的验证最优
**Segment 11 Best 权重**开始，只继承模型权重，不继承旧 optimizer 或 scheduler。

## 1. 当前结论

warmup 实验使用：

```text
继承参数：1e-6 -> 1e-5 Warmup，再全局 Cosine
条件参数：1e-5 -> 1e-4 Warmup，再全局 Cosine
Warmup 占两遍覆盖总 step 的 2%
```

该实验完整持久化至 Segment 105，Best 为 Segment 11，Objective Loss 为
`2.439993896484375`。Segment 1 的 validation forecast loss 为 `2.3923`，Segment 105
为 `2.4021`；完整曲线显示 Warmup 后验证损失持续抬升。Segment 105 时条件 LR 仍约
`9.84e-5`、继承参数 LR 仍约 `9.85e-6`，证明按 1,058 segments 展开的全局余弦在
前 100 segments 基本处于高位。旧 Last 和 State 只保留用于审计，不能作为新实验起点。

## 2. 新正式学习率计划

采用从 Best-11 继续适配的差分学习率：

```text
继承参数：1e-6 -> 3e-6，前 0.5% step Warmup，随后全局余弦降至 5e-7
条件参数：1e-5 -> 3e-5，前 0.5% step Warmup
条件参数：Warmup 后单调快速衰减，在全局 7.5% step 处降到 1e-5
条件参数：之后继续单调余弦尾段，最终降至 1e-6
不使用余弦重启，不重新升到 1e-4，不设置高 LR 平台期
```

原因：Best-11 的条件层已完成初步学习，不再是零初始化冷启动；重新升至 `1e-4` 会
重复已经观察到的高 LR 破坏。短 Warmup 只用于新 optimizer 的平滑接入，条件组随后
在约 Segment 80 前完成快速回落，继承参数峰值同时降为旧实验的 30%。

所有阶段都必须按两遍覆盖的**全局 optimizer step**计算，不能按 segment、
Kaggle Kernel 或 continuation chunk 重新开始。正式诊断配置固定 `batch=32`，按当前
`10,564,072` 个窗口计算，两遍约 `660,256` 个全局 step；Warmup 约为前 `3,301`
step、约 5 个 segment，条件快速衰减里程碑约为 `49,519` step、约 80 个 segment。
实际值仍必须由代码根据数据集和 batch 动态计算并打印，不能只依赖文档示例值。

仍然保持：

- 输入输出为 120 日 -> 10 日；
- AdamW，`betas=(0.9, 0.95)`；
- P100 使用 `batch=32`；predictor 前向使用 FP16 AMP 和 GradScaler，冻结 tokenizer
  编码保持 FP32，避免改变离散 token 目标；
- 目标函数为 `forecast_loss + 0.02 * history_loss`；
- 两遍完整覆盖，不启用 early stopping；
- 每段结束验证并原子持久化 State、Best、Last、日志和指标；
- 中断后只从上一完整 segment 恢复，全局 scheduler 位置随 State 恢复。
- `last_state.pt` 同时持久化 AMP scaler；后续接力的 batch 和 AMP 开关必须通过
  resume guard，不能在同一实验中途修改。

## 3. 参数组优化

正式实现必须显式拆分以下参数组：

1. V6 继承的可训练层、预测头和依赖层；
2. 新增的 `sector_emb` 和 `size_mlp`；
3. no-decay 参数：embedding、norm、bias。

`weight_decay=0.1` 只应用于适合衰减的矩阵权重；embedding、norm 和 bias 的
`weight_decay` 固定为 0。继承参数峰值固定为 `3e-6`，条件参数峰值固定为 `3e-5`。
全局梯度裁剪从 `3.0` 收紧到 `1.0`。SwanLab 和 `metrics.jsonl` 必须分别记录两组
实际学习率。

每 100 step 在全局裁剪前记录：

- 条件输出范数、Layer 10 注入前主干范数和两者比例；
- 条件组与继承组的总梯度 L2、按参数量归一化的梯度 RMS；
- 两组权重范数、权重 RMS 和 `LR * grad_rms / weight_rms` 更新比例；
- 行业 embedding 与市值百分位 MLP 输出层的独立权重范数。

范数比例不使用 `0.3` 或 `0.5` 作为硬阈值，只分析轨迹、突变点及其与验证损失的
同步关系。每 10 segments 以及新实验第 1 segment 执行一次验证消融：真实条件、完全
无条件、batch 内联合平移打乱行业和市值条件。`full-none < 0` 和
`full-shuffled < 0` 才表示条件在当前验证分布上有正贡献；该结论仍需 2026 年 8 月
真正未来数据确认。

## 4. 执行顺序

1. 停止并确认 warmup Kernel 不再 active，旧输出完整保留。
2. 旧 Warmup 被手工取消后没有发布可挂载 Output，因此先用完全相同的数据、V6 SHA、
   seed、batch 与 Warmup 参数重跑前 11 段，并让
   `luckfu/kronos-v1-beta-warmup-best11-recovery` 正常结束；首段验证指标必须与旧日志
   `2.4432 / 2.3923 / 2.5445 / 2.5318` 对齐。新 Kernel 再将该恢复 run 作为 Kaggle
   `kernel_source`，在服务端读取 Segment 11 Best。
3. 新实验只加载 Best 的模型权重，重建 optimizer/scheduler，从新实验 Segment 1 开始。
4. 使用新的 Kernel slug、SwanLab run id、output 名和 manifest；不经本地中转权重。
5. 首个正式诊断 chunk 运行 120 segments，覆盖条件快速衰减里程碑，不运行 smoke。
   旧任务 `batch=16` 实测每段约 170 秒、峰值显存仅 `1.06/1.19 GB`；因此本轮固定
   `batch=32 + predictor AMP`。预计首个 chunk 约 3-4 小时，实际以 Segment 1-3
   的完整耗时重新估算。
6. 每段验证并保存 Best；Segment 20、50、80、100、120 为人工质量检查点，不启用自动
   early stopping。
7. 确认验证趋势和条件消融有效后，按完整接力契约继续两遍 1,058 段。

## 5. 选择指标

不能只看随机 batch 的 token loss。统一评估按以下优先级决定：

- 样本外 Rank IC、IC 均值和 ICIR；
- 头尾收益差、方向准确率和平衡准确率；
- forecast token loss、MAE 和预测波动率；
- 行业和市值分组稳定性；
- 相对 V6 及当前 v1-beta 基线的历史能力退化。

训练完成后仍需样本外评估才能替换 V6 生产模型；本次启动前不执行 V6 对照评估。

## 6. Kaggle 执行约束

- 新调度器使用新的实验 slug、SwanLab run id、output 树和 manifest。
- 首次启动必须加载 warmup Segment 11 Best 模型权重，禁止加载 warmup Last、optimizer
  或 scheduler State；后续双速率实验接力才恢复本实验自己的完整 State。
- 每个 Kernel 仍按不超过 12 小时的安全 chunk 运行，但 chunk 上限不得改变全局
  Warmup/Cosine 的总 step 或当前位置。
- `last_state.pt` 必须保存 scheduler 类型、warmup ratio、全局总 step、已完成 step、
  各参数组峰值/最低学习率和 weight decay；任一项不匹配时拒绝续训。
- SwanLab 同时记录每个参数组的实际学习率，而不是只记录第一个参数组。
- 仍执行 `KAGGLE_RUNBOOK_CN.md` 的唯一 active Kernel、服务端接力和完整输出契约。

## 7. 启动前检查

- [ ] 旧 v1-beta Kernel 已停止且没有其他 active Kernel。
- [ ] 父实验 Best 指标严格等于 Segment 11 / `2.439993896484375`。
- [ ] 新实验从父 Best 权重开始，没有错误加载父 optimizer/scheduler。
- [ ] Warmup 为全局 step 的 0.5%，只执行一次。
- [ ] 条件 LR 在全局 7.5% step 前单调降到 `1e-5`，之后不发生重启。
- [ ] 四个 optimizer group、各自 LR 和 weight decay 已在启动日志中逐项打印。
- [ ] 启动日志明确打印 `batch=32`、`Predictor AMP: enabled (float16)`，tokenizer
  保持 FP32；State 中包含 scaler，接力 guard 包含 batch 和 AMP。
- [ ] embedding、norm、bias 的 weight decay 为 0。
- [ ] Kaggle 接力后 scheduler 的全局 step 与上一完整 segment 完全一致。
- [ ] 监控指标每 100 step 写入 `metrics.jsonl` 和 SwanLab；条件消融按计划运行。
- [ ] 评估股票池、日期、seed、采样和指标与 warmup 实验一致。

## 8. Best 109 正式训练分支（2026-08-24）

阶段评估已决定从双速率实验的 Best 109 建立正式分支。首次启动使用 Best 109 模型权重、
新 optimizer，并将全局 scheduler 精确定位到已完成 109 segments 后，从 Segment 110
继续覆盖；历史 Last 120 及其 optimizer State 不进入正式分支。正式 SwanLab run 固定为
`kronos-v1-beta-best109-official-120d-to-10d`，先回填旧指标的 Segment 1-109，再连续追加
110+；旧分支 110-120 排除。后续接力必须恢复正式分支自己的 Best、Last、State、日志和
manifest，且来源 manifest 必须带 `KRONOS_BRANCH_ORIGIN_SEGMENT=109`。

## 9. 条件层初始化完成后的统一学习率阶段（待执行）

当前 Best 109 正式分支继续按既定双速率方案完成，不因本节调整、停止或重启。该阶段的
主要目标是让新增行业与市值条件层完成初始化和早期适配；完成并持久化 Best、Last 与
完整 State 后，再进入统一学习率微调阶段。

下一阶段固定遵循以下原则：

- 从当前阶段评估后选定的可靠 checkpoint 加载模型权重，重新建立 optimizer 和
  scheduler；不直接沿用双速率 optimizer 状态。
- 所有可训练的继承参数、条件参数与预测头使用同一个低学习率和同一条调度曲线，不再
  保留条件组 10 倍学习率。具体峰值、最低学习率、Warmup 长度和训练覆盖量必须根据
  当前阶段 Best/Last 的正式评估结果确定，本文不提前猜测数值。
- 训练数据仍按正式训练数据分布使用；“均衡”首先约束验证集，不能通过改变验证分布
  偷换训练目标。
- 验证样本在整个统一学习率阶段保持固定，禁止每个 Segment 重新抽样。固定验证集用于
  各 Segment 的趋势判断和 Best 选择，确保相邻点可直接比较。
- 以每个窗口最后一个输入交易日到第 10 个预测交易日收盘价的收益划分方向：空头
  `< -1%`、中性 `[-1%, +1%]`、多头 `> +1%`；三档等量抽样。
- 在方向等量之外，同时约束日期、行业和市值层级的覆盖，避免少数市场日期、热门行业
  或单一市值区间主导结果。构建后必须输出各层样本数、占比和覆盖率供审计。
- 使用两级验证：每 Segment 运行固定的均衡 quick validation，用于监控与保存 Best；
  每隔固定里程碑运行更大的固定均衡 validation，用于确认趋势。最终模型选择还必须在
  未参与调参、保持封存的 final test set 上完成。
- 因验证分布发生变化，统一学习率阶段必须使用新的 SwanLab run，或至少使用完全独立的
  指标命名空间。不得把新验证指标续接到当前 run 的旧曲线上，也不得直接用两阶段的
  validation loss 数值判断模型提升或退化。
- 新阶段启动前先修复 SwanLab 实时转发缺失 `validation/train_average` 和
  `validation/best_loss` 的问题；本地 `metrics.jsonl` 与 SwanLab 必须同时完整记录。

启动前必须形成并保存均衡验证集 manifest，至少包含候选池版本、方向阈值、随机 seed、
样本索引或其稳定哈希、日期范围、行业/市值覆盖、各方向计数以及 final test 隔离证明。
任何 continuation chunk 都只能恢复同一份验证 manifest；不匹配时必须拒绝续训。
