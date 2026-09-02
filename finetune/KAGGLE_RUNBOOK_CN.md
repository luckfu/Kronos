# Kronos Kaggle 统一运行手册

本手册适用于本项目今后所有 Kaggle 训练、续训、调参、网格搜索、回测和验证任务，不限于 V6/V6.01。任何新任务都必须先按本文件做 preflight；历史版本的专项说明只能补充本手册，不能覆盖这里的安全规则。

可直接复制的提交和接力模板见 [`KAGGLE_CONTINUATION_TEMPLATE_CN.md`](KAGGLE_CONTINUATION_TEMPLATE_CN.md)。

v1-beta 旧 Warmup 实验停止后的 Best-11 双速率 optimizer 参数组优化见
[`V1_BETA_FUTURE_TRAINING_PLAN_CN.md`](V1_BETA_FUTURE_TRAINING_PLAN_CN.md)；
它作为独立实验从旧 Warmup 的 Segment 11 Best 权重启动，禁止加载旧实验的
Last、optimizer 或 scheduler State。

## 1. 核心原则

1. 一个实验目标同时只允许一个 active Kernel。
2. 提交前先确认账号、Kernel slug、数据源、模型来源、代码版本和运行目标。
3. 训练必须同时保留 State、Best、Last；评估必须保留每个子任务结果和总汇总。
4. 跨任务大文件只在 Kaggle/模型仓库服务端传递，不经本地电脑中转。
5. 先设计完整生命周期，再启动 GPU：启动、监控、正常分段结束、接力、最终验证必须一次规划清楚。
6. `RUNNING`、日志有输出、某个 loss 较低都不等于任务完成；完成只由输出契约决定。
7. Segment 结束只把断点写到当前 Worker 的 `/kaggle/working`；只有训练程序主动正常退出、
   Kaggle 状态成为 `COMPLETE` 且远端文件列表非空，断点才真正跨 Kernel 持久化。

## 2. 每次提交前必须形成实验清单

提交前必须明确并记录：实验名称和唯一 slug、当前 Kaggle 账号、任务类型、基础模型仓库/版本/SHA-256、数据集 slug、训练/验证日期区间、输入/预测长度、loss 口径、coverage passes、每段样本数、单次最大 segment 数、batch/seed/采样参数、GPU/PyTorch 版本、单次与总耗时、输出目录、完成判定和接力方式。

不能只检查父进程环境变量。若仓库脚本有硬编码值，必须检查并打印最终生效配置；日志必须能看到实际 lookback、predict、loss、日期、总 segment 数和 resume 状态。

## 3. 并发和版本发布纪律

- push 新版本前先执行 `kaggle kernels status <slug>`。
- 新建 Kernel 的 slug 控制在 50 个字符以内；超长 slug 可能只返回无正文的
  `400 Bad Request`，不能靠重复 push 解决。
- 旧版本为 `RUNNING`、`QUEUED` 或仍在取消时，禁止 push 新版本。
- 要替换旧任务时，先取消并确认 `CANCEL_ACKNOWLEDGED`、`COMPLETE` 或 `FAILED`，再提交新版本。
- 一次 push 可能保留旧版本继续运行；不能假设新版本会自动替换旧版本。
- 页面出现多个版本时，只保留唯一正确版本。
- 不允许“先启动再补设计”；需要改变 checkpoint、接力或输出逻辑时，先停止错误任务，再完成脚本和 preflight。

## 4. 训练任务的输出契约

每个正常结束、可用于续训的 chunk 必须包含：

```text
<output>/
├── checkpoints/
│   ├── last_state.pt
│   ├── best_model/model.safetensors + config.json + best_metric.json
│   └── last_model/model.safetensors + config.json
├── metrics.jsonl
├── progress.json
└── summary.json
```

- `last_state.pt` 保存模型、optimizer、scheduler、RNG 和最后完整 segment 位置；v1-beta
  使用 AMP 时还必须保存 GradScaler，不保存或恢复段内 step 状态。
- `best_model` 是整个实验历史上验证指标最优的推理模型，不能被 Last 覆盖。
- `last_model` 是最近一次完整 checkpoint 对应的推理模型。
- 不能只保存 State，不能用 Last 伪造 Best；Best 缺失时拒绝续训。
- 训练结束必须显式导出 Last，并执行输出完整性校验。
- 正常 chunk 结束仍必须满足上面的完整契约。若 Kaggle 强杀导致
  `last_model`/`summary.json` 来不及导出，续训只可在完整 State、Best、日志、指标、
  progress 和 manifest 均通过校验时恢复；Last 必须由 `last_state.pt` 自动重建，
  summary 由下一正常 chunk 重写，不能因此丢弃有效 State，也不能把 Last 当恢复源。
- 改变 loss、预测长度、结构或全局学习率计划时，必须新建 optimizer/scheduler。

## 5. 分段训练和跨 Kernel 接力

- 启动前计算总 segment 数、每段实测耗时和 Kaggle 单次时限。
- 设置安全的 `max_segments_per_run`，让任务主动正常结束。
- 对 v1-beta 当前 P100 基准（约 2 分 31 秒至 2 分 38 秒/segment），朋友账号按 250
  segments/chunk（约 10.5 至 11 小时训练，另留启动/导出/发布余量）执行；不能把“安全结束”
  误设成每个 chunk 只跑 1 个 segment。若硬件或单步时长变化，必须按实测重新计算。
- **首个正式 chunk 启动前必须完成接力设计**：确认后续 Kernel 使用的
  `kernel_sources` 或 continuation Dataset 已经存在、权限可读、目录结构和
  `last_state.pt`/Best/Last 契约已验证。不能先跑一个 segment，再在本地临时
  下载 checkpoint 后上传。
- 下一 Kernel 优先用 `kernel_sources` 挂载上一个完整输出，由 Kaggle 服务端传递。
- 如需 Dataset 中转，也应由 Kaggle 服务端生成/复制；禁止本地下载数百 MB checkpoint 再上传。
- 接力 runner 必须从 `/kaggle/input` 找到唯一完整 V6/实验输出树，并打印恢复路径和 segment。
- 全局 schedule、覆盖顺序和总 segment 数必须与上一 chunk 一致。

### 5.1 日志和指标连续性

- 接力不是只复制 `checkpoints/`：必须复制上一个 chunk 的**完整 output 树**，包括
  `run.log`、`metrics.jsonl`、`progress.json`、`summary.json` 和全部 checkpoints。
- 新 Kernel 打开 `run.log` 和 `metrics.jsonl` 时必须使用追加模式；不能生成一份只含
  当前 chunk 的同名文件后覆盖历史记录。
- `metrics.jsonl` 中每条记录必须保留全局 `segment`、`global_step`、`pass` 和时间戳，
  这样才能跨 chunk 绘制连续 loss 曲线。
- 接力启动日志必须打印：输入 output 树路径、恢复 segment、历史 metrics 行数、Best
  验证 loss 和 Last 状态。任一项缺失则拒绝启动。
- 不能把 Kaggle 页面日志当作唯一历史记录；页面日志只用于实时观察，持久日志必须
  随 output 树发布并在下一 chunk 继承。

## 6. 调参、网格和验证输出契约

- 模型、holdout 优先从 `/kaggle/input` 挂载；不要每个组合重复下载或加载。
- 每个参数组合独立输出 `summary.json`，完成后生成 `combined_summary.json`。
- 恢复时只跳过已存在且通过校验的组合。
- 不能偷偷减少股票数、日期、截面数、seed、采样数或指标口径。
- 参数选择必须基于真实运行结果，不能用插值或单一指标替代实际验证。

## 7. 环境、输入、取消和失败

- P100 必须确认 PyTorch 包含 `sm_60`；已验证组合为 `torch==2.5.1+cu121`。
- Kaggle CLI 2.2.4 的 `--accelerator P100` 会被 `SaveKernel` 以 400 拒绝；实际成功
  metadata 使用 `machine_shape: "Gpu"`。正式 P100 任务必须在入口读取 `nvidia-smi`
  并强制要求 P100，若分配到 T4 等其他 GPU 必须在训练前立即退出，不能静默降级。
- 输入同名文件通常必须唯一；带有正式 `data_manifest.json` 的数据集必须按
  manifest 绑定的根目录选择文件。未被有效 manifest 引用的历史副本只能被记录并
  忽略；若发现多个 SHA 匹配的有效 manifest，必须拒绝启动。模型权重必须校验
  SHA-256。
- Kaggle 账号切换后重新核对私有资源 owner/权限。
- 计划停止不得点击 `Stop Session`，而应由 `max_segments_per_run` 让程序在段末主动退出，
  完成 Last 导出和输出契约校验，并等待 Kaggle 状态变为 `COMPLETE`。
- 手工取消、强制 kill、OOM、机器崩溃只能回到**上一份已发布的 COMPLETE Kernel Output**；
  当前 Worker 即使刚完成多个 segment，也不能把其中的 `last_state.pt` 当作云端断点。
- Kaggle `/kaggle/working` 不是跨 Kernel 持久盘。手工取消或 CLI push 产生的
  `CANCEL_ACKNOWLEDGED` 版本若 `kaggle kernels files <ref>` 返回空列表，就没有可供
  `kernel_source` 挂载的 Output；页面曾经显示 checkpoint 不代表下一个 Kernel 能读取。
  此时禁止引用空 Output、禁止用另一实验的 continuation Dataset 冒充来源。若历史
  checkpoint 没有发布，只能由相同数据清单、底座 SHA、seed、batch 和训练参数重建，
  并用历史首段及目标 Best 指标逐项核对后让恢复任务正常结束。
- 失败先读完整日志，分类根因后只修复一个问题，禁止连续 push 试错版本造成并发。

## 8. 监控和完成判定

- 监控 phase、segment/total、step/total、最新验证指标、历史 Best、GPU 显存和每段耗时。
- 用已完成 segment 平均时间估算剩余时间，并计入启动、下载、导出和接力开销。
- `RUNNING` 不等于完成；只有输出契约全部满足才算 chunk 成功。
- 最后一个 chunk 完成全部 segment 并通过校验后，才称训练完成；随后再做样本外评估。

## 9. 提交前检查表

- [ ] 当前账号和权限正确；同目标没有 active Kernel。
- [ ] slug/title/代码版本一致；实验清单和耗时估算完成。
- [ ] 已实际打开并逐条执行本手册，而不是依赖记忆；P100 任务已固定
  `torch==2.5.1+cu121`，启动日志已验证 `torch.cuda.get_arch_list()` 包含 `sm_60`。
- [ ] 最终生效参数已打印，无隐藏硬编码覆盖。
- [ ] 数据集对象最终长度与实验口径一致；“全量验证”必须在日志中打印验证候选总数、
  最终 Dataset 长度和每段实际验证样本数，三者相等才算通过。
- [ ] 所有配置约束已在本地构造 `Config()` 做过 smoke test，例如 quick/large 样本数关系；
  不能把 Kaggle GPU 任务当配置语法检查器。
- [ ] SwanLab 已在与 Kaggle 相同的环境变量集合下完成初始化 smoke test；需要规避 SDK
  环境变量解析时，必须同时处理当前进程 `os.environ` 和传给训练子进程的 `env`。
- [ ] GPU/PyTorch 兼容，输入唯一，模型哈希正确。
- [ ] 首训/续训模式和 optimizer/scheduler 处理正确。
- [ ] State/Best/Last 或评估结果契约已实现。
- [ ] 正常 chunk 结束、接力和最终完成流程已提前设计。
- [ ] 后续接力输入已提前创建并做过一次挂载验证；没有“训练结束后再临时上传 checkpoint”的步骤。
- [ ] 大文件不会经过本地网络。
- [ ] 接力脚本会继承完整 output 树，日志/metrics 采用追加模式，历史 loss 行数已验证。
- [ ] 提交目录中的 metadata 已实际读取并核对：id、title、code_file、dataset_sources、kernel_sources 均为本次目标值。

## 10. 历史教训

本手册来自 V3 到 V6.01 及 v1-beta 的实际问题：账号切换后资源不可访问、P100 缺少 `sm_60`、硬编码覆盖外部参数、metadata 与实际提交目标不一致、同时启动多个版本、停止状态未确认就重提、只保留 State 导致 Best 丢失、忘记导出 Last、误把临时工作目录当持久盘、让本地中转大文件、首个 chunk 后才补接力方案，以及只继承 checkpoint 导致跨 chunk 日志和 loss 曲线断裂。V6.01 的 chunk2/chunk3 已验证了“预置 continuation 输入后直接接力”的正确流程；v1-beta 必须在首个正式 chunk 前完成该设计，并继承完整 output 树。今后所有 Kaggle 工作都必须遵守本手册。

## 11. v1-beta 已落实的硬化项

- runner 对训练、验证、metadata 输入做唯一性检查，并生成组合 SHA-256 数据清单；续训时清单或实验配置变化会立即拒绝。
- runner 使用 `Popen` 逐行转发训练子进程输出，因此 Kaggle 页面和持久化 `run.log` 保持同一份 step-level 日志；SIGINT/SIGTERM 会转发给训练器。
- `last_state.pt` 携带 resume guard，覆盖窗口、日期、数据清单、loss、学习率、调度器、条件结构和覆盖计划。
- `progress.json` 是实时观测文件，`last_state.pt` 是持久化权重。段末
  `segment=N, step=total_steps` 与 State 的 `segment=N+1, step=0` 先规范化为同一位置；
  段内 `observed_step>0` 与 State 的同段 `step=0` 表示当前段未完成，应丢弃该段日志并
  从 step 1 重跑；跨 segment、反向或无法解释的不一致必须退出。
- 上述“持久化”在单个 Worker 内成立；若 Worker 被取消，只有此前已经发布为
  `COMPLETE` Kernel Output 的 State 能用于下一轮。禁止用页面日志中的最新 segment
  推断存在一个 Kaggle 实际未发布的断点。
- 2026-08-22 实测确认 Kaggle `Stop Session` 可能直接结束容器，不向训练进程传递可捕获的
  `SIGTERM`。v1-beta 因此采用单一、明确的 segment 级恢复：首次训练前保存 Segment 0，
  每段训练和验证完成后原子保存完整 State；中断在段内时重跑整段，不做 step checkpoint。
- `progress.json.current_step` 是持久化位置，段内固定为 0；页面实时位置写入
  `observed_step` 和日志。续训只读取 `last_state.pt.next_epoch`，并强制要求
  `resume_step == 0`，从结构上杜绝根据页面 step 猜恢复点。
- 同一 segment 内“观测位置领先 State”必须写入 `recovery_events.jsonl`、记录并删除该段
  未持久化的 metrics；这是重跑未完成 segment，不是模型回退。跨 segment、反向或无法解释
  的不一致仍在训练前失败。
- 首次验证前先初始化 `best_model`，防止强制中断造成 Best 缺失；后续验证只在指标改善时覆盖 Best。
- `val_data.pkl` 保留 120 日上下文前缀，验证目标仍由 `KRONOS_VAL_SIGNAL_START/END` 单独过滤。

### 11.1 SwanLab 看板接入规则

- v1-beta 专用 runner 优先从环境变量 `SWANLAB_API_KEY` 读取凭据；未设置时从同名
  Kaggle Secret 读取。提交前必须把 Secret 授权给 Kernel，禁止把密钥写入 runner、
  Dataset、`run.log` 或 `metrics.jsonl`。日志只能打印看板是否启用、project、
  experiment 和固定 run id；若两种外部来源都不可用，runner 必须在分配 GPU 前退出。
- 所有 chunk 使用同一组固定标识：`SWANLAB_PROJECT`、`SWANLAB_EXPERIMENT_NAME`
  和 `SWANLAB_RUN_ID`。Best-11 阶段实验的历史 run id 为
  `kronos-v1-beta-twospeed-best11-120d-to-10d`；从 Best 109 建立的正式分支固定使用
  `kronos-v1-beta-best109-official-120d-to-10d`，接力时不得
  为每个 Kernel 生成随机实验名，否则会得到多条断开的曲线。
- runner 启动看板时先读取已持久化的 `metrics.jsonl`，按全局
  `segment * total_steps + step` 回填历史 train/validation 指标，再上传当前
  chunk 的实时 loss、forecast loss、history loss、learning rate 和 segment。
  验证日志本身没有 segment，必须沿用最近一条训练日志的 segment；不能用正则从
  验证行强行提取不存在的字段。
- 正式训练把 SwanLab 作为启动前硬性 preflight：凭据缺失、登录失败或
  `swanlab.init()` 失败时必须在模型下载和训练子进程启动前退出，禁止打印
  `SwanLab disabled` 后继续消耗 GPU。训练开始后的短暂上传失败仍以本地
  `metrics.jsonl` 为准，并在下一次接力时回填。
- v1-beta 固定使用 workspace/project `roc_fu/finance`；启动日志必须出现
  `SwanLab credential configured` 和 `{"swanlab": "enabled", ...}` 后才算通过。
- 新 Kernel 只在确认上一 Kernel 已正常结束并继承完整 output 树后接入同一 run；
  不要在当前 active Kernel 运行时 push 改版代码，避免产生并发版本和重复曲线。

### 11.2 Best-11 双速率实验及后续接力

- runner 必须从环境变量或同名 Kaggle Secret 取得 SwanLab 凭据；启动日志必须先出现
  `SwanLab credential configured`，随后出现 `{"swanlab": "enabled", ...}`。
- 训练器用科学计数法保留 10 位小数输出学习率，避免余弦调度的连续变化在 SwanLab
  中被六位小数压成假台阶；训练 loss 的定义和 scheduler 状态不得因此改变。
- SwanLab `run.log()` 的 payload 只发送数值指标；不得发送 `phase: "train"` 或
  `phase: "validation"` 等字符串。训练与验证使用 `train/*`、`validation/*` 命名空间
  区分，避免 SDK 每次上报都产生 `Unsupported scalar string value`。
- 修改可读源码后必须重建 runner 内嵌归档，并运行测试确认内嵌
  `Kronos/finetune/train_predictor.py` 与预期一致，禁止只改外层脚本。
- Best-11 双速率实验的首次运行只挂载正常发布的
  `kronos-v1-beta-warmup-best11-recovery` Kernel Output。runner 必须核对旧实验
  Best 的 `segment=11`、`objective_loss=2.439993896484375` 和权重 SHA；旧实验实际完成到
  Segment 105 的 Last/State 只用于审计，不得恢复其 optimizer、scheduler 或模型状态。
- 首次运行以 Best-11 模型权重新建 optimizer/scheduler，`resume=False`，从双速率实验的
  Segment 1 开始。只有来源 manifest 已明确为 `KRONOS_SCHEDULER=two_speed` 时，才允许
  进入本实验的 continuation 路径。
- 后续接力用 `kernel_sources` 直接挂载上一轮完整 Kernel Output；不创建 continuation
  Dataset、不经本地下载或上传 checkpoint。接力时只从该 Output 内唯一、完整的
  `last_state.pt` 恢复 optimizer、scheduler、模型及 `next_epoch`，且 `resume_step`
  必须为 0；不根据页面日志猜 segment。
- 复制并追加完整的历史 `run.log` 和 `metrics.jsonl`，复用同一 SwanLab run id；启动后
  核对全局 step 连续，不能生成第二条实验曲线。
- 每 100 step 记录条件/主干输出范数比、两组梯度和权重的 L2/RMS、相对更新强度，
  以及行业 embedding、市值百分位 MLP 输出层权重范数；这些指标必须同时写入
  `metrics.jsonl` 和 SwanLab。
- 新实验第一段及此后每 10 segments 运行条件消融，比较真实条件、无条件、batch 内联合
  打乱条件的 forecast loss，并记录 `Full-None`、`Full-Shuffled` 两个有符号差值。
- 不能把 Kaggle `Stop Session` 当作保存按钮；需要临时改代码时只能让预设 chunk 正常
  结束并形成 `COMPLETE` Output。异常强停只能恢复到上一份已发布的 COMPLETE chunk，
  不是当前 Worker 的上一完整 segment；若没有外部远端断点，本 chunk 内所有进度都会丢失。
- 提交前确认旧 Kernel 已结束且当前没有其他 active Kernel；新版本仍按不超过 12 小时
  的 chunk 运行，Best、Last、State 同时持久化。当前首个正式诊断 chunk 为 120 segments，
  用于完整覆盖 7.5% 条件快速衰减区间；这不是 1-segment smoke test。正式配置固定
  `batch=32`，只对 predictor 启用 FP16 AMP，tokenizer 保持 FP32；batch、AMP 开关和
  scaler 都属于不可漂移的接力状态。

### 11.3 从历史 Best 建立正式分支

- 阶段评估确认 Best 109 后，正式分支首次只读取历史 `best_model`，不得恢复历史
  Last 120 或其 optimizer/scheduler/RNG。必须同时核对 `segment=109`、
  `objective_loss=2.4350784543960815` 和模型 SHA-256
  `134e33d48dcd7dd8a4b59ea1c90d94ad579ddefb151d1364b672fc76bbc27dc0`。
- 新 optimizer 必须把 scheduler 精确定位到前 109 个完整 segment 的全局 step；计算时
  必须逐段计入每遍最后一个不足 20,000 窗口的短 segment，不能使用
  `109 * 1250` 近似。启动日志必须明确打印 `completed_segments=109`、全局 step、两组
  实际 LR 和 `next coverage segment=110`。
- 首次正式分支把历史 `metrics.jsonl` 裁剪到 `segment <= 109` 后回填新 SwanLab run；
  历史分支的 Segment 110-120 不得进入正式曲线。旧 `run.log` 保存为
  `parent_run.log`，正式 `run.log` 从分支事件开始。
- 正式分支 manifest 固定写入 `KRONOS_BRANCH_ORIGIN_SEGMENT=109`。后续只有来源 manifest
  带同一标记时才允许普通 State 接力；缺少标记表示历史父分支，其他值表示错误分支，
  两者均不得被误当作正式 continuation。
- 正式分支每个 chunk 通过 `kernel_sources` 直接挂载上一轮完整 Output，并复用
  `kronos-v1-beta-best109-official-120d-to-10d`。朋友账号首批按 250 segments 上限运行，
  以实测约 158 秒/segment 预留环境安装、验证、Last 导出和 Kaggle 发布时间。

## 12. small_0.1 Kaggle 首训事故复盘（2026-09-03）

本次从提交到真正进入训练耗时约一至两小时，先后产生多个失败版本。问题并不复杂，
主要原因是提交前没有完整执行本手册的 preflight，而是在 Kaggle 上串行暴露本可本地发现的错误。
以后出现失败必须先读取完整 traceback、一次只修一个已确认根因，并重新执行整套 preflight；
不得看到 `RUNNING` 就判断成功，也不得连续猜参数后 push。

### 12.1 失败链路与根因

1. **Kernel 找不到 runner**：入口假定目标脚本已经存在于 Kaggle 工作目录，但实际只上传了
   外层 Kernel 文件。以后提交前必须在一个空临时目录模拟 Kaggle 文件布局，验证入口能够取得
   runner；代码来自 GitHub 时必须打印实际 commit。
2. **数据路径硬编码错误**：错误假定 Dataset 直接挂载在短路径，实际路径带
   `/kaggle/input/datasets/<owner>/<slug>/...`。runner 必须按必需文件和 manifest 自动发现唯一
   数据根，发现零个或多个候选时立即失败，禁止再写固定挂载路径。
3. **SwanLab 环境变量解析失败**：只从传给训练子进程的 `env` 删除了
   `SWANLAB_PROJECT/WORKSPACE/EXPERIMENT_NAME`，但 `swanlab.init()` 在当前进程读取的是
   `os.environ`，因此错误仍然存在。正确处理是先保存显式参数，再同时清理两套环境，最后通过
   `swanlab.init(project=..., workspace=..., experiment_name=...)` 传值。必须在下载模型和启动
   GPU 训练前完成登录与初始化 preflight。
4. **验证配置内部矛盾**：bootstrap 设置 quick validation 为 2,000，同时把 large samples
   设为 0，触发 `large_samples >= quick_samples` 的配置约束。这个错误通过本地构造 `Config()`
   即可发现，不应提交到 Kaggle。更重要的是，临时把 large 改成 2,000 虽能启动，却违背了
   “每个 segment 全量验证”的实验要求，不能把通过配置校验等同于满足实验语义。
5. **误判全量验证开关**：只设置 `KRONOS_VALIDATION_FULL_ONLY=1` 并不会自动取消 Dataset 的
   `KRONOS_VALIDATION_SAMPLES=2000` 上限。日志虽然显示有 123,836 个候选窗口，最终 Dataset
   仍只有 2,000。正确配置还必须设置 `KRONOS_VALIDATION_SAMPLES=0`，并以日志中的
   `Found 123836`、`Using 123836`、`Full-only validation size: 123836` 三项一致为准。
6. **未按文档固定 P100 PyTorch**：Kaggle 最新镜像的 PyTorch 只包含 `sm_70+`，P100 是
   `sm_60`，因此首个 CUDA batch 报 `no kernel image is available for execution on the device`。
   本手册此前已经记录已验证组合 `torch==2.5.1+cu121`，但提交前没有执行。P100 Kernel 必须
   强制安装该版本并在训练前打印、断言 `sm_60`；只看到 `torch.cuda.is_available()` 为真不够。
7. **GitHub clone 瞬时失败**：一次运行在第 2 秒出现 GitHub 认证/连接错误，而同一公开地址前后
   均可访问。入口应使用浅克隆、禁用交互式凭据提示、清理残缺目录并做有限退避重试；同时记录
   commit，避免重试期间分支漂移。网络失败不能与训练代码错误混为一谈。

### 12.2 small_0.1 已验证的正确启动证据

第 9 版最终通过以下证据后才进入训练：

```text
torch: 2.5.1+cu121
cuda_arches: [..., sm_60, ...]
Trainable predictor parameters: 24,819,392
[TRAIN] Found 10661560 possible samples. Using 20000 unique samples per segment.
[VAL] Found 123836 possible samples. Using 123836 unique samples per segment.
Full-only validation size: 123836
Running fixed large validation at Segment 1: 123,836 samples.
```

以后同类任务至少必须看到：代码 commit、数据根、模型 SHA、GPU 型号、PyTorch 完整版本、
CUDA arch、最终训练/验证 Dataset 长度、实际双学习率、SwanLab run URL、首个训练 step、首个
全量验证启动。缺少任一项只能称为“已提交”或“正在启动”，不能称为“训练成功”。

### 12.3 固定执行顺序

1. 本地运行语法检查、单元测试、`Config()` 构造和最小 Dataset 配置检查。
2. 在无 GPU 的 preflight 中验证代码取得方式、数据根唯一性、模型/数据 SHA 和 SwanLab 初始化。
3. P100 环境先固定 `torch==2.5.1+cu121`，打印并断言 `sm_60`，再下载模型、构造 Dataset。
4. 对全量验证核对“候选数 = Dataset 长度 = 实际验证数”，不能只检查布尔开关。
5. 只允许一个 active Kernel；失败后先保存完整日志和根因，再提交一个针对性修复版本。
6. 进入首个训练 step 后继续监控到首个全量验证完成并成功保存 State/Best/Last，才确认首训链路闭环。
