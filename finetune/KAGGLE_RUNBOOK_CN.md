# Kaggle 训练宪法

本文件是本项目所有 Kaggle 训练、续训、评估和参数实验的强制规程。它定义“必须做什么、何时可以继续、何时必须停止”。专项计划只能补充参数，不能降低本文件的要求。

接力命令模板见 [`KAGGLE_CONTINUATION_TEMPLATE_CN.md`](KAGGLE_CONTINUATION_TEMPLATE_CN.md)。

## 第一条：先完成实验定义，再启动 GPU

每次提交前必须形成可复核的实验清单，并在启动日志打印最终生效值：唯一实验名和 slug、Git commit、模型/数据/manifest SHA-256、数据隔离规则、日期、lookback、predict、loss、总 segment、coverage、每段样本数、`max_segments_per_run`、batch、seed、AMP、optimizer、scheduler、双学习率、GPU/PyTorch/CUDA、输出位置、接力来源和 SwanLab 标识。配置、Dataset 构造和输入发现必须在本地 smoke test；不能把 Kaggle GPU 当作语法检查器。

## 第二条：同一实验只能有一个活动 Kernel

提交或改版前必须执行：

```bash
kaggle kernels status <owner>/<slug>
kaggle kernels files <owner>/<slug> --format json
```

目标处于 `RUNNING`、`QUEUED` 或取消未完成状态时，禁止 push 新版本。替换任务时先取消，再确认 `CANCEL_ACKNOWLEDGED`、`COMPLETE` 或 `FAILED`，然后提交下一版。`CANCEL_ACKNOWLEDGED` 不是成功，也不代表 Output 已发布。slug 不超过 50 个字符；metadata 的 id、title、code_file、dataset_sources、kernel_sources 必须核对。

## 第三条：输入必须唯一、可验证、可复现

- 数据和模型优先从 `/kaggle/input` 挂载，不经本地电脑中转大文件。
- runner 按必需文件和 manifest 自动发现唯一数据根；0 个或多个候选立即退出。
- 只有被正式 manifest 引用的文件有效；数据、模型和 manifest 必须校验 SHA-256。
- 账号切换后重新检查 owner 和权限。
- 全量验证必须核对并打印“候选窗口数 = 最终 Dataset 长度 = 实际验证样本数”，不相等即失败。

## 第四条：环境先验收，后训练

P100 固定使用已验证环境 `torch==2.5.1+cu121`。入口读取 `nvidia-smi` 确认 GPU 为 P100，并断言 `torch.cuda.get_arch_list()` 包含 `sm_60`；不符就退出，禁止静默降级。GitHub 代码使用浅克隆、禁用交互式凭据提示、有限退避重试，并打印实际 commit；残缺目录清理后再重试。

## 第五条：SwanLab 是启动前硬条件

- 凭据优先读 `SWANLAB_API_KEY`，否则读同名 Kaggle Secret；授权情况下可以写入代码、Dataset 或日志。
- 所有 chunk 固定复用同一 project、experiment name 和 run id，不得随机创建新曲线。
- 在下载模型和启动训练前完成登录及 `swanlab.init()` smoke test；失败必须退出。
- 启动日志必须出现凭据已配置、看板已启用及 project/experiment/run id。
- 上报只发送数值 payload，使用 `train/*`、`validation/*` 命名空间；本地 `metrics.jsonl` 是事实来源，短暂网络故障由下一段回填。

## 第六条：只允许 segment 级断点恢复

首次训练建立 Segment 0 State。每个 segment 的训练和验证全部完成后，原子保存 `last_state.pt`，内容包括模型、optimizer、scheduler、RNG、AMP GradScaler（如启用）、实验配置、数据清单、loss/学习率/coverage guard 和 `next_epoch`。`next_epoch=N` 表示前 N 个 segment 已完整完成，续训从 N+1、`resume_step=0` 开始。禁止根据页面日志或最近 step 猜断点；段内中断整段重跑。计划停止必须由 `max_segments_per_run` 让程序段末正常退出；不得把 `Stop Session` 当保存按钮。强停、OOM、崩溃或空 Output 只能恢复上一份已发布的 `COMPLETE` Output。

## 第七条：正常 chunk 必须发布完整 Output

只有状态为 `COMPLETE`、`kaggle kernels files` 非空，且以下契约全部满足时，chunk 才能接力：

```text
<output>/
├── run.log
├── metrics.jsonl
├── progress.json
├── summary.json
├── experiment_manifest.json
└── checkpoints/
    ├── last_state.pt
    ├── best_model/model.safetensors
    ├── best_model/config.json
    ├── best_model/best_metric.json
    ├── last_model/model.safetensors
    └── last_model/config.json
```

`last_state.pt` 是唯一续训来源；`best_model` 是全实验历史最优模型；`last_model` 是最近完整 checkpoint 的推理模型。三者不可互换。首次验证前必须初始化 Best。

## 第八条：接力必须挂载上一份完整 Output

- 首个正式 chunk 启动前准备好后续 `kernel_sources` 或服务端 continuation Dataset，并完成挂载验证。
- 下一 Kernel 在 `/kaggle/input` 必须找到恰好一个 continuation State，并验证同一 Output 含 State、Best、Last、日志、指标、progress、summary、manifest；0 个或多个均失败退出。
- 必须复制完整 output 树，不能只复制 `checkpoints/`。
- `run.log`、`metrics.jsonl` 追加写入，记录全局 segment、global_step、pass 和时间戳。
- 启动日志打印 `continuation_output`、`resume_state`、历史 metrics 行数、Best loss、`next_epoch` 和下一 segment。
- schedule、coverage 顺序、总 segment、数据清单和实验配置必须与来源一致。

## 第九条：所有 Kaggle 代码必须实时输出且可审计

所有子进程设置 `PYTHONUNBUFFERED=1`；外层逐行转发必须使用：

```python
print(line, end="", flush=True)
```

页面日志和 `kaggle kernels logs --follow` 必须能实时观察当前阶段；持久化 `run.log` 和 `metrics.jsonl` 是历史事实来源。

本条适用于训练、续训、评估、数据构建、模型下载、依赖安装和任何其他 Kaggle Kernel；不能只在训练循环里加 `print`。每个入口必须在每个可能持续超过 10 秒的阶段前后打印带时间戳的状态行（例如 `phase=install_dependencies`、`phase=download_model`、`phase=load_dataset`、`phase=training`、`phase=validation`、`phase=export`），并将相同内容以行缓冲方式追加到持久化 `run.log`。入口进程必须设置 `PYTHONUNBUFFERED=1`；调用子进程必须使用 `-u` 或等价无缓冲设置、`stdout=PIPE`、`stderr=STDOUT`、`text=True`、`bufsize=1`，父进程逐行 `print(line, end="", flush=True)` 转发，同时写入 `run.log`。禁止使用 `-q` 掩盖关键阶段错误，禁止只依赖 tqdm 的回车刷新，禁止让下载/安装/模型加载阶段无状态输出。

**硬门槛（不可例外）**：入口第一条带时间戳的 `phase=started` 必须在进程启动后立即输出。Kernel 进入 `RUNNING` 后，允许平台采集最多 60 秒的宽限；超过 60 秒 `kaggle kernels logs --follow` 仍无任何事件，且页面也没有日志，或任一长阶段超过 60 秒没有心跳，才按“实时日志契约失败”处理，优先检查并修复代码的缓冲、管道转发和阶段埋点。状态判定仍以 `kaggle kernels status` 为准；不得把任何日志接口的空结果直接解释为任务失败或完成，也不得在旧 Kernel 仍活动时并发 push。每次监控必须同时记录状态、最后一条日志时间、当前 phase、Output 文件列表和检查时间。

### 实时日志命令不可省略 `--follow`

Kaggle CLI 2.2.4 中，下面两个命令调用的不是同一个服务端接口，禁止混用：

```bash
# 运行中实时日志：正确。连接 SSE 日志流，行为类似 tail -f。
kaggle kernels logs --follow <owner>/<slug>

# 一次性持久化日志：运行中可能合法返回空字符串，仅适合任务结束后读取。
kaggle kernels logs <owner>/<slug>
```

不带 `--follow` 时，CLI 调用 `list_kernel_session_output` 并只打印返回对象中的 `log` 字段。Kernel 仍处于 `RUNNING` 时，Kaggle 可能尚未生成最终持久化日志 Blob，因此命令退出码为 0、stdout 长度为 0；即使浏览器页面正在正常显示日志，也会出现这个结果。这不是训练代码丢失输出，不能据此重启、取消或提交新版本。

带 `--follow` 时，CLI 连接 `/api/v1/kernels/logs/stream/<owner>/<slug>` 的 SSE 实时流。服务端会先重放当前 session 已有日志，再持续等待新事件，直到任务结束或用户按 `Ctrl-C`。只想取短快照时可在外层限制时长，例如：

```bash
timeout 20 kaggle kernels logs --follow <owner>/<slug>
```

macOS 默认没有 GNU `timeout`；可直接运行 `--follow` 后按 `Ctrl-C`，或使用下面的 Python 只读回退。回退调用的也是实时 SSE 接口，不会修改 Kernel：

```python
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
for event in api.kernels_logs_stream("<owner>/<slug>"):
    data = event.get("data")
    if data:
        print(data, end="" if data.endswith("\n") else "\n", flush=True)
```

实时日志排查顺序固定如下：

1. `kaggle config view` 确认 CLI 账号与 Kernel owner 一致。
2. `kaggle kernels status <owner>/<slug>` 确认当前状态。
3. 运行 `kaggle kernels logs --follow <owner>/<slug>`，不得先用无 `--follow` 的空结果下结论。
4. 若 `--follow` 能看到 `phase=started`、设备、segment/step 和验证心跳，日志契约正常。
5. 若 `--follow` 超过 60 秒无事件但页面有日志，归类为 CLI/SSE 传输差异；保持任务运行，使用页面或 Python SSE 回退交叉验证，禁止改训练代码。
6. 只有 `--follow` 与页面均超过 60 秒无入口/阶段心跳，才检查缓冲和管道转发代码。

## 第十条：监控和完成判定

```bash
kaggle kernels status <owner>/<slug>
kaggle kernels logs --follow <owner>/<slug>  # RUNNING 时实时监控
kaggle kernels logs <owner>/<slug>           # COMPLETE 后读取持久化完整日志
kaggle kernels output <owner>/<slug> -p /tmp/<check> -o
```

持续记录 phase、segment/total、step/total、训练/验证 loss、历史 Best、GPU、单段耗时和剩余时间，并预留安装、下载、导出和发布时间。`RUNNING` 不等于完成；只有完整 Output 契约通过才算 chunk 成功，最后一个 chunk 通过全部 segment 和最终校验后才算训练完成。

## 第十一条：失败处置顺序

1. 停止继续提交，保留 Kernel 引用。
2. 用 CLI 拉取完整日志、状态和 Output 文件列表。
3. 归类为输入/路径、代码/配置、依赖/GPU、SwanLab、资源/超时或发布/接力。
4. 只修复一个已证实根因，在本地重新执行全部 preflight 和测试。
5. 确认旧 Kernel 已结束后，只提交一个新版本。

禁止连续猜参数、并发 push、用页面日志猜 checkpoint、用 Last 冒充 Best、用空 Output 接力，或用另一实验 Dataset 冒充来源。

## 最小提交检查表

- [ ] 实验清单、最终配置和 commit 已记录
- [ ] 唯一 active Kernel、metadata、账号权限已确认
- [ ] 数据/模型/manifest 唯一且 SHA 通过
- [ ] P100、PyTorch 2.5.1+cu121、`sm_60` 通过
- [ ] Config、Dataset、SwanLab 已 smoke test
- [ ] 验证候选数 = Dataset 长度 = 实际验证数
- [ ] State/Best/Last、日志、指标、progress、summary、manifest 契约已实现
- [ ] `max_segments_per_run` 和接力来源已提前验证
- [ ] 所有入口无缓冲；安装/下载/加载/执行/导出均有时间戳心跳并同步写入 `run.log`
- [ ] 所有子进程 `stdout/stderr` 合并、逐行转发、父进程 `flush=True`；无静默长阶段
- [ ] 实时监控使用 `kaggle kernels logs --follow`；未把无 `--follow` 的空结果误判为无日志
- [ ] 监控命令、完成判定和失败回滚点已写入任务记录

## 附录：规则来源

历史事故已归并为以上规则：入口路径、manifest 重复、SwanLab 环境变量、验证上限与全量语义冲突、P100 缺少 `sm_60`、GitHub 瞬时网络、并发 push、手工停止未发布 Output、只继承 checkpoint 导致日志断裂，以及 stdout 缓冲。今后执行硬规则，不再以流水账替代流程。
