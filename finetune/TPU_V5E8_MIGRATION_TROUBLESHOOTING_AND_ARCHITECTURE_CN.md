# Kaggle TPU v5e-8 迁移与 8 核大规模分布式训练 — 工程实战与技术全书

本文档系统性总结在 **Kaggle TPU v5e-8（8 Chips, 128GB HBM）** 上迁移 Kronos 金融时间序列模型（Beta v2.1 架构）、实现 8 核心全硬件算力释放、全量验证集评测、9 小时分块接力（Chunking Relay）及生产级稳定性落地的**全套工程经验、排障复盘与核心设计决策**。

---

## 目录
1. [Kaggle TPU v5e-8 硬件与现代 PJRT 架构特性](#1-kaggle-tpu-v5e-8-硬件与现代-pjrt-架构特性)
2. [全生命周期故障复盘与排障流水账](#2-全生命周期故障复盘与排障流水账)
   - [故障 1：MpDeviceLoader 导致属性缺失](#故障-1mpdeviceloader-导致属性缺失)
   - [故障 2：SwanLab 看板 404 引发自杀式断言](#故障-2swanlab-看板-404-引发自杀式断言)
   - [故障 3：子线程注册系统信号报错](#故障-3子线程注册系统信号报错)
   - [故障 4：现代 PJRT 废弃旧 API 与 World Size 误判](#故障-4现代-pjrt-废弃旧-api-与-world-size-误判)
   - [故障 5：分母漂移与验证评分除零崩溃（Smoke v11）](#故障-5分母漂移与验证评分除零崩溃smoke-v11)
3. [TPU 8 核心分布式工程关键设计](#3-tpu-8-核心分布式工程关键设计)
   - [启动模式：为什么坚持 Thread-per-Device？](#启动模式为什么坚持-thread-per-device)
   - [DataLoader 进程模型：为什么必须 num_workers=0？](#dataloader-进程模型为什么必须-num_workers0)
   - [多核心同步与权重写盘竞态消除](#多核心同步与权重写盘竞态消除)
4. [9 小时硬限制下的分块接力（Chunking Relay）机制](#4-9-小时硬限制下的分块接力chunking-relay机制)
   - [7.5 小时安全超时守护机制](#75-小时安全超时守护机制)
   - [段边界（Segment Boundary）停机与持久化状态契约](#段边界segment-boundary停机与持久化状态契约)
   - [跨 Chunk 挂载与无缝断点续训](#跨-chunk-挂载与无缝断点续训)
5. [验证集评估策略：100% 全量校验的最佳实践](#5-验证集评估策略100-全量校验的最佳实践)
6. [学习率调度深度决策：Warmup Constant 的必然性](#6-学习率调度深度决策warmup-constant-的必然性)
7. [SwanLab 在线监控的高可用与看板隔离实践](#7-swanlab-在线监控的高可用与看板隔离实践)
8. [模型架构血统与目录规范化说明](#8-模型架构血统与目录规范化说明)
9. [自动化单测与工程质量保障体系](#9-自动化单测与工程质量保障体系)

---

## 1. Kaggle TPU v5e-8 硬件与现代 PJRT 架构特性

* **硬件拓扑**：单机 4 个张量处理芯片（双 Core 设计），共 **8 个独立可寻址 TPU Core**（Device: `xla:0` ~ `xla:7`），配备 **128GB HBM** 超高带宽显存。
* **软件栈**：PyTorch/XLA 2.x（基于 Google 现代统一开放硬件接口 **PJRT (Partially Just-in-Time)** 运行时架构），废弃了早期基于 XRT (XLA Runtime) 的旧式 API。
* **计算特性**：依赖 MXU (Matrix Multiply Unit) 脉动阵列，对 Batch Size 和维度具有极高的矩阵对齐要求（推荐为 8、16、64 的倍数）。单核 Batch 设为 64 时，全局 Batch Size 为 $64 \times 8 = 512$，能最大化榨干张量算力。

---

## 2. 全生命周期故障复盘与排障流水账

### 故障 1：`MpDeviceLoader` 导致属性缺失
* **发生阶段**：初次提交，完成模型初始化刚进入第 1 个训练 epoch 时。
* **错误日志**：
  ```python
  File ".../finetune/train_predictor.py", line 1767, in train_model
    if isinstance(train_loader.sampler, DistributedSampler):
  AttributeError: 'MpDeviceLoader' object has no attribute 'sampler'
  ```
* **根本原因**：PyTorch/XLA 的 `MpDeviceLoader` 用于向 TPU 异步推流，它封装了原生 DataLoader，但并没有向外代理透传 `.sampler` 属性，真正的采样器位于内部私有属性 `_loader` 中。
* **解决办法**：实现属性穿透探测：
  ```python
  train_sampler = getattr(train_loader, 'sampler', None)
  if train_sampler is None and hasattr(train_loader, '_loader'):
      train_sampler = getattr(train_loader._loader, 'sampler', None)
  ```

---

### 故障 2：SwanLab 看板 404 引发自杀式断言
* **发生阶段**：Version 8 提交，容器拉起尚未进入 TPU 初始化阶段。
* **错误日志**：
  ```text
  [HTTP] POST https://api.swanlab.cn/api/project/roc_fu/finance/experiment -> 404 (306ms)
  [ERR] code=Disabled_Resource message=实验已被删除
  RuntimeError: SwanLab initialization failed; refusing to start TPU training
  ```
* **根本原因**：代码中硬编码了固定的 `run_id`。一旦用户在 Web 前端手动清理或重置了历史看板，远端服务会将该 ID 置为不可恢复的 `Disabled_Resource`；外层代码又引入了过度防御机制（抛异常退出），导致排队十几分钟后秒崩。
* **解决办法**：
  1. **自愈容错**：优先允许以动态全新 ID 启动，若显式传参失败则自动回退以无 ID 方式初始化；
  2. **监控降级**：看板仅作为遥测辅助，其失败绝不可阻断 TPU 本地训练和 Checkpoint 落盘。

---

### 故障 3：子线程注册系统信号报错
* **发生阶段**：Version 9 提交，TPU 成功拉起 8 个核心线程进入 `main()` 时。
* **错误日志**：
  ```python
  [TPU Launcher] Using pjrt.spawn_threads (thread-per-device)
  ...
  File ".../finetune/train_predictor.py", line 2566, in main
    signal.signal(signal.SIGINT, request_safe_stop)
  ValueError: signal only works in main thread of the main interpreter
  ```
* **根本原因**：在单进程 8 线程模式（`pjrt.spawn_threads`）下，8 个 TPU worker 运行在独立的子线程中。Python 标准库规定信号处理器（SIGINT / SIGTERM）**只能在主解释器的主线程中注册**。
* **解决办法**：添加主线程归属检测隔离：
  ```python
  try:
      import threading
      if threading.current_thread() is threading.main_thread():
          signal.signal(signal.SIGINT, request_safe_stop)
          signal.signal(signal.SIGTERM, request_safe_stop)
  except (ValueError, AttributeError):
      pass
  ```

---

### 故障 4：现代 PJRT 废弃旧 API 与 World Size 误判
* **发生阶段**：Version 10 提交，8 核心初始化执行多核同步时。
* **错误日志**：
  ```text
  [XLA Setup] Global Rank: 0/1, Local Rank: 0 -> Using device: xla:0
  File ".../finetune/train_predictor.py", line 2609, in main
    distributed_barrier(device, 'kronos_model_initialized')
  File ".../finetune/train_predictor.py", line 99, in distributed_barrier
    if xm.xrt_world_size() > 1:
  AttributeError: module 'torch_xla.core.xla_model' has no attribute 'xrt_world_size'
  ```
* **根本原因**：
  1. PJRT 架构下，旧的 `xm.xrt_world_size()` 已被彻底删除；
  2. 在多线程模式下，操作系统进程只有一个，如果不显式调用 PJRT 运行时设备探测，系统会误判为 `world_size = 1`，导致数据集无法分片，各核心做重复运算。
* **解决办法**：
  1. 使用现代兼容的 `torch_xla.runtime.addressable_device_count()` 替代已废弃的旧接口；
  2. 在 `setup_ddp()` 中建立线程模式识别逻辑，确保获取真实的 `world_size = 8`。

---

### 故障 5：分母漂移与验证评分除零崩溃（Smoke v11）
* **发生阶段**：Version 11 冒烟训练第 1 段顺利跑完并成功写入 `best_model` 后，主线程回收线程结果时。
* **错误日志**：
  ```python
  File ".../finetune/train_predictor.py", line 1215, in evaluate_validation
      result['beta_v21_score'] = beta_v21_validation_score(result, config)
  File ".../finetune/train_predictor.py", line 826, in beta_v21_validation_score
      + 0.20 * metrics['return_loss'] / returns
  ZeroDivisionError: float division by zero
  ```
* **根本原因**：
  1. `beta_v21_validation_score` 对 5 个分母直接裸除，无安全防线；
  2. 自动校准（Auto-Calibrate）阶段由 Rank 0 生成分母并持久化，但在多线程内存中未将更新后的配置同步给其他 7 个核心，导致非 Rank 0 核心的分母为 0。
* **解决办法**：
  1. 实施安全除法保护：`max(denominator, 1e-6)`；
  2. 校准分母设硬性底限：`max(val, 1e-5)`；
  3. Rank 0 写入基准文件并通过 `distributed_barrier` 后，**所有核心强制从文件重新加载统一的分母配置**。

---

### 故障 6：全量 12 万验证集大张量累积与 `xm.all_gather` 导致 OOM / `<Signals.SIGKILL: 9>`（Chunk 1 Version 1）
* **发生阶段**：正式训练首跑（Chunk 1 v1），顺利跑完第 1 个段的全部 40 步训练，进入全量验证 `Running fixed large validation at Segment 1: 123,836 samples` 后，进程突然以退出码 137 / `<Signals.SIGKILL: 9>` 阵亡。
* **错误日志**：
  ```text
  subprocess.CalledProcessError: Command '['/usr/local/bin/python', '-u', '.../tpu_train_entry.py']' died with <Signals.SIGKILL: 9>.
  ```
* **根本原因（从第一性原理深挖）**：
  1. **无节制内存累积**：在 `evaluate_validation` 循环中，代码原本对每一个 batch 都在 Python 列表中 append 预测和 labels（`validation_auxiliary.append(...)`）。在 Smoke 时仅 512 条样本，内存无感知；但在全量 123,836 条样本下，8 个核心（同属一个单进程内的 8 个线程）在 Host RAM 中累积了近 2000 个包含多维张量的 Python 字典，内存呈线性暴涨；
  2. **致命的跨核心集合通信**：循环结束后，代码试图将 8 个核心各自持有的 15,480 条多维数据重新 `.to(device)` 塞回 TPU 芯片，并由 8 个核心同时调用 `xm.all_gather(..., dim=0)` 强行在跨芯片互联网络中拼接一个 123,836 行的超大张量，再 `.cpu()` 拉回内存，重复执行 6 个字典键！这引发了 XLA 巨型图编译与跨核传输内存雪崩，Kaggle 宿主机内存耗尽，触发 Linux 内核 OOM Killer 发送 SIGKILL: 9 将子进程强制杀死；
  3. **重复多余的设计**：事实上，各 batch 的 auxiliary losses 已经在循环体内被加权累加进 `sums` 字典中，通过后续一行极度轻量（仅十几个浮点数）的 `totals = xm.all_reduce(xm.REDUCE_SUM, totals)` 即可在毫秒级内无内存损耗地得到全量验证集的加权平均损失。后面的 `xm.all_gather` 完全是冗余且致命的。
* **解决办法**：
  1. **解除内存累积**：在 TPU（`device.type == 'xla'`）下彻底跳过 `validation_auxiliary.append`，使全量验证循环在内存中“只算损失、不存数据”，零内存增长；
  2. **跳过大张量 all_gather**：当 `local_validation` 为 None 时，跳过 `xm.all_gather`，直接使用由 `xm.all_reduce` 汇总的各分量加权平均损失计算 `beta_v21_score`；
  3. **增加进度反馈**：验证循环中每 50 个 batch 打印一次 `[VAL] Processed X/242 batches...`，并在训练环境变量中将 `KRONOS_LOG_INTERVAL` 默认设为 10，让控制台输出实时可见。

---

## 3. TPU 8 核心分布式工程关键设计

### 启动模式：为什么坚持 Thread-per-Device？
在 Kaggle TPU 环境中，启动 8 核心主要有两种方式：
1. **进程模式（Process-per-Device，如 `xmp.spawn`）**：在 Linux 下使用 `fork()` 启动子进程。由于底层的 `libtpu.so` 驱动及 PJRT C-API 状态在父进程中已被加载初始化，fork 后的子进程无法安全继承硬件句柄，极易引发死锁、内存泄露或报驱动已被占用的错误。
2. **线程模式（Thread-per-Device，`pjrt.spawn_threads`，推荐定案）**：单个主进程内拉起 8 个线程，各自独占绑定一个 TPU Core。共享内存空间使得进程间通信开销极低，消除了进程序列化和驱动抢占问题，是 Google 推荐并经我们验证 100% 稳定的方案。

### DataLoader 进程模型：为什么必须 `num_workers=0`？
在 TPU 训练中，必须将 `DataLoader` 的工作进程数设为 0：
- **原因**：若 `num_workers > 0`，PyTorch 会在各个 TPU 核心线程中再次 fork 子进程加载数据。在 PJRT 运行时中，子进程会继承父线程的 XLA 设备上下文，导致数据读取阶段不可预测地卡死在管道等待上。
- **性能补偿**：配合 `MpDeviceLoader` 的后台双缓冲队列异步推流，在 TPU 上 `num_workers=0` 即可完全吃满计算阵列，无需多进程预加载。

### 多核心同步与权重写盘竞态消除
- **集合同步（Rendezvous）**：段间同步和校准阶段的等待必须使用 PyTorch/XLA 原生的 `xm.rendezvous('tag')`。这能在 XLA 运行时层面挂起各核心直到集合通信完成。
- **写盘去竞态**：所有 Checkpoint（`best_model`、`last_model`、`last_state.pt`、`progress.json`）的写盘操作**严格限定只有 `rank == 0` 执行**。非 0 核心在同步点通过 barrier 等待写盘完成，严禁多核心并发写入同一目录破坏权重文件。

---

## 4. 9 小时硬限制下的分块接力（Chunking Relay）机制

Kaggle 对 TPU 容器施加了严苛的 **9 小时最长运行时间限制（Wall-clock Time Limit）**，超时容器将被强制杀除（SIGKILL），且无法保存任何中间产物。为了完成多个 Epoch 的全量深度训练，必须依靠分块接力。

### 7.5 小时安全超时守护机制
在环境变量中配置 `KRONOS_MAX_RUNTIME_SECONDS="27000"`（7.5 小时）：
- 为全量验证评估（每次约 20~25 秒）、权重 safetensors 转换、指标落盘预留充足的安全余量；
- 避免触碰 Kaggle 平台的 9 小时硬上限。

### 段边界（Segment Boundary）停机与持久化状态契约
训练主循环在每个 segment 结束且完成验证评估后，检测已消耗时长：
```python
if max_runtime_seconds > 0 and (time.time() - start_time) >= max_runtime_seconds:
    print("Reached maximum runtime budget; stopping safely at segment boundary...")
    # 触发安全停机逻辑
```
- **原子落盘**：
  1. `checkpoints/last_state.pt`：包含优化器、学习率调度器、RNG 种子及当前完成段号；
  2. `checkpoints/best_model/` 与 `checkpoints/last_model/`：完整的 safetensors 权重；
  3. `progress.json`：标明 `status: "stopped"` 和已完成的全局步数；
  4. `metrics.jsonl`：追加写入各段训练与验证损失。

### 跨 Chunk 挂载与无缝断点续训
- **Chunk 1**：从初始预训练权重从头启动，训练至 7.5 小时在段边界安全退出；
- **Chunk 2**：在 Kaggle 界面将 Chunk 1 的 Output 作为 Dataset 挂载到 `/kaggle/input` 下；
- **自愈加载**：脚本中的 `copy_continuation_if_requested` 会自动探测 input 中的 `last_state.pt`，校验 manifest 契约后将其拷贝至工作区，自动设置 `KRONOS_RESUME_TRAINING=1`，无需人工修改代码即实现无缝接力续训。

---

## 5. 验证集评估策略：100% 全量校验的最佳实践

针对模型评估的无偏性与计算成本，我们做了严格对比：

| 维度 | 方案 1：100% 全量校验（当前定案） | 方案 2：抽样校验（如固定 20K 样本） |
| :--- | :--- | :--- |
| **样本量** | **123,836 条**（全量验证池） | 20,480 条（约 1/6） |
| **TPU 8核耗时** | **约 20 ~ 25 秒**（单核仅 242 步） | 约 4 ~ 5 秒 |
| **评测偏差** | **绝对零偏差**，反映全局泛化能力 | 存在局部时间段/截面的抽样随机抖动 |
| **综合性价比** | **极高**。相对于每段训练几分钟，20秒开销极低 | 节省时间微弱，但破坏了最佳模型选取的精确性 |

**实践定案**：
将 `KRONOS_VALIDATION_SAMPLES="0"` 且 `validation_full_only="1"`，在 TPU 强大的分布式吞吐下，每次段评估直接跑全量 123,836 条，确保 `best_model` 是全局最真实的优质权重。

---

## 6. 学习率调度深度决策：Warmup Constant 的必然性

在深度微调过程中，放弃传统的余弦退火（Cosine Annealing），坚定采纳 **`warmup_constant`**：

1. **峰值学习率**：提升至 **`3e-5`**（初始学习率为 `3e-6`）；
2. **预热区间**：占总步数的前 1%（约几十步），快速平滑过渡；
3. **恒定保持**：预热完成后，**全程保持 3e-5 恒定不变直到训练结束**。

### 为什么在分块接力训练中禁用余弦衰减？
- 余弦衰减依赖固定的总步数计划。如果在 Chunk 1 中按 3 个 Epoch 计划余弦衰减，学习率会在当前 Chunk 结束时过早塌缩至接近 0，严重抑制模型参数更新；
- 而常数保持（Warmup Constant）让模型在整个跨 Chunk 训练生命周期内持续保有适度的特征适应能力，非常适合金融多因子时间序列的增量微调。

---

## 7. SwanLab 在线监控的高可用与看板隔离实践

为防止历史看板被覆盖冲撞，并避免远端异常阻断训练：
1. **看板隔离**：
   - 实验命名规范化：`beta_v2_1_c1_tpu_chunk1`；
   - 默认将 `run_id` 设为 `None`，交由 SwanLab 自动分配全局唯一的 Run ID，每次全新训练均创建独立看板；
   - 标签对齐为 `["kronos-base", "beta_v2.1", "tpu_v5e8", "chunk1"]`。
2. **断网与异常降级**：
   - SwanLab 初始化代码全部置于独立 `try...except` 块中；
   - 即使 Kaggle 容器偶发 DNS 解析失败或 SwanLab 接口波动，本地训练依然继续运行，所有指标依然完整落盘于 `metrics.jsonl` 和 `run.log`。

---

## 8. 模型架构血统与目录规范化说明

### 模型血统澄清
- 当前训练的架构是标准的 **Base 模型（12 层 Transformer, d_model=832, n_heads=16, 约 102M 参数）**，即 **`Beta v2.1`**；
- 脚本中曾出现的 `kronos_small_v21` 仅是旧实验模板的环境变量默认字面量，已在本次改造中彻底重构。

### 工作区与输出目录规范
- **默认工作区**：`/kaggle/working/kronos_beta_v21`；
- **输出模型路径**：`/kaggle/working/kronos_beta_v21/outputs/models/beta_v2_1_c1_tpu`；
- **元数据落盘**：同时生成 `beta_v21_manifest.json` 与向前兼容的 `small_v21_manifest.json`。

---

## 9. 自动化单测与工程质量保障体系

在本地建立了严格的测试断言验证集：[`tests/test_tpu_training_support.py`](file:///Users/fupengcheng/Documents/Kronos/tests/test_tpu_training_support.py)。

```bash
/opt/miniconda3/bin/pytest tests/test_tpu_training_support.py -v
```

**覆盖的 17 项核心工程断言**：
1. `test_tpu_entry_bootstraps_project_import_paths_before_worker_imports`：验证 Python 导入路径与 XLA 环境变量的初始化顺序；
2. `test_tpu_entry_spawns_single_worker_and_runs_trainer_config`：验证单核与多核启动分支以及环境变量注入；
3. `test_kaggle_runner_uses_extracted_repo_for_child_pythonpath`：验证解压后的仓库路径注入；
4. `test_c1_runner_uses_hardcoded_swanlab_api_key`：验证凭证隔离与无交互登录；
5. `test_c1_kernel_requests_tpu_machine_shape`：验证 Kaggle 元数据中的 `TpuV5E8` 硬件机型与无 GPU 标记；
6. `test_xla_launcher_cannot_be_overridden_by_torchrun_env`：验证 XLA 启动器免疫外部进程干扰；
7. `test_xla_loader_skips_disabled_quick_validation_loader`：验证全量校验模式下快速加载器的安全跳过；
8. `test_c1_runner_environment_passes_config_validation`：验证 20,480 样本/段、0 快速校验、3 Epochs、7.5 小时超时守护参数的正确性；
9. `test_xla_resume_checkpoint_uses_xla_serializer_and_rng_state`：验证 XLA 专用检查点序列化与随机数恢复；
10. `test_xla_export_uses_cpu_state_dict_serializer`：验证 safetensors 导出时的 CPU 状态字典序列化；
11. `test_xla_setup_supports_pjrt_runtime_topology_api`：验证 PJRT 运行时设备拓扑 API 现代接口调用；
12. `test_xla_setup_can_force_single_process_topology`：验证单卡回退拓扑；
13. `test_xla_setup_reads_pjrt_runtime_topology`：验证 8 核心多线程拓扑世界大小与设备编号识别；
14. `test_manifest_records_parent_model_architecture`：验证 `tpu_v5e8_b64` 配置与父模型 12 层架构记录；
15. `test_tpu_trainer_safely_resolves_sampler_from_mp_device_loader`：验证分布式采样器的穿透解析；
16. `test_tpu_trainer_registers_signals_only_in_main_thread`：验证主线程系统信号安全注册；
17. `test_beta_v21_validation_score_safe_against_zero_denominators`：验证分母异常或为 0 时 `beta_v21_score` 的计算鲁棒性。
