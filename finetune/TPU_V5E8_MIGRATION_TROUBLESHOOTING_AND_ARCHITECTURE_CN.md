# Kaggle TPU v5e-8 迁移与 8 核分布式训练 — 故障复盘与技术全书

本文档系统性记录在 Kaggle TPU v5e-8 上迁移 Kronos 模型训练、实现 8 核心全硬件释放过程中遇到的**每一次关键报错、底层根本原因剖析及终极修复方案**。

---

## 故障复盘与修复流水账

### 故障 1：`AttributeError: 'MpDeviceLoader' object has no attribute 'sampler'`

* **发生阶段**：初次提交，完成模型初始化与验证集校准后，刚进入第 1 个训练 epoch 时。
* **原始报错日志**：
  ```python
  File ".../finetune/train_predictor.py", line 1767, in train_model
    if isinstance(train_loader.sampler, DistributedSampler):
  AttributeError: 'MpDeviceLoader' object has no attribute 'sampler'
  ```
* **根因剖析**：
  在 PyTorch/XLA 运行环境中，常规 PyTorch `DataLoader` 会被包装为 `torch_xla.distributed.parallel_loader.MpDeviceLoader`，用于异步向 TPU 芯片推流数据。包装类并未透传 `.sampler` 属性，真正的 DataLoader 和 Sampler 被存放在其内部的 `_loader` 私有属性中。
* **修复方案**：
  在 `finetune/train_predictor.py` 中实现安全的属性穿透探测：
  ```python
  train_sampler = getattr(train_loader, 'sampler', None)
  if train_sampler is None and hasattr(train_loader, '_loader'):
      train_sampler = getattr(train_loader._loader, 'sampler', None)
  if isinstance(train_sampler, DistributedSampler):
      train_sampler.num_samples = math.ceil(len(train_dataset) / world_size)
      train_sampler.total_size = train_sampler.num_samples * world_size
      train_sampler.set_epoch(epoch_idx)
  ```

---

### 故障 2：SwanLab 看板 404 引发训练进程“自杀式断言”

* **发生阶段**：Version 8 提交，容器刚拉起、尚未进入 TPU 初始化阶段。
* **原始报错日志**：
  ```text
  [HTTP] POST https://api.swanlab.cn/api/project/roc_fu/finance/experiment -> 404 (306ms)
  [ERR] code=Disabled_Resource message=实验已被删除

  RuntimeError: SwanLab initialization failed; refusing to start TPU training:
  RuntimeError: Failed to start run: API Request Failed: [Disabled_Resource] 实验已被删除
  ```
* **根因剖析**：
  1. 代码中硬编码了固定 `SWANLAB_RUN_ID = "beta-v2-1-c1-tpu-smoke"`，当该实验曾在 Web 界面被手动删除后，SwanLab 远端将该 ID 标为 `Disabled_Resource`，后续试图以该 ID resume 时直接报 404；
  2. 此前的代码修改引入了过度防御机制：在 catch 到 SwanLab 异常后执行了 `raise RuntimeError(...)`，认为“没有看板就不配跑训练”，直接导致排队十几分钟后秒崩。
* **修复方案**：
  在 `finetune/kaggle_beta_v21_c1_tpu.py` 中：
  1. **自动自愈重试**：当带指定 ID 初始化失败时，自动以全新 Run ID 重新发起创建；
  2. **解耦看板与核心训练**：恢复外层降级容错（仅打印警告），保证即使第三方看板服务异常，TPU 训练和 Checkpoint 存储也绝对不受影响。

---

### 故障 3：`ValueError: signal only works in main thread of the main interpreter`

* **发生阶段**：Version 9 提交，TPU 成功拉起 8 个核心线程进入 `main()` 时。
* **原始报错日志**：
  ```python
  [TPU Launcher] Using pjrt.spawn_threads (thread-per-device)
  ...
  File ".../finetune/train_predictor.py", line 2566, in main
    signal.signal(signal.SIGINT, request_safe_stop)
  File "/usr/local/lib/python3.12/signal.py", line 58, in signal
    handler = _signal.signal(_enum_to_int(signalnum), _enum_to_int(handler))
  ValueError: signal only works in main thread of the main interpreter
  ```
* **根因剖析**：
  在单机 8 核心多线程模式（`pjrt.spawn_threads`）下，8 个 TPU worker 分别运行在不同的 Python 子线程中。Python 原生标准库 `signal` 有硬性限制：**系统信号（SIGINT / SIGTERM）只能在主线程中注册**。子线程执行到该行时直接抛出 `ValueError`。
* **修复方案**：
  在 `finetune/train_predictor.py` 中添加主线程隔离保护：
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

### 故障 4：`AttributeError: module 'torch_xla.core.xla_model' has no attribute 'xrt_world_size'` 与 World Size 误判

* **发生阶段**：Version 10 提交，8 个核心已全部绑定设备（`xla:0` ~ `xla:7`），模型初始化完毕执行多核同步时。
* **原始报错日志**：
  ```text
  [XLA Setup] Global Rank: 0/1, Local Rank: 0 -> Using device: xla:0
  ...
  [XLA Setup] Global Rank: 7/1, Local Rank: 7 -> Using device: xla:7
  File ".../finetune/train_predictor.py", line 2609, in main
    distributed_barrier(device, 'kronos_model_initialized')
  File ".../finetune/train_predictor.py", line 99, in distributed_barrier
    if xm.xrt_world_size() > 1:
  AttributeError: module 'torch_xla.core.xla_model' has no attribute 'xrt_world_size'
  ```
* **根因剖析**：
  1. **现代 PJRT 废弃旧 API**：在 PyTorch/XLA 2.x（PJRT 架构）中，旧版 `xla_model` 模块上的 `xm.xrt_world_size()` 早已被彻底废弃移除，调用直接抛 AttributeError；
  2. **多线程拓扑 World Size 误判为 1**：在单进程多线程模式下，进程级 world size 为 1，若未调用设备级 API，会导致系统将 8 核环境误判为 `world_size = 1`，破坏分布式数据分片。
* **修复方案**：
  1. 在 `finetune/train_predictor.py` 中构建现代兼容解析器 `get_xla_world_size()`，优先从现代 PJRT 运行时 `torch_xla.runtime.addressable_device_count()` 获取真实设备数，淘汰所有 `xm.xrt_world_size()`；
  2. 在 `finetune/utils/training_utils.py` 的 `setup_ddp()` 中，确保 `thread_per_device` 模式正确解析 `world_size = 8`。

---

## 终态架构与参数配置（方案 B）

| 维度 | 配置值 | 设计意图 |
| :--- | :--- | :--- |
| **硬件规格** | **TPU v5e-8** (8 Chips, 128GB HBM) | 全硬件利用，激活全部 8 颗芯片 |
| **单卡 Batch Size** | **64** | 充分贴合 TPU MXU 脉动阵列计算密度 |
| **全局 Batch Size** | **512** ($64 \times 8$) | 释放 8 核总吞吐 |
| **Smoke 测试规模** | **1 个 Segment** (512 样本) | 8 核下 1 步完成梯度反向、AllReduce 及 Checkpoint 验证 |
| **多核启动模式** | **双保险自适应** | 优先 thread-per-device，平滑回退 process-per-device |
| **多核同步保障** | **`distributed_barrier`** | 基于 `xm.rendezvous` 解决段间同步与 Checkpoint 写盘去竞态 |

---

### 5. 验证评分计算 ZeroDivisionError 与全核心分母漂移（Version 11）
- **现象**：Rank 0 已经 100% 完成第 1 个 segment 的全部训练与验证，且最优权重 `best_model` 已成功持久化落盘并输出 `Training finished`。但在主线程收集子线程结果时，非 Rank 0 线程抛出崩溃：
  ```python
  File ".../finetune/train_predictor.py", line 2236, in train_model
      large_metrics = evaluate_validation(...)
  File ".../finetune/train_predictor.py", line 1215, in evaluate_validation
      result['beta_v21_score'] = beta_v21_validation_score(result, config)
  File ".../finetune/train_predictor.py", line 826, in beta_v21_validation_score
      + 0.20 * metrics['return_loss'] / returns
  ZeroDivisionError: float division by zero
  ```
- **根本原因**：
  1. `beta_v21_validation_score` 中对 5 个分母 `path, _history, returns, barrier, ranking` 直接进行浮点数裸除，缺失安全防护；
  2. 在自动校准阶段（`beta_v21_auto_calibrate`），分母未设安全下限保护；且 Rank 0 写入 `beta_v21_validation_denominators.json` 后，多线程环境下的非 Rank 0 核心未能同步加载该文件，导致各核心配置分母不一致或为 0。
- **修复措施**：
  1. 在 `beta_v21_validation_score` 中对 5 个分母全部施加 `max(denominator, 1e-6)` 防护，彻底杜绝除零崩溃；
  2. 校准阶段对 5 个分母设置 `max(val, 1e-5)` 保护；
  3. 在 Rank 0 将校准文件落盘并通过 `distributed_barrier` 步进后，所有 rank 统一重新加载持久化 json 文件中的基准 csv，确保 8 核心分母完全一致。

---

## 自动化测试保障

全量测试套件执行通过：
```bash
/opt/miniconda3/bin/pytest tests/test_tpu_training_support.py -v
============================== 17 passed in 1.09s ==============================
```
覆盖用例包括：
1. TPU 启动脚本环境变量引导测试；
2. 单核与多核启动分支测试；
3. `MpDeviceLoader` sampler 安全透传测试；
4. 多线程信号安全注册测试；
5. PJRT 运行时拓扑读取与强制单核模式测试；
6. `beta_v21_validation_score` 除零安全防御与校准分母保护测试。
