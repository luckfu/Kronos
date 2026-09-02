# Kronos-small `small_0.1`（24.7M）从头建立 A 股模型方案

## 目标与边界

本方案的“从头开始”定义为：不继承任何 Beta v1/v2/v2.1 的模型、优化器、学习率调度器或验证结论，从公开的原始 `NeoQuasar/Kronos-small` 权重开始，重新建立 A 股训练血缘。

有一个必须先澄清的架构事实：原始 `Kronos-small` 是 8 层、`d_model=512`、`ff_dim=1024`、8 头、约 24.7M 参数。历史方案中的“增加两层”不是把模型物理扩成 10 层，而是在 12 层的 `Kronos-base` 上把最后两层设为可训练，并在第 10 层前注入条件。若物理增加两层，参数量和 checkpoint 形状都会改变，不再是 24.7M 的 `Kronos-small`。本方案保持 8 层不变，并将条件注入点设为 `context_layer=6`，对应最后两层适配。

本方案只采用 Beta v2.1 以前已经落地的能力：

- 6 维 OHLCVA tokenizer 与原始自回归路径头；
- 静态行业条件和信号日连续市值百分位条件；
- 新条件分支的零输出初始化；
- 条件层/主干双学习率启动；
- forecast/history 分离的 token loss；
- Beta v2.1 的收益、Barrier、同日偏差和排序辅助 loss。

明确不采用 Beta v3 的动态市值路径、历史市值序列或其数据合同。

## 经验总结

### 1. 架构改动必须与参数目标分开

条件输入应通过独立分支在固定层注入，不能把“最后两层可训练”写成“新增两层”。对于 24.7M 目标，模型配置必须保持 `n_layers=8`，新增参数仅限行业 embedding、市值 MLP、归一化/依赖层和 Beta v2.1 两个辅助头。

### 2. 新条件分支必须从 no-op 开始

行业 embedding、市值 MLP 输出层初始化为零，保证第一个 step 等价于原始 Kronos-small；否则新增条件的随机噪声会掩盖真正的训练收益。条件分支可以使用更高学习率，但必须监控条件输出范数/主干范数比，防止条件能量失控。

### 3. 双学习率只用于启动，不应成为长期解释变量

早期实验用主干 `1e-6`、条件 `1e-5` 让新增分支快速获得可用梯度；后续若仍长期保持十倍差距，容易让模型主要依赖条件层并产生校准漂移。新方案把双速率限定为固定启动阶段，主训练阶段回到统一峰值学习率，所有参数共同更新。

### 4. 训练目标必须区分历史上下文与未来预测

全序列 loss 会被 120 个历史 token 主导，不能准确反映未来 10 日预测质量。主指标使用未来 10 日加权 forecast loss，历史重建只保留小权重 `0.02` 作为锚点；旧的 full-sequence loss 只做兼容记录，不能单独决定模型晋级。

### 5. V2.1 辅助目标必须固定量纲

收益、Barrier 和排序 loss 的数值尺度不同。使用 detached EMA 做训练期无量纲化，辅助目标前 1,000 个 step 线性升权；验证时先用未训练辅助头校准一次五项固定分母，之后所有 segment 和 continuation 都复用同一组分母，禁止每个 checkpoint 重新归一化。

### 6. 抽样验证只能做遥测，不能做最终选择

随机抽样验证适合检查吞吐和代码，但会导致 checkpoint 选择噪声和验证集反复使用。正式训练使用固定、签名、整股票隔离的验证集；若为了速度保留 2,000 条 quick validation，只能作为日志，Best 必须由完整隔离验证或预先约定的完整验证里程碑决定。

### 7. 增训结果暴露了灾难性遗忘风险

只用近期窗口的增训改善了同期指标，却损害旧年份能力。因此这次从头训练不使用“近期增训”语义；完成基础模型后若要滚动更新，必须固定保留 10%--30% 的分层历史回放，并继续保留股票隔离和时间隔离两条验证线。

## 数据合同

Kaggle 数据源使用用户提供的 **Kronos Beta v3 Dynamic Size Continuation 01**，这里只取其中的静态 A 股训练资料，不采用 Beta v3 动态市值路径实现。Kernel 启动后必须先发现并打印实际文件，再分配 GPU：

```text
train_data.pkl       训练股票面板
val_data.pkl         固定验证股票面板
asset_metadata.csv   point-in-time 行业/市值条件（若数据包提供）
data_manifest.json   文件清单和 SHA-256（若数据包提供）
```

训练前必须自动断言：

- 每个窗口为 120 日观察、未来 10 日标签，并保留完整的 `120 + 10 + 1` 行上下文；
- `train_data.pkl` 与 `val_data.pkl` 的股票集合交集为 0；
- 训练信号截止日为最后一个拥有完整未来 10 日标签的日期；
- 行业 ID 的有效范围和 unknown ID 已固定；市值百分位为 `[0, 1]`，缺失值有显式 known 标志；
- 训练窗口不包含固定验证 manifest 中的任何 `(symbol, start_index)`；
- 面板、metadata、验证 manifest 的 SHA-256 写入实验 manifest，continuation 时完全匹配。

如果数据包只有训练面板和一个普通 `val_data.pkl`，必须在第一轮训练前生成一份不可变的 symbol-holdout manifest；不能把训练面板随机切一部分后称为隔离验证。

## 模型合同

### 原始底座

```text
predictor: NeoQuasar/Kronos-small
tokenizer: NeoQuasar/Kronos-Tokenizer-base
d_model: 512
n_layers: 8
n_heads: 8
ff_dim: 1024
s1_bits/s2_bits: 10/10
lookback/predict: 120/10
```

本地对应文件为 `small_model/config.json` 与 `small_model/model.safetensors`。Kaggle 不上传本地大文件，直接从 Hugging Face 下载并记录下载后 `model.safetensors` 的 SHA-256；本地模型只用于启动前的结构和回归检查。

### A 股条件

- `num_sectors=86`，实际 embedding 保留 1 个 unknown 行；
- 不使用离散市值桶作为正式条件：`num_size_buckets=0`；
- `use_size_percentile=True`，用两层 MLP 将 `[percentile, is_known]` 映射到 512 维；
- `context_layer=6`，在第 6 层之后广播行业和市值条件，保留最后两层吸收条件；
- 条件输出层全零初始化；不重置原始 tokenizer 或原始 predictor 权重；
- Beta v2.1 `return_head=4`、`barrier_head=3`，ranking 不新增独立 head，而由收益和 Barrier 概率推导。

## 训练阶段

### 阶段 0：零训练基线与数据验收

1. 加载原始 `Kronos-small` 和 tokenizer，在固定 symbol-holdout 验证集上跑一次：full-sequence、forecast、MAE、Rank IC、方向准确率和预测涨跌比例。
2. 以固定 seed 初始化 A 股条件层和两个 V2.1 辅助头，校准 path/history/return/barrier/ranking 五项验证分母并持久化。
3. 保存 `baseline_manifest.json`、数据 SHA、模型 SHA 和完整验证输出。任何验收失败都不进入 GPU 长训。

### 阶段 1：条件启动（固定 10 个 segment）

目的只是让新增条件获得稳定梯度，不以这一阶段的 Best 晋级。

```text
主干/原始参数学习率：1e-6
行业、市值、辅助头学习率：1e-5
scheduler：warmup_cosine，warmup ratio=1%
trainable：全部 predictor 参数（按参数族使用不同 LR）
loss：forecast + 0.02 * history；V2.1 auxiliary ramp 保持关闭
每 segment：20,000 个无放回训练窗口
验证：固定隔离集 quick 2,000 条，仅作遥测
```

阶段末检查条件输出范数/主干范数比、两族梯度 RMS 和相对更新强度。若条件分支在 10 个 segment 内占据主干能量，先降低条件峰值或延长 warmup，不能直接进入主训练。

### 阶段 2：基础预测主训练（1 遍完整覆盖）

重新建立 optimizer/scheduler，不继承阶段 1 的 scheduler 状态；模型权重从阶段 1 末点开始。

```text
主干与条件统一峰值学习率：5e-6 起步，验证稳定后可升至 1e-5
scheduler：warmup_cosine 或 OneCycle，整个阶段只选一种
loss：weighted forecast + 0.02 * history
forecast horizon 权重：1.364,1.364,1.364,1.136,1.136,0.909,0.909,0.682,0.682,0.455
```

该阶段目标是先恢复 A 股路径预测，不让随机初始化的决策头影响主干。每个 segment 保存 `last_state.pt`；固定验证集每个 segment 至少运行一次完整 forecast loss，Best 以完整隔离 forecast loss 选择。

### 阶段 3：Beta v2.1 决策目标（再 1 遍完整覆盖）

从阶段 2 的 Best 或 Last（必须在 manifest 中明确选择，建议从 Best 开始）建立新 optimizer。启用：

```text
return loss    = 0.80 * horizon Huber + 0.20 * same-date return bias
barrier loss   = inverse-sqrt class-balanced cross entropy
ranking loss   = 同一 signal date、utility gap >= 0.5% 的 pairwise softplus
总权重        = path/history/return/barrier/ranking
                = 0.68/0.02/0.15/0.10/0.05
auxiliary ramp = 前 1,000 step 线性升权
EMA            = detached normalizer，decay=0.99
```

收益标签使用下一交易日开盘为 entry，D1/D3/D5/D10 收盘收益除以 `max(sigma20, 0.005)*sqrt(h)` 并截断到 `[-3,3]`；Barrier 为 `+5%/-3%`，同日双触发样本训练时屏蔽。每个完整验证还要在固定 2,048 个窗口上做 deterministic greedy 10 步生成，记录辅助收益头与生成路径的 MAE、符号一致率、相关性和 bias。

## 验证与晋级

验证分三层，职责不能混淆：

1. **遥测集**：固定 seed 的 2,000 条窗口，快速发现 NaN、OOM、loss 爆炸；不用于 Best。
2. **开发定版集**：完整 symbol-holdout 验证集，每个 segment 运行；Best 使用固定分母的 `beta_v21_score`，并同时保存 path/forecast/history/return/barrier/ranking。
3. **封存未来集**：训练开始前锁定最近至少 20 个完整 signal date，训练和调参都不可读取；只在候选模型完成后运行一次，报告 Rank IC、方向/平衡准确率、MAE、预测涨跌比例、头尾分位收益差、换手和含成本收益。

候选晋级至少满足：

- `beta_v21_score` 相对零训练基线有稳定改善，且不是只靠辅助项下降；
- path forecast loss 相对阶段 0 基线恶化不超过 1%；
- 封存未来集 Rank IC、头尾收益差和含成本收益不低于基线，不能只看方向准确率；
- 预测下跌比例与实际比例的偏差没有扩大，平衡准确率不低于 0.50；
- 2,048 条 consistency 统计没有持续的辅助头/生成路径背离；
- 保存 Best、Last、last_state、manifest、分母文件和 SHA，上一候选保留为回滚点。

## Kaggle 执行与接力

先在 Notebook 中完成一次小规模 smoke test，再提交正式 Kernel。所有 Kernel 只运行一个训练进程，使用 P100/T4 时先检查 CUDA 架构和 AMP 类型；P100 不兼容的 PyTorch wheel 必须切换到支持 `sm_60` 的版本。

```bash
!git clone https://github.com/luckfu/Kronos.git /kaggle/working/Kronos
%cd /kaggle/working/Kronos
!pip install -q -r requirements.txt
!python finetune/verify_colab_setup.py --base-model /kaggle/working/Kronos/small_model
```

正式 runner 应从 Hugging Face 下载 predictor/tokenizer，把数据集挂载路径解析结果写入 `experiment_manifest.json`，并设置：

仓库已新增独立入口 [`kaggle_kronos_small_v21.py`](kaggle_kronos_small_v21.py)，它只调用现有 `train_predictor.py` 和 `export_last_model.py`，不改动任何 Beta v2.1 启动脚本。Notebook 中可直接运行：

```bash
%cd /kaggle/working/Kronos
!KRONOS_SMALL_V21_STAGE=bootstrap python -u finetune/kaggle_kronos_small_v21.py
```

后续阶段显式指定上一阶段模型目录；同一阶段跨 Kernel 接力时，再指定上一 Kernel 的完整 output 树：

```bash
!KRONOS_SMALL_V21_STAGE=main \
  KRONOS_SMALL_V21_PARENT_MODEL=/kaggle/input/<stage1-output>/checkpoints/last_model \
  python -u finetune/kaggle_kronos_small_v21.py

!KRONOS_SMALL_V21_STAGE=v21 \
  KRONOS_SMALL_V21_PARENT_MODEL=/kaggle/input/<stage2-output>/checkpoints/best_model \
  python -u finetune/kaggle_kronos_small_v21.py
```

数据包中存在多个 `train_data.pkl` 时设置 `KRONOS_SMALL_V21_DATA_ROOT` 指向包含 `processed_datasets/` 的唯一目录；metadata 不完整时 runner 会在启动前失败，不会静默训练 unknown 条件。

```bash
export KRONOS_PREDICTOR_PATH=/kaggle/working/models/Kronos-small
export KRONOS_TOKENIZER_PATH=/kaggle/working/models/Kronos-Tokenizer-base
export KRONOS_LOOKBACK_WINDOW=120
export KRONOS_PREDICT_WINDOW=10
export KRONOS_CONTEXT_LAYER=6
export KRONOS_NUM_SECTORS=86
export KRONOS_NUM_SIZE_BUCKETS=0
export KRONOS_USE_SECTOR_FEATURES=1
export KRONOS_USE_SIZE_FEATURES=0
export KRONOS_USE_SIZE_PERCENTILE=1
export KRONOS_USE_BETA_V21_AUXILIARY=1
export KRONOS_TRAINABLE_TRANSFORMER_LAYERS=-1
export KRONOS_TRAIN_SAMPLES_PER_SEGMENT=20000
export KRONOS_VALIDATION_FULL_ONLY=1
export KRONOS_VALIDATION_QUICK_SAMPLES=2000
export KRONOS_VALIDATION_LARGE_SAMPLES=0
export KRONOS_BATCH_SIZE=32
export KRONOS_USE_AMP=1
export KRONOS_AMP_DTYPE=float16
```

正式长训按 `ceil(train_windows / 20,000)` 个 segment 完成两遍覆盖。每个 Kaggle 会话限制在约 100--120 个完整 segment，段末原子保存完整 output 树；下一会话只通过 `kernel_sources` 挂载上一会话的 COMPLETE Output，复制整个 output 树后读取唯一的 `last_state.pt` 接力。不得从页面日志猜 segment，不得只上传权重而丢失 optimizer、scheduler、RNG、metrics 和 manifest。

## 结果记录

每个候选至少保留：

```text
experiment_manifest.json
baseline_manifest.json
data_manifest.json / SHA-256
model_sha256.json
beta_v21_validation_denominators.json
metrics.jsonl
run.log
checkpoints/best_model/
checkpoints/last_model/
checkpoints/last_state.pt
evaluation/sealed_future/
```

最终模型名称建议为 `kronos-small-a-share-v21-from-scratch`。只有封存未来集完成、指标报告和模型 SHA 固化后，才允许把它称为候选版本；训练 loss 下降本身不构成发布依据。
