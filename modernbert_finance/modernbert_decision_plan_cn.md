# Kronos Kairos 金融决策模型方案

版本：v1.6
日期：2026-09-29
状态：第一轮完成、验收未通过；第二轮已完成双 T4 DDP、tokenizer 固化与 dead embedding 修复，等待 smoke 复跑

> 命名约定：从 2026-09-27 起，本方案中的 ModernBERT Decision
> 决策模型正式称为 **Kairos 模型**。代码目录和历史 Kaggle Kernel 名称暂不改动，
> 以避免破坏正在运行的 chunk 接力；后续新任务、报告和看板说明统一使用 Kairos。

## 1. 定位

这是一个独立的 Kronos 金融决策模型，针对 A 股 120D 历史序列和未来 10D
价格事件进行全新训练。

设计采用概率决策模型的两个思想：

1. 输出概率分布，而不是单一分数；
2. 使用严格的决策头表达机会、风险和不确定性。

模型本身是 Kronos 系统中的独立决策器：

```text
120D 历史行情
    -> 冻结 Kronos Tokenizer
    -> ModernBERT 风格序列编码器
    -> 金融事件决策头
    -> 未来 10D 机会/风险概率
```

它不读取 Kronos predictor 的预测结果，也不替代 Kronos 的价格路径预测。

## 2. 第一版明确不做的事情

- 不使用 ModernBERT 的英文文本权重；
- 不把行情序列拼成自然语言 prompt；
- 不使用文本 prompt 或文本 option-marker 输入协议；
- 不做交易成本、成交限制或收益回测；
- 不使用 LightGBM 或其他传统模型作为对比基线；
- 不在第一版一开始就加入 MLM 自监督预训练；
- 不把 CPU 作为训练平台；
- 不在 E1 吞吐和稳定性验证前启动全量 Kaggle 训练。

正式训练平台是 Kaggle GPU。CPU 只用于 E0 的极小规模前向、反向和泄露测试。
第一版先证明金融序列输入和决策头是否有增量价值，再决定是否增加复杂训练阶段。

## 3. 输入契约

每条样本严格使用：

- `T-119 ... T`：120 个交易日输入；
- `T+1 ... T+10`：只用于训练标签；
- 输入不包含任何未来字段；
- 归一化参数只由 120D 历史计算；
- 训练链路使用 `float32` 数值数组和整数 token，不经过文本序列化；
- JSONL 只作为抽样审计和人工检查产物，不作为模型训练输入。

正式模型输入由三部分组成：

### 3.1 行情序列

冻结 `NeoQuasar/Kronos-Tokenizer-base`，将 120D 归一化 OHLCVA 编码为：

```text
s1[120], s2[120]
```

这里的“改 tokenizer”不是修改 BERT 的中文文本分词器，而是把 Kronos 的
`s1/s2` 离散码作为新的金融 token。模型不走文本 tokenizer，而是使用：

```text
Embedding(1024, d) for s1
Embedding(1024, d) for s2
concat -> fusion projection -> inputs_embeds
```

R1 中这些 embedding、融合层、ModernBERT 主干和决策头全部从随机初始化开始训练；
若 R2 门禁确认 R1 仍有可复用信号，R2 Chunk 1 才从 R1 模型权重 warm-start，
后续 chunk 从 R2 checkpoint 接力。
模型只接收 token id 的嵌入，不接收未来数据。

正式训练样本在内存或分片文件中保持为：

```text
s1:      int16[120]
s2:      int16[120]
size:    float32[1]
sector:  int64[1]
target:  int8 / float32
```

不把 120D 序列转成字符串，也不使用小数位数限制训练精度。

### 3.2 市值百分位

只使用信号日 T 的市值百分位：

- `size_percentile`：`float32`；
- 缺失时填充为 `0.5`。

当前训练代码直接使用 `[0, 1]` 范围内的 `size_percentile`，不再额外做
训练集 z-score。

### 3.3 行业标签

只使用信号日 T 的行业标签：

- `sector_id`：`int64`；
- 训练运行时从 train/validation panel 收集行业名并排序生成运行内词表；
- 未知行业使用固定 unknown id。

原始中文行业名只用于审计，不进入 ModernBERT 文本输入。

## 4. 模型结构

### 4.1 主干

使用 ModernBERT 风格的双向 Transformer 编码器，但通过 `inputs_embeds` 输入金融 token：

```text
[COND] + 120 个行情 token
```

早期 E1 采用过小配置作为方法验证实验，但该配置已被全量执行方案替代：

```text
hidden_size = 256
layers = 4
heads = 4
intermediate_size = 512
```

该小配置不代表当前全量 Kairos 训练配置；第一轮实际执行的是 §10.1
所列的 ModernBERT-base 风格大配置。

### 4.2 条件注入

市值百分位经过小型 MLP，行业 id 经过 embedding，合并为一个 `COND` token。

第一版只在输入端注入条件，不改写 ModernBERT 的中间层 forward，减少工程复杂度和泄露风险。

### 4.3 输出头

第一版的主输出是两个概率组：

1. `P(MFE10 >= threshold)`：未来 10D 最大上冲幅度超过阈值的概率；
2. `P(-MAE10 >= threshold)`：未来 10D 最大下探幅度超过阈值的概率。

建议使用：

```text
mfe thresholds: 3%, 5%, 8%, 12%
mae thresholds: 3%, 5%, 8%, 12%
```

上冲和下探分别使用单调序数头，保证：

```text
P(MFE10 >= 3%) >= P(MFE10 >= 5%) >= ...
P(-MAE10 >= 3%) >= P(-MAE10 >= 5%) >= ...
```

`first_touch` 只作为后续可选辅助头，不作为第一版的主要决策输出。

输出同时提供：

- 类别概率；
- 最大机会概率；
- 最大风险概率；
- 置信度；
- 是否满足决策阈值。

## 5. 标签与因果性

所有标签使用原始面板价格，不使用 clip 后的归一化数据反推。

```text
mfe10 = max(high[T+1:T+10]) / close[T] - 1
mae10 = min(low[T+1:T+10]) / close[T] - 1
```

如果后续启用 `first_touch` 辅助头，按交易日顺序扫描：

- 先达到上冲阈值：`upside_first`；
- 先达到下探阈值：`downside_first`；
- 10D 内都未达到：`neither`；
- 同一天同时达到：按保守规则记为 `downside_first`。

该模型描述的是价格事件概率，不直接代表可实现收益。

## 6. 数据切分

直接复用 Kaggle 上现有的 **A-share 120D Temporal Symbol Holdout** 数据集，
不在本地另造一份相同 OHLCVA 数据：

```text
训练：
  现有 train_data.pkl 中全部有效窗口

验证：
  现有 val_data.pkl 中全部有效窗口，按信号日期切成前段和后段

最终测试：
  独立的时间外 sealed 样本
```

训练集不再人为截断到某个日期，以最大化利用现有 Kronos 训练窗口。
由于验证股票已经从训练股票中隔离，验证集可用于检查跨股票泛化；
真正的时间外泛化由 sealed 测试集检查。

开发验证集用于：

- 早停；
- 模型选择；
- 结构和超参数对照。

校准验证集用于：

- 温度校准；
- 决策概率阈值冻结。

最终测试集只运行一次，不用于调参。

## 6.1 Kaggle 数据复用与合并原则

这里的“合并”是**训练时逻辑合并**，不是把两个大数据集解包后重新拼成第三份。
标签已发布为 Kaggle 配套数据集：
`luckfu/ashare120d-modernbert-targets`。

```text
挂载 luckfu/a-share-120d-temporal-symbol-holdout
  train_data.pkl / val_data.pkl / asset_metadata.csv
                         +
挂载 luckfu/ashare120d-modernbert-targets
  train_targets.parquet / validation_targets.parquet / manifest
                         +
冻结 Kronos tokenizer + ModernBERT 决策模型
```

原始 OHLCVA 面板只保留一份，不上传新的窗口 JSON、文本 prompt、120D 序列
或预先 token 化数据集。已发布的 sidecar 仅含未来事件标签，manifest 记录
源 panel SHA-256、窗口定义和时间切分。训练启动时必须校验 panel 哈希和
标签窗口数；不匹配就停止，不能静默错位。不得复用由
`a_share_full_market_v1_beta_symbol_holdout_90_10_v1` 生成的旧 sidecar。

运行时直接读取：

```text
/kaggle/input/a-share-120d-temporal-symbol-holdout/
  processed_datasets/train_data.pkl
  processed_datasets/val_data.pkl
  asset_metadata.csv
```

数据处理方式：

- 窗口身份按现有 Kronos 规则在线枚举；
- 未来 10D 标签从配套 Parquet sidecar 读取，不在训练期间重复生成；
- 归一化、行业/市值条件按面板和历史窗口在线读取/计算；
- Kronos tokenizer 权重固定、参数全部冻结，不参与 Kairos 反向传播；
- 每个训练 segment 只执行一次 tokenizer 编码，生成 CPU `uint16` token cache，
  该 segment 的所有训练 batch 直接复用；
- 完整验证集首次使用时按 rank 分片编码并保存带 SHA-256 身份的 cache，
  同一 Kaggle 任务内的每次 segment 验证直接复用；后续接力若挂载该 cache，
  先校验 panel、标签和 tokenizer 指纹后再复用；
- 训练样本保持数值张量，不生成 JSON；
- 只保存 manifest、checkpoint、验证 token cache、随机状态、标签统计和断点位置；
- 不上传全量训练 token 数据集；训练 token cache 只在当前 segment 生命周期内存在，
  避免产生第二份 900 万行训练数据。

这样不会复制现有数据，也不会产生第二份大规模训练集。

## 7. 实验顺序

### E0：CPU 代码验证

完成：

- 标签构造器；
- 数据窗口对账；
- 未来扰动测试；
- tokenizer 因果性测试；
- 小模型前向和反向；
- 输出概率范围与标签一致性检查。

E0 只跑几十个 step 的极小合成或截断数据，不用于判断模型效果。

### E1：小规模 GPU 实验

在 Kaggle T4 GPU 上，使用正式的金融 token 输入进行新模型训练，先使用固定的
小规模窗口子集。只验证四件事：

1. token + 市值百分位/行业条件是否比 metadata-only 更好；
2. ModernBERT 小配置是否能稳定训练；
3. 最大涨跌幅概率是否可校准；
4. 训练窗口规模和吞吐是否可接受。

E1 不是 CPU 训练，也不是使用 ModernBERT 官方权重微调，而是：

```text
随机初始化金融 embedding
+ 随机初始化 ModernBERT 主干
+ 随机初始化金融决策头
-> Kaggle GPU 训练
```

对照只使用同一模型的输入消融：

```text
A：token + 市值百分位/行业条件
B：mask token + 市值百分位/行业条件
C：仅市值百分位/行业条件
```

A、B、C 使用相同模型结构、数据、随机种子和训练预算。

### E2：Kaggle 全量训练

只有在 E1 明确显示 token 有价值且吞吐可接受后，才进行：

- 训练窗口流式读取或分片读取；
- 全量训练窗口池；
- 256/4 配置与 512/8 配置对照；
- 必要时再加入 MLM span 预训练；
- 更复杂的单调序数头。

MLM 是否有效只看下游验证概率指标，不以 MLM loss 单独决定。

## 8. 评价指标

主指标：

- 多分类 log loss；
- Brier score；
- ECE；
- reliability curve。

辅助指标：

- 各事件 AUC；
- 按行业、市值分位和日期分组；
- 按信号日做 block bootstrap；
- 预测概率与实际事件频率的一致性。

不使用单一命中率作为主指标，也不根据最终测试集结果反向修改模型。

## 9. 方案结论

这套方案的核心是：

```text
概率决策输出
+ Kronos 的 120D 行情 tokenizer
+ ModernBERT 风格的序列编码
+ 面向未来 10D 事件的金融专用输出头
```

第一阶段不追求一次性加入所有设计。先用小模型和严格消融确认：

```text
120D OHLCVA token 是否提供市值百分位和行业标签之外的有效信息
```

如果 E1 没有稳定增益，停止扩大模型；如果有增益，再进入 MLM、大模型和全量训练阶段。

## 10. 全量训练两轮执行计划

本节是当前 Kaggle 全量训练的实际执行合同，优先级高于前文中尚未更新的
“E2 方案稿”描述。当前日期为 2026-09-29。

### 10.1 第一轮：随机初始化基线

第一轮 8 个 chunk 已于 2026-09-29 完成。Chunk 8 的训练脚本完成不等于
模型质量通过；完整验收结果见 `kairos_round1_chunk8_review_cn.md`。

第一轮固定配置：

```text
模型：ModernBERT-base 风格，hidden=768，layers=22，heads=12
初始化：模型参数随机初始化
训练数据：全量 9,010,965 条
训练方式：8 个 Kaggle chunk 串行接力
batch size：16
数据顺序 seed：20260925
看板：modernbert-decision-full-gated-v1
```

第一轮的目标是取得一个完整的全量训练 warm-start 和基线，不在训练过程中
选择 `best_model`。每个 chunk 只保存：

- `last_checkpoint.pt`：供下一个 chunk 接力；
- `chunk_report.json`：记录覆盖范围、样本数和顺序 hash；
- SwanLab 训练进度；
- 最后一个 chunk 的完整验证报告。

第一轮结束时必须确认：

1. 8 个 chunk 均完成，且没有重复或跳过 row group；
2. 每个 checkpoint 的 `shuffle_seed` 和 `group_order_hash` 一致；
3. 最后一个 chunk 完成 `123,836` 条验证集评估；
4. 产出第一轮的 `final_model.pt` 和完整验证指标；
5. 第一轮模型只作为 baseline/warm-start，不称为验证集 best。

### 10.2 第一轮结束后的接力操作

第一轮第 8 个 chunk 完成后，先执行第一轮质量复核。
2026-09-29 的 Chunk 8 报告显示全量验证 macro log loss `0.9182`、
macro Brier `0.2700`、8 头平均 ECE `0.2216`，暂不自动启动第二轮；
详见 `kairos_round1_chunk8_review_cn.md`。只有排除数据/推理错位并确认
继续训练有合理依据后，才按以下顺序进入第二轮：

1. 下载或挂载第一轮第 8 个 chunk 的最终 checkpoint；
2. 校验第一轮输出中的模型文件、训练报告和数据 manifest；
3. 只有在审计确认 R1 仍有可复用的排序信号、且失败主要来自训练策略或
   概率尺度时，才以第一轮最终模型作为第二轮初始化；若排序接近随机、
   学到反信号或根因仍不明，则先做最小复现实验，不启动 R2；
4. 新建第二轮首个 segment smoke 任务，仍然只提交一个 Kaggle 训练任务；
5. smoke 完成后，根据实测时长设置本次接力的 `MAX_SEGMENTS_THIS_RUN`；
6. 每次接力完成后，将输出发布为下一次接力的输入，直到全部 segment 完成；
7. 每次提交前确认前一个接力任务已停止或完成，不能并行占用 Kaggle GPU；
8. 第二轮所有正式接力使用同一个新的 SwanLab run。

第一轮和第二轮不共用看板，避免把“无中途验证的 warm-start 基线”和
“带分层验证的正式训练”混在同一组曲线中。单 segment smoke 作为 R2 的
首个正式 shakedown，沿用 R2 看板，不另建看板；其结果会保留在同一条
R2 曲线上，并在日志中标记 `purpose=single-segment-smoke`：

```text
第一轮：modernbert-decision-full-gated-v1
第二轮：modernbert-decision-full-gated-round2-v2（v1 看板已删除，重建后复用）
```

同一轮内部的正式接力必须复用同一个 run id，并使用
`resume="allow"`。

### 10.3 第二轮：换顺序并加入分层验证

第二轮不是重复第一轮的数据顺序。固定使用新的数据顺序 seed。训练 segment
和 Kaggle chunk 接力边界分开定义：

```text
第一轮：SHUFFLE_SEED = 20260925
第二轮：SHUFFLE_SEED = 20260927
```

```text
SEGMENT_SAMPLES = 20,000
MAX_SEGMENTS_THIS_RUN = 按本次 GPU 时长预算设置
TOTAL_SEGMENTS = ceil(9,010,965 / 20,000) = 451
```

segment 固定覆盖 20,000 条训练样本；chunk 只决定本次 Kaggle 任务最多连续
完成多少个 segment。约 5 小时时设置较小的 `MAX_SEGMENTS_THIS_RUN`，接近
10 小时时设置更大的值。不能按 row group 平均切 segment，也不能在 segment
中间伪造 checkpoint。首个正式 R2 前先提交一个
`MAX_SEGMENTS_THIS_RUN = 1` 的单 segment smoke，实测训练、完整验证、显存、
输出大小和每段耗时，再决定后续每次接力的 segment 数。每个完整 segment 都要
执行全量验证。最后不足 20,000 条的剩余训练样本仍作为一个真实的尾 segment
训练和验证，不丢弃、不补造样本。提交前必须确认上一段已经完成或停止，不能并行提交。

第二轮优先使用 Kaggle 双 T4，通过 `torch.distributed.run --nproc_per_node=2`
显式启动两个 DDP worker；全局 batch 为 16，每卡 local batch 为 8。rank 0
独占看板、日志、JSON 报告、checkpoint 和 cache 发布，其他 rank 只负责计算并
通过 barrier/collective 同步。若运行时只有一张 GPU，脚本支持单卡降级，保持
全局 batch 语义不变但速度约减半；若检测到两张或更多 GPU 却无法建立恰好两个
worker，则直接失败，不静默降级。四个 chunk 脚本除 `CHUNK_INDEX` 外保持同一
实现，`MAX_SEGMENTS_THIS_RUN` 仍可按本次 GPU 时长预算调整。

第二轮 Chunk 1 只加载第一轮 `final_model.pt` 中的模型权重，重置
optimizer/scaler，并使用新的数据顺序；它不加载第一轮 optimizer 状态，也不重新
训练或修改冻结的 Kronos tokenizer。tokenizer 只在 cache 构建阶段运行，后续 chunk
优先加载带身份校验的验证 cache。后续 chunk 才从上一段的
`last_checkpoint.pt` 恢复完整训练状态。
若 Kaggle 在 10 小时上限前终止任务，下一次不得从头重跑该次接力：必须挂载
该任务最后成功写出的 `last_checkpoint.pt`，保持相同的接力配置、
`SHUFFLE_SEED` 和 `group_order_hash`，从已保存的 segment/row offset 继续。若任务没有
成功写出 checkpoint，则该 chunk 必须拆分为临时更小的恢复段后再继续，不能把
不完整结果发布为下一段输入。
第二轮只改变数据访问顺序和验证策略，不改变第一轮已经确定的输入契约。
第二轮 checkpoint 必须记录新的 `shuffle_seed` 和
`group_order_hash`，各次接力之间严格校验，不能混用第一轮 checkpoint。

当前实现中 `vocab_size=1024` 只满足 ModernBERT 配置接口；模型实际通过
`inputs_embeds` 输入金融 token，因此 ModernBERT 自带的词表 embedding 不参与
forward。该 embedding 已显式冻结，避免 DDP 将其识别为未参与反传的可训练参数；
它仍占用约 0.79M 参数空间，但不影响优化动力学。

第二轮每个 segment 完成后执行完整验证集。chunk 只是 Kaggle 单次运行边界，
不承担模型选择语义：

```text
每个 chunk：由 `MAX_SEGMENTS_THIS_RUN` 决定
每次接力可根据可用 GPU 时长修改 `MAX_SEGMENTS_THIS_RUN`，不固定 chunk 数
总 segment：451
每个 segment 验证：完整 123,836 条
指标：macro log loss、macro Brier、ECE、8 个阈值的明细、
      每阈值 reliability curve 所需的分桶统计
```

如果当前 segment 的完整验证指标优于历史最佳：

```text
保存 best_model.pt
更新 best_metric.json
```

训练接力仍然使用：

```text
last_checkpoint.pt
```

不能用 `best_model.pt` 替代接力 checkpoint，否则会改变优化器状态和训练轨迹。
第二轮最后一个 segment 的完整验证即为最终验证，不再单独执行；最后一个
chunk 同时保存：

- `best_model.pt`：第二轮全部 segment 的完整验证中表现最佳；
- `final_model.pt`：第二轮最后训练状态；
- 完整验证报告：覆盖全部 `123,836` 条验证样本；
- 按 `2025H2`、`2026H1` 和 8 个阈值拆分的指标。

### 10.3.1 第二轮校准与可靠性报告

第二轮不得只报告 8 个输出头的 macro ECE。每个阈值都必须独立保存：

```text
up_003、up_005、up_008、up_012
down_003、down_005、down_008、down_012
```

每个输出头至少记录：

- reliability curve 的置信度分桶；
- 每桶样本数、平均预测概率和实际正例率；
- per-threshold ECE；
- per-threshold Brier；
- per-threshold log loss；
- 整体正例率和预测概率均值。

R2 训练脚本只负责输出未校准概率和分桶统计，不在训练 chunk 内拟合温度。
校准报告必须同时按时间和波动状态拆分，不能只在全体验证样本上汇总。
至少要统计按月或季度的正例率，并记录对应窗口的波动率分位区间，以识别
`3%/5%/8%/12%` 事件在高、低波动阶段的聚集和漂移。

训练完成后单独运行校准阶段，温度校准的执行约束：

1. 校准集必须覆盖高、低波动阶段和主要时间区间；
2. 校准集与最终评估集严格分离，不能用最终测试集拟合温度；
3. 先评估一个全局温度，再根据 per-threshold reliability 结果决定是否使用
   每阈值独立温度；
4. 温度拟合后必须在未参与拟合的评估集上重新报告上述全部指标；
5. 校准前后的结果必须并列保存，不能只保留校准后的数字。

### 10.3.2 第二轮稀有阈值与类别不平衡

不得在没有统计正例率前直接引入 `pos_weight` 或 focal loss。第二轮首先输出
8 个阈值的整体、时间分块和波动状态正例率，再比较：

```text
普通 BCE
按阈值加权 BCE
focal loss（仅作为候选实验）
```

所有候选损失必须同时比较：

- 8 个阈值的 log loss；
- 8 个阈值的 Brier；
- 8 个阈值的 reliability curve / ECE；
- 高、低波动阶段的时间稳定性；
- 单调序数约束是否仍然满足。

如果加权损失改善稀有头的召回，却明显恶化概率校准，则不能仅依据命中率
选择加权版本。

### 10.4 对照实验安排

C 对照不与主模型并行运行，也不插入当前第一轮。待第二轮主模型完成并确认
训练链路稳定后，再单独安排 C 对照：

```text
C：金融 token 路径置零
保留：同一 ModernBERT 主干、行业、市值、数据顺序和验证契约
看板：单独的 C-control run，不能混入主模型看板
```

这样可以回答两个独立问题：

1. 第二轮训练和分层验证后，模型是否比第一轮基线更稳定；
2. 金融 token 是否在相同主干、行业和市值条件下提供增量信息。

### 10.5 当前状态记录

截至 2026-09-29：

```text
当前阶段：第一轮已完成，质量验收未通过；R2 单 segment smoke V4 在 DDP 启动后因 SwanLab v1 看板已删除而于训练前失败
当前位置：Chunk 8 / 8 已完成
最近 R2 Kernel：wynstonliu/kairos-r2-modernbert-chunk-1 V4（ERROR；SwanLab `Disabled_Resource`，切换 v2 看板后待复跑）
R1 最后 Kernel：wynstonliu/modernbert-decision-full-chunk-8
当前看板：modernbert-decision-full-gated-round2-v2（待 smoke 重建并初始化）
第一轮验收：FAIL（模型质量，不是脚本执行失败）
第二轮：脚本改用新看板后等待 smoke 复跑
C 对照：尚未开始，禁止并行提交
```

### 10.6 第二轮日志与看板进度契约

第一轮当前日志主要记录累计样本和 row group，阅读长时间运行进度不够直观。
从第二轮 Kairos 训练开始，所有 chunk 必须同时记录全局进度和 segment 进度。
一个 `segment` 是一次固定 20,000 条训练样本的覆盖单元；chunk 包含多少个
segment 由 `MAX_SEGMENTS_THIS_RUN` 决定，第二轮总计 451 个 segment。每个
segment 完成后都必须执行完整验证。

训练日志和 SwanLab 指标必须包含：

```text
segment_total                 = 451
segment_index                 = 1..451
segment_samples               = 当前 segment 的样本数
segment_processed_samples     = 当前 segment 已处理样本数
segment_progress              = 当前 segment 完成比例
global_processed_samples      = 全局累计已处理样本数
global_total_samples          = 9,010,965
```

每个 segment 结束时还必须记录：

```text
segment_complete = 1
validation_samples = 123,836
validation_macro_log_loss
validation_macro_brier
validation_ece
best_updated = 0/1
```

其中 `last_checkpoint.pt` 用于下一 segment 接力，`best_model.pt` 只在完整验证
指标改善时更新。看板曲线必须能同时回答：

1. 当前做到第几个 segment；
2. 当前 segment 已完成多少；
3. 全局 451 个 segment 完成了多少；
4. 最近一次验证是否刷新 best。

第一轮已完成，不强行回溯修改；从第二轮脚本开始，
必须遵守上述字段命名。

### 10.7 导师评审锁定项

以下项目是 Kairos 第二轮的强制验收项，不因第一轮训练结果较好而跳过：

1. 每个输出阈值的 reliability curve，而不是只看 macro ECE；
2. 每个输出阈值的正例率及其时间漂移；
3. 高、低波动阶段的事件聚集性统计；
4. 校准集覆盖不同波动状态，且与最终评估集隔离；
5. 校准前后 per-threshold 指标并列报告；
6. 依据 E1 预先定义的常数先验、C 对照、时间稳定性和 ECE 标准判断是否
   继续扩大训练，不因已经消耗算力而降低门槛。

### 10.8 第一轮失败后的 R2 门禁

第一轮 Chunk 8 的 `PASS` 只表示 Kaggle 脚本完成，不代表模型质量通过。
在训练/验证鸿沟没有解释前，禁止消耗第二轮 GPU 配额。

R2 启动前必须完成以下审计：

1. 固定小批样本的训练模式与验证模式逐张量对照；
2. 抽样人工复核 `down_012`、`up_012` 原始 OHLCVA 标签；
3. 以训练集正例率建立固定常数基线；
4. 每个输出头增加 ROC-AUC、PR-AUC 和 reliability 分桶；
5. 完成 E1 与全量训练在数据、模型、优化器、AMP、chunk 接力和验证构成
   上的差异表；
6. 温度缩放只能在独立校准集拟合，不能使用最终验证/测试标签。

判定逻辑：

- 若训练/验证管线或标签存在错误，先修复并重新跑最小复现实验；
- 若排序指标有信号但概率尺度失真，再评估温度缩放；
- 若排序和校准都失败，停止 R2，不继续扩大模型或训练预算；
- 只有修复后模型在常数基线、C 对照、时间分块和校准指标上同时达到门槛，
  才恢复第二轮动态 segment 接力。
