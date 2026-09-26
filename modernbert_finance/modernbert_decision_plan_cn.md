# Kronos ModernBERT 金融决策模型方案

版本：v1.0
日期：2026-09-25
状态：方案稿，不启动 GPU 训练

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

这些 embedding、融合层、ModernBERT 主干和决策头全部从随机初始化开始训练。
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
- 缺失时使用固定 unknown 值。

市值百分位经过固定的训练集标准化后进入条件 token。

### 3.3 行业标签

只使用信号日 T 的行业标签：

- `sector_id`：`int64`；
- 使用固定的行业词表；
- 未知行业使用固定 unknown id。

原始中文行业名只用于审计，不进入 ModernBERT 文本输入。

## 4. 模型结构

### 4.1 主干

使用 ModernBERT 风格的双向 Transformer 编码器，但通过 `inputs_embeds` 输入金融 token：

```text
[COND] + 120 个行情 token
```

第一版采用小配置作为主实验：

```text
hidden_size = 256
layers = 4
heads = 4
intermediate_size = 512
```

原因是第一阶段的目标是验证方法，不是追求参数规模。大配置只在小配置确认有效后再做。

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
- s1/s2 token 按 batch 在线编码；
- 训练样本保持数值张量，不生成 JSON；
- 只保存小型 manifest、checkpoint、随机状态、标签统计和断点位置；
- 如在线 tokenization 吞吐不足，只在 Kaggle 工作目录生成分片缓存，不上传为新的输入数据集。

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
