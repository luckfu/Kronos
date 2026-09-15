# Kronos-small `small_0.1` 模型与训练技术报告

## 1. 摘要与范围

本文记录 Kronos-small `small_0.1` 从 `small_0.1_bootstrap_timing` 初始化、经 `small_0.1_main` 主训练、`small_0.1_stage2_extend_01` 增训、两轮 Warmup-Constant 训练，到 `small_0.1_stage2_cosine_refinement` 退火收口的模型设计、数据处理、优化配置和实测结果。报告面向论文撰写，优先记录可由代码和训练日志复核的事实。

Stage 2 已完成至 Cosine refinement 267/267 segments，最终 best 为 Segment 179。2026-09-15 对已有 OOS 预测统一原始价格和方向口径后，完整 19 个 signal dates、97,916 个窗口/模型上，C2 best 的 D10 方向准确率为 50.694%，pooled Rank IC 为 0.16910，日均 Rank IC 为 0.16163（第 8.3 节）。这仍是短时间区间的预测证据，不是净交易收益证明。

报告同时纳入 Stage 3 多周期路径对齐的实现、正确性验收与单段结果（第 6.7 节）。修正版 Kaggle 单段测试已完成双 T4 预检、20,000 窗口训练与 123,836 窗口全量验证；尚无受控训练改善或 Stage 3 OOS 结论。工程验收通过不等于预测性能改善。

`small_0.1_stage2_wc_last` 是从 Extend 01 后继续进行的 Warmup-Constant 训练，本轮已完成 534/534 个 segments；其后的 `small_0.1_stage2_wc_dual_t4` continuation 也已完成 534/534 segments，随后进行的 `small_0.1_stage2_cosine_refinement` 退火训练完成 267/267 segments。三者均作为独立阶段记录，不混合不同 scheduler 的结果。

## 2. 版本血缘与研究目标

研究目标是在保持 Kronos-small 预训练时序建模能力的基础上，引入 A 股市场的股票行业、市值和时间条件，并使用严格的时间/股票隔离验证方案进行全参数微调。版本关系如下：

```mermaid
flowchart LR
    B[NeoQuasar/Kronos-small<br/>预训练底座] --> BT[Bootstrap<br/>双学习率<br/>534/534 segments]
    BT --> BB[Bootstrap best_model<br/>约segment 516]
    BB --> M[Stage 2 Main<br/>统一学习率<br/>534/534 segments]
    M --> L[Main last_model]
    L --> E[Stage 2 Extend 01<br/>新 seed 20260907<br/>190/534 segments]
    E --> WC[Stage 2 WC Last<br/>Warmup-Constant<br/>534/534 segments]
    WC --> WCN[Stage 2 WC Dual T4<br/>Warmup-Constant<br/>534/534 segments]
    WCN --> AN[Cosine refinement 退火<br/>267/267 segments]
    AN --> C2B[Cosine C2 best<br/>Segment 179]
    C2B -.仅加载权重.-> PA[Stage 3 Path Alignment<br/>fresh optimizer / step 0<br/>单段工程验收通过]
```

Bootstrap 从预训练底座开始，以双学习率完成条件化初始化；Main 继承 Bootstrap 的 `best_model`，切换为统一学习率并完成一轮主训练；Extend 01 再继承 Main 的 `last_model`，使用不同窗口排列进行续训；WC first round 继承 Extend 01 C2 的 `last_model`，改用 Warmup-Constant；WC dual T4 再从上一轮 WC checkpoint 连续训练一轮完整 coverage；Cosine refinement 最后从 WC dual T4 的 checkpoint 退火收口。六者是串行训练血缘，不是从底座出发的并列实验。

Stage 3 则从 Cosine C2 的 best 权重建立新的目标函数实验，不从 C2 last 或旧 Stage3 C1 恢复训练状态。其优化器、step、EMA 统计和 coverage 游标重新初始化，不应拼接进 Stage 2 的同一条优化轨迹。

## 3. 模型架构

### 3.1 预训练底座

- 模型：`NeoQuasar/Kronos-small`
- tokenizer：`NeoQuasar/Kronos-Tokenizer-base`
- Transformer 层数：8
- `d_model=512`
- attention heads：8
- feed-forward hidden dimension：1024
- 原始模型规模约 24.7M；加入条件模块后的 predictor 可训练参数为 24,819,392（日志显示 100% 可训练，约 24.8M）。

本项目没有物理新增两层 Transformer。条件信息通过 `context_layer=6` 在第 6 层位置注入，因此条件信息会被后续两层继续处理；这应表述为“中间层条件注入”，而不是“增加两层”。

### 3.1.1 模型结构图

可以把底座模型理解为“把一段历史价格压缩成 token，再用 Transformer 预测下一个 token”的通用时序模型。`small_0.1` 保留这条主干，只在输入和中间层增加股票静态信息，并保留原有的双 token 预测头：

```mermaid
flowchart TD
    A[120日 OHLCVA 历史窗口] --> B[窗口内标准化]
    B --> C[冻结 Kronos Tokenizer]
    C --> D[层级 token embedding<br/>s1 与 s2 合并为 512维]
    T[时间特征<br/>minute/hour/weekday/day/month] --> E[时间 embedding]
    D --> E
    E --> F[Transformer Block 1-6<br/>原始 Kronos 主干]
    S[股票行业 ID<br/>86类 + unknown] --> S1[Sector Embedding<br/>87 x 512]
    M[市值百分位 + 是否缺失] --> M1[两层 Size MLP<br/>2 -> 64 -> 512]
    S1 --> G[条件向量相加]
    M1 --> G
    F --> G
    G --> H[Transformer Block 7-8<br/>吸收行业/市值条件]
    H --> I[RMSNorm]
    I --> J[s1 预测头<br/>粗粒度 token logits]
    I --> K[Dependency-aware Layer]
    J --> K
    K --> L[s2 预测头<br/>细粒度 token logits]
    J --> N[未来10日 token 概率]
    L --> N
    N --> O[Tokenizer 解码与反归一化]
    O --> P[未来10日 OHLCVA 预测]
```

图中“新增”的部分只有三类：

1. **行业条件**：行业 ID 经过一个 512 维 embedding，得到一条代表行业的向量。
2. **市值条件**：市值百分位和缺失标记经过 `2 -> 64 -> 512` 的 MLP，得到一条代表规模的向量。
3. **条件注入位置**：行业向量与市值向量相加后，加到第 6 个 Transformer block 之后的每个时间步，因此最后两个 block 可以利用这些静态信息。

这些条件模块的输出层在初始化时置零，所以刚加载底座时，新增分支近似“不改变原模型”；训练过程中它们再逐渐学习行业和市值的影响。训练日志显示新增条件参数约 78,016 个，主干及原有预测路径仍占绝大多数参数；因此“24.8M 模型”不是一个从零设计的新网络，而是 Kronos-small 主干加小规模条件适配模块。

### 3.1.2 底座与改造后网络对照

| 组件 | 原始 Kronos-small | `small_0.1` 改造后 |
|---|---|---|
| 连续输入 | OHLCVA，经 tokenizer 转 token | 相同；额外使用时间和股票元数据 |
| Transformer 主干 | 8 blocks，512 hidden，8 heads，FFN 1024 | 完整保留，8 blocks 均可训练 |
| 时间信息 | temporal embedding | 保留并继续相加到 token embedding |
| 行业信息 | 无 | 87 类 sector embedding，输出 512 维 |
| 市值信息 | 无 | percentile + known flag，经 `2->64->512` MLP |
| 条件注入 | 无 | 第 6 层后注入，影响最后 2 个 blocks |
| 输出 | s1/s2 双 token 预测 | 保留原输出头；本实验未启用 return/barrier auxiliary head |
| 参数更新 | 预训练时更新 | 微调时 24,819,392/24,819,392（100%）更新 |

这里的“最后两个 blocks 吸收条件”是信息流位置描述，不代表前六个 blocks 被冻结；三段正式训练日志都报告全部 predictor 参数可训练。

**参数统计注记：** 表中的 100% 仅指 Predictor 主干及条件模块；Kronos Tokenizer 在数据预处理阶段作为静态映射字典完全冻结，不计入 Predictor 的梯度图与可训练参数总量。

### 3.2 输入与条件分支

每个样本由 120 日历史窗口和未来 10 日预测窗口组成。六个连续市场特征为 `open/high/low/close/volume/amount`；时间戳特征包括 minute、hour、weekday、day、month。股票行业使用 86 个行业类别及 unknown 类别。市值不使用离散 bucket，而使用 `[size_percentile, is_known]` 输入的两层 MLP。

条件分支输出层采用零初始化，使模型刚接入条件时近似保持预训练模型的输出，降低突然改变预训练分布的风险。`return_head` 和 `barrier_head` 仅在 Beta v2.1 auxiliary 配置开启时创建；本报告覆盖的 Bootstrap、Main 和 Extend 01 日志均未启用该 auxiliary objective。

### 3.3 预测路径

历史 OHLCVA 经 tokenizer 编码为离散 token，Transformer 对历史及 teacher-forcing 序列进行预测；训练目标是未来 10 日 token 的交叉熵，同时保留历史 token 的辅助重建损失。推理阶段使用自回归生成，并将归一化空间的输出还原到原始量纲。

上述损失描述适用于 Bootstrap/Stage 2。Stage 3 保留同一生产推理接口，但使用第 6.7 节定义的未来 token CE 与数值路径代理损失，不沿用 history 辅助项。

## 4. 数据窗口、隔离与归一化

### 4.1 窗口构造

`QlibDataset` 每个窗口包含 `120 + 10 + 1 = 131` 行：120 行 lookback、10 行 forecast，以及用于边界/时间对齐的额外行。训练数据共 10,661,560 个可用窗口，每个 segment 无放回抽取 20,000 个窗口，因此一轮完整 coverage 为 534 segments。

验证不是从全市场所有股票中随机抽窗口。数据首先按行业、市值分位和历史长度等分层，从源股票池中固定抽取 **520 只验证股票**；另有 4,678 只股票作为主要训练股票。随后对这 520 只验证股票做时间切分：2024-12-31 及以前的历史可留在训练侧，2025-01-01 起的数据只进入验证侧。因此这是“股票 cohort + 时间”的隔离，而不是简单的全股票随机窗口切分。

由于部分验证股票在有效日期范围内不足以构成完整的 120 日 lookback + 10 日 forecast 窗口，最终实际进入验证窗口池的是 **516 只股票**。因此 manifest 中的 `val_data.pkl` 仍有 520 只股票，但候选窗口身份只覆盖 516 只。时间切分后，训练面板统计为 5,189 只股票（4,678 只主要训练股票，加上 520 只验证股票在 2024-12-31 以前仍保留的历史部分）；验证面板为 520 只股票，训练和验证的有效窗口不重叠。这 516 只有效验证股票在 2025-07-03 至 2026-07-02 的 242 个 signal dates 上产生 123,836 个候选验证窗口。为了快速完成 Bootstrap 初始化，Bootstrap 使用其中按方向、月份、行业和市值分层抽取的 19,998 个固定验证窗口；Main 和 Extend 01 才使用这 123,836 个候选窗口的全量。Extend 01 日志明确记录 `replay_samples=0`。

4 只未形成窗口的验证股票是自然数据长度不足，而非人工按结果淘汰：`sh.603352`（121 行）、`sh.688811`（67 行）、`sz.301669`（28 行）和 `sz.301680`（91 行）均不足以构成最小 131 行样本。它们保留在验证股票清单中，但不出现在有效窗口身份集合中。

```mermaid
flowchart TD
    U[源股票池<br/>5,198只] --> S[分层抽取股票 cohort]
    S --> TR[4,678只训练股票<br/>完整可用历史]
    S --> VH[520只验证股票]
    VH --> VT[2024-12-31及以前<br/>历史可留在训练侧]
    VH --> VV[2025-01-01以后<br/>只进入验证侧]
    VV --> E[满足120+10窗口条件的516只]
    E --> C[123,836个候选验证窗口]
    C --> BS[Bootstrap固定抽样子集<br/>19,998个<br/>快速初始化]
    C --> FM[Main/Extend全量验证<br/>123,836个]
```

```mermaid
flowchart LR
    A[验证股票在2024及以前的数据<br/>可以形成训练窗口] --> B[2025-01起进入隔离区]
    B --> C[2025年上半年<br/>只用于提供最长120日 lookback]
    C --> D[2025-07-03至2026-07-02<br/>123,836个验证窗口]
    D --> E[2026-08-03至2026-08-27<br/>19个signal dates的时间 OOS]
    style B fill:#fff2cc,stroke:#8a6d00
    style D fill:#d9ead3,stroke:#38761d
    style E fill:#cfe2f3,stroke:#0b5394
```

上图表达的是同一批验证股票的时间角色，而不是把 520 只股票误认为全市场。2025 年上半年的记录可以出现在验证窗口的历史输入中，但不能形成训练样本；OOS 数据既不训练，也不参与 best checkpoint 选择。验证 cohort 的固定股票清单来自 `symbol_split.csv`，窗口级 19,998 子集另有独立 manifest 和样本身份哈希。

验证窗口之间允许自然的滚动重叠：同一股票相邻 signal dates 的 120 日 lookback 通常共享约 119 日历史。因此 123,836 是窗口数量，不应被解释为 123,836 个 IID 独立观测；它适合估计验证期平均 NLL，但显著性分析应按 signal date、股票或 block bootstrap 估计有效样本量。

### 4.2 窗口内标准化

对每个样本，只使用 lookback 的 120 行计算每个 OHLCVA 列的均值和标准差，并将同一组统计量应用于未来 10 日：

\[
x' = \operatorname{clip}\left(\frac{x-\mu_{1:120}}{\sigma_{1:120}+10^{-5}},-5,5\right).
\]

因此未来目标不会参与标准化统计，避免信息泄漏。时间特征、行业 ID 和市值百分位不进行上述 OHLCVA Z-score；归一化后的连续序列再交给冻结 tokenizer 编码。

```mermaid
flowchart LR
    H[过去120日 OHLCVA] --> S[计算历史均值和标准差]
    H --> ZH[历史部分 Z-score<br/>并裁剪至 -5到5]
    S --> ZH
    F[未来10日 OHLCVA] --> ZF[未来目标使用同一组<br/>历史均值和标准差]
    S --> ZF
    ZH --> TK[冻结 Tokenizer]
    ZF --> Y[未来 token 监督目标]
    F -.禁止参与统计量计算.-> S
```

### 4.3 coverage 顺序与可复现性

Dataset 初始化时以固定 seed 生成 `coverage_order`，DataLoader 使用 `shuffle=False`。一轮 coverage 中每个窗口原则上恰好出现一次；启用均衡策略时先在 bucket 内打乱，再交错分配并打乱 segment 内顺序。不同训练阶段可以使用不同 `coverage_seed`：Bootstrap/Main 使用默认 seed `100`，Extend 01 使用 `20260907`，Warmup-Constant 使用 `20260908`。因此新阶段会重新排列同一批训练窗口，但不会改变训练数据集合。

## 5. 损失函数与模型选择

本节记录 Bootstrap/Stage 2 的目标函数和历史选模口径；Stage 3 的训练、验证和选模定义单列于第 6.7 节，两者数值不能混称。

对 history 和 forecast 分别计算 token-level NLL。未来 10 个 horizon 的 forecast loss 使用配置权重：

```text
1.364, 1.364, 1.364, 1.136, 1.136,
0.909, 0.909, 0.682, 0.682, 0.455
```

代码会先将 horizon 权重归一化。训练目标为：

\[
L = L_{forecast}^{weighted} + 0.02 L_{history}.
\]

checkpoint 的 best 选择指标是 `forecast`。日志中的 `Validation Forecast` 是未加 horizon 权重的原始 forecast NLL 汇总值；不能把该打印值误称为 weighted forecast objective。本文正式范围没有加入 Beta v2.1 的 return/barrier auxiliary loss。

```mermaid
flowchart TD
    P[模型输出 s1与s2 token logits] --> FH[历史120日 token NLL]
    P --> FF[未来10日逐日 token NLL]
    W[H1至H10权重<br/>归一化后使用] --> WF[Weighted Forecast Loss]
    FF --> WF
    FH --> HW[乘以0.02]
    WF --> TOTAL[训练目标<br/>Weighted Forecast + 0.02 History]
    HW --> TOTAL
    TOTAL --> BP[反向传播<br/>更新全部24.819M参数]
    VF[验证集原始 Forecast NLL] --> Q{是否刷新历史最低值?}
    Q -- 是 --> BEST[保存 best_model]
    Q -- 否 --> KEEP[保留原 best_model]
```

## 6. 训练阶段与 Stage 3 路径对齐实验

| 阶段 | 初始化 | batch | 学习率 | scheduler | 验证 | 结果 |
|---|---|---:|---|---|---:|---|
| Bootstrap | Kronos-small 底座 | 32 | adaptation/backbone `1e-6`；condition `1e-5` | 1% warm-up + cosine | 19,998 固定抽样窗口 | 534/534 segments；forecast 约 2.8758 -> 2.6287；最佳约 segment 516 |
| Main | Bootstrap `best_model` | 64 | backbone 与 condition 统一峰值 `1e-5`，warm-up 起点 `1e-6` | 1% warm-up + cosine | 123,836 | 534/534；forecast 2.638442 -> 2.505295；最佳 segment 528，2.505244 |
| Extend 01 | Main 的 `last_model` | 64 | 统一峰值 `1e-5`，fresh optimizer | 1% warm-up + cosine | 123,836 | 新 seed；完成 190/534；forecast 2.505623 -> 2.437912 |
| WC first round | Extend 01 C2 的 `last_model` | 64 | warm-up 后统一保持 `1e-5`，fresh optimizer | 1% warm-up + constant | 123,836 | seed=20260908；完成 534/534；forecast 2.437121 -> 最佳 2.353069，末段 2.354255 |
| WC dual T4 | WC first round `last_model` | 32/GPU × 2，global 64 | warm-up 后统一保持 `1e-5`，fresh optimizer | 1% warm-up + constant | 123,836 | seed=20260910；完成 534/534；forecast 2.355044 -> 最佳 2.304162，末段 2.306788 |
| Cosine refinement | WC dual T4 checkpoint | 64 | 统一 LR 从约 `1e-5` 退火至约 `1e-6` | uniform cosine，无 warm-up | 123,836 | seed=20260912；完成 267/267；forecast 2.306393 -> 最佳 2.294402，末段 2.295665 |
| Stage 3 Path Alignment（试验） | Cosine C2 best，Segment 179；仅加载权重 | 32/GPU × 2，global 64 | 统一 `2e-6`，fresh AdamW | fixed，无 warm-up | 123,836，全量、显式因果 | seed=20260915；1 segment/313 steps 完成，CE 2.339598；尚无性能改善结论 |

Bootstrap 的双学习率只用于启动阶段：条件分支以较快速度适应，而主干以较小步长保持预训练能力。Main 将两类参数统一到同一峰值学习率，作为正式全参数微调。Extend 01 不改变目标函数和验证集，只改变续训起点及 coverage 顺序。WC first round 在此基础上还改变 scheduler，但仍保持相同 batch、loss、数据集合和全量验证定义。WC dual T4 保持 Warmup-Constant 训练定义不变，只将执行方式改为双 T4 DDP；每卡 batch 32、global batch 64，因此优化器每步看到的总样本量没有改变。

#### Main、Extend 01 与两轮 WC 的日志串联

下图展示 Stage 2 的四次连续训练。箭头表示实际的 checkpoint 继承关系；色块表示训练阶段，而不是并列的独立实验。四次训练均更新全部 24,819,392 个 predictor 参数，验证集均为全量 123,836 个窗口。

```mermaid
flowchart LR
    MB[Main last_model<br/>534/534 segments<br/>forecast 2.505295<br/>global step 166,854]
    M[Stage 2 Main<br/>warm-up + cosine<br/>统一峰值 LR = 1e-5<br/>warm-up 起点 = 1e-6<br/>batch = 64]
    ML[Main best_model<br/>forecast 2.505244<br/>约 segment 528]

    EB[Extend 01 输入<br/>继承 Main last_model]
    E[Stage 2 Extend 01<br/>warm-up + cosine<br/>统一峰值 LR = 1e-5<br/>coverage seed = 20260907<br/>batch = 64]
    EL[Extend 01 last_model<br/>190/534 segments<br/>forecast 2.437912]

    WB[WC 输入<br/>继承 Extend 01 C2 last_model]
    W[Stage 2 WC first round<br/>warm-up + constant<br/>warm-up 后恒定 LR = 1e-5<br/>warm-up 起点 = 1e-6<br/>coverage seed = 20260908<br/>batch = 64]
    WL[WC last_model<br/>534/534 segments<br/>forecast 2.354255]
    WBest[WC best_model<br/>约 segment 527<br/>forecast 2.353069]

    W2[Stage 2 WC dual T4<br/>warm-up + constant<br/>恒定 LR = 1e-5<br/>2×T4，global batch = 64]
    W2L[WC dual T4 last_model<br/>534/534 segments<br/>forecast 2.306788]
    W2B[WC dual T4 best_model<br/>segment 512<br/>forecast 2.304162]

    MB --> M --> ML
    M --> EB --> E --> EL
    EL --> WB --> W --> WL
    W --> WBest
    WL --> W2 --> W2L
    W2 --> W2B

    classDef main fill:#dbeafe,stroke:#2563eb,color:#111827,stroke-width:2px;
    classDef extend fill:#dcfce7,stroke:#16a34a,color:#111827,stroke-width:2px;
    classDef wc fill:#fef3c7,stroke:#d97706,color:#111827,stroke-width:2px;
    classDef wc2 fill:#f3e8ff,stroke:#7c3aed,color:#111827,stroke-width:2px;
    classDef checkpoint fill:#f3f4f6,stroke:#6b7280,color:#111827;
    class M,MB,ML main;
    class E,EB,EL extend;
    class W,WB,WL,WBest wc;
    class W2,W2L,W2B wc2;
```

图中 `Main` 与 `Extend 01` 的 cosine 训练都使用统一峰值 `1e-5`；WC first round 将 scheduler 改为 warm-up 后恒定 `1e-5`，并使用新的 coverage seed。由于 Extend 01 的日志只完成 190/534 个 segments，WC 的实际输入标注为 Extend 01 C2 的 `last_model`，不能把 Extend 01 的中间日志末点误写成 WC 的初始化 checkpoint。

对应的 segment-level 全量验证损失如下。横轴将五份日志按实际训练顺序连续拼接：Main 为 1--534，Extend 01 为 535--724，WC first round 为 725--1258，WC dual T4 为 1259--1792，Cosine refinement 为 1793--2059；虚线为阶段边界，背景色与上图血缘图一致。

![small_0.1 Stage 2 validation loss trajectory](assets/small_0_1_stage2_loss_trajectory.png)

图中可以直接看到：Main 和 Extend 01 均为 warm-up + cosine、峰值学习率 `1e-5`；两轮 WC 为 warm-up 后恒定 `1e-5`；Cosine refinement 再将统一学习率从约 `1e-5` 退火至约 `1e-6`。退火阶段完成 267/267 segments，forecast 从 `2.306393` 降至最佳 `2.294402`。该图描述的是验证轨迹，不构成控制变量意义上的 scheduler 因果比较，因为各 continuation 阶段同时使用了新的 coverage seed。

```mermaid
flowchart TD
    Q1[新增条件分支能否稳定接入?] --> B[Bootstrap<br/>主干峰值1e-6<br/>条件峰值1e-5]
    B --> R1[完成534 segments<br/>Forecast约2.8758到2.6287]
    R1 --> Q2[提高主干步长能否充分适配A股?]
    Q2 --> M[Main继承Bootstrap best<br/>全部参数峰值1e-5<br/>改用全量验证]
    M --> R2[完成534 segments<br/>Forecast 2.6384到2.5053]
    R2 --> Q3[一轮末端仍下降<br/>继续训练是否有收益?]
    Q3 --> E[Extend 01从Main last续训<br/>新seed + fresh optimizer]
    E --> R3[完成190/534 segments<br/>Forecast 2.5056到2.4379]
    R3 --> C[支持继续增训<br/>但不能单凭曲线断言未饱和]
```

所有阶段都训练完整网络，差异是学习率和初始化起点，而不是是否冻结主干：

```mermaid
flowchart LR
    subgraph BS[Bootstrap]
        BA[原有主干约24.74M<br/>峰值LR 1e-6] --> BO[共同优化]
        BC[新增条件约78K<br/>峰值LR 1e-5] --> BO
    end
    subgraph MS[Main]
        MA[原有主干约24.74M<br/>峰值LR 1e-5] --> MO[共同优化]
        MC[新增条件约78K<br/>峰值LR 1e-5] --> MO
    end
    subgraph ES[Extend 01]
        EA[从Main last加载全部权重] --> EO[统一峰值LR 1e-5<br/>fresh optimizer与新顺序]
        EC[条件分支继续更新] --> EO
    end
```

### 6.1 第一次训练：Bootstrap 初始化

#### 为什么需要 Bootstrap

直接把刚加入的行业、市值条件分支和 24.7M 预训练主干同时用同一个较大学习率更新，存在破坏底座输出分布的风险。因此 Bootstrap 被设计成一次“稳定启动”：让新增条件分支先获得可用梯度，同时让预训练主干以更保守的步长适应 A 股数据。这里的目标不是宣称 Bootstrap 已经得到最终模型，而是验证条件接入、归一化、tokenizer、loss 和 Kaggle 接力流程能够稳定工作。

#### 具体参数与数据

- 初始化：`NeoQuasar/Kronos-small`；行业 embedding 和市值 percentile MLP 的输出层重置为零，使新增条件初始近似 no-op。
- 可训练参数：24,819,392/24,819,392（100%），没有冻结 Transformer 主干。
- lookback/forecast：120/10 日；batch size 32；AMP float16，tokenizer 编码保持 float32。
- 训练集：10,661,560 个窗口；每 segment 20,000 个唯一窗口；完整 coverage 534 segments。
- 验证集：固定 manifest 的 19,998 个窗口（从可用的 123,836 个验证窗口中抽取），其中 2025H2 为 9,820、2026H1 为 10,178。
- 优化器学习率：主干/原有 adaptation 参数峰值 `1e-6`，条件参数峰值 `1e-5`；两者 warm-up 起点分别为 `1e-7` 和 `1e-6`。
- scheduler：warm-up 占全局计划的 1%，随后 cosine 衰减；loss 为 `weighted forecast + 0.02 * history`；best 按 forecast 选择。

Bootstrap 最终完成 534/534 segments。固定验证 forecast 从首段约 2.8758 降到末段约 2.6287，最佳约出现在 segment 516（约 2.628729）。这证明模型和条件分支能够稳定训练，但由于主干学习率较低且验证集是固定子集，该阶段结果主要承担初始化和管线验证职责。

### 6.2 第二次训练：Stage 2 Main

#### 与 Bootstrap 的差异

Main 从 Bootstrap 的 `best_model` 继续，而不是从原始 Kronos-small 重新开始。运行脚本明确查找 Bootstrap best 的 `model.safetensors`，并记录 `selection=bootstrap_best`。进入 Main 时只继承模型权重，创建新的 optimizer 和全局学习率计划；Main 与 Bootstrap 默认使用同一个 coverage seed=100，因此并不是通过换 seed 产生新的训练窗口顺序。Main 的作用是让已经完成条件化初始化的模型进入统一学习率主训练，而不是一个与 Bootstrap 相互独立的对照实验。核心改变有三点：

1. backbone/adaptation 与 condition 使用相同的峰值学习率 `1e-5`，不再让主干长期处于 `1e-6` 的保守步长；
2. batch size 从 32 提升到 64，以提高吞吐并降低单个 segment 的执行时间；
3. 验证由 19,998 个固定抽样窗口改为全量 123,836 个窗口，覆盖验证期内全部可用 segment。

#### 具体参数与数据

- 初始化：Bootstrap `best_model`（约 segment 516）；只载入模型权重，并为 Main 建立新的 optimizer/scheduler；24,819,392 个 predictor 参数全部更新。
- lookback/forecast：120/10 日；batch size 64；AMP float16。
- 训练集：10,661,560 个窗口，20,000/segment，534 segments，完整一轮 coverage。
- 验证集：123,836 个窗口，signal 日期 2025-07-03 至 2026-07-02，共 242 天；无 replay 样本。
- 学习率：backbone 和 condition 的 warm-up 起点均 `1e-6`，峰值均 `1e-5`；warm-up ratio 1%。
- scheduler、loss、horizon 权重和 best 指标与 Bootstrap 保持一致，以便比较时主要改变训练动力学和验证覆盖，而不是改变任务定义。

#### 训练表现

Main 完成 534/534 segments。全量验证 forecast 从 segment 1 的 2.638442 降至 segment 534 的 2.505295，最佳 checkpoint 出现在约 segment 528，forecast 为 2.505244。全量验证比 Bootstrap 的固定抽样更慢，但给出的指标覆盖完整验证窗口，因此作为 `small_0.1` 的主要 Stage 2 基准。

这里存在一个必须保留的实验限制：Bootstrap best 是在 19,998 个固定抽样窗口上选择的，Main 则从该 checkpoint 开始，在 123,836 个全量窗口上继续训练。因此 Bootstrap 的 `best` 只表示抽样子集意义下的最优，不能直接与 Main 的全量 forecast 数值比较。Bootstrap best 与 last 在抽样集上的 forecast 分别约为 `2.628663` 和 `2.628686`，差异约 `0.000023`；但两者在同一 123,836 全量验证集上的对照尚未执行，后续应补充该诊断以检验子集选模的稳定性。

新阶段重新排列训练窗口后，验证 loss 仍会在上一阶段末端附近衔接，因为模型权重是连续继承的：Main 的 Segment 1 在加载 Bootstrap best 权重后开始；Extend 01 的 Segment 1 在加载 Main last 权重后开始，因此 Extend 01 的起始 forecast 约 2.5056，与 Main 末段约 2.5053 同量级。Extend 01 的新 seed 只改变后续训练窗口的访问顺序，不改变模型初始化；同理，Warmup-Constant 继承前一阶段 checkpoint 后，即使使用新的窗口排列和 scheduler，起始验证 loss 也应与上一阶段末端相近，随后才沿新的优化轨迹发展。

### 6.3 第三次训练：Stage 2 Extend 01

#### 为什么增加 Extend 01

Main 完整覆盖一轮后，验证 forecast 仍在下降，且最佳点位于末段附近，日志没有显示明确平台。这个现象提供了“当前模型可能仍能从新窗口组合中继续学习”的经验信号，但不能单独证明模型已经或尚未达到容量上限。因此 Extend 01 的定位是**受控增训实验**：从 Main 的 `last_model` 续训，在不改变数据、loss 和验证定义的情况下，仅更换窗口访问顺序，观察改善是否能够持续。

#### 具体参数与数据

- 初始化：Stage 2 Main 的 `last_model`；重新创建 fresh optimizer，避免沿用上一轮 optimizer 动量状态。
- coverage seed：`20260907`，不同于 Main 的 seed=100；窗口集合不变，排列改变。
- 可训练参数：24,819,392/24,819,392（100%）。
- batch size 64；统一峰值学习率 `1e-5`；warm-up ratio 1%；scheduler 仍为 warm-up cosine。
- 训练数据：同一 10,661,560 个训练窗口，每 segment 20,000 个窗口；本次只完成 190/534 segments，属于中间状态，不是完整第二轮 coverage。
- 验证数据：保持全量 123,836 个窗口，`replay_samples=0`，验证日期仍为 2025-07-03 至 2026-07-02。

#### 训练表现与结论边界

Extend 01 的 forecast 从 segment 1 的 2.505623 降至 segment 190 的 2.437912，190 个 segment 内持续改善。这个结果支持“在新的 coverage 顺序和相同任务定义下仍有可学习增益”的判断，因此值得继续训练；但因为只完成 190/534，不能把它和 Main 的完整 coverage 做成严格的最终优劣比较，也不能据此断言模型一定没有吃饱。后续应完成相同 coverage 计划，或使用固定计算预算进行可比实验。

### 6.4 第四次训练：Stage 2 Warmup-Constant first round

#### 实验动机与设置

Extend 01 C2 结束后，模型的全量验证 forecast 仍在下降，cosine 学习率则会随预设 global step 逐步衰减。为区分“真实平台期”和“步长衰减造成的表观放缓”，本轮从 Extend 01 C2 的 `last_model` 继续训练，将 scheduler 改为 warm-up 后恒定学习率。该设计只改变优化器的学习率轨迹；模型结构、loss、batch、训练窗口集合和验证定义保持不变。

- checkpoint 起点：Extend 01 C2 的 `last_model`；重新创建 fresh optimizer。
- coverage seed：`20260908`，与之前阶段不同；窗口集合仍为同一批 10,661,560 个训练窗口，仅访问顺序重新排列。
- 模型参数：24,819,392/24,819,392（100%）可训练。
- batch size：64；AMP 为 float16，tokenizer 编码保持 float32。
- loss：`forecast` 模式，训练目标仍为 weighted forecast loss + `0.02 * history loss`；验证选择指标为原始 forecast NLL。
- scheduler：`warmup_constant`；warm-up ratio 1%，起点 `1e-6`，达到 `1e-5` 后保持不变；global plan 为 166,854 steps，其中 1,669 steps warm-up。
- 训练覆盖：每 segment 20,000 个窗口，共 534 个 segments；本轮完成 534/534。
- 验证：全量 123,836 个窗口，验证期 2025-07-03 至 2026-07-02（242 个 signal dates）。

#### 训练结果

| 指标 | Segment 1 | 本轮最佳（约 Segment 527） | Segment 534 / last |
|---|---:|---:|---:|
| Validation forecast | 2.437121 | **2.353069** | 2.354255 |
| Validation history | 2.640016 | 未单独记录 | 2.578719 |
| Validation full | 2.622678 | 未单独记录 | 2.559669 |

本轮总耗时约 2 小时 18 分钟。末段 forecast 比历史最佳高约 0.00119，说明最后几个 segment 未刷新 best，但仍处于同一低值区间；这属于末段小幅波动，不能称为明确反弹。与 Extend 01 末端约 Segment 190 的 forecast `2.437912` 相比，WC Segment 1 为 `2.437121`，起点在连续继承权重后正常衔接；随后完整一轮继续下降至最佳 `2.353069`。

#### 解释边界

结果表明，warm-up 后恒定 `1e-5` 在完整一轮内没有发散，并且全量验证指标继续改善；截至本轮结束，loss 尚未进入可确认的平台期。因此下一轮 continuation 具有继续观测的价值。但本轮同时使用了新的 coverage seed，不能把结果解释为纯粹的 cosine/constant scheduler A/B，也不能仅凭持续下降断言模型“没有吃饱”。严格结论仍需在相同起点、相同 seed 和相同预算下比较两种 scheduler，并结合 OOS 指标。

### 6.5 第五次训练：Stage 2 WC dual T4

#### 为什么继续第二轮 Warmup-Constant

WC first round 完成时，最佳 forecast 出现在约 segment 527，且末端只比最佳值高约 0.00119，没有形成持续横盘或系统性反弹。因而第二轮的研究问题不是重复证明训练可以运行，而是检验：在统一 `1e-5` 不衰减的条件下，再完整覆盖一次重新排列的训练窗口，验证 loss 是否仍能获得可复核的增益。该阶段沿用相同模型、目标函数、训练集和验证集，计算平台改为双 T4 DDP，以提高单位时间吞吐。

#### 实验设置

- 初始化：继承 WC first round 的 `last_model`；模型权重连续，因此 segment 1 的 forecast `2.355044` 与上一轮 last `2.354255` 正常衔接。
- coverage seed：正式 coverage 使用 `20260910`。日志中还出现一次 seed `20260911` 的启动尝试，但该尝试未形成纳入连续 coverage 的有效 segment；最终记录的 534 个 segment 按 1--534 连续且无缺号。
- 硬件与并行：2×NVIDIA T4，PyTorch DDP/NCCL；每卡 batch 32，global batch 64，与上一阶段的有效 batch size 相同。
- 模型参数：24,819,392/24,819,392（100%）可训练；tokenizer 仍冻结并以 float32 编码。
- 混合精度：predictor 使用 float16 AMP 和 gradient scaling。
- loss：`forecast` 模式，history weight `0.02`；best checkpoint 继续按原始 validation forecast NLL 选择。
- scheduler：`warmup_constant`；global plan 166,854 optimizer steps，1% warm-up（1,669 steps），从 `1e-6` 升至 `1e-5` 后保持恒定。
- 数据：10,661,560 个训练窗口，每 segment 20,000 个窗口，共 534/534 segments；验证仍为固定的全量 123,836 个窗口、242 个 signal dates。
- 日志中各 segment 的训练与验证耗时合计约 27.47 小时；该值为跨 Kaggle invocation 的 segment 时间求和，不等于任意单个 Kernel 的连续运行时长。

#### 完整训练结果

| 指标 | Segment 1 | 本轮最佳（Segment 512） | Segment 534 / last |
|---|---:|---:|---:|
| Validation forecast | 2.355044 | **2.304162** | 2.306788 |
| Validation history | 2.578852 | 2.541234 | 2.541959 |
| Validation full | 2.559856 | 2.521239 | 2.522110 |

从本轮起点到最佳点，forecast loss 下降 `0.050882`，相对下降约 2.16%；到 last 的净下降为 `0.048256`，相对下降约 2.05%。最佳点出现在完整 coverage 的后段，segment 534 比最佳点高 `0.002626`。因此，该轮再次证明恒定 `1e-5` 仍能在新的完整 coverage 中继续降低固定验证集 loss；同时，末段围绕最佳值波动加大，提示继续维持同一学习率的边际收益正在减弱。

#### 阶段结论与退火依据

WC dual T4 的完成将证据从“第一轮 constant 仍下降”推进到“第二轮 constant 仍有收益，但末段开始在低值附近波动”。这仍不能证明模型达到容量上限，也不能单独证明 Constant 优于 Cosine，但已足以支持从探索阶段转入**退火阶段**：降低学习率，在保留当前表征的前提下减小参数更新噪声，检验能否获得低于 `2.304162` 的稳定验证最优值。

退火已作为独立阶段完成，结果见第 6.6 节，不回写 WC dual T4 结果。其模型结构、loss、global batch 64、训练窗口集合、全量验证集和 best 选择指标保持不变；但同时改变 coverage seed，因此不能把全部改善归因于学习率。

### 6.6 第六次训练：Cosine refinement 退火

#### 实验设置

Cosine refinement 从 WC dual T4 的连续 checkpoint 开始，目标是检验降低学习率后能否在当前低损失区域进一步收口。该轮使用新的 coverage seed `20260912`，保持 global batch 64、全量验证集和 forecast best 选择规则不变。日志记录的 scheduler 为 `uniform_cosine`，无额外 warm-up；统一学习率从约 `1e-5` 逐步退火到约 `1e-6`。global learning-rate plan 为 83,571 steps，warm-up 为 0 steps。

本轮完成 `267/267 segments`，每 segment 20,000 个训练窗口；训练集合仍为 10,661,560 个窗口，验证集合仍为 123,836 个窗口。24,819,392 个 predictor 参数全部参与优化，tokenizer 继续冻结。

#### 训练结果

| 指标 | Segment 1 | 本轮最佳（Segment 179） | Segment 267 / last |
|---|---:|---:|---:|
| Validation forecast | 2.306393 | **2.294402** | 2.295665 |
| Validation history | 2.5416 | 2.5318 | 2.5332 |
| Validation full | 2.5217 | 2.5118 | 2.5131 |

相对于 WC dual T4 的 best forecast `2.304162`，Cosine refinement 的 best 进一步下降 `0.009760`（约 0.42%）；相对于本轮 Segment 1，下降 `0.011991`（约 0.52%）。last 比 best 高 `0.001263`，说明退火后段仍有小幅波动，但整体保持在更低的损失区间。该结果支持“降低学习率可以继续改善并收口”的经验判断，但由于 refinement 同时使用了新 coverage seed，不能把全部改善归因于 scheduler 本身。

#### 退火阶段结论

Cosine refinement 已完成预注册的 267 个 segments，并刷新了当前全量验证 forecast best。至此，统一 `1e-5` 的探索阶段与低学习率退火阶段均有实测结果；后续若继续训练，应从 refinement 的 best/last 明确分叉，并单独记录新的学习率计划，避免把不同退火周期拼接成一个不可复核的阶段。

### 6.7 Stage 3：结构保持的多周期路径对齐

#### 6.7.1 研究问题与实验边界

Stage 3 研究的问题是：在不增加模型结构、不改变生产自回归推理接口的条件下，连续数值路径上的附加监督，能否使未来 H1–H10 的 OHLCVA 预测更贴近真实轨迹，并改善或至少保持独立的金融评价指标？它不是把第 10 日方向作为唯一判别目标，也不是将 Kronos 改造成带收益头或排序头的选股网络。

当前版本保留 C2 的 8 层 Transformer、条件注入模块及原有 s1/s2 heads，24,819,392 个 predictor 参数均可训练；tokenizer 参数冻结且保持 eval。**没有新增 return/barrier head、ranking loss 或模型权重 EMA teacher。** 文中的 EMA 仅指 Path Loss 的标量归一化统计。此前 `v21` 辅助头方案不作为本轮 Stage 3 的实现定义。

起点固定为 `Kronos Small 0 1 Stage2 Cosine Refinement C2` 的 `best_model`，即 Cosine Segment 179，权重 SHA-256 为 `4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a`。只加载权重；AdamW、固定学习率调度、新阶段 step=0 和 loss EMA 都重新建立。旧 `Kronos Small 0 1 Stage3 Path Alignment C1` 因实现缺陷排除出正式结果，仅保留为实现验证记录；不作为修正版父模型或比较基线，其 steps 不计入修正版训练进度，但已消耗算力仍属于项目实验成本。

#### 6.7.2 数据、预测位置与目标单位

本阶段继续使用公开数据集 `luckfu/a-share-120d-temporal-symbol-holdout`，通过原有 `QlibDataset('train')`、`QlibDataset('val')` 读取 `processed_datasets`，不另造数据拆分。训练窗口池仍为 10,661,560，单段取 20,000 个窗口；新 coverage seed 为 `20260915`。验证仍为固定的 123,836 个窗口，不改为 2,000 或 20,000 个抽样窗口。股票/时间隔离规则及仅使用历史 120 行计算归一化统计的约束沿用第 4 节。

使用从 0 开始的行索引时，131 行样本的 next-token 对齐为：

| 对象 | 当前切片 | 含义 |
| --- | --- | --- |
| predictor 输入 | token `[:, :-1]` | 行 0–129，共 130 个输入位置 |
| H1–H10 对应的预测位置 | logits/context `[:, 119:129]` | 位置 119–128 分别预测下一行 |
| token 与连续目标 | token/x `[:, 120:130]` | 行 120–129，即未来十日 |
| 不可替代的错误切片 | logits `[:, -10:]` | 对应目标行 121–130，会整体错位一天 |

训练采用 teacher forcing：预测 Hh 时，模型可使用历史及此前真实的未来前缀，而不是以自身生成的前缀滚动十次。因此这里的路径代理指标不是自由自回归预测的直接准确率，二者必须另行比较。

当前 Path Loss 的目标是**经过历史 Z-score 并沿用既有 `[-5,5]` 裁剪后的六维 OHLCVA**，六个特征等权。它不是原始价格 MAE、收益率 Huber，也没有可微反归一化环节；`delta=0.02` 是归一化单位，不能解释为 2% 收益。未额外使用目标分位数截断或残差 clamp，但也不能声称目标完全未裁剪，因为原数据预处理仍然生效。

#### 6.7.3 条件联合候选与冻结解码

Kronos 的 s2 是依赖 s1 的条件分布，而非独立第二个分类器。记第 h 个预测位置的 teacher-forced 因果上下文为 \(\mathcal C_h\)，则单位置联合概率为：

\[
p_h(i,j)=p_\theta(s_{1,h}=i\mid\mathcal C_h)
\,p_\theta(s_{2,h}=j\mid\mathcal C_h,s_{1,h}=i).
\]

实现先从 s1 分布选 Top-16；对每个 s1 候选重新经过 dependency layer 计算条件 s2 分布，再各取 s2 Top-16。每个 horizon 得到 256 个候选配对，并在这 256 个配对内按 joint log-probability 取 Top-16。不能把同一份 teacher-forced s2 logits 与所有候选 s1 作笛卡尔积。

共享 Transformer context 只计算一次。候选分支使用 `[B*K, T, D]` 的完整位置序列运行 dependency layer，然后取指定 horizon，以保留 RoPE 位置、因果遮罩和 residual 对齐；不是把单个 query 放到位置 0。C2 的 s1/s2 词表均为 1,024，以上 shortlist 不是完整联合词表，也不保证覆盖其全部概率质量。

```mermaid
flowchart TD
    C[Teacher-forced 因果 context] --> CE[均匀 H1-H10 Token CE]
    C --> S1[s1 Top-16]
    S1 --> S2[逐个 s1 候选重算条件 s2<br/>各取 Top-16]
    S2 --> J[每个 horizon 256 个 joint 配对<br/>在 shortlist 内取 Top-16]
    J --> ID[离散候选 ID<br/>按各 horizon 的相同 rank 串成十日锚点]
    ID --> D[冻结 tokenizer decode<br/>no_grad]
    J --> W[入选 joint logp 的 softmax 权重]
    D --> MIX[逐 horizon 概率加权数值路径]
    W --> MIX
    MIX --> H[六特征均匀 Huber<br/>全局 loss EMA 归一化]
    CE --> L[总损失]
    H --> L
    L --> BP[DDP backward<br/>更新 predictor，不更新 tokenizer]
```

令入选配对的 joint log-probability 为 \(\ell_{h,n}\)，每个 horizon 的归一化权重为：

\[
w_{h,n}=\frac{\exp(\ell_{h,n})}{\sum_{m=1}^{16}\exp(\ell_{h,m})}.
\]

实现将各 horizon 的第 n 名 token 配对串成长度 10 的候选序列，直接送入冻结 tokenizer decoder，得到数值锚点 \(A^{(n)}_{h,f}\)。decoder 此处接收候选十日序列，没有拼接历史 120 日 token。预测代理量为：

\[
\widehat y_{h,f}=\sum_{n=1}^{16}w_{h,n}\,\operatorname{stopgrad}(A^{(n)}_{h,f}).
\]

锚点每次 forward 重新生成，但本次 backward 将其视作常量；梯度经 joint 概率权重回到 s1 head、候选条件 s2 head、dependency layer 和共享主干，不穿过离散索引及 tokenizer decoder。Top-K 支持集变化处并非光滑；有限差分验收验证的是当前支持集邻域内的计算，而不是证明整个离散选择过程处处可微。

**近似性质必须保留：** 各 horizon 的候选排序被按 rank 串接，权重也是逐 horizon 的，而不是整条序列概率。由于 decoder 具有时间依赖，这一方法是“teacher-forced、逐日截断候选混合”的路径代理目标，不是完整十日自回归分布的精确期望。Soft 与 Hard decode 的差异小，也不能单独证明该近似无偏或预测更准确。

#### 6.7.4 训练损失、EMA 与验证选模

对 batch 大小 B、十个 horizon 和六个特征，当前 token CE 为两路 CE 的等权平均，各 horizon 等权：

\[
L_{\mathrm{CE}}=-\frac{1}{20B}\sum_{b=1}^{B}\sum_{h=1}^{10}
\left[\log p_\theta(s^*_{1,bh}\mid\mathcal C_{bh})+
\log p_\theta(s^*_{2,bh}\mid\mathcal C_{bh},s^*_{1,bh})\right].
\]

这里 CE 的 s2 分支仍以真实 s1 为条件；Path Loss 分支则以各候选 s1 为条件，两者不能混淆。原始路径损失为：

\[
L_{\mathrm{path}}=\frac{1}{60B}\sum_{b,h,f}\rho_{0.02}(\widehat y_{bhf}-y_{bhf}),
\qquad
\rho_\delta(e)=
\begin{cases}
\tfrac12 e^2,& |e|\le\delta,\\
\delta(|e|-\tfrac12\delta),& |e|>\delta.
\end{cases}
\]

令 \(\bar L_t\) 为各 rank 按本地 batch 样本数加权汇总的 detached Path Loss，全局共享归一化分母为：

\[
m_0=\bar L_0,\qquad m_t=0.99m_{t-1}+0.01\bar L_t,
\qquad
L_{\mathrm{train}}=L_{\mathrm{CE}}+0.05\frac{L_{\mathrm{path}}}{\max(m_t,10^{-6})}.
\]

EMA 不参与求导，各 rank 使用同一个分母，避免局部 EMA 对 shard 隐式重加权。首次 normalized path loss 通常接近 1，但其梯度并不为零；分母采用 detach，不能把它视作对分子求导后相消。EMA 也不是必然压小梯度的“防爆开关”，原始 loss 较小时可能放大该分支，因此仍记录裁剪前梯度并使用全局 `max_norm=1.0`、非有限值报错。

验证阶段不更新 EMA，当前 smoke test 的 checkpoint 比较量固定为：

\[
L_{\mathrm{val,select}}=L_{\mathrm{CE,val}}+0.05L_{\mathrm{path,val}}.
\]

同时单独报告 CE、raw Path Huber 和 H1–H10 MAE。这个比较量既不是 Stage 2 的 forecast-only best，也不是训练时 EMA-normalized total loss，不能用它们的绝对数值直接比较阶段优劣。单段测试产出的 best 只表示该次已完成验证的候选，不代表已经得到经过充分选模的 Stage 3 最终模型。

与 Stage 2 相比，本版还**取消了 history loss，并把 token CE 的 horizon 权重改成均匀权重**。因此即使后续结果改善，也不能直接归因于 Path Loss 单一因素。若进行机制归因，需要从同一 C2 best 出发，以相同 seed、学习率、预算、因果验证及均匀 CE/no-history 定义比较 `lambda_path=0` 与 `0.05`；该对照尚未完成。

#### 6.7.5 DDP 与因果验证的实现约束

完整目标由 `DDP(Stage3TrainingModel(...)).forward()` 组织，训练代码调用 DDP 对象后再 backward，而不是绕过它调用 `.module.encode_context()` 等方法。包装器不改变 predictor 权重结构；导出仍为原 Kronos `save_pretrained` 格式。

Stage 3 对 CE 和 candidate 两条 s2 分支都显式传入 `is_causal=True`；验证使用 eval 关闭 dropout。这样将“是否因果”与“是否训练模式”解耦。默认 `is_causal=None` 仍保留旧 Kronos/C2 行为，不静默改写历史基线。此前 eval dependency attention 的非因果行为会让早期 horizon 读取后续真实 context，因此新的 causal validation 不能与旧日志无条件拼接；这也不自动推翻按原生产自回归 evaluator 得到的历史 OOS 指标。

全量验证按互不重叠的索引分片分配到各 rank，不用补齐重复样本的 DistributedSampler；允许两 rank 的末批大小或批次数不同，在末尾汇总计数和指标，断言实际验证数等于 123,836。模型参数、optimizer、loss EMA、step、coverage 信息和 RNG 随 segment 边界保存；CPU 存取验收与 GPU 多进程接力验收分开表述。

#### 6.7.6 已获得的正确性证据

本地验收使用 Conda Python 3.12、PyTorch 2.13.0、CPU；16 项自动测试通过。C2 best 的真实数据复核从原 123,836 个验证窗口取 8 个窗口、batch=2，只用于工程检查，未更新 C2 权重，也没有替代正式全量验证。

| 验收项 | 已确认结果 | 结论边界 |
| --- | --- | --- |
| 条件候选 s2 对显式因果 reference forward | 四批最大偏差均为 0 | 证明当前条件与位置映射一致 |
| 扰动未来 context 第 129 行 | 对 query 119:129 的影响均为 0 | 对被测位置满足因果隔离 |
| Path-only backward | s1/s2/dependency/backbone 梯度均有限且非零；tokenizer 无梯度 | 证明梯度可达，不证明预测收益 |
| 固定支持集的有限差分 | 真实窗口 log-probability 差分对 autograd 误差小于 `4e-14` | 局部数值一致性，不是全局可微性证明 |
| 两进程 Gloo 对单进程 global batch | 两步梯度最大差分别 `3.73e-8`、`5.96e-8` | 合成小模型、关闭 dropout 的确定性对照 |
| 两 rank 更新后参数 | 两步最大差均为 0 | CPU DDP 同步验收，不替代 NCCL/T4 验收 |
| checkpoint 存取及下一步更新 | 确定性 CPU 对照逐参数一致，原 Kronos 可加载导出模型 | GPU RNG/设备行为另测 |
| 非整除验证分片 | 5 个样本按 3/2 分片，精确统计 5/5 | 验证无重复补齐逻辑 |

在真实 C2 best 的一次完整包装器 backward 中，示例 batch 的 Token Loss 为 `2.523433`，raw Path Huber 为 `0.008134`，首次 EMA-normalized Path 为 `1.0`，总损失为 `2.573433`。四组裁剪前梯度范数分别约为 s1 `1.5684`、s2 `0.8653`、dependency `1.0532`、backbone `1.5658`；所有可训练 predictor 参数均有有限梯度，tokenizer 仍为 eval 且无梯度。这是数值检查，不能解释为训练后的 loss 或已经收敛。

显式因果复核的四批 Top-16 joint 概率质量均值约为 `51.00% / 57.29% / 76.63% / 55.19%`。它们不同于单独 s1 Top-K 的覆盖率，也不足以决定 K 的最优值；后续同时记录 joint mass、s1 熵、在入选 s1 内加权的条件 s2 熵，并观察路径方差和分 horizon 误差。

#### 6.7.7 Kaggle 单段测试设计与结果状态

修正版单段测试以独立 Kernel 和新 SwanLab run 执行，不接续废弃 C1 的曲线。以下为此次完成任务的配置：

| 配置 | 本次值 |
| --- | --- |
| Kernel / version | `smmt315/kronos-small-0-1-stage3-joint-path-smoke` / Version 1 |
| 训练源码 commit | `a854641420068dbbc6daf2dcfa329c8ecbaa425a` |
| 提交入口 commit | `378077062cd42861370b23402be8221cf108bf3c` |
| 父模型 | Cosine C2 best，Segment 179，仅加载权重 |
| 训练方式 | 双 Tesla T4，DDP/NCCL，全部 predictor 参数更新 |
| batch / LR / optimizer | 32/卡，global 64；统一 `2e-6`；fresh AdamW，weight decay=0.01 |
| scheduler / 精度 | fixed、无 warm-up；本次 FP32，AMP 关闭；tokenizer FP32 |
| 数据与预算 | 20,000 训练窗口；1 segment；123,836 全量验证窗口 |
| 覆盖顺序 | `coverage_seed=20260915`，不改变原训练窗口集合 |
| 运行上限 | Kernel 硬超时 18,000 秒（5 小时），低于 Kaggle 12 小时限制 |
| 看板 | `small_0.1_stage3_joint_path_alignment_from_c2_best_v2` |

执行顺序为：输入唯一性及 SHA-256 检查、环境/CPU 测试、双 T4 的可丢弃微步预检、重新从 C2 best 建立 fresh 状态、一个完整 segment、全量验证和输出契约检查。GPU 预检中的临时更新不进入正式训练血缘。若显存或预估耗时超过预设门槛，入口应在训练 segment 前失败退出，不自动改 batch、K 或验证样本数。

2026-09-15 的 Output 验收确认任务 `COMPLETE`，日志无 traceback，实际设备为两张 Tesla T4，`torch==2.6.0+cu124`，输入哈希与父 C2 best 一致。GPU 预检及正式单段结果如下：

| 验收指标 | 实测结果 |
| --- | --- |
| DDP 梯度与 global-batch reference 的最大差 | `7.15e-7` |
| 两 rank 更新后参数最大差 | `0` |
| GPU checkpoint roundtrip | 通过 |
| GPU 预检峰值 allocated 显存 | `3.061 GiB`/被测设备 |
| tokenizer 非空梯度数 | `0` |
| 正式训练 | Segment 1，313 optimizer steps，20,000 窗口 |
| 正式验证 | 123,836 窗口，无补齐重复样本 |
| 验证 Token CE / raw Path Huber | `2.339598417 / 0.006650186` |
| 验证选模量 | `2.339931488` |
| 验证 s1 熵 / 条件 s2 熵 | `2.689964 / 1.844142` |
| 验证入选 joint 概率质量均值 | `0.567784` |
| 验证最大残差（归一化六特征） | `7.440997` |
| segment 训练和验证耗时 | `663.24 s`，约 11.05 分钟（不含环境安装） |

H1–H10 六特征平均绝对误差依次为 `0.381719, 0.356383, 0.344300, 0.338616, 0.335842, 0.334546, 0.333850, 0.333058, 0.332736, 0.332248`，单位为既有归一化空间，不是价格或收益百分比。末批裁剪前梯度范数约 `0.775704`；不能将末批数值当成整段梯度分布。

本地进一步读取完整 `last_state.pt`，确认 `segment=next_epoch=1`、`resume_step=0`、`step=313`、132 个参数的 optimizer state、两份 rank RNG 和 Path Loss EMA 均存在，coverage 已消费 20,000 个唯一窗口；State 内模型与 `last_model/model.safetensors` 逐 tensor 一致且有限。此时 best 与 last 相同，权重 SHA-256 为 `d1ac9e49ff2ac4f139939211f48ce815c753ab4381dc0c1fd7b411a634aa6780`；State SHA-256 为 `cf4a62e4084e6d380cd39165cee2bd6a5c48535264a01ab1960d01e6d8fe53f8`。

以上证明完整训练/验证链路可运行并可保存状态，不证明一段训练已改善预测。尚缺 C2 best 在同一 causal Stage3 口径下的全量起点结果；不能拿旧 C2 forecast `2.294402` 与本次 CE `2.339598` 直接判定退化。下一步短程观察将在相同口径补测 C2 起点及 Segment 1，并保持原 seed、LR、optimizer/EMA、global step 和看板，从 Segment 2 接续至 Segment 6。验证新增预测/目标逐 horizon 方差监控及 float64 汇总仅用于观测，不改变目标函数。

- 测试 Kernel：[Kronos Small 0 1 Stage3 Joint Path Smoke](https://www.kaggle.com/code/smmt315/kronos-small-0-1-stage3-joint-path-smoke)
- 新看板：[Stage3 joint path alignment](https://swanlab.cn/@roc_fu/finance/runs/small_0.1_stage3_joint_path_alignment_from_c2_best_v2)

#### 6.7.8 结果判定与后续评价

单段测试首先回答运行正确性、梯度同步、显存/耗时、日志和全量验证能否完成，不足以判断泛化或批准长期增训。后续如开展受控探针，应同时检查 Token CE、raw Path Huber、H1–H10 MAE、预测截面/时间方差、概率熵和 joint mass，防止仅靠输出向均值收缩降低平均误差。不能从单一熵阈值、梯度范数或 MAE 降幅推导模型成功或失败。

最终比较 C2 best 与 Stage3 候选时，两者必须使用同一生产自回归 inference/evaluator、窗口、采样配置与收益定义；按 H1–H10 报告数值路径误差，并独立报告方向准确率、pooled/daily Rank IC、ICIR、正 IC 日期比例及 Top-Bottom spread。换手、毛/净收益、Sharpe 与最大回撤需要另行固定交易规则和成本口径，不能由当前 Path Loss 推算。

MAE 下降不保证 Rank IC 上升；二者背离时应报告路径校准与排序的权衡，不能宣称整体金融预测能力改善。Stage 3 不使用 OOS 调权重、学习率或 K；第 8 节已知的 C2 OOS 结果保持为历史证据，已参与方法讨论的数据也不能重新宣称为完全未见的最终确认集。条件允许时，应再用未参与设计选择的新时间区间确认结论。

## 7. 训练执行与 Kaggle 接力

每个 segment 包含固定数量的唯一窗口，segment 完成后进行验证并保存推理权重 `last_model` 和完整训练状态 `last_state.pt`；若该阶段预先声明的指标刷新则保存 `best_model`（Stage 2 为 forecast，Stage 3 单段测试为第 6.7.4 节的比较量）。Kaggle 单次任务受时限约束，因此同阶段接力按 segment 边界从完整 State 恢复模型、优化器、scheduler、全局 step、coverage cursor 和 best 指标。Stage 3 首次启动则是从 C2 best 权重建立新实验，不适用“继承 C2 optimizer”的同阶段恢复规则。

日志应同时输出到终端和文件，使用无缓冲、逐段 flush 的方式，确保 Kaggle CLI 能实时读取；任何训练、评估或诊断 Kernel 都必须遵守该约定。训练日志中的 global schedule 不因 chunk 截断而重置，scheduler 进度按全局 optimizer step 恢复。

```mermaid
flowchart LR
    A[读取固定 coverage_order] --> B[取下一个20,000窗口 segment]
    B --> C[训练一个 segment]
    C --> D[运行固定验证集]
    D --> E[保存 last_model 推理权重<br/>及 last_state.pt 完整训练状态]
    D --> F{本阶段预定义<br/>选模指标是否刷新?}
    F -- 是 --> G[保存 best_model]
    F -- 否 --> H[保留已有 best_model]
    E --> I{达到本次 Kaggle<br/>时间或segment上限?}
    I -- 否 --> B
    I -- 是 --> J[在segment边界安全退出]
    J --> K[下一账号或下一chunk<br/>恢复全局step与cursor]
    K --> B
```

## 8. Stage 2 训练结果与 OOS 证据

第 8.1–8.2 节保留早期评估的原始统计，以便追溯；2026-09-15 发现两次汇总混用了裁剪还原目标、原始标签和零收益处理规则。**最终 C2 对比以第 8.3 节统一原始价格口径的复算结果为准。** Stage 2 Main 的旧指标尚未按新口径复算，不与第 8.3 节直接作数值差分。

在独立 OOS 日期 2026-08-03 至 2026-08-10（6 个 signal dates，30,930 个窗口）上，Stage 2 checkpoint 相比原始底座取得以下结果：

| 模型 | 10 日方向准确率 | pooled Rank IC | 日均 Rank IC |
|---|---:|---:|---:|
| 原始 Kronos-small | 48.765% | -0.02248 | -0.03673 |
| Stage 2 best（segment 530） | 50.970% | 0.08651 | 0.08104 |
| Stage 2 last（segment 534） | 50.947% | 0.08627 | 0.08135 |

历史动量基线（10 日/20 日）方向准确率分别为 45.45%/38.98%，pooled Rank IC 为 -0.17906/-0.34855。十分位分组中，Stage 2 best/last 的 Top-minus-Bottom 平均约 +3.55%，原始底座约 -1.70%。这些结果只覆盖 6 个 OOS 日期，应作为初步证据，不能外推为长期稳定性或可交易性证明。

本文同时报告两种 Rank IC，定义必须区分：`pooled Rank IC` 将所有股票×signal-date 行拼接后计算一次 Spearman 相关；`daily Rank IC` 则先按 signal date 对当日横截面股票计算 Spearman，再对各日期的 IC 取平均。pooled 统计可能受各日期股票数量影响，因此不能替代 daily 统计；扩展 OOS 后还应报告每日期 IC、标准差、ICIR 和正 IC 比例。旧版 6 日期结果存在大量同日横截面和相邻日期相关性；扩展到 19 日期后，有效时间观测增加但仍不等同于 IID 样本。

### 8.1 Cosine C2 的首次 OOS 评估（历史口径）

Cosine refinement C2 完成后，使用与训练完全隔离的 OOS 包进行重新评估。评估范围为 2026-08-03 至 2026-08-10，共 6 个 signal dates、30,930 个股票窗口；每个模型使用同一 tokenizer、同一窗口集合、同一随机采样设置和同一自回归解码流程。C2 best 的实际 checkpoint 是 Cosine 阶段 Segment 179，C2 last 是 Segment 267；评估器输出中沿用了旧的 `best_segment_530`/`last_segment_534` 标签，不能据此误认为它们属于 Stage 2 Main。

| 模型 | D10 方向准确率 | pooled Rank IC | 日均 Rank IC | 正 IC 日期 | ICIR（按日） | Top-Bottom 10% 平均收益差 |
|---|---:|---:|---:|---:|---:|---:|
| 原始 Kronos-small | 48.765% | -0.0225 | -0.0367 | 1/6 | -0.83 | -1.71% |
| Stage 2 Main best | 50.970% | 0.0865 | 0.0810 | 6/6 | 2.46 | 3.55% |
| Stage 2 Main last | 50.947% | 0.0863 | 0.0813 | 6/6 | 2.48 | 3.55% |
| Cosine C2 best（Segment 179） | **54.455%** | **0.1569** | **0.1354** | **6/6** | 2.20 | **5.35%** |
| Cosine C2 last（Segment 267） | 54.336% | 0.1480 | 0.1244 | 6/6 | **2.30** | 5.04% |

其中，方向准确率表示预测收益与实际收益的涨跌符号相同的比例；Rank IC 表示模型预测排序与实际未来收益排序的 Spearman 相关；Top-Bottom 10% 是按预测分数选出的最高十分位与最低十分位的实际 D10 收益差。ICIR 在本表中定义为 6 个 signal-date 横截面 IC 的均值除以样本标准差，仅作描述性统计。

相对于 Stage 2 Main best，Cosine C2 best 的 D10 方向准确率提高约 3.49 个百分点，pooled Rank IC 从 0.0865 提高到 0.1569，日均 Rank IC 从 0.0810 提高到 0.1354，Top-Bottom 收益差从约 3.55% 提高到约 5.35%。原始模型和 10 日动量基线均为负 Rank IC；因此本次改善不是简单动量规则可以解释的。C2 last 略低于 C2 best，但仍保持相近的 OOS 排序能力，说明退火阶段的改善不是单个 checkpoint 的孤立异常。

这组历史口径结果支持短 OOS 区间内的排序改善，但不能证明长期稳定 Alpha，也不能直接推出扣除手续费后的可交易收益。OOS 已扩展并统一复算至 19 个 signal dates（第 8.3 节）；D10 目标仍存在重叠，更长时间及交易成本检验仍未完成。

### 8.2 扩展 OOS：2026-08-11 至 2026-08-27（历史本地汇总）

在原有 2026-08-03 至 2026-08-10 的 6 个 signal dates 之外，使用数据集 `A-share 120D Temporal Symbol Holdout` 中新增的全量 OOS 包进行扩展评估。新增区间覆盖 13 个 signal dates（2026-08-11、12、13、14、17、18、19、20、21、24、25、26、27），共 66,986 个窗口；三种模型使用完全相同的窗口集合、tokenizer 和自回归解码流程。OOS 仍不参与训练、scheduler 决策或 best checkpoint 选择。

| 模型 | 窗口数 | D10 方向准确率 | pooled Rank IC | 日均 Rank IC | 日 IC 标准差 | 正 IC 日期 |
|---|---:|---:|---:|---:|---:|---:|
| 原始 Kronos-small | 66,986 | 47.801% | 0.0256 | 0.0209 | 0.0306 | 8/13 |
| C2 best（Cosine Segment 179） | 66,986 | 48.697% | **0.1864** | **0.1741** | 0.0694 | **13/13** |
| C2 last（Cosine Segment 267） | 66,986 | 48.553% | 0.1822 | 0.1700 | 0.0666 | **13/13** |

新增 13 个日期上，C2 best/last 的 Rank IC 均显著高于原始底座，且每日横截面 IC 全部为正；但方向准确率仍接近 50%，说明主要改善体现为横截面排序而非逐股票涨跌命中率。C2 best 的 pooled Rank IC 仅比 C2 last 高约 0.004，二者差异不大，支持保留 best 与 last 两个 checkpoint 进行后续稳健性比较。

本次 Kaggle Kernel 的合并文件包含三个模型各 66,986 行、13 个 signal dates，共 200,958 行，模型×identity 无重复。`summary.json` 实际存在且可解析；此前“未生成”的描述不成立。上表引用 `summary_local.json`，不是 Kernel 原始汇总：前者使用原始 `return_10d` 标签和 `>0` 二分类，后者使用裁剪还原的 `actual_return_d10` 和 `sign` 三值比较。因此不能混用这两套数字；第 8.3 节从完整预测文件及原始 panel 统一复算。

将本次 13 个日期与旧的 6 个日期合并后，OOS 总覆盖应为 19 个 signal dates、97,916 个窗口/模型。由于日期之间的 D10 标签存在重叠，19 个日期仍不是 97,916 个 IID 观测；正式结论应同时报告逐日 IC、ICIR、正 IC 比例、Top-minus-Bottom、换手和交易成本，并继续扩展更长时间跨度。这一证据限制不妨碍第 6.7 节的 Stage 3 工程单段测试，但不能据此预先批准长期增训或宣称 Stage 3 改善泛化。

评估产物与可复核日志如下：

- Kaggle Kernel：[smmt315/kronos-small-0-1-c2-alpha-oos-evaluation](https://www.kaggle.com/code/smmt315/kronos-small-0-1-c2-alpha-oos-evaluation)
- 本次扩展 OOS 本地汇总：[summary_local.json](/Users/fupengcheng/Documents/Kronos/artifacts/kronos_small_0_1_c2_alpha_oos_output_smmt315_20260914/kronos_small_0_1_stage2_oos/summary_local.json)
- 本次扩展 OOS 预测：[predictions.csv.gz](/Users/fupengcheng/Documents/Kronos/artifacts/kronos_small_0_1_c2_alpha_oos_output_smmt315_20260914/kronos_small_0_1_stage2_oos/predictions.csv.gz)
- 预测分片合并文件：[predictions.csv.gz](/Users/fupengcheng/Documents/Kronos/artifacts/kronos_small_0_1_c2_alpha_oos_output/kronos_small_0_1_stage2_oos/predictions.csv.gz)

验证 signal 结束于 2026-07-02，OOS signal 从 2026-08-03 开始，中间存在 2026-07-03 至 2026-08-02 的日历间隔。evaluation manifest 使用 2026-07-31 作为父数据训练目标截止边界，不能由此反推本轮实际训练窗口的最大目标日期，也未声明专门设计的 31 日 embargo。首次 OOS 的 D10 target 日期为 2026-08-17 至 2026-08-24，新增区间为 2026-08-25 至 2026-09-10；不能将旧 6 日 target 范围写成全部 OOS 范围。

```mermaid
flowchart LR
    M[训练后模型] --> G[自回归生成未来10日 OHLCVA]
    G --> R[形成预测收益或横截面分数]
    R --> DA[方向准确率<br/>涨跌判断是否正确]
    R --> IC[Rank IC<br/>预测排序与真实排序相关性]
    R --> DEC[按预测分成十组]
    DEC --> TB[Top-minus-Bottom<br/>最高组减最低组收益]
    DA --> L[当前覆盖19个signal dates<br/>仍需更长时间验证]
    IC --> L
    TB --> L
```

### 8.3 完整 19 日 OOS：统一原始价格口径复算

#### 评估修正与可复现定义

复算不重新运行模型、不改变 checkpoint 或采样参数。它读取两次已有 `predictions.csv.gz`，按 `model + identity` 去重断言，并依据 identity 中的股票、起始行、signal/target 日期恢复原数据窗口。三模型在各区间拥有完全相同的 identity 集；两区间互不重叠，合计 97,916 个窗口/模型、19 个 signal dates。

旧 evaluator 从已裁剪到 `[-5,5]` 的 normalized 输入还原真实价格与 signal 收盘价，极端行情下会改变收益标签及预测收益分母。新增 13 日中，157 个窗口的旧 D10 目标与原始标签之差超过 `1e-5`；最大例子的旧目标约 `17.78%`、原始目标约 `91.01%`。这不是浮点舍入。复算采用：

\[
r_h=\frac{C^{raw}_{t+h}}{C^{raw}_t}-1,\qquad
\widehat r_h=(1+\widehat r^{saved}_h)\frac{C^{old}_t}{C^{raw}_t}-1.
\]

其中 \(C^{old}_t\) 按旧 evaluator 的 float32、历史均值/标准差和裁剪逻辑重建；\(C^{raw}\) 为评估 panel 中未做窗口裁剪的收盘价。恢复保留已有预测文件的有限精度，没有重新生成预测。D10 原始目标逐样本对照 manifest 标签，误差须小于 `1e-8`。后续 evaluator 应直接保存原始 signal 收盘价与完整真实价格路径，避免再靠重建。

方向准确率统一定义为 `(pred_return > 0) == (actual_return > 0)`，即上涨与不涨二分类，零收益归“不涨”；不是 long/neutral/short 三分类。Rank IC 为 average-tie-rank Spearman；daily IC 按日期等权，ICIR 为日均 IC 除以样本标准差，不年化。Top-Bottom 使用每日按预测排序最高/最低 `ceil(10% × 当日股票数)`，计算其原始累计 horizon 收益差后按日期等权。它不是扣费后的多空策略收益。

#### D10 主结果

| 区间 | 模型 | 窗口数 | 方向准确率 | pooled Rank IC | 日均 Rank IC | ICIR | 正 IC 日期 | Top-Bottom |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 首次 6 日 | 原始 small | 30,930 | 49.311% | -0.02329 | -0.03752 | -0.841 | 1/6 | -1.879% |
| 首次 6 日 | C2 best 179 | 30,930 | 55.021% | 0.15601 | 0.13446 | 2.202 | 6/6 | 5.066% |
| 首次 6 日 | C2 last 267 | 30,930 | 54.905% | 0.14701 | 0.12351 | 2.297 | 6/6 | 4.793% |
| 新增 13 日 | 原始 small | 66,986 | 47.803% | 0.02575 | 0.02106 | 0.687 | 8/13 | 0.626% |
| 新增 13 日 | C2 best 179 | 66,986 | 48.697% | 0.18647 | 0.17416 | 2.509 | 13/13 | 3.447% |
| 新增 13 日 | C2 last 267 | 66,986 | 48.555% | 0.18233 | 0.17006 | 2.551 | 13/13 | 3.332% |
| 完整 19 日 | 原始 small | 97,916 | 48.279% | 0.00917 | 0.00256 | 0.058 | 9/19 | -0.165% |
| 完整 19 日 | C2 best 179 | 97,916 | **50.694%** | **0.16910** | **0.16163** | **2.381** | **19/19** | **3.958%** |
| 完整 19 日 | C2 last 267 | 97,916 | 50.561% | 0.16289 | 0.15536 | 2.380 | 19/19 | 3.794% |

完整区间的 pooled IC 是拼接原始行后重新计算，不是两个区间 IC 的加权平均。C2 best 的 D10 收益 MAE 为 `7.829` 个百分点，原始 small 为 `8.503` 个百分点，C2 last 为 `7.921` 个百分点。当前证据支持 C2 的排序与平均误差改善，但二分类命中率仅略高于 50%；应保留这两种结论的区别。

#### H1–H10 完整路径评价

以下均为完整 19 日、相同 97,916 窗口的累计 horizon 收益。MAE 单位为收益百分点；这不是 Stage 3 的归一化六特征 MAE。

| Horizon | 原始方向准确率 | C2 best 方向准确率 | 原始 pooled IC | C2 best pooled IC | 原始 MAE | C2 best MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H1 | 50.412% | 54.730% | 0.0136 | 0.1335 | 2.836 | 2.743 |
| H2 | 48.734% | 52.771% | -0.0073 | 0.1169 | 4.005 | 3.798 |
| H3 | 47.791% | 50.222% | -0.0223 | 0.0807 | 4.870 | 4.591 |
| H4 | 47.314% | 49.671% | -0.0294 | 0.0820 | 5.672 | 5.272 |
| H5 | 47.013% | 49.613% | -0.0309 | 0.0981 | 6.301 | 5.707 |
| H6 | 47.531% | 50.248% | -0.0276 | 0.1195 | 6.826 | 6.137 |
| H7 | 47.821% | 50.831% | -0.0214 | 0.1361 | 7.328 | 6.600 |
| H8 | 47.988% | 50.207% | -0.0103 | 0.1350 | 7.734 | 7.037 |
| H9 | 48.232% | 50.206% | 0.0020 | 0.1468 | 8.137 | 7.462 |
| H10 | 48.279% | 50.694% | 0.0092 | 0.1691 | 8.503 | 7.829 |

三模型所有 horizon 的统计、D10 逐日期 IC/Top-Bottom、数据与预测 SHA-256 见随仓库保存的 [统一复算报告](/Users/fupengcheng/Documents/Kronos/finetune/reports/small_0_1_oos_raw_audit_20260915.json)，实现见 [audit_small_0_1_oos.py](/Users/fupengcheng/Documents/Kronos/finetune/audit_small_0_1_oos.py)；脚本还会输出所有 horizon 的逐日期明细。19 日仍短，且相邻 D10 标签重叠；未在此声称统计显著性、长期稳定 Alpha 或可交易净收益。该 OOS 已用于研究讨论，不能在 Stage 3 再次使用时声称为完全未见的确认集。

## 9. 训练健康性与解释边界

Stage 2 的五段正式训练均更新全部 predictor 参数。Main 的全量验证 forecast 从 2.6384 降至约 2.5052，Extend 01 在 190 segments 内继续降至约 2.4379，WC first round 最佳约 2.3531，WC dual T4 最佳约 2.3042，最终 Cosine refinement 最佳为 2.294402。两轮 WC 未发散，随后退火也已完成；这些是已完成阶段的优化轨迹，不是当前继续训练指令。

但“线性下降”不能单独证明模型没有吃饱，也不能证明学习率过小或金融信号已达到信息论上限。Cosine scheduler、窗口顺序、验证噪声和多步预测难度都会影响曲线形状。论文中应将 entropy floor、token 类别不平衡、层级 drift 等作为待检验假设，并使用权重位移、分 horizon loss 和预测分布熵等诊断提供证据。

### 9.1 Batch 噪声与优化速度的区分

Warmup-Constant 运行期间，dashboard 中的 raw `train/forecast_loss`、`train/loss` 和 `train/history_loss` 是 batch-level 指标。当前 batch size 为 64，而不同 batch 在股票、行业、市值、波动率、市场状态和历史窗口上差异很大，因此 train 曲线在约 2.2--2.7 区间高频抖动，不能直接用肉眼判断 `1e-5` 是否过小。

相反，固定的 123,836 个验证窗口提供了更低噪声的长期指标。WC first round 从 full/forecast `2.622678/2.437121` 降至 `2.559669/2.354255`，最佳 forecast 为 `2.353069`；WC dual T4 又从 `2.559856/2.355044` 降至 last 的 `2.522110/2.306788`，最佳 forecast 为 `2.304162`。这支持“统一 `1e-5` 在两轮完整 coverage 中持续推动优化”的判断，但仍不能证明它是单位计算量最优的学习率，也不能把末段波动直接等同于模型容量耗尽。

```mermaid
flowchart LR
    B[batch=64<br/>股票与窗口异质性高] --> N[raw train loss<br/>高频噪声]
    V[固定123,836验证窗口] --> S[validation loss<br/>低噪声长期趋势]
    N --> Q{不能单独判断<br/>LR是否过小}
    S --> Q2[历史WC验证轨迹<br/>用于当时的预算决策]
    Q --> A[按segment或1000 step求均值]
    Q2 --> A
    A --> C[与validation trajectory<br/>放在同一图中比较]
```

建议将训练指标按每个 segment（20,000 个窗口、约 312--313 个 batch）或每 1,000 个 optimizer step 聚合，额外记录：`train/forecast_loss_segment_mean`、`train/loss_segment_mean` 和 `train/history_loss_segment_mean`。聚合后若 train forecast 也下降，说明训练和验证趋势一致；若 train 长期横盘而 validation 继续下降，则应优先检查 train/validation 的 loss 定义、teacher forcing、dropout/eval 状态和 forecast 权重是否完全同构。

还必须核对 dashboard 中 `train/loss` 的精确定义：理论训练目标是 `weighted_forecast_loss + 0.02 * history_loss`，而 `forecast_loss` 可能是 weighted 或 raw/unweighted 统计。验证日志中的 `Validation Forecast` 已明确是原始 forecast NLL；训练指标若采用不同定义，不能直接用数值大小互相比较。

### 9.2 尚未完成的学习率归因对照

Stage 2 已结束，未执行统一起点的 `1e-5`、`2e-5`、`3e-5` 学习率对照。若未来研究学习率效率，需要相同数据、seed、验证与预算的独立实验；不能把已完成的不同 continuation 当作该对照。这不是当前 Stage 3 的训练计划。

## 10. 导师评审意见与论文补强路线

导师评审认为，本方案的主要优点是数据规模、窗口内归一化、训练/验证隔离、完整 coverage 和 Kaggle 可复现接力流程；当前需要补强的是回答“性能提升究竟来自哪里、在哪些市场状态下有效、能否稳定复现”。以下内容是尚未完成的归因与稳健性研究建议，不替代已经完成的 Stage 2 和 OOS 结果。

### 10.1 先明确 split protocol

源股票池 5,198 只股票按分层规则抽出 520 只验证股票和 4,678 只主要训练股票；520 只验证股票在 2024-12-31 前的历史仍可用于训练侧，2025-01-01 后的数据只用于验证侧；最终 516 只验证股票形成有效验证窗口。验证 signal 日期为 2025-07-03 至 2026-07-02，当前 OOS signal 覆盖 2026-08-03 至 2026-08-27 的 19 个交易日，属于全市场时间 OOS，不限于 520 只验证股票。标准化统计量只来自各窗口过去 120 日。未来还需进一步审查行业/市值统计的时点可得性及重叠窗口对统计推断的影响。

### 10.2 条件分支归因实验（尚未执行）

当前 OOS 提升同时包含“全参数 A 股微调”和“行业/市值条件”两个因素，尚不能证明提升单独来自条件信息。建议补充以下消融矩阵：

| 实验 | Sector | Size | Full FT | 要回答的问题 |
|---|---:|---:|---:|---|
| A | 关 | 关 | 开 | 纯 domain adaptation 的收益 |
| B | 开 | 关 | 开 | 行业条件的增量收益 |
| C | 关 | 开 | 开 | 市值条件的增量收益 |
| D | 开 | 开 | 开 | 完整 `small_0.1` |

同时建议比较 condition-only、最后两个 Transformer block + condition、全参数微调及 LoRA/adapter。由于条件在第 6 层后注入，这组实验可检验高层条件调制是否足够，还是必须重塑底层 token 表征。

这组 A/B/C/D 用于区分一般 A 股微调与行业/市值条件的增量影响，应同时比较 loss、横截面 Rank IC、Top-minus-Bottom 和逐 horizon 指标。它属于后续研究建议，不是已经启动的 Stage 3 路径对齐实验，也不能在未完成时作为条件分支有效性的证据。

```mermaid
flowchart TD
    BASE[同一Kronos-small底座<br/>同一数据与训练预算] --> A[实验A<br/>无Sector 无Size<br/>Full FT]
    BASE --> B[实验B<br/>Sector only<br/>Full FT]
    BASE --> C[实验C<br/>Size only<br/>Full FT]
    BASE --> D[实验D<br/>Sector + Size<br/>Full FT]
    A --> CMP[统一验证与OOS比较]
    B --> CMP
    C --> CMP
    D --> CMP
    CMP --> R1[D-A<br/>完整条件的增量]
    CMP --> R2[B-A<br/>Sector贡献]
    CMP --> R3[C-A<br/>Size贡献]
```

### 10.3 训练目标与金融评价对齐

Stage 2 优化加权 token NLL 与 history 辅助项，而金融评价报告方向准确率、Rank IC 和 Top-minus-Bottom spread，两者之间存在目标差异。此前讨论过 return/ranking auxiliary 和 `return_head`/`barrier_head`，但这些属于未采用的备选方案，不是当前 Stage 3 的实施计划。

当前采用第 6.7 节的结构保持方案：均匀 H1–H10 token CE 加归一化 OHLCVA 路径 Huber，不增加收益头或排序头。应逐 horizon 报告路径误差、方向准确率和 Rank IC；为了识别路径损失的增量效果，优先补充相同 C2 best、均匀 CE、无 history、因果验证及训练预算下的 CE-only 对照。优化路径代理损失不等同于直接优化金融指标，二者都需要独立实测。

### 10.4 Warmup-Constant 的公平比较

Extend 01 更换 coverage seed 的性质是 optimization continuation，而不是独立泛化实验。Warmup-Constant 的两轮 continuation 已分别完成 534 个 segments；但由于起点和 coverage seed 均不同，仍不能把它们与 cosine 结果当作严格的 scheduler A/B。严格比较应使用相同 checkpoint 起点、数据集合、batch、seed、验证集和 segment 预算，至少比较 100%、200% 和完整 coverage 的验证轨迹，再进行 OOS 比较。

已经完成的 `small_0.1_stage2_wc_last` 是本项目用于回答“Cosine 后期衰减是否过早限制优化”的直接实验。它保持统一峰值学习率 `1e-5`，warm-up 后不再衰减，且不改变 checkpoint 起点、loss、batch、数据集和验证定义；本次按计划使用新的 coverage seed `20260908`，因此它同时包含 scheduler 改变和窗口访问顺序改变两个因素，不能被表述为纯 scheduler A/B。完整 534/534 segments 后 validation forecast 最佳为 `2.353069`。其后的 `small_0.1_stage2_wc_dual_t4` 又完成一轮 534/534 segments，最佳 forecast 为 `2.304162`；这一 continuation 已在第 6.5 节单独报告。

在 WC first round 结束时，validation full loss 从 `2.622678` 降至 `2.559669`，forecast loss 从 `2.437121` 降至 `2.354255`，最佳 forecast 为 `2.353069`；WC dual T4 的最佳 forecast 进一步降至 `2.304162`。两轮曲线存在正常 batch/segment 抖动，第二轮末段开始围绕低值波动。随后已完成 Cosine refinement，最佳 forecast 为 `2.294402`。这一先后变化不构成 Constant/Cosine 的控制变量比较。

因此本实验的执行纪律是：训练完成前不改学习率、不切换 scheduler、不修改 loss 权重、不更换 batch 或 seed、不手动挑选中间 checkpoint。训练完成后统一比较：同一训练进度下的 validation trajectory、best/last checkpoint、最终是否仍有下降、以及独立 OOS 的方向准确率和 Rank IC。只有在相同预算或相同 coverage 进度下比较，才能区分“单纯继续训练的收益”和“Constant 学习率本身的收益”。从优化理论角度，本实验还可视为对 **River Valley 损失地形假说** 的探索：在 cosine 后期步长衰减可能把模型限制在浅层局部区域的假设下，warm-up 后保持较大的恒定步长，可能通过持续的随机梯度噪声（SGN）帮助参数离开浅层局部最优，并为最终的物理平台期退火提供依据。River Valley 和 SGN 在本项目中仍是待验证的解释性假说，不是已由当前曲线证明的机制。

### 10.5 OOS、统计稳定性与市场状态

当前已完成 19 个 signal dates、97,916 个窗口/模型的统一原始价格 OOS 复算。C2 best 的 pooled IC 为 0.16910、日均 IC 为 0.16163；不再以早期 Main 的 0.0865 或仅 6 日结果代表最终模型。窗口与日期均有相关性，有效时间样本远小于窗口数量。进一步工作是更长 walk-forward、市场状态分组、置信区间与交易成本评价，而不是把已完成 OOS 重列为待执行。

```mermaid
flowchart LR
    T1[训练截止点1] --> O1[后续时间段OOS 1]
    O1 --> T2[训练窗口向前滚动<br/>截止点2]
    T2 --> O2[后续时间段OOS 2]
    O2 --> T3[训练窗口继续滚动<br/>截止点3]
    T3 --> O3[后续时间段OOS 3]
    O1 --> AGG[聚合各期独立结果]
    O2 --> AGG
    O3 --> AGG
    AGG --> METRIC[Mean/Median IC<br/>ICIR<br/>正IC比例<br/>Top-Bottom稳定性]
```

### 10.6 论文中的结论边界

在补充实验完成前，建议使用以下表述：“在当前短 OOS 窗口和给定数据快照下，A 股条件化全参数微调相较原始 Kronos-small 显示出方向准确率和横截面排序指标改善。”不应写成“模型已证明具有稳定 Alpha”或“改善必然来自行业/市值条件”。

## 11. 未完成实验与后续工作

### 11.1 Stage 3 五段稳定性观察与受控探针

第 6.7 节的单段 GPU 及完整 Output 验收已通过。当前下一步是保持原实验定义，从 Segment 1 的完整 State 接续 Segment 2–6，同时补测同口径 C2 best 与 Segment 1 全量基线，增加预测/目标方差观测。此处不是从 C2 重新训练，也不重置 optimizer、EMA、seed、step 或看板。

该短程任务已提交为 [Kronos Small 0 1 Stage3 Joint Path C2](https://www.kaggle.com/code/smmt315/kronos-small-0-1-stage3-joint-path-c2)，Version 1，源码固定为 `3595dc2381fd6c920cac9bad7514ebba46e490f0`，继续原 run `small_0.1_stage3_joint_path_alignment_from_c2_best_v2`。计划从 global step 313 接续至 1,878（每段 313 steps），硬超时 18,000 秒；五段实训/验证按首段估算约 55 分钟，另计双基线验证、安装和输出时间。结果尚待验收。

短程观察完成后，再判断是否进入 40–60 segments 的受控探针，并补充同设置 CE-only 对照。更长训练预算和 Stage 3 OOS 尚未执行，不因单段成功而自动批准。

### 11.2 其余后续实验

退火完成后，仍建议在相同起点、相同 seed、相同验证集和相同预算下做 cosine 与 constant 的公平比较；按 forecast horizon 分解验证 loss；比较初始底座与最终 checkpoint 的分层 relative weight drift；扩展 OOS 日期后再评估方向准确率、Rank IC、分组收益及统计显著性。

## 12. 可复核文件

- 模型定义：[model/kronos.py](/Users/fupengcheng/Documents/Kronos/model/kronos.py)、[model/module.py](/Users/fupengcheng/Documents/Kronos/model/module.py)
- 数据与窗口：[finetune/dataset.py](/Users/fupengcheng/Documents/Kronos/finetune/dataset.py)
- 训练与损失：[finetune/train_predictor.py](/Users/fupengcheng/Documents/Kronos/finetune/train_predictor.py)
- 配置：[finetune/config.py](/Users/fupengcheng/Documents/Kronos/finetune/config.py)
- Stage 3 条件候选与损失：[stage3_path_alignment.py](/Users/fupengcheng/Documents/Kronos/finetune/stage3_path_alignment.py)
- Stage 3 DDP 目标包装与训练：[stage3_training_model.py](/Users/fupengcheng/Documents/Kronos/finetune/stage3_training_model.py)、[train_stage3_path_alignment.py](/Users/fupengcheng/Documents/Kronos/finetune/train_stage3_path_alignment.py)
- Stage 3 验收报告：[条件联合梯度验收](/Users/fupengcheng/Documents/Kronos/finetune/STAGE3_CONDITIONAL_JOINT_VERIFICATION_CN.md)、[训练框架与因果复核](/Users/fupengcheng/Documents/Kronos/finetune/STAGE3_TRAINING_FRAMEWORK_VERIFICATION_CN.md)；前者的早期数值以后一份显式因果复核为补充
- Stage 3 自动测试：[test_stage3_conditional_joint.py](/Users/fupengcheng/Documents/Kronos/tests/test_stage3_conditional_joint.py)、[test_stage3_training_framework.py](/Users/fupengcheng/Documents/Kronos/tests/test_stage3_training_framework.py)
- Stage 3 GPU 预检与提交入口：[stage3_gpu_probe.py](/Users/fupengcheng/Documents/Kronos/finetune/stage3_gpu_probe.py)、[stage3_joint_path_smoke.py](/Users/fupengcheng/Documents/Kronos/finetune/kaggle_stage3_joint_path_smoke/stage3_joint_path_smoke.py)
- 训练日志目录：`small_train_log/`；WC first round 完整日志：[small_0.1_stage2_wc_last-2026-9-9_23_24_00.log](/Users/fupengcheng/Documents/Kronos/small_train_log/small_0.1_stage2_wc_last-2026-9-9_23_24_00.log)；WC dual T4 完整日志：[small_0.1_stage2_wc_dual_t4-2026-9-12_18_47_04.log](/Users/fupengcheng/Documents/Kronos/small_train_log/small_0.1_stage2_wc_dual_t4-2026-9-12_18_47_04.log)
- Stage 2 Main OOS 汇总：[summary.json](/Users/fupengcheng/Documents/Kronos/artifacts/kronos_small_0_1_stage2_oos_base_kaggle/kronos_small_0_1_stage2_oos/summary.json)
- Cosine C2 OOS 汇总：[summary.json](/Users/fupengcheng/Documents/Kronos/artifacts/kronos_small_0_1_c2_alpha_oos_output/kronos_small_0_1_stage2_oos/summary.json)
- Stage 2 最终统一原始价格 OOS 报告：[small_0_1_oos_raw_audit_20260915.json](/Users/fupengcheng/Documents/Kronos/finetune/reports/small_0_1_oos_raw_audit_20260915.json)
- Stage 3 已完成单段 Output：[summary.json](/Users/fupengcheng/Documents/Kronos/artifacts/stage3_joint_path_smoke_v1/stage3_joint_path_smoke/summary.json)、[gpu_probe.json](/Users/fupengcheng/Documents/Kronos/artifacts/stage3_joint_path_smoke_v1/stage3_joint_path_smoke/gpu_probe.json)
- Stage 3 单段验收归档（随仓库保存）：[stage3_joint_path_smoke_v1_acceptance.json](/Users/fupengcheng/Documents/Kronos/finetune/reports/stage3_joint_path_smoke_v1_acceptance.json)

本文以日志和当前代码为准；若历史计划文档与实测配置冲突，应优先引用本报告中的代码/日志事实，并在论文实验设置中注明具体 commit、seed、数据快照和 checkpoint 标识。
