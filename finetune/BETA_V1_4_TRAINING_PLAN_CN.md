# Beta v1.4 下一轮训练规划

## 状态

本文件记录当前 `1e-5` 两轮覆盖实验完成后的下一轮方案。它是训练规划，
不是 Beta v1.4 发布声明，也不改变正在 A800 上运行的实验。

## 核心目标

下一轮不再使用同一股票的相邻滑动窗口同时承担训练和验证。验证的核心问题改为：

> 从一组股票学到的当前市场周期规律，能否迁移到同期另一组股票？

历史周期共享是有意设计。项目需要持续增训以适应当前 Market Regime；本轮重点隔离
同一股票相邻窗口的高度重叠，而不是把当前市场周期整体排除在训练之外。

## 固定 80/20 股票拆分

- 以股票代码为不可分割分组，固定 80% 股票用于增训，另外 20% 用于验证。
- 验证股票在本轮增训区间内的所有窗口必须完全排除出训练候选池。
- 同一股票不得跨越训练组和验证组；禁止按 `(symbol, asof_date)` 单窗口随机拆分。
- 拆分必须按行业、市值分位和上市时间分层，使两组的截面分布尽量一致。
- 使用固定 seed，并保存训练股票、验证股票、源数据和生成参数的 SHA-256 manifest。
- 启动时必须校验两组股票交集为零，并输出各层分布与候选窗口数量审计。

Beta v1.3 基础权重历史上已经看过完整股票池，因此该设计检验的是“本轮增训更新的
跨股票迁移能力”，不是“模型从未见过该股票”。这与持续增训和现有股票池部署目标一致。

## 已生成的数据集

2026-08-31 已使用 `finetune/build_symbol_holdout_dataset.py` 生成 80/20 物理隔离数据：

`data/a_share_full_market_v1_beta_symbol_holdout_80_20_v1/`

- 总股票数：5,198。
- 训练股票：4,158；验证股票：1,040；交集：0。
- 训练候选窗口：8,443,283；验证候选窗口：2,120,789。
- 分层字段：行业、2026-06-30 市值十分位、源 panel 首次可用年份桶。
- 数据 manifest SHA-256：
  `f1d00238c3168de64927f011098fc65e49e5ce02120d1512d3e711f7848365ec`。
- 训练 panel SHA-256：
  `cda8ca052e37023602733076c638565776dcf737bb633027e104a401bb04b0a3`。
- 验证 panel SHA-256：
  `5699baca43d0fb0a43b66023f93968c79524f857094d521f8a56a4ee6e38dffa`。

验证股票 panel 上另行固定抽取了 24,000 个 2025H2–2026H1 tuning 窗口：

`data/a_share_full_market_v1_beta_symbol_holdout_80_20_v1/natural_validation_2025h2_2026h1_symbol_holdout_v1/`

- 固定验证 manifest SHA-256：
  `2f812ddfaa47cb2e171bb5ea6136881a4a824c064b825a7bc3d3ccda38a6e51e`。
- 24,000 个窗口覆盖 1,029 只验证股票。
- 属于训练股票的样本：0。
- 在训练候选池中的精确窗口：0；全部 24,000 个样本因股票隔离而天然不可训练。
- 该 24k 仍是 tuning set，不是 sealed future。
- `QlibDataset` 实际加载核验通过：训练候选 8,443,283，固定验证 24,000，
  2025H2/2026H1 分别为 12,496/11,504，输入窗口形状为 131x6。
- 数据目录内的 `SHA256SUMS` 覆盖 7 个关键文件并已全部校验通过。

先前生成的 `a_share_full_market_v1_beta_symbol_holdout_50_50_v1` 保留用于审计和
可复现性，但已被本 80/20 方案取代，不再作为下一轮默认输入。

下一轮 A800 启动时必须使用以下路径与哈希契约：

```bash
export KRONOS_TRAIN_DATA_PATHS="$DATA_ROOT/processed_datasets/train_data.pkl"
export KRONOS_VAL_DATA_PATHS="$DATA_ROOT/processed_datasets/val_data.pkl"
export KRONOS_DATA_MANIFEST_SHA256=f1d00238c3168de64927f011098fc65e49e5ce02120d1512d3e711f7848365ec
export KRONOS_FIXED_VALIDATION_MANIFEST_PATH="$DATA_ROOT/natural_validation_2025h2_2026h1_symbol_holdout_v1/natural_validation_manifest.json"
export KRONOS_FIXED_VALIDATION_MANIFEST_SHA256=2f812ddfaa47cb2e171bb5ea6136881a4a824c064b825a7bc3d3ccda38a6e51e
export KRONOS_EXCLUDE_FIXED_VALIDATION_FROM_TRAINING=1
```

其中 `DATA_ROOT` 指向新的 symbol-holdout 数据根目录，不得混用原 full-market panel。

## 验证与发布边界

- 当前固定 24k 验证集降级为 tuning 指标，只用于观察训练过程和产生候选节点。
- 20% 验证股票集用于本轮 Best 选择，指标至少包括 objective loss、forecast loss、
  月度 Rank IC/ICIR、分组单调性、换手率和含成本 PnL。
- 共享日历周期不视为本轮污染；同股票窗口进入两侧才视为拆分违规。
- 用于模型选择的数据一经查看，不得继续称为最终 sealed test。
- Beta v1.4 发布前仍需在未参与调参和候选选择的独立数据上做最终复评。

## 执行顺序

1. 让当前 Beta v1.3 Best@343 派生的 `1e-5` 两轮覆盖实验完整退火结束。
2. 冻结当前实验的 Best、第一轮结束点和最终点，整理学习率与验证走势。
3. 使用并复核已生成的分层固定 80/20 股票 manifest，不复用当前 24k 的窗口拆分语义。
4. 依据当前实验结果确定下一轮学习率，不在看到股票验证结果后追加候选配置。
5. 完成股票验证和独立最终复评后，才决定是否定版 Beta v1.4。

## 明确不做

- 不修改或重启当前正在运行的 1056 Segment 两轮训练。
- 不把当前固定 24k 的下降直接解释为严格样本外提升。
- 不用股票验证集反复试探学习率后仍将其宣传为 sealed future。
