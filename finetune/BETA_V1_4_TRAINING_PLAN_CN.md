# Beta v1.4 下一轮训练规划

## 状态

本文件记录当前 `1e-5` 两轮覆盖实验完成后的下一轮方案。它是训练规划，
不是 Beta v1.4 发布声明，也不改变正在 A800 上运行的实验。

## 核心目标

下一轮不再使用同一股票的相邻滑动窗口同时承担训练和验证。验证的核心问题改为：

> 从一组股票学到的当前市场周期规律，能否迁移到同期另一组股票？

历史周期共享是有意设计。项目需要持续增训以适应当前 Market Regime；本轮重点隔离
同一股票相邻窗口的高度重叠，而不是把当前市场周期整体排除在训练之外。

## 固定 90/10 股票拆分

- 以股票代码为不可分割分组，固定 90% 股票用于增训，另外 10% 用于验证。
- 验证股票在本轮增训区间内的所有窗口必须完全排除出训练候选池。
- 同一股票不得跨越训练组和验证组；禁止按 `(symbol, asof_date)` 单窗口随机拆分。
- 先按行业分配验证名额，股票数不少于5的行业至少保留1只验证股；再在行业内按
  市值分位和上市时间分层。更小的行业按行业代码大类合并分配。
- 使用固定 seed，并保存训练股票、验证股票、源数据和生成参数的 SHA-256 manifest。
- 启动时必须校验两组股票交集为零，并输出各层分布与候选窗口数量审计。

Beta v1.3 基础权重历史上已经看过完整股票池，因此该设计检验的是“本轮增训更新的
跨股票迁移能力”，不是“模型从未见过该股票”。这与持续增训和现有股票池部署目标一致。

## 已生成的数据集

2026-08-31 已使用 `finetune/build_symbol_holdout_dataset.py` 生成 90/10 物理隔离数据：

`data/a_share_full_market_v1_beta_symbol_holdout_90_10_v1/`

- 总股票数：5,198。
- 训练股票：4,678；验证股票：520；交集：0。
- 训练候选窗口：9,457,646；验证候选窗口：1,106,426。
- 分层字段：行业、2026-06-30 市值十分位、源 panel 首次可用年份桶。
- 验证集覆盖全部72个至少有5只股票的行业；缺失必保行业：0。
- 数据 manifest SHA-256：
  `17afbeede658c13787043e601aa355717dda4d11719b51fb3ce368fb138e627a`。
- 训练 panel SHA-256：
  `b2f2a861f651321efd38761c65ffcff4d14290cd580325298c4b9a7bc915f832`。
- 验证 panel SHA-256：
  `748a9205714ee9714525872e65432063edd26d6afbef05391919e8d9d8811115`。

验证股票 panel 上固定纳入2025H2–2026H1全部123,982个同期窗口：

`data/a_share_full_market_v1_beta_symbol_holdout_90_10_v1/natural_validation_2025h2_2026h1_symbol_holdout_full_v1/`

- 固定验证 manifest SHA-256：
  `ea29ecdb318adf9789ddd47eb4c5d3df7cdadcbbd60471305241a469d357d184`。
- 123,982个窗口覆盖516只同期有足够历史数据的验证股票。
- 2025H2/2026H1分别为64,441/59,541个窗口。
- 24,000个分层样本仅保留为可选看板前缀；Best只认123,982全量验证。
- 属于训练股票的样本：0。
- 在训练候选池中的精确窗口：0；全部123,982个样本因股票隔离而天然不可训练。
- 该全量股票验证集仍是 tuning set，不是 sealed future。
- `QlibDataset` 实际加载核验通过：训练候选9,457,646，固定验证
  123,982/123,982全量映射，输入窗口形状为131x6。
- 数据目录内的 `SHA256SUMS` 覆盖 7 个关键文件并已全部校验通过。

先前生成的50/50和80/20数据目录继续保留用于审计和可复现性，但已被本90/10方案
取代，不再作为下一轮默认输入。

下一轮 A800 启动时必须使用以下路径与哈希契约：

```bash
export KRONOS_TRAIN_DATA_PATHS="$DATA_ROOT/processed_datasets/train_data.pkl"
export KRONOS_VAL_DATA_PATHS="$DATA_ROOT/processed_datasets/val_data.pkl"
export KRONOS_DATA_MANIFEST_SHA256=17afbeede658c13787043e601aa355717dda4d11719b51fb3ce368fb138e627a
export KRONOS_FIXED_VALIDATION_MANIFEST_PATH="$DATA_ROOT/natural_validation_2025h2_2026h1_symbol_holdout_full_v1/natural_validation_manifest.json"
export KRONOS_FIXED_VALIDATION_MANIFEST_SHA256=ea29ecdb318adf9789ddd47eb4c5d3df7cdadcbbd60471305241a469d357d184
export KRONOS_EXCLUDE_FIXED_VALIDATION_FROM_TRAINING=1
export KRONOS_VALIDATION_FULL_ONLY=1
export KRONOS_VALIDATION_QUICK_SAMPLES=24000
export KRONOS_VALIDATION_LARGE_SAMPLES=123982
```

其中 `DATA_ROOT` 指向新的 symbol-holdout 数据根目录，不得混用原 full-market panel。

## 验证与发布边界

- 24k前缀仅用于可选看板监控，不得参与Best选择。
- 10%验证股票的123,982个同期全量窗口用于本轮Best选择，指标至少包括
  objective loss、forecast loss、
  月度 Rank IC/ICIR、分组单调性、换手率和含成本 PnL。
- 共享日历周期不视为本轮污染；同股票窗口进入两侧才视为拆分违规。
- 用于模型选择的数据一经查看，不得继续称为最终 sealed test。
- Beta v1.4 发布前仍需在未参与调参和候选选择的独立数据上做最终复评。

## 执行顺序

1. 让当前 Beta v1.3 Best@343 派生的 `1e-5` 两轮覆盖实验完整退火结束。
2. 冻结当前实验的 Best、第一轮结束点和最终点，整理学习率与验证走势。
3. 使用并复核已生成的分层固定90/10股票 manifest，Best只认同期全量验证。
4. 依据当前实验结果确定下一轮学习率，不在看到股票验证结果后追加候选配置。
5. 完成股票验证和独立最终复评后，才决定是否定版 Beta v1.4。

## 明确不做

- 不修改或重启当前正在运行的 1056 Segment 两轮训练。
- 不把当前固定 24k 的下降直接解释为严格样本外提升。
- 不用股票验证集反复试探学习率后仍将其宣传为 sealed future。
