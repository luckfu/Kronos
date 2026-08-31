# Beta v1.3.2 Clean-V1.1 训练计划

## 父权重选择

现有 Beta v1.1、v1.2、v1.3 和 v1.3.1 都曾使用完整股票池训练，因此没有一个
正式版本能称为严格干净。选择 **Beta v1.1 Best@818**，因为它是现有正式版本中
最早、后续全市场增训层数最少的 checkpoint；Beta v1.2、v1.3 和 v1.3.1 都是在
它之上继续叠加训练。

- 父权重 SHA-256：`b890771368737c6c93825165695afc16b57870f4692f87a563392cc96e405673`。
- 完整继承该 checkpoint 中已经训练过的 `sector_emb` 和连续市值 `size_mlp`
  权重；不重新初始化行业层或市值层。
- 结论边界：该线路比 Beta v1.3.1 父线相对干净，但仍不能称为股票从未见过。
- 真正严格干净需要回到本项目全市场增训之前的基础模型重新训练。

## 训练与数据

- 使用固定 `a_share_full_market_v1_beta_symbol_holdout_90_10_v1`。
- 4,678 只训练股票、520 只验证股票、交集为 0。
- 全 Predictor、BF16、batch size 64、OneCycle。
- Predictor 和 conditioning 使用统一峰值学习率 `1e-5`；不再走新增层双学习率
  初始化阶段。
- `KRONOS_RESET_SECTOR_EMBEDDING=0`、`KRONOS_RESET_SIZE_EMBEDDING=0`。
- 两遍覆盖，共 946 Segment；每段运行 123,982 个 full-only 股票隔离验证窗口。
- 启动训练前先记录 Beta v1.1 Best@818 在同一验证集上的零训练基线。

当前 Beta v1.3.1 父线只保留为历史污染/遗忘诊断，不与本线路混用 checkpoint、
optimizer、SwanLab run 或 Best 结论。
