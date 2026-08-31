# Beta v2.0 训练与定版计划

## 版本定义

- 正式发布目标为 **Beta v2.0**。
- `beta_v1_3_2_clean_v11_best818_symbol_holdout_90_10_aggressive_1e5_twopass`
  仅作为正在运行的内部实验 ID，保留其远端目录、tmux、SwanLab run 和 resume
  血缘，不作为最终版本名。
- Beta v1.2、v1.3 和 v1.3.1 保留为旧验证体系下的历史实验，不再作为可信泛化
  版本继续晋级，也不删除或覆盖其权重和指标。
- Beta v2.0 是首个以固定股票隔离验证作为主要定版依据的正式 Beta 版本。

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

## 版本血缘语义

- Beta v1.1 Best@818 作为 bootstrap 底座：提供已经完成初始化和初步训练的
  `sector_emb` 与连续市值 `size_mlp`，避免 Beta v2.0 再从随机新增层和双学习率
  初始化阶段开始。
- Beta v2.0 是新验证体系下第一版正式全量增训：全 Predictor 参数与 conditioning
  参数共同训练，使用相同 OneCycle 学习率调度和相同 `1e-5` 峰值。
- 此处“全量增训”指模型参数全量解冻，不表示使用 100% 股票训练；520 只验证股票
  始终隔离，只有 4,678 只训练股票参与梯度更新。

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

## 定版规则

- 训练结束后以 123,982 窗口 Combined objective loss 最低的 Best checkpoint
  作为 Beta v2.0 候选，不直接以 Last 定版。
- 对候选 Best 独立重跑同一 full-only 股票隔离验证，并记录 Combined、2025H2、
  2026H1、模型 SHA、父权重和数据 manifest。
- 最终 release 目录、模型卡、评估报告和发布 manifest 统一使用 `beta-v2.0`；内部
  实验路径只记录在 lineage 中。
