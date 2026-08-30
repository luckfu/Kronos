# A-share Full-Market Beta v1.1 发布说明

发布日期：2026-08-29。

## 发布内容

Beta v1.1 同时发布 `Best@818` 和 `Last@1058`。两者来自 V6 Segment 568 权重重新初始化行业层
和连续市值条件层后的同一次 BF16 双学习率两轮训练：

- `Best@818`：Natural Validation 最优点，SHA-256 `b890771368737c6c93825165695afc16b57870f4692f87a563392cc96e405673`。
- `Last@1058`：完整两轮训练终点和恢复锚点，SHA-256 `59ac7999625e247e2211738ec42da4759c8fdb1991d683ec2de67e40f2a94bcc`。

稳定本地入口：

```text
models/a_share_v1_beta/releases/beta_v1.1/best_model
models/a_share_v1_beta/releases/beta_v1.1/last_model
models/a_share_v1_beta/releases/beta_v1.1/tokenizer
```

别名只指向不可变训练目录，不复制或改名原始权重，因此训练 segment、optimizer 终点和评估结果
仍可完整追溯。两个 checkpoint 都是 Beta v1.1 的正式产出物；报告结果时必须写成
`Beta v1.1 Best@818` 或 `Beta v1.1 Last@1058`，不能只写 `Beta v1.1`。

## 选择依据

- 固定 24k Natural Validation objective：Best@818 为 2.447197，Last@1058 为 2.448726。
- 严格未来多头 Top 20%：Best@818 为 55.12%，自然基准为 46.83%。
- 严格未来多头 Rank IC：Best@818 为 0.0844，Last@1058 为 0.0736。
- Best 与 Last 的未来多头命中差只有 0.37 个百分点，尚不能证明两者存在显著差异，因此两者
  共同发布。具体应用可以优先试用 Best，但必须保留 Last 作稳定终点对照。

## 使用边界

- Beta v1.1 可进入多头候选研究和后续封存评估，不代表可直接实盘。
- 空头 Top 20% 命中为 14.19%，低于 16.70% 自然基准，空头分数禁止用于选股。
- 严格未来集只有 6 个信号日；后续必须在全新日期上按原口径追加评估，不得反复使用现有未来集调参。
- 本次只确定模型版本，没有自动替换现有 Web、Modal 或其他生产服务。
