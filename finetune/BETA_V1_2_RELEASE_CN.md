# A-share Full-Market Beta v1.2 发布说明

发布日期：2026-08-30。

## 发布内容

Beta v1.2 是从正式 Beta v1.1 `Best@818` 开始的全 Predictor 增量微调结果，同时发布：

- `Best@871`：Quick Validation 选出的全局 Best；固定 24k objective 为 `2.4165296555`，SHA-256 `965c3bc40f6ca17d2fbff16bcb7b1c15ab4ac5d47204d5714108b813462393dd`。
- `Last@1056`：完整两遍覆盖终点；固定 24k objective 为 `2.4168839455`，SHA-256 `7323e0cb76204fafb040b97524dd8ec69de101f33a99cff15d776543ea2aa216`。

稳定本地入口：

```text
models/a_share_v1_beta/releases/beta_v1.2/best_model
models/a_share_v1_beta/releases/beta_v1.2/last_model
models/a_share_v1_beta/releases/beta_v1.2/tokenizer
```

两个 checkpoint 都属于 Beta v1.2；评估和部署时必须写明 `Beta v1.2 Best@871` 或
`Beta v1.2 Last@1056`。Beta v1.1 的发布目录与权重保持不变。

## 训练配置

- 父权重：Beta v1.1 Best@818，仅加载模型权重。
- 全 Predictor 可训练：102,437,248 / 102,437,248 参数。
- 行业条件层和连续市值条件层保留，不重新初始化。
- fresh AdamW；Predictor 与条件分支统一 OneCycle，峰值学习率 `1e-6`。
- `basemodel_epochs=1`、`coverage_passes=2`、`batch_size=64`。
- 单卡 A800 GPU 0、BF16、seed 100，共 1,056 segments。

## 固定 24k 复评

使用 manifest SHA-256 `f329b1ca2a6f94c373403563d941d05c66de3754e7e3206c002cb5e475be1ac7`，
objective 为 `forecast_loss + 0.02 × history_loss`。

| checkpoint | Combined | 2025H2 | 2026H1 |
|---|---:|---:|---:|
| Beta v1.1 Best@818 | 2.447197 | 2.429536 | 2.458560 |
| Beta v1.2 Best@871 | **2.416530** | **2.403448** | **2.423210** |
| Beta v1.2 Last@1056 | 2.416884 | 2.404488 | 2.423992 |

Best@871 三项均略优于 Last@1056，因此作为默认候选；Last@1056 保留为完整训练终点和续训锚点。

## 使用边界

- Beta v1.2 尚未获得全新严格未来时间段的评估，不能沿用 Beta v1.1 的旧未来集成绩作为自身成绩。
- 固定 24k 验证集参与了训练期观察，只用于同口径 checkpoint 比较，不等同于封存未来评估。
- 本次发布不自动替换 Web、Modal 或其他生产服务。
- 本版本是研究候选，不构成实盘交易建议。
