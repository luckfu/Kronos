# A-share Full-Market Beta v1.3 发布说明

发布日期：2026-08-30。

## 发布内容

Beta v1.3 从正式 Beta v1.2 `Best@871` 继续训练，同时发布：

- `Best@343`：本轮固定 24k full-only 验证全局 Best，objective 为 `2.4104652405`，SHA-256 `b2e90710d7619f3ae3a1b488726b2885adff11dcd45834f2560766f49bbc20f3`。
- `Last@528`：完整单遍覆盖终点，objective 为 `2.4107725620`，SHA-256 `fde230a9a370c495c00063cf31bae7036e77a7efa21ea2e3f869f9ca887c4e82`。
- `last_state.pt`：完整 optimizer/scheduler/RNG 恢复状态，SHA-256 `4bf824bcc9150b5171afb8daa8e47de0b114e13e15c0b2dec3f6f9a5e871aeba`。

稳定本地入口：

```text
models/a_share_v1_beta/releases/beta_v1.3/best_model
models/a_share_v1_beta/releases/beta_v1.3/last_model
models/a_share_v1_beta/releases/beta_v1.3/last_state.pt
models/a_share_v1_beta/releases/beta_v1.3/tokenizer
```

`models/a_share_v1_beta/current` 指向该版本。评估和部署必须写明
`Beta v1.3 Best@343` 或 `Beta v1.3 Last@528`。

## 训练配置

- 父权重：Beta v1.2 Best@871，仅加载模型权重。
- 全 Predictor：102,437,248 / 102,437,248 参数可训练。
- 行业层和连续市值层保留，不重新初始化。
- fresh AdamW；Predictor 与 condition 统一 OneCycle，峰值学习率 `1e-6`。
- 单次全市场覆盖，528 Segment，每段 20,000 个训练窗口。
- batch size 64、BF16、seed 100、单卡 A800 GPU 0。
- 每个 Segment 只执行固定 24k full-only 验证；不运行 quick。

## 固定 24k 独立复评

manifest SHA-256：`f329b1ca2a6f94c373403563d941d05c66de3754e7e3206c002cb5e475be1ac7`。

| checkpoint | Combined | 2025H2 | 2026H1 |
|---|---:|---:|---:|
| Beta v1.2 Best@871 | 2.416530 | 2.403448 | 2.423210 |
| Beta v1.3 Best@343 | **2.410465** | 2.399127 | **2.415734** |
| Beta v1.3 Last@528 | 2.410773 | **2.398807** | 2.416429 |

Best@343 相对 Beta v1.2 Combined 改善约 `0.251%`，作为默认模型。训练结束后独立重新加载
Best 与 Last 权重所得结果与训练记录完全一致。

## 使用边界

- 固定 24k 验证集参与了 checkpoint 观察，不是全新封存未来评估。
- Beta v1.3 不能继承 Beta v1.2 在旧未来时间段上的指标。
- 当前 `1e-5` 激进诊断是 Beta v1.3 的子实验，不属于本次发布。
- 本次发布不自动替换 Web、Modal 或其他生产服务。
- 本版本不构成实盘交易建议。
