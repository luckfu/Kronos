# Beta v1.2 Best@343 全 Predictor 单遍续训结果

## 结论

本轮从正式 Beta v1.2 Best@871 仅加载权重，以 fresh optimizer、OneCycle、
Predictor/condition 峰值学习率 `1e-6` 完成一次全市场覆盖。训练于
2026-08-30 完成 528/528 Segment，最佳固定 24k 验证结果出现在 Segment 343。

| 模型 | Combined | 2025H2 | 2026H1 |
|---|---:|---:|---:|
| 正式 Beta v1.2 Best@871 | 2.4165296555 | 2.4034476280 | 2.4232099056 |
| 本轮 Best@343 | **2.4104652405** | **2.3991272449** | **2.4157335758** |
| 本轮 Last@528 | 2.4107725620 | **2.3988070488** | 2.4164288044 |

Best@343 的 Combined 相对正式基线下降 `0.0060644150`，约 `0.251%`。
Last@528 没有超过 Best@343，但仍显著优于正式基线。

## 完整性

- 状态：`completed`，528/528 Segment。
- 训练指标记录：1,649 条。
- `validation_large`：528 条，每条固定 24,000 样本。
- `validation/quick`：0 条。
- NaN/Inf：0 条。
- Best：Segment 343，`2.4104652404785156`。
- Last：Segment 528，`2.4107725620269775`。
- 固定验证 manifest SHA-256：
  `f329b1ca2a6f94c373403563d941d05c66de3754e7e3206c002cb5e475be1ac7`。

## 独立复评

训练进程退出后，使用 `evaluate_fixed_validation_full.py` 对 Best 和 Last 分别
重新加载权重并执行一次独立 CUDA 前向评估。评估器只构造完整 24k loader，
不运行也不写入 quick 结果。复评结果与训练时记录逐项完全一致。

- Best 模型 SHA-256：
  `b2e90710d7619f3ae3a1b488726b2885adff11dcd45834f2560766f49bbc20f3`
- Last 模型 SHA-256：
  `fde230a9a370c495c00063cf31bae7036e77a7efa21ea2e3f869f9ca887c4e82`
- Resume state SHA-256：
  `4bf824bcc9150b5171afb8daa8e47de0b114e13e15c0b2dec3f6f9a5e871aeba`

## 本地归档

完整归档位于：

`models/a_share_v1_beta/beta_v1_2_best871_full_every_segment_onecycle_1pass_seed100/`

该目录包含 Best、Last、`last_state.pt`、`metrics.jsonl`、`launcher.log`、
`progress.json`、`summary.json`、独立复评、实验 manifest、A800 实际源码快照
以及覆盖全部 29 个文件的 `SHA256SUMS`。该目录由 Git 忽略，应通过 U 盘传输。

## 下一轮

下一轮从本轮 Best@343 仅加载权重，Predictor 和 condition 均使用峰值
`1e-5` 的 OneCycle，先跑 50 Segment 激进诊断。保持全 Predictor、BF16、
batch size 64、行业层和连续市值层不重置，并继续执行每 Segment 固定 24k
full-only 验证。

## 血缘

- Handoff lineage：`lin-d2a39f2905ecfe8f`
- Parent handoff：
  `20260830-175409-codex-kronos-local-migration-and-beta-v1.2-tra-01a04158-9ea1-7921-b9c2-`
- Git 基线：`450f913dcaa52fda65ba0e2e749c55c265d20310`

