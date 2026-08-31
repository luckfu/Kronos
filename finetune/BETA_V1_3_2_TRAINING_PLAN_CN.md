# Beta v1.3.2 训练计划

## 血缘

- 父版本：Beta v1.3.1 Best@895。
- 父权重 SHA-256：`ad6f2ffc84536795a5c88e0dfa74d5486aa720a2296bce4f669ff5e99bc8ed6a`。
- Beta v1.3.1 来自 Beta v1.3 Best@343 的全 Predictor `1e-5` 两遍覆盖实验；
  由用户在完成 Segment 953 后停止，Best 出现在 Segment 895。

## 数据边界

- 数据集：`a_share_full_market_v1_beta_symbol_holdout_90_10_v1`。
- 4,678 只训练股票，520 只验证股票，股票交集为 0。
- 训练候选窗口 9,457,646；两遍覆盖共 946 Segment。
- Best 只认验证股票在 2025H2 至 2026H1 的 123,982 个固定全量窗口。
- 该验证集用于调参与选 Best，不是 sealed future test。

## 训练配置

- 全 Predictor，BF16，batch size 64，seed 100。
- OneCycle；Predictor 和 condition 峰值学习率均为 `1e-5`。
- fresh optimizer；从 Beta v1.3.1 Best@895 只加载模型权重。
- 两遍覆盖；行业层与连续市值层不重置。
- full-only 验证，不运行或记录 quick validation。

启动前先对父权重运行同一 123,982 窗口的零训练基线评估。训练中的 loss 只表示
本轮更新向未参与增训的股票迁移得如何；最终发布仍需新的未来时间数据复评。

执行入口：

```bash
bash finetune/run_a800_beta_v1_3_2_symbol_holdout_baseline.sh
bash finetune/run_a800_beta_v1_3_2_symbol_holdout_90_10_aggressive_1e5_twopass.sh start
```
