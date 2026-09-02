# Beta v2.1 训练契约

Beta v2.1 从 **Beta v2.0 Best@687** 初始化，保留原有 10 日 OHLCVA
自回归路径头，并从第 120 个历史 token 的因果隐藏态增加两个辅助头：4 维
收益头与 TP/SL/未触发三分类头。旧 checkpoint 不含辅助头参数时允许缺失并按
固定 seed 初始化；旧训练入口默认不启用这些头。

## 标签与目标

- 信号日收盘后决策，D1 开盘价作为入场价。
- D1/D3/D5/D10 收益除以 `max(sigma20, 0.005) * sqrt(h)`，截断到
  `[-3, 3]`。
- TP/SL 为 `+5%/-3%`；同日双触发在训练中屏蔽，回测中按 SL 处理。
- Return loss 为 `0.80 * horizon Huber + 0.20 * return bias`。bias 优先按
  同一信号日的截面均值计算。
- ranking score 由收益头与 barrier 概率推导，不增加独立 ranking head。
- 总损失权重为 path/history/return/barrier/rank =
  `0.68/0.02/0.15/0.10/0.05`；训练损失用 detached EMA 无量纲化，辅助项在
  1,000 step 内线性升权。

## 验证与定版

首次启动先用未训练辅助头的 Best@687 在固定 full-only 验证集上校准一次五项
分母，并写入 `beta_v21_validation_denominators.json`。后续 Segment 与 resume
必须复用该文件，禁止按 checkpoint 重新归一化。Best 使用固定分母的
`beta_v21_score`。

每次验证另取固定 2,048 个窗口做 deterministic greedy 10 步生成，逐周期记录：

- 辅助收益头与生成路径收益的 MAE、符号一致率和相关性；
- 两者的均值、标准差及相对真实收益的 bias；
- token forecast loss、return、return bias、barrier 与 ranking loss。

启动脚本：

```bash
finetune/run_a800_beta_v2_1_best687_decision_heads_twopass.sh start
```
