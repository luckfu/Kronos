# Beta v2.1 训练契约

Beta v2.1 从 **Beta v2.0 Best@687** 初始化，保留原有 10 日 OHLCVA
自回归路径头，并从第 120 个历史 token 的因果隐藏态增加两个辅助头：4 维
收益头与 TP/SL/未触发三分类头。旧 checkpoint 不含辅助头参数时允许缺失并按
固定 seed 初始化；旧训练入口默认不启用这些头。

直接父权重是 Beta v2.0 `best_model`，不是 `last_model`。其模型文件 SHA-256 为：

```text
e603fe3178d61ee7feb8a5b0ad520d13166d533785f12d0a4f51d85db0a91ed3
```

## 本轮运行合同

- 数据拆分：symbol-holdout 90/10，训练 4,678 只，验证 520 只；
- 训练窗口：9,457,646，每个 Segment 无放回取 20,000 个窗口；
- 覆盖计划：每遍 473 个 Segment，共两遍、946 个 Segment；
- 验证：每个 Segment 只运行 123,982 条 full-only 验证，禁止 quick validation；
- 模型：完整 Predictor 解冻，BF16，单卡 GPU 0，batch size 64；
- 调度器：OneCycle，统一峰值学习率 `1e-5`；
- 10 日 Path 权重：
  `1.364/1.364/1.364/1.136/1.136/0.909/0.909/0.682/0.682/0.455`。

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

### 反向传播边界

Beta v2.1 不是只增加验证指标。下列目标组成实际训练 Loss，并通过第 120 个
历史 token 的隐藏态反向更新辅助头和完整 Predictor Backbone：

| 项目 | 是否反向传播 | 参数作用 |
| --- | --- | --- |
| 加权 10 日 Path token loss | 是 | 更新原路径头与 Backbone，前 3-5 日权重更高 |
| History token loss | 是 | 以低权重保留历史重建锚点 |
| Return Huber loss | 是 | 更新 4 维收益头与 Backbone |
| Same-date Return bias loss | 是 | 抑制同日截面整体偏多或偏空 |
| Barrier class-balanced loss | 是 | 更新三分类头与 Backbone |
| Same-date Ranking loss | 是 | 通过收益头和 Barrier 概率推导的 Utility 分数更新两头与 Backbone |

Return 与总训练目标为：

```text
return_loss = 0.80 * return_huber + 0.20 * return_bias

ramp = min(1, global_step / 1000)
base = 0.68 * norm(path) + 0.02 * norm(history)
aux  = 0.15 * norm(return) + 0.10 * norm(barrier) + 0.05 * norm(ranking)
training_loss = (base + ramp * aux) / (0.70 + 0.30 * ramp)
```

`norm(loss)` 使用 detached EMA 作为分母。EMA 值不进入计算图，但归一化后的
Loss 仍保留梯度。Return bias 按同一信号日的截面平均误差计算；Ranking 只比较
同日且 Utility 差至少 `0.5%` 的样本对。

## 验证与定版

首次启动先用未训练辅助头的 Best@687 在固定 full-only 验证集上校准一次五项
分母，并写入 `beta_v21_validation_denominators.json`。后续 Segment 与 resume
必须复用该文件，禁止按 checkpoint 重新归一化。Best 使用固定分母的
`beta_v21_score`。

`beta_v21_score` 越低越好，公式为：

```text
beta_v21_score =
    0.50 * weighted_forecast_loss / fixed_path_denominator
  + 0.20 * return_loss            / fixed_return_denominator
  + 0.20 * barrier_loss           / fixed_barrier_denominator
  + 0.10 * ranking_loss           / fixed_ranking_denominator
```

History 分母会随校准文件一起固化，以保持完整验证合同，但当前
`beta_v21_score` 不包含 History 项。某个 Segment 的 score 低于历史最低值时，
训练器覆盖 `best_model`。

本轮校准后锁定的分母为：

```text
path    = 2.3113956451416016
history = 2.509082317352295
return  = 0.6631228923797607
barrier = 2.0452120304107666
ranking = 0.6932359337806702
```

每次验证另取固定 2,048 个窗口做 deterministic greedy 10 步生成，逐周期记录：

- 辅助收益头与生成路径收益的 MAE、符号一致率和相关性；
- 两者的均值、标准差及相对真实收益的 bias；
- token forecast loss、return、return bias、barrier 与 ranking loss。

### 纯观察项与终审边界

以下项目在 `torch.no_grad()` 验证阶段运行，不参与反向传播：

| 项目 | 用途 |
| --- | --- |
| 2,048 窗口 deterministic 10 步生成 | 取得实际生成路径，不改变参数 |
| D1/D3/D5/D10 MAE | 检查路径头与收益头的数值差距 |
| Sign agreement | 检查两套预测的涨跌方向一致率 |
| Correlation | 检查两套预测的截面协同变化 |
| Generated/Auxiliary bias | 分别检查生成路径和收益头是否系统性偏多或偏空 |
| `beta_v21_score` | 只负责 checkpoint 排序，不产生梯度 |

当前实现按照 `beta_v21_score` 保存 Best，不会在保存 checkpoint 时自动执行其他
一票否决。候选模型最终晋级仍必须同时满足：

- Path forecast loss 相对 Beta v2.0 基线的恶化不超过 1%；
- 生成路径和辅助收益头没有重新出现明显偏空漂移；
- 两套预测的一致性没有持续背离；
- 通过未参与调参的 Sealed Future Set，并报告 Rank IC、方向指标和含成本 Top5
  回测；token loss 或 `beta_v21_score` 单独下降不能证明可盈利。

### SwanLab 看板

本地 relay 每 10 秒通过 `ssh A800` 增量上传 `metrics.jsonl`。主要指标路径为：

```text
progress/completion_percent
validation/full/beta_v21_score
validation/full/weighted_forecast_loss
validation/full/return_loss
validation/full/return_bias_loss
validation/full/barrier_loss
validation/full/ranking_loss
validation/full/return_path_consistency/{mae,sign_agreement,correlation}/{D1,D3,D5,D10}
```

启动 relay：

```bash
finetune/run_a800_beta_v2_1_swanlab_relay.sh
```

固定 SwanLab run id：

```text
kronos-beta-v2-1-best687-decision-heads-twopass-a800
```

启动脚本：

```bash
finetune/run_a800_beta_v2_1_best687_decision_heads_twopass.sh start
```
