# Cosine 退火没有发生：调度表长度被 520 段撑满

发现时间：2026-11-09。第一轮 `base_clean_0.1_cosine`（kernel `user281434/kronos-base-clean-0-1-cosine`）
跑完 88 段后 forecast 只从 2.24355 动到 2.24236（−0.0012），基本等于没训。原因不是模型到顶了，
是**学习率压根没退下来**。

## 证据

第一轮末尾（第 88 段）打印的 LR：

```text
Adaptation LR 9.3805320592e-06
```

按 cosine 从 `1e-5` 退到 `1e-6` 反推表长：

| 表长 | 第 88 段的理论 LR |
|---|---|
| 90 段 | 1.0110e-06 |
| **520 段** | **9.3789e-06** ← 实测 9.3805e-06 |

80 段下去 LR 只降了 6%，等于全程都在峰值附近跑。所以那 88 段实质是
「WC 用新 seed 在 1e-5 上继续训 88 段」，不是退火精调。

## 根因

`finetune/train_predictor.py`：

```python
minimum_coverage_segments = segments_per_coverage * coverage_passes      # 520
required_segments = minimum_coverage_segments + patience if require_full_coverage else 0
effective_epochs = max(int(config['epochs']), required_segments)         # 520
...
scheduler_steps = sum(ceil(...) for segment in range(effective_epochs))  # 520 段的步数
```

`KRONOS_MAX_SEGMENTS_PER_RUN` 只是**单次调用的停车位**，trainer 自己会打印
`"the global schedule is unchanged"`。调度表长度由 `effective_epochs` 决定，而
`require_full_coverage=1` + `epochs=1` 把它钉在 520。所以「跑 90 段」只走了余弦表的前 17%。

## 修复

`push_main.py` 原来只注入固定的一批 `KRONOS_*`，`KRONOS_EPOCHS` 与
`KRONOS_REQUIRE_FULL_COVERAGE` 根本到不了 kernel（`launch_main.py` 用
`env=os.environ.copy()` 传给 runner，所以只要 kernel 里有就能透传）。
已改为：这两个变量存在时才注入。

重跑参数：`KRONOS_EPOCHS=90` + `KRONOS_REQUIRE_FULL_COVERAGE=0`
→ `effective_epochs = max(90, 0) = 90`，余弦表就是 90 段。

重跑 kernel `user281434/kronos-base-clean-0-1-cosine-a90` 启动日志：

```text
[Rank 0, Segment 1/90, Step 200/313] Adaptation LR 9.9988806896e-06
```

段数显示从 `Segment 22/520` 变成了 `Segment 1/90`，表长确认生效。

## 对照关系

两次跑除学习率调度外完全同构，可当作受控 A/B：

| 项 | cosine（已跑完） | cosine-a90（重跑） |
|---|---|---|
| 父权重 | WC last@520 | WC last@520 |
| coverage seed | 20260912 | 20260912 |
| 段数 | 88 / 90（session 到点） | 90 |
| 调度表长 | **520** | **90** |
| 第 88 段 LR | 9.38e-06 | 1.01e-06（预期） |

## 遗留

- Small 的 Cosine 是 267 段、覆盖一轮的 50%，退火表长也是 267。本轮 a90 只覆盖 17%。
  若 a90 证明退火有效，再考虑拉长到 267 段（约 34 小时 GPU）才是与 Small 同比例。
- 已跑完的那次 cosine 不是废数据：它是「WC 在峰值 LR 上多训 88 段」的干净对照，
  可以用来回答「多训有没有用」。
