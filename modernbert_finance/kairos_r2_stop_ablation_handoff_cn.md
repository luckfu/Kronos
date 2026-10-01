# Kairos R2 停训交接 + 便宜消融计划

日期：2026-10-01（北京时间）。

## 决策

停止长训。证据显示 R2（lr=3e-5，global permutation）验证 macro log loss
在常数基线附近徘徊（约 `0.5966`），继续堆 GPU 小时不划算。
下一步改为**短消融 / 管线 sanity**，不再延长本轮 Kaggle 训练。

## 停训对象

| 项 | 值 |
| --- | --- |
| Kaggle kernel | `user281434/kairos-r2-r1-restart-lr-3e-5` |
| SwanLab | `roc_fu/finance` / `kairos-r2-r1-global-permutation-lr3e5-20261001-v2` |
| 不停 / 不删 | 不删除 kernel；不碰同事 TPU WIP（`beta_v21_c1*`） |

### 停训尝试记录（北京时间 2026-10-01 约 16:46–16:50）

1. `kaggle kernels status user281434/kairos-r2-r1-restart-lr-3e-5`
   → `KernelWorkerStatus.RUNNING`
2. Kaggle CLI **没有** `kernels cancel` / `stop` 子命令（仅 list/status/logs/push/delete 等）。
3. SDK `KernelsApiService/CancelKernelSession`（需 `kernelSessionId`）用当前 OAuth
   （scope `resources.admin:*`，用户 `user281434`）调用返回：
   `403 Permission 'kernelSessions.cancel' was denied`
4. 未找到公开 REST 路径可在无 session id 时安全取消；日志流可跟读但
   **不含** session id。未对 kernel 执行 `delete`。

**结论：** 自动取消被 API 权限挡住。请在 Kaggle 网页对该 kernel 点 Stop/Cancel，
然后用下面命令确认终态为 `CANCEL_ACKNOWLEDGED` / `COMPLETE` / `ERROR` 之一：

```sh
kaggle kernels status user281434/kairos-r2-r1-restart-lr-3e-5
```

勿删 kernel。日志快照显示段 1–3 best 已到约 `0.5977`，贴近训练先验常数
`≈0.5966`，支持“卡住基线”判断，但不替代网页停训。

## 下一阶段：便宜消融（优先 CPU / 本地 smoke）

实现见 `modernbert_finance/ablations/`（本交接同期提交）。

1. **Label shuffle sanity**  
   同一特征、打乱标签；若管线与损失 OK，训练/拟合应接近常数先验、无明显可学信号。
2. **Simple baseline（logistic / linear）**  
   同一窗口汇总特征 → 预测同一批 head（或简化为 `up_005`）；作为 ModernBERT
   是否值得再训的下限对照。
3. **特征/标签时间对齐核验**  
   核对 `(symbol, start_index, asof_date)` 与 sidecar parquet 行序一致；
   lookback 只用 `asof` 及之前，horizon 标签只用未来 10 日。

跑法（本地 CPU，合成数据 smoke 默认可用）：

```sh
python -m modernbert_finance.ablations.run_cheap_ablations --smoke
# 有真实 panel/targets 时：
python -m modernbert_finance.ablations.run_cheap_ablations \
  --panel path/to/train_data.pkl \
  --targets path/to/train_targets.parquet \
  --sector-vocab path/to/sector_vocabulary.json
```

## 明确不做

- 不再对同一 R2 配置追加长训段数 / 多 GPU 小时。
- 不提交、不推送同事 TPU（`finetune/*beta_v21_c1*`、`tpu_*`）改动。
- 不把验证集上偶然低于基线的单点当成泛化证明。
