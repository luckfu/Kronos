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

**当时结论：** 自动取消被 API 权限挡住，需网页 Stop/Cancel。

### 停训确认（北京时间 2026-10-01 约 16:57–16:59）

| 检查 | 结果 |
| --- | --- |
| `kaggle kernels status user281434/kairos-r2-r1-restart-lr-3e-5` | **`KernelWorkerStatus.CANCEL_ACKNOWLEDGED`**（终态已确认） |
| 中间态 | 约 16:56 曾见 `CANCEL_REQUESTED`，随后变为 `CANCEL_ACKNOWLEDGED` |
| SwanLab | 看板 URL 可打开：`https://swanlab.cn/@roc_fu/finance/runs/kairos-r2-r1-global-permutation-lr3e5-20261001-v2`；本机无 `SWANLAB_API_KEY`，未能 API 拉取最终曲线点 |
| 操作 | **未重启训练**；**未改动**同事 TPU dirty 文件 |

```sh
kaggle kernels status user281434/kairos-r2-r1-restart-lr-3e-5
# -> KernelWorkerStatus.CANCEL_ACKNOWLEDGED
```

勿删 kernel。日志快照显示段 1–3 best 已到约 `0.5977`，贴近训练先验常数
`≈0.5966`，支持“卡住基线”判断。

## 下一阶段：便宜消融（已跑真实验证集）

实现见 `modernbert_finance/ablations/`。

1. **Label shuffle sanity**  
   同一特征、打乱标签；若管线与损失 OK，训练/拟合应接近常数先验、无明显可学信号。
2. **Simple baseline（logistic / linear）**  
   同一窗口汇总特征 → 预测同一批 head（或简化为 `up_005`）；作为 ModernBERT
   是否值得再训的下限对照。
3. **特征/标签时间对齐核验**  
   核对 `(symbol, start_index, asof_date)` 与 sidecar parquet 行序一致；
   lookback 只用 `asof` 及之前，horizon 标签只用未来 10 日。

### 数据与路径

| 用途 | 路径 / 来源 |
| --- | --- |
| Panel（真实） | Kaggle `luckfu/a-share-120d-temporal-symbol-holdout` → `processed_datasets/val_data.pkl`（SHA-256 `4cce31bc…19bf7`，与 manifest `validation_panel_sha256` 一致） |
| Targets（真实） | Kaggle `luckfu/ashare120d-modernbert-targets` → `validation_targets.parquet`（123 836 行） |
| Manifest | 同目录 `decision_targets_manifest.json` |
| Sector vocab | 由 val panel 行业列现场生成（73 类） |
| 本地缓存（不入库） | `scratch/kairos_ablation_data/` |
| 结果 JSON | `modernbert_finance/ablations/kairos_cheap_ablation_validation.json` |

说明：完整 `train_data.pkl` ≈875 MB、`train_targets` ≈9.0M 行；本机用 `_dataset_matrices` 全量枚举不可行。
本次采用**最大可行真实子集 = 官方 validation split**（非合成 smoke）。

跑法：

```sh
python -m modernbert_finance.ablations.run_cheap_ablations \
  --panel path/to/val_data.pkl \
  --targets path/to/validation_targets.parquet \
  --sector-vocab path/to/sector_vocabulary.json \
  --out-json modernbert_finance/ablations/kairos_cheap_ablation_validation.json
```

### 消融结果摘要（真实 validation，n=123 836）

对齐：`ok=true`（抽查 64 行无 mismatch）。

**Label shuffle（head=`up_005`）**

| 指标 | 值 |
| --- | --- |
| 常数先验 log loss | 0.69314 |
| 打乱标签拟合并 log loss | 0.69329（−先验 ≈ +0.00015，接近先验 ✓） |
| 真标签拟合并 log loss | 0.67984（−先验 ≈ −0.0133，有弱可学信号） |
| shuffle sanity | **ok** |

**Simple logistic vs 常数先验（75/25 切分；train 92 877 / test 30 959）**

| Head | Prior LL | Model LL | Δ(model−prior) | Beats prior? |
| --- | ---: | ---: | ---: | --- |
| up_003 | 0.6401 | 0.6323 | −0.0079 | 是 |
| up_005 | 0.6931 | 0.6798 | −0.0133 | 是 |
| up_008 | 0.6336 | 0.6151 | −0.0185 | 是 |
| up_012 | 0.4964 | 0.4775 | −0.0188 | 是 |
| down_003 | 0.6480 | 0.6342 | −0.0138 | 是 |
| down_005 | 0.6919 | 0.6726 | −0.0193 | 是 |
| down_008 | 0.5955 | 0.5767 | −0.0189 | 是 |
| down_012 | 0.3727 | 0.3535 | −0.0192 | 是 |

线性回归（无先验 LL）：`mfe10` MSE≈0.00842；`mae10` MSE≈0.00256。

**解读（相对 Kairos R2 验证 ≈0.5966 卡住基线）：**

- 管线/对齐/shuffle 正常：打乱后贴先验，说明损失与标签接线可信。
- 窗口汇总特征 + 逻辑回归在各 binary head 均小幅打赢常数先验（约 0.008–0.019 LL）。
- 这只说明**极简统计特征有一点可分性**，**不**证明 ModernBERT R2 该继续长训；与「深度模型未稳定超越训练先验」并不矛盾。
- 下一步应做结构/标签/切分诊断，而不是同一 R2 配置加 GPU 小时。

## 明确不做

- 不再对同一 R2 配置追加长训段数 / 多 GPU 小时。
- 不提交、不推送同事 TPU（`finetune/*beta_v21_c1*`、`tpu_*`）改动。
- 不把验证集上偶然低于基线的单点当成泛化证明。
