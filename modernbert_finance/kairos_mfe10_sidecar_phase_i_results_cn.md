# Kairos MFE≥10% Phase I 短 Sidecar 结果（gate=1 + bias=logit prior）

日期：2026-10-01 21:20 CST（北京时间）。  
Kernel：`user281434/kairos-mfe10-sidecar-short-phase-i` → **COMPLETE**  
SwanLab：`roc_fu/finance` / `kairos-mfe10-sidecar-short-phase-i-20261001` → **FINISHED**  
URL：https://swanlab.cn/@roc_fu/finance/runs/kairos-mfe10-sidecar-short-phase-i-20261001

## 一句话

**未过闸。** 全程 `gate_passed=false`。补丁生效（init Δ≈0、末段 gate≈1.01、bias≈logit prior），但深度短训仍无打赢先验。  
**最佳 post-train Δ≈+7.8e-6（seg3，≈0）**；`best_metric` 记的是 **init Δ≈+1.4e-7**（常数先验对齐）。  
对照：破损短跑 best Δ=**+0.008**；Logistic Base+xsection Δ≈**−0.034**；线性 BCE 冻结特征 Δ≈**−0.031**。闸门 **Δ≤−0.04**。

## 运行事实

| 项 | 值 |
| --- | --- |
| 启动 | 2026-10-01 20:20:01 CST（GPU 2×T4） |
| 结束 | 2026-10-01 21:01:45 CST |
| 墙钟 | ≈42.4 min（`runtime_elapsed_seconds`≈2545；预算 5400s） |
| 补丁 | `gate_init=1.0`，`head.bias=logit(train_prior)≈−1.0808` |
| 预算 | 4×20 000；LR `3e-5`；fresh 22 层 ModernBERT |
| stop_reason | **`segment_limit`**（非 early stop） |
| processed | 80 000 |
| train prior | 0.25336 |
| val n / 正类率 | 123 836 / **0.25294** |

## 闸门对照

| 指标 | 常数先验 | init（★best_metric） | seg3 最近 | 末次 seg4 | 闸门 |
| --- | ---: | ---: | ---: | ---: | --- |
| val log_loss | **0.565542** | **0.565542** | 0.565550 | 0.573485 | — |
| Δ vs prior | 0 | **+1.38e-7** | **+7.84e-6** | +0.007943 | 需 **≤ −0.04** |
| gate_passed | — | **false** | **false** | **false** | 全程 0 |

## 分段验证（all）

| seg | 时间 CST | model_ll | prior_ll | Δ | gate |
| ---: | --- | ---: | ---: | ---: | --- |
| init ★best_metric | 20:26:02 | 0.565542 | 0.565542 | +1.38e-7 | false |
| 1 | 20:34:55 | 0.573869 | 0.565542 | +0.008327 | false |
| 2 | 20:43:48 | 0.577159 | 0.565542 | +0.011617 | false |
| 3 | 20:52:46 | 0.565550 | 0.565542 | +7.84e-6 | false |
| 4 | 21:01:41 | 0.573485 | 0.565542 | +0.007943 | false |

## Checkpoint 审计（last @ 80k；best=init 故未单独产出有用 best_model）

| 项 | Phase I last | 破损短跑 best |
| --- | ---: | ---: |
| gate | **1.0112**（保持打开） | 0.0020（≈关） |
| sigmoid(head.bias) | **0.2543**（≈prior） | 0.5012（≈0.5） |
| \|bias − logit(prior)\| | **0.0052** | ≈1.085 |
| head_w_norm | 0.0447（几乎未学） | 0.561 |

**结论**：H1/H2 补丁在短训中 **保持住了**；失败转移到 **H3：fresh 22 层 backbone 在 80k 预算下洗不掉噪声 / 学不到线性已有的 −0.03 信号**。

## 对比摘要

| 跑次 | best Δ vs prior | gate_passed |
| --- | ---: | --- |
| 破损短跑（gate≈0, bias@0.5） | **+0.008** | false |
| Phase I（本跑，补丁） | **≈0**（init / seg3） | false |
| Logistic Base+xsection | **≈−0.034** | false（相对更好） |
| 线性 BCE 冻结特征 | **≈−0.031** | false |
| 闸门 | ≤ **−0.04** | — |

Phase I 相对破损跑略好（不再稳定劣于先验 +0.008），但仍 **远未过闸**，且 **未超过** logistic/线性基线。

## 判读与下一步（已执行）

- **不要**在可训 22 层设定下加长训。
- **下一步（Phase J，便宜）**：`FREEZE_BACKBONE=True`，只训 fusion/gate/sector/size/cond/head（+token embeds）；LR `1e-4`；新 slug 短冒烟 4×20k。  
- **不**动 TPU WIP；**不**开全量 R2。

## 产物

- Kaggle：`/kaggle/working/kairos_mfe10_sidecar/{report,best_metric,validation_history}.json`、`last_checkpoint.pt`
- 本地：`/workspace/kaggle_kairos_mfe10_sidecar_phase_i_out/kairos_mfe10_sidecar/`
