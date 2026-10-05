# Kairos MFE≥10% 短 Sidecar 结果（2026-10-01）

日期：2026-10-01 19:15 CST（北京时间）。  
Kernel：`user281434/kairos-mfe10-path-touch-sidecar-short` → **COMPLETE**  
SwanLab：`roc_fu/finance` / `kairos-mfe10-sidecar-short-20261001` → **FINISHED**  
URL：https://swanlab.cn/@roc_fu/finance/runs/kairos-mfe10-sidecar-short-20261001

## 一句话

**未过闸。** 全程 `gate_passed=false`；最佳 val Δ vs 常数先验为 **+0.00802**（劣于先验），闸门要求 **Δ≤−0.04**。停因 `segment_limit`（4/4），**非** `stuck_at_constant_prior` 早停。  
**建议：不要加长训 / 不加下一短 chunk**；回消融或特征，勿扩预算。

## 运行事实

| 项 | 值 |
| --- | --- |
| 启动 | 2026-10-01 18:30:24 CST（GPU 2×T4） |
| 结束 | 2026-10-01 19:07:12 CST |
| 墙钟 | ≈36.9 min（`runtime_elapsed_seconds`≈2214.7；预算 5400s） |
| 目标 | `y=1{mfe10≥0.10}`，`mfe10=max(high[T+1:T+10])/close[T]-1` |
| 预算 | 4 segments × 20 000 samples；LR `3e-5`；fresh 单 logit 头 |
| stop_reason | **`segment_limit`**（非 early stop） |
| processed | 80 000 |
| train prior | 0.25336 |
| val n / 正类率 | 123 836 / **0.25294** |

## 闸门对照

| 指标 | 常数先验 | 最佳模型（seg1） | 末次（seg4） | 闸门 |
| --- | ---: | ---: | ---: | --- |
| val log_loss | **0.565542** | **0.573564** | 0.606550 | — |
| Δ vs prior | 0 | **+0.008022** | +0.041008 | 需 **≤ −0.04** |
| gate_passed | — | **false** | **false** | 全程 0 |

最佳点来自 `best_metric.json` / SwanLab `validation/log_loss` 最低步（step=20000）。  
注：`report.json` 的 `best_delta_vs_prior` 误写成末次 Δ（+0.041）；以 `best_metric.json` 为准。

## 分段验证（all）

| seg | 时间 CST | model_ll | prior_ll | Δ | gate |
| ---: | --- | ---: | ---: | ---: | --- |
| init | 18:36:30 | 0.668275 | 0.565542 | +0.102733 | false |
| 1 ★best | 18:44:07 | 0.573564 | 0.565542 | +0.008022 | false |
| 2 | 18:51:47 | 0.596397 | 0.565542 | +0.030855 | false |
| 3 | 18:59:28 | 0.576145 | 0.565542 | +0.010603 | false |
| 4 | 19:07:10 | 0.606550 | 0.565542 | +0.041008 | false |

SwanLab 摘要与上表一致：`validation/gate_passed` 全程 0；`delta_vs_prior` min=+0.008022、max=+0.102733。

## 早停？

- 规则：连续 2 次 eval `|model_ll − prior_ll| ≤ 1e-3` → `stuck_at_constant_prior`
- 实际 |Δ|：0.008 / 0.031 / 0.011 / 0.041，均 **> 1e-3**
- **未触发早停**；跑满 4 segments 后 `segment_limit` 正常结束

## 判读与建议

- 最佳模型仍 **劣于常数先验**（Δ=+0.008），距离闸门 −0.04 尚差约 **0.048**
- Logistic Base+xsection 消融曾达 Δ≈−0.034；深度 sidecar 本轮 **未能超过先验**
- 后续 seg 变差（末次 Δ=+0.041），加长训无望不大

**推荐：否（不要 longer train）**  
下一步应回 Phase 消融 / 特征或标签协议，而不是扩 sidecar 预算或开全量 R2。

## 产物

- Kaggle output：`/kaggle/working/kairos_mfe10_sidecar/{report,best_metric,validation_history}.json`、`best_model.pt`
- 本地拉取：`/workspace/kaggle_kairos_mfe10_sidecar_short_out/artifacts/kairos_mfe10_sidecar/`
