# Kairos MFE≥10% Phase J 短 Sidecar 结果（freeze-backbone）

日期：2026-10-01 22:20 CST（北京时间）。  
Kernel：`user281434/kairos-mfe10-sidecar-short-phase-j-freeze-bb` → **COMPLETE**  
SwanLab：`roc_fu/finance` / `kairos-mfe10-sidecar-short-phase-j-freeze-bb-20261001` → **FINISHED**  
URL：https://swanlab.cn/@roc_fu/finance/runs/kairos-mfe10-sidecar-short-phase-j-freeze-bb-20261001

## 一句话

**未过闸。** 全程 `gate_passed=false`。冻结 22 层 fresh ModernBERT（只训 fusion/gate/sector/size/cond/head/+embeds）后仍打不赢先验，更远未到 Δ≤−0.04。  
**最佳 post-train Δ≈+8.5e-4（seg3）**；`best_metric` 仍是 **init Δ≈+1.4e-7**。  
对照：Phase I best≈0；Logistic Base+xsection Δ≈**−0.034**；线性 BCE 冻结特征 Δ≈**−0.031**；tok_full logistic Δ≈**−0.026**。闸门 **Δ≤−0.04**。

## 运行事实

| 项 | 值 |
| --- | --- |
| 启动 | 2026-10-01 21:21:55 CST（GPU 2×T4） |
| 结束 | 2026-10-01 21:50:46 CST |
| 墙钟 | ≈29.7 min（`runtime_elapsed_seconds`≈1783；预算 5400s） |
| 设定 | `FREEZE_BACKBONE=True`，LR `1e-4`，gate_init=1，bias=logit(prior) |
| 冻结审计 | trainable 13 / frozen 134；trainable_numel≈1.47M / frozen≈111M |
| 预算 | 4×20 000；fresh 22 层（冻结） |
| stop_reason | **`segment_limit`** |
| processed | 80 000 |
| train prior | 0.25336 |
| val n / 正类率 | 123 836 / **0.25294** |

## 闸门对照

| 指标 | 常数先验 | init（★best_metric） | seg3 最近 | 末次 seg4 | 闸门 |
| --- | ---: | ---: | ---: | ---: | --- |
| val log_loss | **0.565542** | **0.565542** | 0.566397 | 0.572169 | — |
| Δ vs prior | 0 | **+1.38e-7** | **+0.000855** | +0.006627 | 需 **≤ −0.04** |
| gate_passed | — | **false** | **false** | **false** | 全程 0 |
| roc_auc | — | 0.500 | 0.549 | 0.534 | — |

## 分段验证（all）

| seg | 时间 CST | model_ll | prior_ll | Δ | gate |
| ---: | --- | ---: | ---: | ---: | --- |
| init ★best_metric | 21:27:33 | 0.565542 | 0.565542 | +1.38e-7 | false |
| 1 | 21:33:20 | 0.567336 | 0.565542 | +0.001794 | false |
| 2 | 21:39:09 | 0.572513 | 0.565542 | +0.006972 | false |
| 3 | 21:44:57 | 0.566397 | 0.565542 | +0.000855 | false |
| 4 | 21:50:45 | 0.572169 | 0.565542 | +0.006627 | false |

## Checkpoint 审计（last @ 80k）

| 项 | Phase J last | Phase I last | 破损短跑 best |
| --- | ---: | ---: | ---: |
| gate | **1.0054**（保持打开） | 1.0112 | 0.0020（≈关） |
| sigmoid(head.bias) | **0.2546**（≈prior） | 0.2543 | 0.5012（≈0.5） |
| \|bias − logit(prior)\| | **0.0067** | 0.0052 | ≈1.085 |
| head_w_norm | 0.189（有学） | 0.0447 | 0.561 |

**结论**：H1/H2 补丁在 freeze 设定下仍 **保持住**；H3「只因可训 backbone 洗信号」**被否**——冻结后同样学不到线性已有的 −0.03。失败转移到 **H7：随机初始化 22 层 ModernBERT 特征本身无可训练于短预算的有用归纳偏置**（相对 tok/tabular 线性探针）。

## 对比摘要

| 跑次 | best Δ vs prior | gate_passed |
| --- | ---: | --- |
| 破损短跑（gate≈0, bias@0.5） | **+0.008** | false |
| Phase I（可训 22 层 + 补丁） | **≈0**（init / seg3） | false |
| Phase J（冻结 22 层，本跑） | **≈0**（init；post≈+8.5e-4） | false |
| tok_full logistic | **≈−0.026** | false（相对更好） |
| Logistic Base+xsection | **≈−0.034** | false |
| 线性 BCE 冻结特征 | **≈−0.031** | false |
| 闸门 | ≤ **−0.04** | — |

Phase J 相对 Phase I **无提升**（略差的 post-train Δ），仍 **远未过闸**，且 **未超过** logistic/线性/tok 探针。

## 判读与下一步（已执行）

- **不要**在 fresh 22 层（可训或冻结）上再加长训。
- **下一步（Phase K，更便宜）**：`BACKBONE_MODE=identity`，绕过 ModernBERT，对 condition+market embeds 做 mean-pool + 单 logit 头；测「去掉随机深栈后能否逼近 tok 线性探针」。  
- **不**动 TPU WIP；**不**开全量 R2。

## 产物

- Kaggle：`/kaggle/working/kairos_mfe10_sidecar/{report,best_metric,validation_history}.json`、`last_checkpoint.pt`
- 本地：`/workspace/kaggle_kairos_mfe10_sidecar_phase_j_out/kairos_mfe10_sidecar/`
