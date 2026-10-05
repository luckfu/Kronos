# Kairos MFE≥10% Phase K 短 Sidecar 结果（identity-backbone）

日期：2026-10-01 22:35 CST（北京时间）。  
Kernel：`user281434/kairos-mfe10-sidecar-short-phase-k-identity` → **COMPLETE**  
SwanLab：`roc_fu/finance` / `kairos-mfe10-sidecar-short-phase-k-identity-20261001` → **FINISHED**  
URL：https://swanlab.cn/@roc_fu/finance/runs/kairos-mfe10-sidecar-short-phase-k-identity-20261001

## 一句话

**未过闸。** 全程 `gate_passed=false`。`BACKBONE_MODE=identity`（绕过 22 层 ModernBERT，mean-pool embeds + 头）后 best Δ≈**−0.00297**（seg4），仍远未到 Δ≤−0.04。  
对照：Logistic Base+xsection Δ≈**−0.034**；线性 BCE ≈**−0.031**；tok logistic ≈**−0.026**；Phase I/J ≈**0**。  
**相对 I/J 略好（首次稳定负 Δ），但未逼近线性探针，更未过闸。** 用户对齐：α 在排序不在二分类 → 下一步廉价排序目标消融。

## 运行事实

| 项 | 值 |
| --- | --- |
| 启动 | 2026-10-01 22:21:58 CST（GPU 2×T4） |
| 结束 | 2026-10-01 22:25:46 CST |
| 墙钟 | 训练段 ≈1.5 min（含安装/数据加载约 4 min）；预算 5400s |
| 设定 | `BACKBONE_MODE=identity`，`FREEZE_BACKBONE=False`，LR `1e-4` |
| 参数 | trainable 13 / frozen 0；numel≈1.47M（无 22 层） |
| 预算 | 4×20 000 |
| stop_reason | **`segment_limit`** |
| processed | 80 000 |
| train prior | 0.25336 |
| val n / 正类率 | 123 836 / **0.25294** |

## 闸门对照

| 指标 | 常数先验 | init | seg4 ★best | 闸门 |
| --- | ---: | ---: | ---: | ---: |
| val log_loss | **0.565542** | 0.565542 | **0.562571** | — |
| Δ vs prior | 0 | **+1.38e-7** | **−0.002970** | 需 **≤ −0.04** |
| gate_passed | — | **false** | **false** | 全程 0 |

## 分段验证（all）

| seg | 时间 CST | model_ll | prior_ll | Δ | gate |
| ---: | --- | ---: | ---: | ---: | --- |
| init | 22:24:19 | 0.565542 | 0.565542 | +1.38e-7 | false |
| 1 | 22:24:41 | 0.575994 | 0.565542 | +0.010452 | false |
| 2 | 22:25:02 | 0.575910 | 0.565542 | +0.010368 | false |
| 3 | 22:25:24 | 0.567209 | 0.565542 | +0.001668 | false |
| 4 ★best | 22:25:46 | 0.562571 | 0.565542 | **−0.002970** | false |

## 对比摘要

| 跑次 | best Δ vs prior | gate_passed |
| --- | ---: | --- |
| 破损短跑（gate≈0, bias@0.5） | **+0.008** | false |
| Phase I（可训 22 层 + 补丁） | **≈0** | false |
| Phase J（冻结 22 层） | **≈0**（post≈+8.5e-4） | false |
| Phase K（identity，本跑） | **−0.00297** | false |
| tok_full logistic | **≈−0.026** | false（相对更好） |
| Logistic Base+xsection | **≈−0.034** | false |
| 线性 BCE 冻结特征 | **≈−0.031** | false |
| 闸门 | ≤ **−0.04** | — |

## 判读

- Identity 绕过随机 22 层后**首次**拿到稳定负 Δ，支持 H7「随机深栈无短预算归纳偏置」。
- 但仍比 tabular/tok 线性探针弱一个数量级，**不能**用更深/更长二分类 sidecar 追闸。
- 用户对齐诊断：**排序非二分类**。下一步 = 廉价本地排序目标消融（Rank IC / TopK vs chance），相对二分类路径是否有清晰抬升。

## 下一步（已执行方向）

- **Phase L（本地 CPU）**：Base+xsection 预测 `fwd_ret_10` / `mfe10` 连续与截面分位 + pairwise；报 Rank IC、TopK vs chance。
- **不**开 22 层长训；**不**动 TPU WIP；**不**全量 R2。

## 产物

- Kaggle kernel logs（metrics）；SwanLab run 如上。
- 本地：`/workspace/kaggle_kairos_mfe10_sidecar_phase_k_out/report_from_logs.json`
- JSON：`modernbert_finance/ablations/kairos_phase_k_results_ranking_next.json`
