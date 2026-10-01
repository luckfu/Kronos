# Kairos 短排序损失探针（Phase N）启动

日期：2026-10-01（北京时间）。  
前置：Phase M identity+MSE 过闸 Rank IC≈**0.085** / TopK lift≈**0.057**。  
Kernel：`user281434/kairos-ranking-probe-short-phase-n`  
SwanLab：`roc_fu/finance` / `kairos-ranking-probe-short-phase-n-20261001`

## 一句话

- 同 identity 浅头 + 连续 `mfe10`，把 **MSE → 同日 pairwise**（默认；脚本亦支持 listwise ListNet）。
- 预算同 Phase M：3×20 000 / 60 min GPU；Kairos-only，**不**碰 TPU WIP，**不**开 22 层二分类。
- 闸门不变：Rank IC≥0.05 或 TopK lift≥0.05；对照 Phase M MSE≈0.085 与 Phase L Ridge≈0.273。

## 设定

| 项 | 值 |
| --- | --- |
| BACKBONE | `identity` mean-pool + rank head |
| LOSS_MODE | **`pairwise`**（同日 softplus；min_gap=0.005） |
| 备选 | `listwise` ListNet（T=0.05）/ `mse` |
| BATCH | 64（段内按 asof_date 排序，增加同日对） |
| LR | 1e-4 |
| 目标 | `mfe10_continuous` |

## 不做

- mfe≥10% 二分类 / 22 层 sidecar / 全量 R2
- 同事 TPU WIP
