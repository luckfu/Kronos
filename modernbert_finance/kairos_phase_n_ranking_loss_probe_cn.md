# Kairos 短排序损失探针结果（Phase N）

日期：2026-10-01 23:14 CST（北京时间）。  
Kernel：`user281434/kairos-ranking-probe-short-phase-n` → **COMPLETE**  
SwanLab：`roc_fu/finance` / `kairos-ranking-probe-short-phase-n-20261001` → **FINISHED**  
URL：https://swanlab.cn/@roc_fu/finance/runs/kairos-ranking-probe-short-phase-n-20261001

## 一句话

- **过闸。** `gate_passed=true`（seg1 起持续）。
- 终值 Rank IC mean≈**0.082**（略低于 Phase M MSE **0.085**）；TopK lift≈**+0.064**（高于 M 的 **+0.057**）。
- 半年度更均衡（H2/H1 均≈0.082），相对 M 的 H1 偏强更稳。
- 对照 Phase L Ridge `mfe_comb`≈**0.273** → identity+pairwise 仍 **弱约 3×**；pairwise **未**相对 MSE 显著抬升 IC。
- 建议下一步：极短 **listwise** 收束损失族（可选），或主推 **冻结 embeds + 浅非 identity rank 头**；**不**开 22 层二分类 / 全量 R2；**不**碰 TPU WIP。

## 运行事实

| 项 | 值 |
| --- | --- |
| 启动 | 2026-10-01 22:54:41 CST（GPU 2×T4） |
| 结束 | 2026-10-01 22:57:00 CST |
| 墙钟 | 训练段 ≈2.5 min（runtime≈147s）；预算 60min |
| 设定 | `BACKBONE_MODE=identity`，**pairwise**→连续 `mfe10`，LR `1e-4`，BATCH 64 |
| 预算 | 3×20 000；实际 processed=60 000 |
| stop_reason | **`segment_limit`** |
| val n | 123 836 |

## 闸门对照（vs Phase M）

| 指标 | Phase M MSE ★ | Phase N pairwise ★ | 闸门 |
| --- | ---: | ---: | --- |
| Rank IC mean | **0.0853** | **0.0817** | ≥ **0.05** |
| TopK lift | +0.0569 | **+0.0644** | ≥ **0.05** |
| gate_passed | true（seg2 起） | **true（seg1 起）** | — |
| vs Ridge 0.273 | ~3× 弱 | ~3× 弱 | — |

## 分段验证（all）

| seg | 时间 CST | Rank IC | TopK lift | gate |
| ---: | --- | ---: | ---: | --- |
| 1 | 22:56:42 | 0.0809 | +0.0664 | true |
| 2 | 22:56:51 | 0.0749 | +0.0606 | true |
| 3 ★ | 22:57:00 | **0.0817** | **+0.0644** | true |

## 半年度拆分（final）

| 窗 | Rank IC mean | TopK lift | gate |
| --- | ---: | ---: | --- |
| 2025H2 | **0.0819** | +0.0644 | true |
| 2026H1 | **0.0815** | +0.0645 | true |

对比 Phase M final：H2 0.052 / H1 0.120 → N 两侧拉平。

## 对比摘要

| 跑次 | Rank IC / 主指标 | TopK lift | gate |
| --- | ---: | ---: | --- |
| Phase K identity 二分类 Δ | −0.003 | — | false |
| Phase L Ridge mfe_comb | **0.273** | +0.174 | true（本地） |
| Phase L Ridge mfe_cs_comb | 0.276 | — | true（本地） |
| Phase M identity+MSE | **0.085** | +0.057 | true |
| **Phase N identity+pairwise** | **0.082** | **+0.064** | **true** |
| 闸门 | IC≥0.05 或 TopK≥0.05 | — | — |

## 判读

- pairwise 同预算可过闸，且 **TopK 略优于 MSE**、半年度更稳；但 **Rank IC 未超过 M**（−0.004 量级，可视作持平）。
- identity 浅头在短预算下天花板约 **IC≈0.08**；换排序损失 alone **不够**追 Ridge。
- val MSE≈0.029（高于 M≈0.009）属预期：pairwise 不优化点级回归。
- SwanLab 侧有 `train/loss_mode` 字符串标量报错（噪声），不影响闸门与 JSON 终值。
- 幅度通道提醒仍在：MFE 排序 ≠ 涨跌方向 α。

## 下一步建议

- **推荐（主）**：极短 **冻结 tokenizer embeds + 浅非 identity rank 头**（同 3×20k 量级），检验能否突破 ~0.08 天花板。
- **可选**：同预算 **listwise**（ListNet）收束损失族对照；期望有限。
- **可选**：接受 neural identity 天花板，转向 Phase L Ridge/表格特征蒸馏路径。
- **明确不做**：mfe≥10% 二分类 22 层 sidecar / 全量 R2；同事 TPU WIP。

## 产物

- Kernel logs + `kairos_ranking_probe/report.json`
- SwanLab run 如上（state=FINISHED）
- 本地：`artifacts/kairos_ranking_probe/phase_n_out/`
- JSON：`modernbert_finance/ablations/kairos_phase_n_ranking_loss_probe_results.json`
