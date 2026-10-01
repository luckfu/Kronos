# Kairos 短排序探针结果（Phase M）

日期：2026-10-01 22:52 CST（北京时间）。  
Kernel：`user281434/kairos-ranking-probe-short-phase-m` → **COMPLETE**  
SwanLab：`roc_fu/finance` / `kairos-ranking-probe-short-phase-m-20261001` → **FINISHED**  
URL：https://swanlab.cn/@roc_fu/finance/runs/kairos-ranking-probe-short-phase-m-20261001

## 一句话

- **过闸。** `gate_passed=true`（seg2 起持续）。
- 终值 Rank IC mean≈**0.085**；TopK lift≈**+0.057**（门槛均为 0.05）。
- 对照 Phase L Ridge `mfe_comb` Rank IC≈**0.273** → identity+MSE 头过闸但 **弱约 3×**。
- 建议下一步：极短 **pairwise/listwise** 或轻量非 identity 浅头；**不**开 22 层二分类 / 全量 R2；**不**碰 TPU WIP。

## 运行事实

| 项 | 值 |
| --- | --- |
| 启动 | 2026-10-01 22:42:45 CST（GPU 2×T4） |
| 结束 | 2026-10-01 22:46:06 CST |
| 墙钟 | 训练段 ≈3.5 min；预算 60min |
| 设定 | `BACKBONE_MODE=identity`，MSE→连续 `mfe10`，LR `1e-4` |
| 预算 | 3×20 000；实际 processed=60 000 |
| stop_reason | **`segment_limit`** |
| val n | 123 836 |

## 闸门对照

| 指标 | init | seg1 | seg2 | seg3 ★final | 闸门 |
| --- | ---: | ---: | ---: | ---: | --- |
| Rank IC mean | NaN* | 0.0450 | 0.0726 | **0.0853** | ≥ **0.05** |
| TopK lift | +0.0055 | +0.0324 | +0.0544 | **+0.0569** | ≥ **0.05** |
| gate_passed | false | false | **true** | **true** | — |

\*init Rank IC `n_days=0` → `best_metric.json` 被 NaN 污染，**以 final/seg3 为准**。

## 分段验证（all）

| seg | 时间 CST | Rank IC | TopK lift | gate |
| ---: | --- | ---: | ---: | --- |
| 1 | 22:45:20 | 0.0450 | +0.0324 | false |
| 2 | 22:45:43 | 0.0726 | +0.0544 | true |
| 3 ★ | 22:46:06 | **0.0853** | **+0.0569** | true |

## 半年度拆分（final）

| 窗 | Rank IC mean | TopK lift | gate |
| --- | ---: | ---: | --- |
| 2025H2 | 0.0519 | +0.0315 | true（IC 刚过） |
| 2026H1 | **0.1203** | **+0.0835** | true |

## 对比摘要

| 跑次 | Rank IC / 主指标 | gate |
| --- | ---: | --- |
| Phase K identity 二分类 Δ | −0.003 | false |
| Phase L Ridge mfe_comb | **0.273** | true（本地） |
| Phase L Ridge mfe_cs_comb | 0.276 | true（本地） |
| **Phase M identity rank head** | **0.085** | **true** |
| 闸门 | IC≥0.05 或 TopK≥0.05 | — |

## 判读

- 短预算 identity+MSE 排序头**确认可过闸**，支持「α 在排序」诊断。
- 但相对表格 Ridge 仍弱一个量级，**不能**据此开深栈/长训追 Ridge。
- 幅度通道提醒仍在：MFE 排序 ≠ 涨跌方向 α；2026H1 明显强于 2025H2。

## 下一步建议

- **推荐**：极短 pairwise / listwise 损失探针（同预算量级），或冻结 embeds + 浅 rank 头（非 22 层全训）。
- **可选**：`TARGET_MODE=mfe_cs_rank` 短对照（注意分位标签泄漏口径）。
- **明确不做**：mfe≥10% 二分类 22 层 sidecar / 全量 R2；同事 TPU WIP。

## 产物

- Kernel logs + `kairos_ranking_probe/report.json`
- SwanLab run 如上
- 本地：`artifacts/kairos_ranking_probe/phase_m_out/`
- JSON：`modernbert_finance/ablations/kairos_phase_m_ranking_probe_results.json`
