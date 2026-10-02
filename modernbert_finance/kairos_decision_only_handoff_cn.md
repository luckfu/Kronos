# Kairos 决策系统交接（Decision-Only Charter）

日期：2026-10-02（北京时间，交接更新 2026-10-02 11:07:54 CST）。  
状态：**产品路径 = 决策标签 only**；时序切分曾过闸（S/T2/U/V/W/Y secondary）；**主协议 train→val 在 U/V/W/X/Y/Y-alt 均 FAIL**；**Phase Z Kronos 分数轴 HARD-REPORT FAIL（train→val BLOCKED；密封 OOS Δ≈−0.012）**；排序探针已停作产品。

## 一句话

闸门 = `Δ logloss vs 常数先验 ≤ −0.04`（主协议 train→val）。  
**表格 Base+xsection 赢家栈**仅稳过 **val_temporal**；主协议在绝对 mfe10≥10%、CS top 五分位、软绝对 ≥8%、Alpha158+LGB 上均未过闸。  
**Kronos C2 生产分数**无法对齐 train/val；密封窗决策头亦远未过闸。  
**当前：HARD-CONCLUDE / 建议 STOP** 本 binary mfe10 决策产品路径。

## 标签契约（历史）

| 项 | 值 |
| --- | --- |
| mfe10 | `max(high[T+1:T+10]) / close[T] - 1`（不变） |
| 曾用标签 | `y=1{mfe10≥0.10}`；`y=1{CS pct(mfe10)≥0.80}`；`y=1{mfe10≥0.08}` |
| 闸门 | `Δ ≤ −0.04` on **train→val** |
| 判读 | 时序次闸可过；主协议未过；Kronos 分数轴 BLOCKED/弱 |

## 诊断摘要

| 路径 | 主协议 Δ | 次协议 temporal | 产品含义 |
| --- | ---: | ---: | --- |
| I/J/K/P 序列 sidecar | ~0 | — | 序列死 |
| **S/T2 时序 enet/blend** | （未主跑） | **−0.0418** | 次闸过 |
| **U/V/W 绝对 ≥10%** | **−0.0285**（W） | −0.0418 | 主 FAIL；表格 HARD-CONCLUDE |
| **X Alpha158+LGB** | **−0.0258** | −0.0352 FAIL | Qlib 轴死 |
| **Y CS top 五分位** | **−0.0314** | −0.0416 | 标签换轴仍主 FAIL |
| **Y-alt 软绝对 ≥8%** | **−0.0219** | −0.0421 | 更差；硬报停 |
| **Z Kronos C2 score** | **BLOCKED** | 密封 OOS **−0.0124** FAIL | 分数轴死；建议 STOP |

## 当前状态（Phase Z COMPLETE → HARD-REPORT）

- **Phase Z COMPLETE（2026-10-02 11:07:54 CST）**：train→val BLOCKED（分数窗 2026-08-11..09-03 vs val 2025-07..2026-07）；密封 temporal best Δ≈−0.012399。  
- **硬结论**：停止 binary mfe10 决策空转；最后一轴仅当父预算全量 C2 分数物化到 val。  
- 结果：`kairos_phase_z_kronos_score_decision_results_cn.md`

## 明确不做


1. 不重启 ranking Phase。  
2. 不复活 ModernBERT / 序列 tokenizer。  
3. 不碰 TPU WIP（`beta_v21_c1*`）。  
4. 不开 Time-Series-Library 序列复刻（高复刻已死轴）。  
5. 不再沿同一表格栈对 mfe10 绝对/CS 变体加长烟测。

## 若重启

需用户显式点名**新证据轴**（特征族/模型族/协议显著不同于已死路径），默认建议停。

## Phase Z2（最后一轴）— HARD-STOP FAIL

- Kernel COMPLETE: https://www.kaggle.com/code/user281434/kronos-c2-seg179-kairos-val-prod-scores
- C2 Seg@179 prod scores on Kairos val 2025-07-03..2026-07-02 (n=123836, 100% mfe10 join)
- val_temporal best Δ≈−0.01986 (gate −0.04) → FAIL
- train→val BLOCKED (no train scores; not worth weeks of GPU given val upper bound)
- **永久停止** binary `mfe10≥10%` 决策产品路径
- 结果：`kairos_phase_z2_kronos_val_score_decision_results_cn.md`

## Phase AA → AA2（成本感知 TopK 可交易规则）— 非 mfe10 决策

- **AA（2026-10-02）**：短密封 OOS 18d 上 `topk50_tp10_nostop` mean_net≈+0.559%；Sharpe 因短窗膨胀；**PROPOSE_LONGER_CONFIRM**。
- **AA2（2026-10-02 12:32 CST）**：冻结先验规则、**不重选** K/止损；Z2 后半 121d + walk-forward 4/4 折成本后仍为正 → **PASS_PROPOSE_PAPER_TRADE**。
  - late mean_net≈**+1.376%**；random50≈−1.06%；EW≈−1.24%；WF 折净期望均 >0。
  - 污染警告：Z2 分在 C2 训练窗内，IC 可能偏乐观；确认的是规则机制而非干净模型 OOS。
- 下一步：廉价纸面/影子盘（Top50 / TP10 nostop / T+1 / 涨停跳过 / 30bps）；**不**深训 / **不**碰 TPU / **不**复活 mfe10。
- 结果：`kairos_phase_aa2_kronos_longer_sealed_confirm_cn.md`
