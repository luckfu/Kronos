# Kairos 决策系统交接（Decision-Only Charter）

日期：2026-10-02（北京时间，交接更新 2026-10-02 11:04:47 CST）。  
状态：**产品路径 = 决策标签 only**；时序切分曾过闸（S/T2/U/V/W/Y secondary）；**主协议 train→val 在 U/V/W/X/Y/Y-alt 均 FAIL**；排序探针已停作产品。

## 一句话

闸门 = `Δ logloss vs 常数先验 ≤ −0.04`（主协议 train→val）。  
**表格 Base+xsection 赢家栈**仅稳过 **val_temporal**；主协议在绝对 mfe10≥10%、CS top 五分位、软绝对 ≥8%、Alpha158+LGB 上均未过闸。  
**当前：HARD-CONCLUDE 标签族空转**（绝对/CS-top/软绝对）+ 已死轴（序列 ModernBERT、Qlib/Alpha158、Ranking 产品）。

## 标签契约（历史）

| 项 | 值 |
| --- | --- |
| mfe10 | `max(high[T+1:T+10]) / close[T] - 1`（不变） |
| 曾用标签 | `y=1{mfe10≥0.10}`；`y=1{CS pct(mfe10)≥0.80}`；`y=1{mfe10≥0.08}` |
| 闸门 | `Δ ≤ −0.04` on **train→val** |
| 判读 | 时序次闸可过；主协议未过 |

## 诊断摘要

| 路径 | 主协议 Δ | 次协议 temporal | 产品含义 |
| --- | ---: | ---: | --- |
| I/J/K/P 序列 sidecar | ~0 | — | 序列死 |
| **S/T2 时序 enet/blend** | （未主跑） | **−0.0418** | 次闸过 |
| **U/V/W 绝对 ≥10%** | **−0.0285**（W） | −0.0418 | 主 FAIL；表格 HARD-CONCLUDE |
| **X Alpha158+LGB** | **−0.0258** | −0.0352 FAIL | Qlib 轴死 |
| **Y CS top 五分位** | **−0.0314** | −0.0416 | 标签换轴仍主 FAIL |
| **Y-alt 软绝对 ≥8%** | **−0.0219** | −0.0421 | 更差；硬报停 |

## 当前状态（Phase Y / Y-alt COMPLETE → HARD-REPORT）

- **Phase Y COMPLETE（2026-10-02 10:57:23 CST）**：CS-top 标签主协议 FAIL（Δ≈−0.031380）；temporal 过闸。  
- **Phase Y-alt COMPLETE（2026-10-02 11:04:30 CST）**：`mfe10≥0.08` 主协议 FAIL（Δ≈−0.021921）；更差。  
- **硬结论**：换决策标签定义（CS-top / 软绝对）未能清主闸 → **停止本标签族 + Base+xsection 表格栈空转**；不加长；不发 kernel。  
- 结果：`kairos_phase_y_cs_top_tabular_results_cn.md`、`kairos_phase_y_alt_mfe08_tabular_results_cn.md`

## 明确不做

1. 不重启 ranking Phase。  
2. 不复活 ModernBERT / 序列 tokenizer。  
3. 不碰 TPU WIP（`beta_v21_c1*`）。  
4. 不开 Time-Series-Library 序列复刻（高复刻已死轴）。  
5. 不再沿同一表格栈对 mfe10 绝对/CS 变体加长烟测。

## 若重启

需用户显式点名**新证据轴**（特征族/模型族/协议显著不同于已死路径），默认建议停。
