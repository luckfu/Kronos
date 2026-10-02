# Kairos Phase Y 启动（截面 top 五分位 path-MFE 标签烟测）

日期：2026-10-02 10:45:33 CST（北京时间）。  
范围：父决策 Phase Y — **换决策标签**，仍 buy/not；复用 S/T2 表格赢家栈（logistic/enet/浅 MLP blend，Base+xsection）。  
禁止：Ranking 产品；ModernBERT 复活；Time-Series-Library 序列复刻；TPU WIP `beta_v21_c1*`；Alpha158/Qlib（Phase X 已死）。

## 标签契约（变更）

| 项 | 值 |
| --- | --- |
| mfe10 | `max(high[T+1:T+10]) / close[T] - 1`（**不变**） |
| **新标签** | `y = 1{ daily CS percentile(mfe10) ≥ 0.80 }`（日内 top 五分位） |
| 旧标签对照 | `y = 1{mfe10 ≥ 0.10}`（G2/S–W；主协议 U/V/W FAIL） |
| 闸门 | `Δ logloss vs 常数先验 ≤ −0.04` |
| **主协议** | train 2023–2024 → 全 val |
| 次协议 | val_temporal 2025H2→2026H1（一并报告） |

## 模型 / 预算

- 栈：`logistic_C0.01` / `enet_C0.01_l1_{0.7,0.5}` / `blend_lr{0.5,0.7,0.8,0.9}_mlp*`（= S/T2 赢家族）
- 特征：Base+xsection（26 维），**非** Alpha158
- 短烟：`train_cap=300k`，`MLP max_iter=60`
- 若主协议 PASS → 适度加长 confirm + 可选 Kaggle；FAIL → 仅一次廉价备选（`mfe10≥0.08` 或行业中性 residual touch）后硬报

## 假设

绝对 10% 触及在远端 train→val 上表格栈过不了闸（W Δ≈−0.0285）；换成**日内相对 top 五分位**可能更可学（正类≈20%、截面可比），且仍是决策二分类而非 ranking 产品。

## 产物路径

- 脚本：`modernbert_finance/ablations/phase_y_mfe10_cs_top_tabular_smoke.py`
- 标签：`buy_profit_mfe_ablations.build_mfe_buy_labels(..., label_mode="cs_top")`
- Launch JSON：`modernbert_finance/ablations/kairos_phase_y_cs_top_tabular_launch.json`
- 结果：跑完后写 `kairos_phase_y_cs_top_tabular_results_cn.md` + `ablations/kairos_phase_y_cs_top_tabular_results.json`
