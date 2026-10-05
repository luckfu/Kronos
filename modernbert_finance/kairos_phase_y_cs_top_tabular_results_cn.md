# Kairos Phase Y 结果（截面 top 五分位 path-MFE 标签烟测）

日期：2026-10-02 10:57:23 CST（北京时间）。  
范围：父决策 Phase Y — 换决策标签；复用 S/T2 表格赢家栈（logistic/enet/浅 MLP blend，Base+xsection）。  
本地短烟：`train_cap=300k`，`MLP max_iter=60`；内存安全：CS 标签在全窗计算后平衡抽样再抽窗。

## 一句话结论（硬结论）

**主协议 train→val FAIL 闸**（best=`blend_lr0.5_mlp0.5` Δ≈**−0.031380**，需 ≤ −0.04）。  
相对 Phase W 绝对 10% 标签（Δ≈−0.028500）略好约 −0.0029，**仍远未过闸**。  
**次协议 val_temporal 仍过闸**（blend Δ≈**−0.041619**）。  
→ 截面 top 五分位标签**不是**主闸解药 → 按协议启动一次廉价备选（`mfe10≥0.08`），见 Phase Y-alt。

## 硬指标

### 主协议 train→val（2023–2024 → 全 val）

| 项 | 值 |
| --- | ---: |
| best_model | `blend_lr0.5_mlp0.5` |
| best_delta_vs_prior | **−0.031379848149291345** |
| gate (≤ −0.04) | **false** |
| logistic_C0.01 | −0.029944 |
| enet_C0.01_l1_0.7 | −0.029902 |
| enet_C0.01_l1_0.5 | −0.029948 |
| n_train / n_test | **300000** / 123836 |
| n_train_raw | 2347269 |
| train_prior | 0.25（平衡抽样） |
| pos_rate_full（窗内） | ≈0.200 |
| pos_rate_test（val） | ≈0.201 |
| feature_dim | 26 |
| vs Phase W (−0.028500) | 略好 |
| vs 闸门缺口 | 约 +0.0086 |

### 次协议 val_temporal（2025H2→2026H1）

| 项 | 值 |
| --- | ---: |
| best_model | `blend_lr0.5_mlp0.5` |
| best_delta_vs_prior | **−0.041618908780075514** |
| gate | **true** |
| n_train / n_test | 63268 / 60568（同 S/T2 切分） |

耗时 ≈208 s。

## 诊断含义

1. **标签轴否证（主协议）**：把绝对 10% 换成日内 CS top 五分位，主协议 Δ 仅从 −0.0285 挪到 −0.0314，**填不满 −0.04 闸**。  
2. **时序次闸仍在**：与 W 同构（主 FAIL / 次 PASS）→ 近端 OOS 可学，远端 train→全 val 仍不够。  
3. 正类率≈20% 健康，不是稀有类问题。  
4. **不加长 confirm**（未过主闸）；进入一次廉价备选 soft absolute `mfe10≥0.08`。

## 产物

- 脚本：`modernbert_finance/ablations/phase_y_mfe10_cs_top_tabular_smoke.py`
- 标签 API：`build_mfe_buy_labels(..., label_mode="cs_top", cs_pct=0.80)`
- 结果 JSON：`modernbert_finance/ablations/kairos_phase_y_cs_top_tabular_results.json`
- 日志：`scratch/kairos_phase_y_outputs/`
