# Kairos 决策表格时序正则（Phase R）结果

日期：2026-10-02（北京时间）。  
设定：val 面板 **时序切分** 2025H2→2026H1；Base+xsection；Logistic / 强正则浅 MLP。  
状态：**COMPLETE**（本地 CPU）

## 一句话

**未过闸，但逼近。** best=`logistic_comb_C0.01` Δ≈**−0.0398**（距闸门 −0.04 仅 ~2e−4）。  
Phase Q 随机切分 mlp_3x 过闸为 **乐观假象**；时序上深 MLP 过拟合，表格 Logistic 才是稳健近闸信号。

## 主结果（时序）

| 模型 | Δ vs prior | gate |
| --- | ---: | --- |
| logistic C=0.01 | **−0.039815** | false |
| mlp1 α=0.1 | −0.039360 | false |
| logistic C=1（G2 式） | −0.037801 | false |
| mlp3 α=0.1 | **+0.097** | false（过拟合） |

## 判读

- 决策产品路径确认在 **表格特征**，不在 tokenizer 序列（P/K 失败）。
- 下一步（Phase S）：C 网格 / elastic-net / Logistic⊕浅MLP 融合，冲过 −0.04。

## 产物

- `modernbert_finance/ablations/kairos_phase_r_tabular_temporal_decision_results.json`
