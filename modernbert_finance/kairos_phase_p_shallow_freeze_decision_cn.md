# Kairos 决策浅层冻结 embeds（Phase P）启动

日期：2026-10-01（北京时间）。  
前置：排序 Phase O COMPLETE → 产品枢轴回决策；Phase K identity Δ≈−0.003 未过闸；表格 Logistic Δ≈−0.034。  
Kernel：`user281434/kairos-mfe10-decision-short-phase-p` → **COMPLETE**  
URL：https://www.kaggle.com/code/user281434/kairos-mfe10-decision-short-phase-p  
Commit：`8e6ff68`  
启动：2026-10-01 23:28:44 CST  
SwanLab：`roc_fu/finance` / `kairos-mfe10-decision-short-phase-p-20261001`  
URL：https://swanlab.cn/@roc_fu/finance/runs/kairos-mfe10-decision-short-phase-p-20261001

## 设定

| 项 | 值 |
| --- | --- |
| 标签 | **`y=1{mfe10≥0.10}`**（路径 MFE，非 ctc） |
| backbone | **`shallow`**（`SHALLOW_LAYERS=2`，**非** 22 层） |
| freeze | **`FREEZE_TOKENIZER_EMBEDS=True`**（冻 s1/s2）；浅 bb 可训 |
| loss | **BCE**（单 logit） |
| 初始化 | gate=1；head.bias=`logit(train_prior)` |
| 指标 | Δ vs 常数先验（闸门 ≤ −0.04） |
| 预算 | 4×20 000 / 90min；LR `1e-4` |
| 对照 | K identity Δ≈−0.003；Logistic xsection ≈−0.034 |

## 目的

检验「冻 tokenizer embeds + 浅非 identity 决策头」能否相对 Phase K identity 抬升 Δ，并逼近表格 Logistic。  
**不做**：Ranking IC；22 层；全量 R2；TPU WIP。

## 产物预期

- Kernel logs + `kairos_mfe10_sidecar/report.json`
- SwanLab：`validation/delta_vs_prior` / `gate_passed`
- 本地 launch JSON：`modernbert_finance/ablations/kairos_phase_p_shallow_freeze_decision_launch.json`

## 结果

见 `kairos_phase_p_shallow_freeze_decision_results_cn.md`（COMPLETE；best Δ≈0，未过闸；劣于 K）。
