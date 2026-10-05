# Kairos 决策浅层冻结 embeds（Phase P）结果

日期：2026-10-02（北京时间）。  
Kernel：`user281434/kairos-mfe10-decision-short-phase-p` → **COMPLETE**  
URL：https://www.kaggle.com/code/user281434/kairos-mfe10-decision-short-phase-p  
SwanLab：`roc_fu/finance` / `kairos-mfe10-decision-short-phase-p-20261001`  
URL：https://swanlab.cn/@roc_fu/finance/runs/kairos-mfe10-decision-short-phase-p-20261001

## 一句话

**未过闸；浅层冻结 tokenizer embeds 也失败。** 全程 `gate_passed=false`。best Δ≈**+1.4e−7**（实质为 init 常数先验）；4 段训练全部把 val logloss **推坏**（Δ 落在 +0.006…+0.038）。相对 Phase K identity Δ≈**−0.003** 更差，相对 Logistic Base+xsection Δ≈**−0.034** 差一个数量级以上。  
**结论：短预算下「冻 embeds + 浅 2 层序列头」无法抬升决策 Δ；tokenizer 序列路径短烟测到此可停。下一步换不同角度（表格特征 + 浅 MLP，不依赖 tok 序列）。**

## 运行事实

| 项 | 值 |
| --- | --- |
| 启动 | 2026-10-01 23:28 CST（GPU 2×T4） |
| 训练段结束 | 2026-10-01 23:35:50 CST（UTC 15:35:50） |
| 设定 | `FREEZE_TOKENIZER_EMBEDS=True`，`BACKBONE_MODE=shallow(2)`，BCE，gate=1，bias=logit(prior)，LR `1e-4` |
| 参数 | trainable 24 / frozen 3；numel≈10.7M / 1.57M |
| 预算 | 4×20 000；stop_reason=`segment_limit`；processed=80 000 |
| train prior | 0.25336 |
| val n / 正类率 | 123 836 / **0.25294** |
| prior_ll | **0.565542** |

## 闸门对照

| 指标 | 常数先验 | init ★best | seg4 | 闸门 |
| --- | ---: | ---: | ---: | ---: |
| val log_loss | **0.565542** | **0.565542** | 0.592594 | — |
| Δ vs prior | 0 | **+1.38e−7** | +0.027052 | 需 **≤ −0.04** |
| gate_passed | — | **false** | **false** | 全程 0 |

## 分段验证（all）

| seg | 时间 CST | model_ll | prior_ll | Δ | gate |
| ---: | --- | ---: | ---: | ---: | --- |
| init ★best | 23:31:23 | 0.565542 | 0.565542 | **+1.38e−7** | false |
| 1 | 23:32:30 | 0.571545 | 0.565542 | +0.006003 | false |
| 2 | 23:33:36 | 0.591569 | 0.565542 | +0.026027 | false |
| 3 | 23:34:44 | 0.603787 | 0.565542 | +0.038245 | false |
| 4 | 23:35:50 | 0.592594 | 0.565542 | +0.027052 | false |

## 对比摘要

| 跑次 | best Δ vs prior | gate_passed |
| --- | ---: | --- |
| Phase I/J（22 层 sidecar） | ≈0 | false |
| Phase K（identity） | **−0.00297** | false |
| **Phase P（浅冻 embeds，本跑）** | **≈0（init）** | **false** |
| tok_full logistic | ≈−0.026 | false |
| 线性 BCE 冻结特征 | ≈−0.031 | false |
| Logistic Base+xsection | ≈**−0.034** | false |
| 闸门 | ≤ **−0.04** | — |

## 判读

- 浅非 identity 头 + 冻 tokenizer embeds **未能**相对 K 抬升 Δ；训练过程单调恶化，best 停在先验。
- 与表格/线性探针（−0.03 量级）对照，强化「短预算序列 ModernBERT 路径对 mfe10 二分类无效」。
- **不做**：更深/更长序列二分类；22 层；Ranking 产品；TPU WIP。

## 下一步（Phase Q，不同角度）

- **表格 Base+xsection → 浅 MLP BCE**（**无** tokenizer 序列），目标仍 `y=1{mfe10≥0.10}`，报 Δ / gate。
- 对标 Logistic −0.034；冲 −0.04 闸门。
- 见 `kairos_phase_q_tabular_shallow_mlp_decision_cn.md`。

## 产物

- Kaggle logs → `/workspace/kaggle_kairos_mfe10_decision_phase_p_out/report_from_logs.json`
- JSON：`modernbert_finance/ablations/kairos_phase_p_shallow_freeze_decision_results.json`
- Launch 状态已改为 COMPLETE。
