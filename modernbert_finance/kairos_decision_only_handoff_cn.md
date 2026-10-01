# Kairos 决策系统交接（Decision-Only Charter）

日期：2026-10-01（北京时间）。  
状态：**产品路径 = 决策标签 only**；排序探针 Phase M/N/O 仅作诊断，**已停作产品**。

## 一句话

下一步一切实验围绕 **`y = 1{mfe10 ≥ 0.10}`**（路径最高价触及 +10%/10 日），用 **Δ logloss vs 常数先验** 判闸；**禁止**再把 Ranking IC 当产品。

## 标签契约（必须遵守）

| 项 | 值 |
| --- | --- |
| mfe10 | `max(high[T+1:T+10]) / close[T] - 1` |
| 标签 | `y = 1{mfe10 ≥ 0.10}` |
| 正类率（val） | ≈ **0.2529**（~25%） |
| 常数先验 logloss | ≈ **0.56555** |
| 闸门 | `Δ = model_ll − prior_ll ≤ −0.04` |
| 早停 | 连续 2 次 eval `|model_ll − prior_ll| ≤ 1e-3` → `stuck_at_constant_prior` |

**不是**：close-to-close `fwd_ret_10`；旧 8 头 R2；Rank IC 产品化。

## 已验证诊断摘要

| 路径 | 结果 | 产品含义 |
| --- | ---: | --- |
| Phase I/J 22 层 sidecar | Δ≈0 | 短预算随机深栈无效 |
| Phase K identity 二分类 | best Δ≈**−0.003** | 首次负 Δ，远未过闸 |
| Logistic Base+xsection | Δ≈**−0.034** | 最接近闸门的表格基线 |
| 线性 BCE 冻结特征 | Δ≈**−0.031** | 同上量级 |
| Phase L Ridge（排序） | Rank IC≈**0.27** | 排序信号在表格侧；**非决策产品** |
| Phase M/N neural 排序 | Rank IC≈**0.08** | 诊断：短预算 neural≪Ridge |
| Phase O 浅冻 embeds 排序 | COMPLETE | **停产品化**；见 stop 笔记 |

## 下一实验原则（Decision-Only）

1. **只**训决策头：BCE / 校准 / 浅头 / 特征增强，目标 = 抬升 **Δ vs prior**。
2. 固定 **gate=1**、**head bias = logit(train_prior)**（避免先验坍塌）。
3. 优先廉价：`freeze tok embeds + shallow(≤4)` 或 `feature→logistic→浅 neural`；短 Kaggle（≤4×20k）或本地 CPU 表格。
4. 对标：先逼近/超过 Logistic Δ≈−0.034，再冲 −0.04 闸门。
5. **禁止**：Ranking IC 作主指标；22 层长训；全量 R2；TPU WIP。

## 当前下一步（已启动）

- **Phase P**：`FREEZE_TOKENIZER_EMBEDS=True` + `BACKBONE_MODE=shallow(2)` + BCE，`y=1{mfe10≥0.10}`。  
- Kernel：`user281434/kairos-mfe10-decision-short-phase-p`  
- 详见 `kairos_phase_p_shallow_freeze_decision_cn.md`

## 代码入口

| 项 | 路径 |
| --- | --- |
| 标签助手 | `modernbert_finance/mfe10_sidecar.py` |
| 训练脚本 | `finetune/kaggle_kairos_mfe10_sidecar/train_mfe10_sidecar.py` |
| Builder / Stage | `finetune/build_kairos_mfe10_sidecar.py` / `stage_kairos_mfe10_sidecar.py` |
| 本交接 | `modernbert_finance/kairos_decision_only_handoff_cn.md` |
| Phase O 停记 | `modernbert_finance/kairos_phase_o_ranking_stop_decision_pivot_cn.md` |

## 明确不做

1. 不重启 ranking Phase / Rank IC 产品实验。  
2. 不重启 `kairos-r2-r1-restart-lr-3e-5` / 旧 8 头。  
3. 不提交同事 TPU WIP（`finetune/kaggle_beta_v21_c1*`、`tpu_train_entry.py` 等 dirty）。  
4. 不开全量 R2 多 chunk relay。
