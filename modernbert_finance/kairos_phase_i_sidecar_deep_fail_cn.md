# Kairos Phase I：深度 Sidecar 失败诊断（线性有信号）

日期：2026-10-01 20:17 CST。

## 一句话

**主因（强）**：`gate≈0.0020195289980620146` 几乎关掉市况 token，且 `head.bias→p≈0.5011777386872409`（应对 prior≈0.25）。
线性 BCE 头在冻结特征上 **Δ=-0.031446**（打赢先验）；深度 sidecar 最佳 Δ=+0.008。
非 tokenizer / 非标签反转 / 非 prior 公式错误。

## 审计要点

### 1) Checkpoint（优化塌缩）

| 项 | best | last |
| --- | ---: | ---: |
| gate | 0.0020195289980620146 | 0.008044314570724964 |
| sigmoid(head.bias) | 0.5011777386872409 | 0.5008390244021114 |
| logit(prior) | -1.0807723798841387 | — |

- 常数 p=0.5 相对 prior 的 Δ≈`0.1276`（init 附近量级）。
- `market = fusion(s1,s2) * gate`，gate 短训后仍 <0.01 → **市况路径实质关闭**。

### 2) 脚本审计

- Loss：BCE-with-logits；**无** pos_weight / label_smoothing / dropout。
- LR=`3e-5`；22 层 fresh ModernBERT；80k samples / 4 segments。
- Eval prior：train_prior≈0.2534 打在 val（pos≈0.2529）— **匹配，非 bug**。
- 标签：`mfe10≥0.10` 现场派生；首 batch `check_labels`；**未反转**。
- `report.json` 的 `best_delta_vs_prior` 在末段未刷新 best 时会写成末次 Δ（已知）。

### 3) Batch / segment 正类率 vs val 25%

| seg | pos_rate | batch_std |
| ---: | ---: | ---: |
| 1 | 0.3064 | 0.1169 |
| 2 | 0.3056 | 0.1174 |
| 3 | 0.2635 | 0.1170 |
| 4 | 0.2301 | 0.1052 |

train_prior=`0.25336`；val_pos=`0.25294`。
前两段≈30.6%（相对 val 偏高）— **弱～中** 贡献，非主因。

### 4) 冻结特征 + 同损失线性头（Phase I-1）

| 设定 | Δ vs prior | beats? |
| --- | ---: | --- |
| BCE linear Base+xsection（bias=logit prior） | -0.031446 | True |
| BCE linear 同特征（bias=0 init） | -0.031439 | True |
| sklearn logistic 同切分 | -0.034130 | True |
| 对照 tok_full logistic | -0.025916 | True |
| 对照 deep sidecar best | +0.008022 | False |

**结论**：同 BCE 损失下线性头可复现打赢先验 → 失败在 **深度优化/门控**，不在损失族或标签。

## 假设强度

- **H1_gate_stuck_near_zero**（strong）：market gate 短训后仍≈0，市况 token 几乎进不了 backbone
- **H2_head_bias_not_prior**（strong）：head.bias≈0 → 默认 p≈0.5，相对 prior≈0.25 的常数先验劣 Δ≈+0.128
- **H3_short_budget_underfit_deep**（medium）：22 层 fresh ModernBERT + LR 3e-5 + 仅 80k 样本，远不够打开门控/学表征
- **H4_segment_pos_rate_shift**（weak_medium）：短预算前两段正类率≈30.6% vs val 25.3%，有偏但不足以单独解释失败
- **H5_label_or_prior_bug**（rejected）：标签反转 / prior 计算错误 / 错列 — 已排除
- **H6_loss_family_blocks_signal**（rejected）：BCE 损失本身不会抹掉线性信号 — 同损失线性头可打赢先验

## 建议下一步

1. **已落地小补丁**（本提交）：`train_mfe10_sidecar.py` 中 `gate` 初值 `1.0`，`head.bias=logit(train_prior)`；并修正 `report.json` 的 `best_delta_vs_prior`。
2. **未**自动推 Kaggle / **未**开长训；仅在你明确授权时再跑 ≤1 segment GPU 冒烟验证补丁。
3. **不要**在 gate≈0 旧设定下加长训；**不要**怪 tokenizer；**不要**动 TPU WIP。

## 用户要点（中文）

- **主因强度**：H1 gate≈0 + H2 bias→0.5 = **强**
- **线性 BCE Δ**：`-0.031446`（打赢先验）
- **深度 sidecar Δ**：`+0.008022`（劣于先验）
- **排除**：标签反转、prior 错配、tokenizer bottleneck、BCE 本身
- **下一步**：补丁已进脚本；等授权再短 GPU 冒烟，否则停
