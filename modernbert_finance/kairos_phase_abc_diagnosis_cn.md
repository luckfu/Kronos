# Kairos Phase A/B/C 诊断结论（为何卡在 ~0.5966）

日期：2026-10-01 17:15 CST。

## 一句话结论

**根因：`deep_model_collapsed_to_constant_train_prior`。** R2 验证 macro LL≈`0.5966` 与「训练正例率常数先验打在验证集上」的 macro LL=`0.596610` 相差仅 `0.000010` —— 深度模型没有学到超越边际/先验的条件信号。

## Phase A — 标签 / 时间 / 切分

### 1. Feature cutoff vs 10D horizon

```
{
  "lookback": 120,
  "horizon": 10,
  "source_window": 131,
  "asof_position_in_window": 119,
  "feature_indices": "[start, start+120) inclusive of asof bar",
  "label_future_indices": "[asof+1, asof+10] = relative [120, 130)",
  "mfe10": "max(high[T+1:T+10]) / close[T] - 1",
  "mae10": "min(low[T+1:T+10]) / close[T] - 1",
  "up_k": "1 if mfe10 >= k/100",
  "down_k": "1 if -mae10 >= k/100",
  "leakage_risk_if_broken": "Any feature using bars after asof, or labels using bars at/before asof, would be leakage. Dataset history slice is values[start:start+LOOKBACK]."
}
```

- 对齐抽查：`ok=True`
- 时间完整性（未来污染探针）：`ok=True`；特征在污染未来后不变 `256/256`，标签改变 `256/256`
- asof 收益 vs mfe10 相关：`0.14173075967546084`（弱相关预期；近 1 才像泄漏）

### 2. 正例率漂移

| Head | train | val | 2025H2 | 2026H1 | val−train |
| --- | ---: | ---: | ---: | ---: | ---: |
| up_003 | 0.6469 | 0.6614 | 0.6758 | 0.6441 | +0.0145 |
| up_005 | 0.4944 | 0.4983 | 0.5044 | 0.4908 | +0.0040 |
| up_008 | 0.3311 | 0.3292 | 0.3263 | 0.3326 | -0.0019 |
| up_012 | 0.1958 | 0.1971 | 0.1879 | 0.2078 | +0.0013 |
| down_003 | 0.6487 | 0.6491 | 0.5710 | 0.7291 | +0.0004 |
| down_005 | 0.4744 | 0.4747 | 0.3718 | 0.5799 | +0.0003 |
| down_008 | 0.2838 | 0.2828 | 0.1806 | 0.3870 | -0.0010 |
| down_012 | 0.1386 | 0.1229 | 0.0570 | 0.1888 | -0.0157 |

Macro prior LL：train=`0.601361`，val=`0.596412`，2025H2=`0.558941`，2026H1=`0.613429`。

**训练先验 → 验证打分 macro LL = `0.596610`** （与 R2 卡住点重合）。

### 3. 八头相关

- within-up mean |ρ| ≈ `0.579`
- within-down mean |ρ| ≈ `0.514`
- up vs down mean ρ ≈ `-0.264`
- Adjacent thresholds are highly correlated (~0.7); equal-weight multi-head loss double-counts nested exceedance events and can dilute rare-tail heads.

### 4. 同股未来信息

Dataset `history = values[start:start+LOOKBACK]`，asof=`start+119`；标签只用 asof 之后 10 根。未来污染实验表明特征不含同股未来 bar。**不是泄漏问题。**

## Phase B — 结构消融（CPU）

- 8-head 独立 logistic macro Δ(model−prior) = `-0.01957`
- MultiOutput 共享特征 macro Δ = `-0.01990`
- 单头 up_005 Δ = `-0.01711`
- 粗标签 Δ：`{"any_up_threshold": -0.01099839391599422, "up_005_only": -0.017109405034147707, "down_005_only": -0.02452610717934911, "direction_up_minus_down": -0.002148553886063609, "signed_move": -0.0014996362961579024}`
- 冻结 embedding 探针：`No local Kairos/ModernBERT checkpoint under scratch/finetune; skip freeze-backbone linear probe (would require HF/Kaggle download + train).`

## Phase C — 决策（无谄媚）

| 项 | 值 |
| --- | --- |
| 根因 | `deep_model_collapsed_to_constant_train_prior` |
| 多头是否主 blockers | `False` |
| 放弃同配置长训 | `True` |
| 泄漏主因假设 | 放弃=`True` |

### 建议下一步（每条一句）

1. **改什么：** 不要重启 R2 / 不要在同一 8-head BCE+同一特征上继续堆 GPU 小时
   - 证据：R2≈0.5966 matches train-prior-on-val macro LL=0.596610 (val self-prior=0.596412); collapse to constant prevalence.
   - 下一实验：训练日志必须显式打印常数先验基线；若 1–2 个 chunk 后仍落在先验 ±1e-3 内则自动停训。

2. **改什么：** 不要把泄漏/错位当作主修方向
   - 证据：temporal_integrity.ok=True, alignment.ok=True; future-corrupt leaves features invariant and changes labels.
   - 下一实验：把因果合同测试留在 CI；精力转到标签信息量与特征内容，而非管线臆测。

3. **改什么：** 八头稀释是次要问题——可减到 2–4 头，但先修信号路径
   - 证据：8-head independent macro Δ=-0.0196; best coarsened down_005_only Δ=-0.0245 (similar order).
   - 下一实验：在同样窗口汇总特征上先做连续 mfe10/mae10（或 first-touch）回归，再考虑 Transformer。

4. **改什么：** 特征侧：当前 OHLCVA 汇总相对先验仅 Δ≈0.008–0.025
   - 证据：Cheap logistic on summarized windows beats prior on all 8 heads, but effect sizes are small; deep model failed to harvest even that.
   - 下一实验：加入截面排序/行业相对收益等便宜特征重跑 logistic；若 Δ 仍 <0.02，先改 10D MFE/MAE 标签定义再谈 ModernBERT。

## 证据强度

| 主张 | 强度 | 依据 |
| --- | --- | --- |
| 模型塌到训练先验 | **强** | 数值与 train-prior-on-val 差 <1e-3 |
| 非泄漏/错位 | **强** | 对齐+未来污染探针通过 |
| 标签有弱可学信号 | **中** | logistic Δ≈0.008–0.019 |
| 八头稀释是主因 | **弱/否** | 粗标签与八头同量级 |
| 值得再上 ModernBERT 长训 | **弱** | 深度模型未吃到线性已有的小信号 |

未重启 R2；未改动同事 TPU WIP。
