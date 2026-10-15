# TimesFM 3.0 V11 Clean C06 训练复盘与非有限梯度专题报告

**项目归属**：Kronos / 走势大模型轨迹预测体系  
**受试架构**：TimesFM 3.0 (Transformer) + Rank-16 LoRA + 8 通道输入 (7 个连续通道 + 行业 Embedding) + 幂律加权采样  
**报告范围**：C01 ~ C06 接力训练与 C06 True OOS 评测  
**归档日期**：2026年10月15日（同日修订）  
**实验追踪**：SwanLab Run ID `tfm3v11rank16powerlaw` / `tfm3v11-clean-c06-oos-audit`  
**数据来源**：`finetune/reports/timesfm3_v11_clean_c06_true_oos_report.json`（C06 唯一存档的评测结果）

> **修订说明**：初版报告中的 FP16 下溢、`tanh(20x)` 梯度陡峭两项 NaN 机制与代码不符，已删除；"NRMSE 倒挂已修复""+10.23% 净优势"两项结论与 OOS 数据矛盾，已更正；C05 的 `reserve_seconds` 等工程细节已按构建脚本更正；无回测支撑的实盘收益表述已移除。

---

## 执行摘要

1. **训练收官**：C06 在双卡 Tesla T4 上跑完预算（`budget_seconds=20000`，`reserve_seconds=1200`），全局步数止于 Step 125,209；最佳验证点为 Step 122,500。
2. **验证集（123,839 条）**：10 天方向胜率 63.01%，轨迹相关度 +0.2460，综合分 0.7532，为本体系验证集最高值。验证集用于多次评估间的最优 checkpoint 选择，数值偏乐观。
3. **True OOS（175,428 条）**：10 天方向胜率 57.03%，市场自然上涨率 46.80%，相对"恒猜下跌"的多数类基线（53.20%）高 **3.8 个百分点**。Day 2 起胜率即达 57.3% 并维持到 Day 10。但横截面排序能力（IC）弱于 C02，NRMSE 仍劣于朴素零预测（`skill_nrmse = -0.307`）。
4. **非有限梯度**：现场 Hook 证明前向有限、NaN 首次出现在 ∂L/∂raw_logits，即 `output_head` 之后的非模块运算中。Loss 侧运算梯度均有界，可排除；目前最可能的来源是 TimesFM `cpm_iterative_revin_refine` 中对零方差统计量的 `sqrt` 求导（**待实证**）。跳批机制保证了权重未被污染，C06 跳批率约 0.74%。

---

## 第一章 训练与 OOS 结果

### 1.1 版本对比

验证集均为 123,839 条，OOS 均为 175,428 条。C06 列数据来自存档 JSON；C02 ~ C05 数据仓库内无评测文件存档，沿用训练期记录，**未经复核**。

| 指标 | C02 (Step 45k) | C03 (Step 92.5k) | C04 (Step 95.5k) | C05 (Step 111k) | C06 (Step 122.5k) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 验证集 10 天方向胜率 | 62.38% | 62.26% | 62.36% | 62.80% | **63.01%** |
| 验证集轨迹相关度 | +0.1756 | +0.2253 | +0.2260 | +0.2370 | **+0.2460** |
| 验证集综合分 | 0.7374 | 0.7352 | 0.7366 | 0.7478 | **0.7532** |
| OOS D1~D5 平均胜率 | 54.09% | 54.81% | -- | -- | **56.80%** |
| OOS D10 胜率 | 58.51% | **58.64%** | -- | -- | 57.03% |
| OOS 相对 46.8% 上涨率的差值 | +11.71% | +11.83% | -- | -- | +10.23% |
| OOS 相对多数类基线 (53.2%) 的差值 | +5.31% | +5.44% | -- | -- | +3.83% |
| OOS 日度 IC 均值 | **0.0969** | 0.0653 | -- | -- | 0.0685 |
| OOS 正 IC 交易日占比 | **85.29%** | 70.59% | -- | -- | 70.59% |
| OOS Pooled IC | **0.1600** | 0.1548 | -- | -- | 0.1383 |
| OOS 轨迹相关度 | 0.1050 | **0.1146** | -- | -- | 0.1088 |
| OOS skill_nrmse | -- | -- | -- | -- | **-0.307** |

同一份 JSON 中的历史基线 Kronos-small C2：OOS 方向胜率 54.42%，Pooled IC **0.2428**，日度 IC **0.1796**。C06 方向胜率更高，但横截面排序能力明显弱于该基线。

### 1.2 分日胜率

```
持有期限   C02 OOS     C03 OOS     C06 OOS     C06 - C03
-------------------------------------------------------
Day 1      51.96%      52.55%      54.28%      +1.73%
Day 2      53.65%      54.34%      57.33%      +2.99%
Day 3      54.00%      55.06%      57.40%      +2.34%
Day 4      54.20%      55.68%      57.42%      +1.74%
Day 5      54.68%      56.54%      57.58%      +1.04%
Day 6      55.22%      57.04%      57.48%      +0.44%
Day 7      56.09%      57.41%      57.36%      -0.05%
Day 8      57.23%      57.83%      57.30%      -0.53%
Day 9      57.85%      58.16%      57.11%      -1.05%
Day 10     58.51%      58.64%      57.03%      -1.61%
```

**观察**：C06 的胜率曲线前移，Day 2 ~ Day 10 稳定在 57.0% ~ 57.6%；C02/C03 则随持有期单调上升，Day 10 更高。

**解读时的限制**：

1. **基线**：OOS 期间市场自然上涨率仅 46.8%，恒猜下跌即得 53.2%。前段胜率的提升可能部分来自模型更偏向预测下跌，需补充"预测为涨的比例"与平衡准确率（balanced accuracy）才能区分。
2. **有效样本量**：正 IC 占比 0.7059 对应约 17 或 34 个交易日（JSON 未记录天数）。17.5 万条样本在时间上高度相关，版本间 1 ~ 3 个百分点的差异尚无置信区间支撑。
3. **方向胜率不等于收益**：尚未做含交易成本的收益回测，C06 是否适合 T+2 ~ T+5 持有期策略需由回测决定，本报告不做结论。

---

## 第二章 工程问题记录

### 2.1 全量验证耗时与评估频率

- 全量验证集 123,839 条，`eval_batch_size=512`，共 ⌈123839 / 512⌉ = 242 批。实测每批约 3.95 秒，单次评估约 242 × 3.95 ≈ 956 秒（**约 15.9 分钟**）。
- C05 使用 `eval_every=500`：按约 1.2 step/s，500 步训练约 7 分钟，评估占用约 69% 的墙钟时间。
- C06 恢复 `eval_every=2500`：2500 步训练约 35 分钟，评估占比降至约 31%。

### 2.2 振幅正则：从单向改为双向宽带

- **原问题**：旧版 `amp_loss = F.relu(pred_std - true_std)` 只惩罚预测波动过大，不惩罚过小。模型预测波动收缩（`pred_std ≈ 0.02` vs `true_std ≈ 0.04`），NRMSE 劣于零预测基线。
- **改动**：
  $$\mathcal{L}_{\text{amp}} = \text{ReLU}(\text{pred\_std} - 1.3 \times \text{true\_std}) + \text{ReLU}(0.7 \times \text{true\_std} - \text{pred\_std})$$
  不除以 `true_std`，避免低波动样本放大该项。
- **效果（部分改善，未解决）**：
  - Step 111,338 现场 `pred_std = 0.0292`，`true_std = 0.0427`，比值 0.68，仍处于下界 0.7 附近；
  - `amp_loss_weight = 0.1` 时该项对总损失贡献约 7e-5 量级，约束力很弱；
  - C06 OOS `nrmse = 0.0903`，`naive_nrmse = 0.0691`，`skill_nrmse = -0.307`，**NRMSE 倒挂仍然存在**；
  - 未做消融实验，无法判断该改动对方向胜率的影响。

### 2.3 Kaggle Dataset 路径平铺

- **现象**：上一个 Kernel 的输出打包为 Dataset 后，`relay/relay.json` 可能被平铺为根目录下的 `relay.json`，导致找不到断点而从 Step 0 重启。
- **修复**：`find_relay_state` 同时扫描两种模式，并按 `run_id` 匹配、取 `global_step` 最大者：
  ```python
  candidates = list(root.glob("**/relay/relay.json")) + list(root.glob("**/relay.json"))
  ```

### 2.4 时间预算与安全存档

- Kaggle 超时为强制终止，`finally` 块不保证执行。`train.py` 在 `deadline = budget_seconds - reserve_seconds` 时主动退出循环并执行 `save_state`。
- 实际配置（见 `build/` 下的构建脚本）：

  | Chunk | budget_seconds | reserve_seconds | eval_every |
  | :--- | :---: | :---: | :---: |
  | C05 | 43200 | 600 | 500 |
  | C06 | 20000 | 1200 | 2500 |

  两段均在预算内完成了正常存档。

---

## 第三章 非有限梯度专题

### 3.1 现场记录（Step 111,338）

C06 首次出现非有限梯度时触发了 Two-shot Lazy Hook 诊断，记录如下：

```json
{
  "event": "warning_skip_non_finite_grad",
  "step": 111338,
  "amp_loss": 0.00071242,
  "pred_std": 0.02916159,
  "true_std": 0.04267717,
  "max_abs_pred_ret": 0.28604638,
  "max_abs_true_ret": 0.45021915,
  "nan_count": 400,
  "total_trainable": 401,
  "sec_nan": true,
  "forward_clean": true,
  "backward_clean": false,
  "first_anomaly": {
    "module": "module.output_head",
    "stage": "backward_grad_output",
    "idx": 0,
    "shape": [16, 8, 6, 576]
  }
}
```

可得出的事实：

1. **前向有限**：所有叶子模块的输出均有限，且该批的 loss 本身有限（否则会走 `warning_skip_non_finite_loss` 分支）。注意 Hook 只覆盖叶子模块，`util.revin`、`cpm_iterative_revin_refine` 等函数式运算不在检查范围内，但 loss 有限说明它们的前向输出也有限。
2. **NaN 产生于反向**。
3. **首个异常为 `output_head` 的 `grad_output`**，即 ∂L/∂raw_logits。`register_full_backward_hook` 按反向顺序触发，`output_head` 是前向最后一个叶子模块，其 Hook 最先触发。因此 NaN 产生在 `raw_logits` 到 loss 之间的**非模块运算**中，`output_head` 本身不是来源。
4. **诊断重跑时 LoRA dropout 掩码与原批不同，NaN 仍复现**，说明触发条件由数据决定，而非随机数。

### 3.2 反向路径与排除

TimesFM `src/timesfm3/torch/model.py` 中 `output_head` 之后的路径：

```python
raw_logits = self.output_head(transformer_output)
revin_mean, revin_std = revin_stats
if self.use_iterative_cpm_revin and patch_cpm_mask is not None:
    refined_mu, refined_sigma = cpm_revin_refine_lib.cpm_iterative_revin_refine(raw_logits, ...)
    revin_mean = torch.where(cpm_bvn, refined_mu, revin_mean)
    revin_std = torch.where(cpm_bvn, refined_sigma, revin_std)
revin_logits = util.revin(raw_logits, revin_mean, revin_std, reverse=True)   # x * sigma + mu
clipped_logits = torch.clamp(revin_logits, -self.value_clip, self.value_clip)
```

之后 `train.py` 取通道 0、中位数分位的前 10 步作为 `pred`，进入 `alpha_loss`。

**Loss 侧可排除**：训练全程为 FP32（`train.py` 无 `autocast` / `GradScaler`，TimesFM 统计量显式使用 `float32`），各运算的梯度均有界：

| 运算 | 导数上界 | 能否产生 NaN |
| :--- | :--- | :---: |
| `smooth_l1_loss(beta=0.02)` | 1 | 否 |
| `tanh(20 · pred_ret)` | 20 | 否 |
| `sqrt(var + 1e-7)`（FP32） | 1 / (2√1e-7) ≈ 1581 | 否 |
| `util.revin(reverse=True)` = x·σ + μ | 线性 | 否 |
| `torch.clamp` | 0 或 1 | 否 |

剩下唯一含奇点的运算是 `cpm_iterative_revin_refine` 内部调用的 `util.update_running_stats`。

### 3.3 最可能的来源：CPM 修正中的零方差开方（待实证）

`cpm_iterative_revin_refine` 把模型对预测区间的中位数估计（依赖 `raw_logits`，带梯度）反归一化后，逐 patch 喂给 `update_running_stats` 更新统计量。后者包含两次开方：

```python
inc_sigma = torch.sqrt(inc_var)
new_sigma = torch.sqrt(torch.where(new_n == 0, 0, (n·σ² + inc_n·inc_σ² + n·(μ − new_μ)² + inc_n·(inc_μ − new_μ)²) / new_n))
```

PyTorch 中 `sqrt` 的反向为 `grad / (2 · sqrt(x))`，当 x 恰为 0 时得到 0/0 = NaN（或 g/0 = Inf），且这个 NaN 不会被后续 `torch.where` 屏蔽，因为 CPM 位置选中的正是 `new_sigma` 分支。

**推断的触发条件**：某个样本的某个通道在整个上下文窗口内取值恒定（冻结后的 σ = 0）：

1. 冻结统计量 σ = 0 时，`revin(reverse=True)` 把预测值还原为常数 μ，梯度经 x·σ 被乘以 0；
2. 喂回 `update_running_stats` 的预测 patch 方差为 0，均值等于 μ，`new_sigma` 的开方底数恰为 0；
3. 反向时开方处出现 0/0，NaN 沿 `refined_sigma → revin_logits → raw_logits` 传入 `output_head` 的 `grad_output`。

**关键点**：CPM 修正对全部 8 个通道都会执行。即使 loss 只使用通道 0，其他通道回传的上游梯度为 0，0/0 依然得到 NaN 并汇入共享的 `raw_logits` 梯度。因此任一协变量通道整窗恒定（如长期停牌、成交量为 0、被 `np.clip` 截断成常数、或市值百分位在窗口内不变）都可能触发。

**这仍是推断**：现场未保存该批数据，尚无法确认是哪个通道。若市值百分位等通道频繁整窗恒定，实际跳批率应远高于 0.74%，所以触发条件可能更窄（例如还需叠加 padding 或 mask 的特定组合）。验证方法见 3.6。

### 3.4 全部参数梯度为 NaN 的原因

∂L/∂raw_logits 含 NaN 后，经 `output_head` 回传的 ∂L/∂transformer_output 也含 NaN，再逐层传播到每一层 LoRA 的 `lora_a`/`lora_b`（共 400 个参数），以及输入端的行业 Embedding。因此 `nan_count = 400`、`sec_nan = true` 是链式传导的结果，不代表某个参数本身出了问题。

### 3.5 跳批机制与影响评估

```python
grad = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
if not torch.isfinite(grad):
    ...
    optimizer.zero_grad(set_to_none=True)
    continue
optimizer.step()
```

- 非有限梯度批次不会执行 `optimizer.step()` 与 `scheduler.step()`，**权重与 AdamW 状态均未被污染**；
- C06 统计（训练期日志，仓库内未存档）：14,209 个有效步、106 次跳批，跳批率 106 / (14,209 + 106) ≈ 0.74%；有限梯度范数 p50 = 0.299，p99 = 0.590，最大 0.668；
- **潜在影响**：如果触发条件是数据层面的（如零方差通道），这类样本会被**系统性**地排除在训练之外，推理时模型对同类样本没有学习信号。比例虽小，但不应视为完全无害。

### 3.6 后续行动

1. **已加入的诊断**（`train.py`）：每次跳批记录按模型通道顺序（close, open, high, low, volume, amount, size_percentile）统计的零方差样本数 `zero_std_per_channel` 和 `zero_std_samples`，并把前 32 个跳批批次的 `contexts / targets / sectors` 保存到 `relay/nan_batches/`；
2. **离线复现**：加载保存的批次，在 CPU 上用 `torch.autograd.detect_anomaly()` 重跑前向与反向，确认 NaN 的具体运算和通道；
3. **确认后的修复方向**（均在模型输入侧，不改 loss）：
   - 数据侧：采样时剔除任一通道整窗恒定的样本，或对这些通道加入极小扰动（如 1e-6 量级）；
   - 模型侧：在 `update_running_stats` 的开方底数上加 epsilon（需修改或 monkey-patch TimesFM 源码，影响与预训练行为的一致性，需评估）；
4. **初版报告建议的"Loss 进入 FP32""`pred_std` 的 eps 提到 1e-4"不针对实际路径，不予采纳。**

---

## 第四章 总结

- **已确认**：C06 是本体系验证集指标最高的 checkpoint；OOS 方向胜率曲线较 C02/C03 前移，Day 2 起稳定在约 57%；跳批机制保证了权重不被污染；四项工程问题已处理。
- **未解决**：OOS NRMSE 仍劣于零预测；OOS 横截面 IC 弱于 C02 与 Kronos-small C2 基线；非有限梯度的具体通道与运算待实证。
- **待补充**：OOS 预测涨跌比例与平衡准确率、按交易日的置信区间、含交易成本的收益回测。在这些完成之前，不对 C06 的实盘价值下结论。
