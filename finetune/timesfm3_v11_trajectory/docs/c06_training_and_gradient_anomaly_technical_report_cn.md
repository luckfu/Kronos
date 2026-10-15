# TimesFM 3.0 V11 Clean C06 训练复盘与脏梯度第一性原理专题技术报告

**项目归属**：Kronos / 走势大模型轨迹预测体系  
**受试架构**：TimesFM 3.0 (Transformer) + Rank-16 LoRA + 8 通道扩展 (行业 Embedding) + 幂律加权采样  
**报告周期**：C01 ~ C06 长周期接力训练与 True OOS 盲测终极大考  
**归档日期**：2026年10月15日  
**实验追踪**：SwanLab Run ID `tfm3v11rank16powerlaw` / `tfm3v11-clean-c06-oos-audit`  

---

## 执行摘要（Executive Summary）

1. **训练闭环与历史突破**：
   - TimesFM 3.0 V11 Clean C06 在双卡 Tesla T4 上圆满跑满预算，全局步数顺利收官于 **Step 125,209**。
   - 在 Step 122,500 验证集（123,839 条样本）上，未来 10 天方向准确率是 **63.01%**，轨迹相关是 **+0.2460**（表内 C02 为 +0.1756），综合分是 **0.7532**。这三项高于表内前几轮。平方误差技能仍为负，见 §2.2。
2. **True OOS 盲测（175,428 条封箱样本）方向指标**（`finetune/reports/timesfm3_v11_clean_c06_true_oos_report.json`）：
   - 逐日方向准确率：Day 2 为 **57.33%**，Day 5 为 **57.58%**，Day 10 为 **57.03%**。Day 1 为 54.28%；Day 2 至 Day 10 落在 57.03%～57.58%。
   - Day 10 为 57.03%，低于本报告表内 C03 的 58.64%。
   - 日度方向准确率是全样本符号一致率。这次评测没有交易成本、涨跌停与流动性约束、标的筛选或仓位，所以它只是一个评测指标。
3. **“脏梯度（非有限梯度 / NaN）” 目前能站住的结论，以及尚未证实的部分**：
   - Step 111,338 的 Two-shot Lazy Hook：`forward_clean: true`。叶子模块的前向输出是有限的。
   - 首个异常模块是 `output_head` 的 `backward_grad_output`。`output_head` 是反传路上**第一个接到脏 `backward_grad_output` 的模块，不是根因**。
   - RevIN 方差开方为零、`pred_std` 的 `1e-7` 在 FP16 下下溢、以及 `tanh(20x)`，都只是**候选假说，有待实验**。饱和区里 `tanh` 的导数趋于 0，不会剧烈震荡。下一步是 `autograd.grad(loss, pred)`，用来判断非有限梯度是否在进入网络之前就在 `alpha_loss` 里形成。
   - `zero_grad(set_to_none=True); continue` 会跳过 `optimizer.step()`，因此这批脏梯度没有写入权重。C06 日志中的跳批率约 0.74%。

---

## 第一章 全周期训练战果与 True OOS 形态演化复盘

### 1.1 历代版本演进全景对比表

我们对从 C02 到 C06 的全周期接力成果建立同口径精密对比（验证集均为 123,839 条样本，True OOS 均为 175,428 条样本）：

| 评估维度与指标 | C02 (Step 45k) | C03 (Step 92.5k) | C04 (Step 95.5k) | C05 (Step 111k) | **C06 (Step 122.5k)** | 演进特征与客观评定 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **【验证集】10天方向准确率 (DirAcc)** | 62.38% | 62.26% | 62.36% | 62.80% | **63.01%** | 表内最高；C03 曾低于 C02，不是全程单调 |
| **【验证集】波形同频度 (TrajCorr)** | +0.1756 | +0.2253 | +0.2260 | +0.2370 | **+0.2460** | 较表内 C02 的 +0.1756 约高 40% |
| **【验证集】走势综合评分 (Score)** | 0.7374 | 0.7352 | 0.7366 | 0.7478 | **0.7532** | 表内最高；中间轮次有过回落 |
| **【OOS 盲测】样本测试规模** | 175,428 条 | 175,428 条 | *(直接接力未单测)* | *(直接接力未单测)* | **175,428 条** | 100% 绝对封箱未见交易日 |
| **【OOS 盲测】D1~D5 均值方向准确率**| 54.09% | 54.81% | -- | -- | **56.80%** | 高于表内 C02 / C03；仍是符号一致率，不是策略收益 |
| **【OOS 盲测】D10 方向准确率** | 58.51% | **58.64%** | -- | -- | **57.03%** | 低于表内 C03 的 58.64% |
| **【OOS 盲测】相对自然上涨率的方向差** | +11.71% | +11.83% | -- | -- | **+10.23%** | C06 为 `alpha_win_rate_edge`；自然上涨率约 46.8%。这不是扣费后的策略优势 |
| **【OOS 盲测】日度 IC 均值** | 0.0969 | 0.0653 | -- | -- | **0.0685** | C06 为 0.0685；高于表内 C03，低于 C02 |
| **【OOS 盲测】正 IC 交易日占比** | 85.29% | 70.59% | -- | -- | **70.59%** | 超七成真实交易日横截面排序为正 |
| **【OOS 盲测】全样本 Pooled IC** | 0.1600 | 0.1548 | -- | -- | **0.1383** | C06 为 0.1383，仍为正，低于表内 C02 / C03 |
| **【OOS 盲测】波形轨迹同频度** | 0.1050 | 0.1146 | -- | -- | **0.1088** | 稳定在 0.11 左右 |

---

### 1.2 分日方向准确率（Day 1 ~ Day 10）

下表是 C02、C03 与 C06 的逐日 OOS 方向准确率。C06 一列与 `finetune/reports/timesfm3_v11_clean_c06_true_oos_report.json` 的 `daily_horizon_accs` 一致。它衡量预测符号与真实符号是否一致，**不是**扣费后的策略净值，也没有涨跌停、选股或仓位。

对比 C02、C03 与 C06 的逐日 OOS 方向准确率：

```
Horizon     C02            C03            C06             C06 - C03
--------------------------------------------------------------------
Day 1       51.96%         52.55%         54.28%          +1.73%
Day 2       53.65%         54.34%         57.33%          +2.99%
Day 3       54.00%         55.06%         57.40%          +2.34%
Day 4       54.20%         55.68%         57.42%          +1.74%
Day 5       54.68%         56.54%         57.58%          +1.04%
Day 6       55.22%         57.04%         57.48%          +0.44%
Day 7       56.09%         57.41%         57.36%          -0.05%
Day 8       57.23%         57.83%         57.30%          -0.53%
Day 9       57.85%         58.16%         57.11%          -1.05%
Day 10      58.51%         58.64%         57.03%          -1.61%
```

**读数说明**：
1. C06 的 Day 2 方向准确率为 **57.33%**，Day 5 为 **57.58%**（十日中的最高点），Day 10 为 **57.03%**。
2. Day 1 为 **54.28%**。Day 2 至 Day 10 落在 **57.03%～57.58%**。Day 10 为 57.03%，低于表内 C03 的 58.64%。
3. 上表是符号一致率。评测没有成本、涨跌停与流动性约束、标的筛选和仓位。方向和轨迹相关可以单独引用；平方误差缺口见 §2.2，仍然为负。

---

## 第二章 关键工程阻断与优化避坑全记录

在约 12.5 万步的 Kaggle 双卡 T4 接力训练里，下面四件事有日志或代码可以核对。振幅正则改过公式，但没有把 `skill_nrmse` 修成正数。

### 2.1 硬件物理瓶颈：约 15.9 分钟全量大考与步频优化

- **现象与危机**：
  在前期测试中，若设置 `--eval-every 500`，训练经常长时间“假死”。
- **第一性原理测算**：
  全量验证集有 123,839 条样本。在双卡 T4 上，即使使用最大安全显存批次 `batch_size=512`（每卡 256），单次前向推断仍需处理：
  $$\text{Batches} = \lceil 123839 / 512 \rceil = 242 \text{ 批}$$
  实测每批前向+指标累加耗时 3.95 秒，单次大考耗时恒定为：
  $$242 \times 3.95\text{s} = 955.9\text{s} \approx \mathbf{15.9 \text{ 分钟}}$$
  若 500 步一验，纯训练仅需约 7 分钟，大考占用了 68.8% 的时间，GPU 被迫陷入漫长的推断循环。
- **解决方案**：
  严格恢复 `--eval-every 2500`。单次 2500 步训练耗时约 36 分钟，大考耗时占比压低至 30% 以下，纯训练吞吐恢复至 1.15~1.23 step/s，确保了余弦退火有效步数的充分推进。

### 2.2 振幅正则：旧公式，以及仍然为负的 skill_nrmse

- **现象**：
  方向准确率可以升到 60% 以上，平方误差技能仍然为负：预测比“未来收益恒为 0”的朴素基线更差（`skill_nrmse < 0`）。
- **旧公式（已核对代码历史）**：
  双向宽带之前，振幅项是只罚上沿的
  `amp_loss = F.relu(pred_std - 1.3 * true_std)`，
  **不是** `F.relu(pred_std - true_std)`。它不惩罚 `pred_std` 低于 `true_std` 的收缩。
- **C06 使用的形式**：
  后来加上了下沿，成为双向宽带：
  $$\mathcal{L}_{\text{amp}} = \text{ReLU}(\text{pred\_std} - 1.3 \times \text{true\_std}) + \text{ReLU}(0.7 \times \text{true\_std} - \text{pred\_std})$$
  该项不除以 `true_std`。
- **这项正则没有把平方误差补到健康区间**：
  - C06 验证集 `skill_nrmse` 仍为负，大约在 **-0.08 到 -0.13**。
  - True OOS（`finetune/reports/timesfm3_v11_clean_c06_true_oos_report.json`）`skill_nrmse` 约为 **-0.307**（`nrmse` 0.090，相对 `naive_nrmse` 0.069）。
  - Step 122,500 验证集上的方向准确率 **63.01%**、轨迹相关 **+0.2460** 可以保留。平方误差缺口没有解决，不能写成“方差已回到健康带，因此 NRMSE 被修好”。

### 2.3 Kaggle 接力文件路径平铺陷阱

- **现象**：
  在 C05 接力 C04 时，脚本偶发找不到上游断点而退化回 Step 0 重新初始化。
- **根因分析**：
  当把上一个 Kernel 的 output 打包为 Kaggle Dataset 时，Kaggle 的后台解压机制会去除层级目录，将原 `/kaggle/working/relay/relay.json` 平铺为根目录下的 `/kaggle/input/.../relay.json`。
- **修复方案**：
  重构 `find_relay_state` 扫描函数，采用多模式通配扫描：
  ```python
  candidates = list(root.glob("**/relay/relay.json")) + list(root.glob("**/relay.json"))
  ```
  通过解析 `run_id` 与比对 `global_step` 最大值，彻底根除了断点失联隐患。

### 2.4 配额控制与优雅安全存档机制

- **机制设计**：
  Kaggle 单实例有严格的 12 小时硬件超时强杀（SIGKILL）。强杀时不会给脚本执行 `finally` 块或保存状态的机会。
- **防御部署**：
  在 `train.py` 中设置 `deadline = budget_seconds - reserve_seconds`（`reserve_seconds = 1200`，即 20 分钟安全裕量）。
  一旦时间到达截止线，循环主动跳出并执行完整的 `save_state` 与安全落盘；C05 与 C06 正是依托该机制在配额耗尽前完成了 100% 完整的断点封箱。

---

## 第三章 “脏梯度（非有限梯度 / NaN）”：已观测事实与待验证假说

### 3.1 现场第一手确凿证据复原（Step 111,338）

在 C06 训练至 Step 111,338 时，系统首次触发了预埋的 **Two-shot Lazy Hook 深度诊断**。现场记录的原始 JSON 事件如下：

```json
{
  "time": "2026-10-14T14:51:55.110949+00:00",
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

从该现场数据可以直接读出三点：
1. **`forward_clean: true`**：这次重放里，叶子模块的前向输出是有限的。Hook 检查的是模块输出，不是注意力内部每一个 QK 中间值；就这次记录而言，发散不是“前向模块输出已经是 NaN”。
2. **`backward_clean: false`**：非有限值出现在反向传播。
3. **`first_anomaly` 落在 `module.output_head` 的 `backward_grad_output`**：张量形状为 `[16, 8, 6, 576]`（16 是批次，8 是通道，6 是 Patch，576 为 $64 \times 9$ 预测输出）。这是第一个接到脏梯度的模块，不是根因。

---

### 3.2 因果链条严格定性：为什么 `output_head` 是受害者而非根因

在 PyTorch 的 Autograd 反向传播机制中，必须严格区分两个 Hook 阶段：
- **`backward_grad_input`**：表示损失函数反传流经当前模块后，当前模块对其**输入张量**计算出的梯度（即向下游层传递的梯度 $\frac{\partial \mathcal{L}}{\partial X}$）；
- **`backward_grad_output`**：表示当前模块的**输出端**接收到的上级反传梯度（即来自损失函数或后级模块传递过来的初始梯度 $\frac{\partial \mathcal{L}}{\partial Y}$）。

深入查阅 TimesFM 3.0 的官方模型源码（`src/timesfm3/torch/model.py` 第 324~353 行）：
```python
# 1. 骨干网络的最后一层 Linear 投影
raw_logits = self.output_head(transformer_output)  # 形状: (b, v, n, 576)

# 2. 输出端后处理: RevIN 反向反归一化与 CPM 迭代校准
refined_mu, refined_sigma = cpm_iterative_revin_refine(raw_logits, ...)
revin_logits = util.revin(raw_logits, revin_mean, revin_std, reverse=True)
clipped_logits = torch.clamp(revin_logits, -self.value_clip, self.value_clip)
final_logits = clipped_logits.view(...)
```

在反向传播时，信息流逆向流动：
$$\text{Loss} \to \text{alpha\_loss} \to \text{clipped\_logits} \to \text{revin\_logits} \to \text{raw\_logits} \to \mathbf{output\_head}$$

**严密因果判定**：
- Hook 捕获到异常的阶段正是 `output_head` 的 `backward_grad_output`（即 $\frac{\partial \mathcal{L}}{\partial \text{raw\_logits}}$）。
- 这表明：**在梯度尚未进入 `output_head` 权重参数进行矩阵乘法之前，这一股反向梯度流从外部灌进来时就已经变成了包含 NaN 的脏张量！**
- `output_head` 是反传链路上从 Loss 往回走最先碰到的第一个神经网络模块，它只是**第一个接到脏梯度的“受害者”**，结构本身没有任何设计缺陷。

---

### 3.3 策源地尚未定位：三条候选假说，下一步用 `autograd.grad(loss, pred)` 判别

Hook 只说明：非有限值在到达 `output_head` 的 `backward_grad_output` 时已经存在。`output_head` 和损失之间还有 clamp、RevIN reverse，以及 `alpha_loss` 内部运算。Module hook 盖不住这些算子，所以下面三条都**不是已证实根因**，只是待实验的候选假说。

**下一步实验**：在同一次非有限反传上，当 `pred` 仍在计算图里时计算 `autograd.grad(loss, pred)`，判断非有限梯度是不是在 `alpha_loss` 里、进入网络之前就形成。

- 若 \(\partial \mathcal{L} / \partial \text{pred}\) 已经非有限：问题在 loss 侧（假说 2、3 所在的 `alpha_loss`）。
- 若 \(\partial \mathcal{L} / \partial \text{pred}\) 有限，而 `output_head` 的 `backward_grad_output` 非有限：问题在 `pred` 与 `output_head` 之间的后处理（假说 1 所在的 RevIN / clamp 一段）。

```
+-----------------------------------------------------------------------------------------+
|                     已观测边界 vs 尚未证实的候选位置                                      |
|                                                                                         |
|  [output_head]                                                                          |
|       ^                                                                                 |
|       |  已观测: backward_grad_output 非有限（第一个接到脏梯度的模块，不是根因）            |
|       |                                                                                 |
|  [util.revin(reverse=True) / clamp]  <--- 候选假说 1: sqrt 在方差为 0 时反传分母为 0      |
|       ^                                                                                 |
|  [pred]   <--- 下一步实验切在这里: autograd.grad(loss, pred)                             |
|       ^                                                                                 |
|  [alpha_loss]  <--- 候选假说 2: pred_std 的 +1e-7 在 FP16 下溢                           |
|                    候选假说 3: tanh(20x)；饱和区导数趋于 0，不是剧烈震荡                    |
+-----------------------------------------------------------------------------------------+
```

#### 候选假说 1：RevIN 统计量开方在方差为零时的反传奇点（未证实）
`util.py` 的 `update_running_stats` 里有：
$$\text{inc\_sigma} = \sqrt{\text{inc\_var}}, \quad \frac{d}{dx}\sqrt{x} = \frac{1}{2\sqrt{x}}$$
若增量方差走到 0，前向 \(\sqrt{0}=0\) 仍是有限数，反向分母会变成 0。这可以产生 NaN，但 Step 111,338 的 Hook **没有**把异常定位到这个算子。在 `autograd.grad(loss, pred)` 给出结果之前，它只是候选。

#### 候选假说 2：`alpha_loss` 里 `pred_std` 的 `+ 1e-7` 在 FP16 下溢（未证实）
`train.py` 里当前写法是：
```python
pred_std = torch.sqrt(torch.var(pred_ret, dim=-1, unbiased=False) + 1e-7).mean()
```
FP16 的最小正正规数约为 \(2^{-14} \approx 6.1 \times 10^{-5}\)，最小非正规数约为 \(2^{-24} \approx 5.96 \times 10^{-8}\)。`1e-7` 靠近半精度下沿，反传时有下溢的可能。这次 Hook 没有单独测量该分母，所以不能写成已经发生的根因。

#### 候选假说 3：`tanh(20x)` 不是“饱和区剧烈震荡”
`train.py` 中方向项使用：
$$\text{soft\_sign} = \tanh(20 \cdot \text{pred\_ret})$$
导数为：
$$\frac{d}{dx} \tanh(20 x) = 20 \times (1 - \tanh^2(20 x))$$
- 在 \(x = 0\) 附近斜率最大，峰值是 20。
- 在饱和区 \(|20x|\) 很大时，\(\tanh(20x) \to \pm 1\)，**导数趋于 0**。梯度被压灭，不会剧烈震荡。

因此原先“饱和区陡峭震荡、并与开方导数叠加后形成非有限梯度”的写法不成立。`tanh(20x)` 至多仍是一条待检验的候选，而且饱和区的行为和“震荡”相反。

---

### 3.4 链式法则“全网染色”现象剖析

用户曾提出关键疑问：**“为什么日志显示 400 个 LoRA 参数和 1 个行业 Embedding 全部染上 NaN？行业 Embedding 是否失效了？”**

答案是：**这是反向传播中链式法则（Chain Rule）的标准传导现象，绝不代表底层行业嵌入失效**。

1. **反传染色机制**：
   损失函数对任何一层权重参数 $W$ 的梯度计算公式为：
   $$\frac{\partial \mathcal{L}}{\partial W_l} = X_l^T \cdot \delta_l, \quad \text{其中 } \delta_l = \delta_{l+1} W_{l+1}^T \odot f'(Z_l)$$
   由于顶端 $\delta_{\text{output\_head}} = \text{NaN}$：
   $$\text{NaN} \times \text{任意有限矩阵} = \mathbf{NaN}$$
   这个非有限标量像墨水一样，顺着反向图一路向下倒灌：
   $$\delta_{\text{output\_head}} \to \text{Layer 23} \to \text{Layer 22} \to \dots \to \text{Layer 0 (400个LoRA)} \to \mathbf{sector\_embedding}$$
   因此在检查每个参数的 `.grad` 时，整张反传树上的所有可训练参数无一幸免，全部呈现为 NaN（`nan_count: 400/401, sec_nan: True`）。
2. **权重的绝对纯净性**：
   在 `train.py` 中，我们的防御逻辑是：
   ```python
   grad = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
   if not torch.isfinite(grad):
       log("warning_skip_non_finite_grad", ...)
       optimizer.zero_grad(set_to_none=True)  # 彻底清空所有参数的梯度为 None
       continue                                # 直接跳过当前批次，严禁 optimizer.step()
   optimizer.step()
   ```
   **结论**：`optimizer.step()` 从未在脏梯度状态下被执行过！受污染的梯度在第 766 行被 `set_to_none=True` 完全物理擦除，**模型权重从始至终从未被沾染哪怕一次，行业 Embedding 与 400 个 LoRA 参数处于 100% 绝对健康状态**。

---

### 3.5 当前跳批防御机制的严密性统计（无害性实证）

我们对 C06 运行日志（14,209 步）进行了完整的事件统计抽检：

- **有效总步数**：14,209 步
- **非有限梯度跳批总次数**：106 批
- **真实跳批率**：
  $$\text{Skip Rate} = \frac{106}{14209} \approx \mathbf{0.74\%}$$
- **梯度范数分位数分布（正常训练步）**：
  - 中位数（p50）：`0.299`
  - 99分位数（p99）：`0.590`
  - 最大有限梯度：`0.668`

**读数**：
在有限梯度的步上，梯度范数中位数约 0.30，p99 约 0.59。跳批率约 0.74%。`zero_grad` 之后不调用 `optimizer.step()`，所以这些批次没有把非有限梯度写进权重。跳批是否改变了收敛路径，这次日志没有单独测量。

---

### 3.6 定位实验之前，不要把数值改动当成已证实修复

先做 §3.3 的 `autograd.grad(loss, pred)`。在它分出 loss 侧和后处理侧之前，下面两段只是**未验证的候选缓解**，不是根因修复，本文也不要求改训练代码。

1. **若实验表明 loss 侧在 FP16 下出问题，再考虑 FP32 隔离**：
   ```python
   with torch.cuda.amp.autocast(enabled=False):
       loss, parts = alpha_loss(pred.float(), t.float(), args.dir_loss_weight, args.amp_loss_weight)
   ```
2. **若实验表明方差开方的底数过小，再考虑抬高下界**（例如把 `+ 1e-7` 换成更大的正常数，或对方差做 `clamp_min`）。在假说 2 被证实之前，这不是应立即落地的补丁。

---

## 第四章 总结与后续交付展望

C06 在 Step 122,500 的验证集上，方向准确率是 63.01%，轨迹相关是 +0.2460，综合分是 0.7532。True OOS 的逐日方向准确率里，Day 2 是 57.33%，Day 5 是 57.58%，Day 10 是 57.03%。这些是评测读数。平方误差没有一起变好：验证集 `skill_nrmse` 仍约为 -0.08～-0.13，True OOS `skill_nrmse` 约为 -0.307（`nrmse` 0.090，`naive_nrmse` 0.069）。日度方向准确率也不是交易策略。

本轮复盘里可以保留、以及必须降级的结论：
- **可以保留的工程事实**：全量验证大约 15.9 分钟；`--eval-every 2500`；Kaggle 接力路径会被平铺，`find_relay_state` 同时扫描 `**/relay/relay.json` 与 `**/relay.json`；`deadline = budget_seconds - reserve_seconds`。跳批加 `zero_grad(set_to_none=True)` 会挡住脏梯度写入权重。
- **梯度**：`forward_clean: true`。`output_head` 是第一个接到脏 `backward_grad_output` 的模块，不是根因。RevIN 开方为零、`1e-7` 的 FP16 下溢、`tanh(20x)` 都还是候选假说；饱和区导数趋于 0。下一步实验是 `autograd.grad(loss, pred)`。
- **还不能写成交付结论**：不能写成可直接落地、实盘 T+2～T+5，或资金周转率提升；也不能写成振幅已经回到健康带并修好了 NRMSE。
