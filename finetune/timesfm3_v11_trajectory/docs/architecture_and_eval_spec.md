# TimesFM 3.0 V11 模型架构、权重加载与评估工程规范

> **警示与铁律**：严禁在任何模型加载环节使用无防御的 `strict=False`！严禁脱离 `train.py` 单独篡改 LoRA 注入逻辑！任何推断与评估必须严格执行数量断言（Fail-Fast）。

---

## 1. 核心架构与参数规范

### 1.1 基础模型
- **底模**：`google/timesfm-3.0-pytorch`
- **上下文长度**：120 天日线 Lookback
- **预测视界 (Horizon)**：未来 10 天累积走势切片
- **输入通道数 (8 维多元张量)**：
  - 通道 0: `close` (归一化收盘价，以第 120 天为基准 1.0)
  - 通道 1: `open`
  - 通道 2: `high`
  - 通道 3: `low`
  - 通道 4: `volume` (以过去 120 天中位数成交量归一化)
  - 通道 5: `amount` (以过去 120 天中位数成交额归一化)
  - 通道 6: `size_percentile` (流通市值百分位，连续动态时序)
  - 通道 7: `sector_embedding` (申万一级行业映射嵌入，维度等于 Patch 长度)

### 1.2 LoRA 规格
- **Rank**：16
- **Alpha**：32
- **Dropout**：0.05 (训练期) / 0.0 (推断期)
- **目标注入模块 (Targets)**：
  TimesFM 3.0 内部模块命名与通用 HuggingFace 模型不同，必须严格匹配以下 6 种核心线性投影层：
  - `query_proj`
  - `key_proj`
  - `value_proj`
  - `out_proj`
  - `ff0`
  - `ff1`
- **参数命名规范**：
  每一个注入层产生两个直接 Parameter：
  - `...<module_name>.lora_a` (Shape: `[16, in_features]`)
  - `...<module_name>.lora_b` (Shape: `[out_features, 16]`)

---

## 2. 权重持久化与加载铁律 (Anti-Gotchas)

### 2.1 权重字典标准结构
由 `get_trainable_state()` 保存的权重字典包含且仅包含以下两部分：
1. **LoRA 核心参数**：前缀为 `lora.`，参数总量严格为 **400 个**；
2. **行业嵌入权重**：前缀为 `sector.`，参数总量严格为 **1 个** (`sector.weight`)。
- **总计可学习参数条目**：**401 个**。

### 2.2 严格防呆加载范式 (强制实施)
任何评估、推断、迁移脚本，必须严格采用以下三道防御栅栏，严禁使用盲目的 `strict=False`：

```python
# 1. 注入 LoRA (必须完全复制 train.py 中的实现)
model = attach_lora(model, rank=16, alpha=32, dropout=0.0)

# 2. 检查模型内生成的 LoRA 参数数量
lora_params_in_model = {k: v for k, v in model.named_parameters() if "lora_" in k}
assert len(lora_params_in_model) == 400, (
    f"[严重错误] 模型内部注入的 LoRA 参数数量异常: 期望 400, 实际 {len(lora_params_in_model)}!"
)

# 3. 提取并校验权重字典
saved_weights = torch.load(weights_path, map_location="cpu", weights_only=False)
lora_weights = {k.replace("lora.", ""): v for k, v in saved_weights.items() if k.startswith("lora.")}
sec_weights = {k.replace("sector.", ""): v for k, v in saved_weights.items() if k.startswith("sector.")}

assert len(lora_weights) == 400, f"[严重错误] 待加载 LoRA 权重字典数量异常: {len(lora_weights)}"

# 4. 加载并断言零遗漏、零多余
missing, unexpected = model.load_state_dict(lora_weights, strict=False)
assert len(unexpected) == 0, f"[严重错误] 发现未匹配的多余权重: {unexpected}"

# 5. 行业嵌入必须 strict=True
sector_embedding.load_state_dict(sec_weights, strict=True)
print("[LoRA Check] 400 个核心 LoRA 权重 + 1 个行业嵌入权重已 100% 精准装载生效!")
```

---

## 3. 评估与大考指标规范

### 3.1 样本与数据分流
- **训练集 (Train)**：`smmt315/a-share-120d-temporal-symbol-holdout` 主集，已剔除 26.4 万行停牌/零成交记录；
- **全量验证集 (Holdout Val)**：`val_data.pkl`，**123,839 条样本**，单次评估耗时约 14.5 分钟；
- **True OOS 盲测集 (True OOS)**：`evaluation_panel.pkl` + `evaluation_samples.jsonl`，**175,428 条样本**，严格未见交易日封箱测试。

### 3.2 核心量化指标定义
1. **未来 10 天方向胜率 (Direction Acc / DirAcc)**：
   $$\text{DirAcc} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{sign}(\hat{R}_{10, i}) == \text{sign}(R_{10, i}))$$
2. **超越市场底噪净胜率 (Alpha Win Rate Edge)**：
   $$\text{Alpha Edge} = \text{DirAcc} - \text{Market Up Rate}$$
3. **全样本 Pooled IC (10 天收益率)**：
   $$\text{Pooled IC} = \text{PearsonCorr}(\hat{R}_{10}, R_{10})$$
4. **日度截面 IC 均值与 ICIR (Daily IC & Daily ICIR)**：
   对同一交易日截面内的所有股票计算当期 IC，求全历史区间的均值（$\mu_{\text{IC}}$）与标准差（$\sigma_{\text{IC}}$）：
   $$\text{ICIR} = \frac{\mu_{\text{IC}}}{\sigma_{\text{IC}}}$$
5. **技巧均方根误差 (Skill NRMSE)**：
   $$\text{Skill NRMSE} = 1.0 - \frac{\text{NRMSE}}{\text{Naive NRMSE}}$$
   （注：Naive 基准恒定预测收益为 0，数值为 0.0721）
6. **走势形态综合评分 (Trajectory Score)**：
   $$\text{Score} = \text{DirAcc} + 0.5 \times \max(\text{TrajCorr}, 0.0)$$

---

## 4. 历史版本踩坑复盘 (Post-Mortem)

| 事件 | 现象 | 根因分析 | 解决方案与防范 |
| :--- | :--- | :--- | :--- |
| **LoRA 静默漏载 (2026-10-13)** | 运行评估后，验证集胜率退化为 48.9%，与 Baseline 一致。 | 评估脚本从通用模板拷贝了 `("q_proj", ...)`，而 TimesFM 目标层名为 `("query_proj", ...)`。且配合了 `strict=False`，导致 400 个权重静默跳过，沦为纯底模评测。 | 统一核心代码为单一数据源；强制增加 400 个参数数量断言（`assert len == 400`），有任何错配立即报错中断，绝不静默运行。 |
| **双卡推断通信瓶颈 (2026-10-11)** | 单次验证耗时达 15 分钟，导致 1 小时训练中 75% 时间在考试。 | 12.3 万条样本在双卡 DataParallel 下频繁触发数据 Scatter/Gather 通信。 | 将大考步频设为 `eval_every = 2500`，确保训练与大考时间比维持在 1.7 : 1，把算力留在有效训练上。 |
