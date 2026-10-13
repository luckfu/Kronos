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
| **预安装静默假死排查 (2026-10-13)** | 容器运行前 50 分钟无日志输出，外部无法判断是运行中还是挂起。 | 入口处 `pip install -q git+https://...` 采用静默模式，且在进入 `main()` 前无任何生命周期打点，导致外部黑盒排查极难。 | 实施“第 0 秒零延迟打点”铁律；移除 `-q` 静默模式；加设 `--timeout 60` 并在各启动阶段打出明确阶段耗时。 |

---

## 5. 启动生命周期与日志透明化规范 (强制实施)

为彻底杜绝“长达数十分钟静默假死、外部无法得知进展”的严重工程问题，后续所有任务必须严格遵循以下启动规范：

### 5.1 零延迟打点 (Zero-Lag Bootstrap)
在脚本 `if __name__ == '__main__':` 的**第一行代码**，必须立即无条件向终端输出唤醒信号：
```python
print(f"[Bootstrap] 任务已启动，当前时间: {datetime.now(timezone.utc).isoformat()}，准备配置环境...", flush=True)
```

### 5.2 依赖安装透明化与防挂起
- **严禁静默模式**：严禁在 `pip install` 中使用 `-q` / `--quiet`；
- **强制超时防护**：必须添加 `--timeout 60`，防止 git clone 握手无限卡死；
- **明确前后打点**：安装前后必须打印耗时。

### 5.3 关键阶段状态全透明
数据解包（pkl读取）、模型初始化、LoRA加载、前向批次推断，每个大步骤必须有明确的 `[Stage Start]` 与 `[Stage Done]` 打点，任何阶段不允许产生超过 60 秒的静默盲区。

---

## 6. 时序窗口拉平与有上限混合加权采样工程规约 (C04 标准)

> **铁律**：彻底废止旧的 $|R_{10}| \ge 5\%$ 拒绝采样（拒绝采样不仅截断了真实概率分布、制造严重的幸存者偏差，还会破坏模型对真实窄幅盘整行情的判别力）。必须严格执行 C04 窗口级拉平加权混合采样机制。

### 6.1 全窗口平铺池 (Window-level Flat Pool)
- **原理**：将全部合格股票在全历史时段内所有满足 $120 \text{ 天上下文} + 10 \text{ 天目标}$ 长度的合法滑窗全部拉平展开，构成全局唯一的窗口池样本集合 $\{(s_i, t_i)\}$。
- **目的**：彻底根除“先等概抽股票、再在股票内抽时点”所引入的**次新股抽样放大偏见**（上市 1 年的新股被抽中概率曾被放大数十倍）。

### 6.2 路径最大波幅定义 (Full-Path Max Amplitude)
针对每个合法窗口 $i$，计算其未来 10 天整个波段的路径最大绝对震荡幅度：
$$A_i = \max_{h \in [1, 10]} \left| \frac{\text{close}[t_i + 119 + h]}{\text{close}[t_i + 119]} - 1.0 \right|$$
相比于仅看第 10 天终点的收益率 $R_{10}$，路径波幅 $A_i$ 能够精准捕获未来 10 天内触及大涨大跌但终点回落的典型行情。

### 6.3 4.0 倍硬封顶幂律相对加权 (Capped Relative Weight)
1. 计算全样本池路径波幅中位数：$A_{\text{med}} = \text{median}(\{A_i\}) + 10^{-4}$；
2. 定义单样本相对放大权重，并实施硬性截断：
   $$w_i = \min\left(1.0 + \frac{A_i}{A_{\text{med}}}, 4.0\right)$$
- **封顶原则**：单样本相对权重最大不得超过基准的 4.0 倍，坚决杜绝极端妖股样本主导梯度，破坏全局泛化。

### 6.4 40% 均匀基底 + 60% 幂律加权混合采样机制 (Dual-Stream Mixture)
全池采样概率分布由两股概率流线性叠加而成：
$$P_i = 0.40 \times \frac{1}{N} + 0.60 \times \frac{w_i}{\sum_{j=1}^N w_j}$$
- **效果**：既保留了 40% 全市场真实宏观统计底噪分布，又对 60% 波动窗口实施适度上采样，使得未来 10 天涨跌幅超过 5% 的样本比例温和提升至 ~78.7%，兼具波段敏锐度与底模抗噪性。

### 6.5 预冻结确定性采样计划与哈希校验 (Pre-frozen Plan with SHA-256)
- **机制**：在训练正式启动前，使用固定随机种子预先一次性生成全量训练所需的样本索引序列（例如 50 万或 400 万步）；
- **SHA-256 校验**：对生成的抽样序列计算 SHA-256 校验哈希并写入日志打点；
- **断点精准接力**：中途断点续训（Relay chunk）通过全局步数指针（`plan_ptr`）直接位移接力，杜绝跨进程采样漂移与重叠。

---

## 7. 行业特征映射与 Unknown 隔离铁律

### 7.1 申万一级行业字典分配
- 全市场申万一级行业去重排序后共 84 个，对应有效 ID 范围为 `0 ~ 83`。

### 7.2 严禁 `fillna(0)` 静默污染
- **致命根因**：`0` 是排序第一位的真实行业（如字典序首位的真实行业）。任何使用 `df.map(sector2id).fillna(0)` 的写法，都会导致缺失行业或新股票静默归入真实第 0 行业，引发表示空间混淆；
- **隔离铁律**：未知、缺失或未匹配行业必须严格统一映射为独立占位 ID：
  $$\text{UNKNOWN\_SECTOR\_ID} = \text{len}(\text{sectors}) = 84$$

### 7.3 嵌入层词表总长度锁死 89 维 (Immutable 89-dim)
- **词表结构**：`num_classes = len(sectors) + 5 = 89`（包含 84 个真实行业、1 个独立 Unknown 行业及 4 个预留冗余位）；
- **动量兼容铁律**：严禁私自将 Embedding 扩容至 94 维或其它尺寸。否则加载上游 Checkpoint 时，恢复出的 AdamW 优化器动量张量（`exp_avg`, `exp_avg_sq`）在底层 C++ multi-tensor 内核中会触发不可恢复的张量尺寸不匹配致命崩溃。

### 7.4 训练、验证与 OOS 测试全链路编码一致性
- `train.py` 中的 `FastStockPool`、`build_validation_pool` 以及 `eval_oos/eval.py` 中的 `extract_full_validation_set`、`extract_full_oos_set`，必须 100% 共享相同的映射字典和一致的 `unknown_sector_id` fallback 逻辑，杜绝任何训练与测试端的特征语义漂移。


