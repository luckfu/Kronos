# GPU Alpha Wave 3A 事前登记（G30–G31）：四家族非线性/跨股因子注意力大一统与 `Kronos-mini` 深度掌控（`dep_layer` / `s1` 解析期望 / 极温和微调 / `C2` 多尺度池化）（2026-11-21）

> **严格防泄漏声明**：
> - 所有训练、PCA 基、2 折 OOF 线性锚、闭式岭回归与模型早停严格截止 **`2026-07-02`**（训练拟合 `<= 2026-04-30`，验证选型 `2026-05-18 .. 2026-07-02`）；
> - 65 天 OOS 评测统一对齐 **`2026-07-17 .. 2026-10-23`（`335,312` 行）**；**`2026-10-24+` 封存验收窗完全不参与任何拟合、选型或调参**；
> - 生产线上 [`deploy/oracle-kronos/head_val_alpha1e4.npy`](../../deploy/oracle-kronos/head_val_alpha1e4.npy) 保持冻结不变。

---

## 1. 研究动机与假设

Wave 2（G24–G29）已将 65 天 OOS Rank IC 推至 **`0.2245`（`100%` 正天数）**、`k=10` 推至 **`+30.04%`**、Calmar 推至 **`2.738`**，并揭示了三大尚未充分挖掘的前沿杠杆：
1. **G27 非线性头（FiLM / Residual Cosine-IC MLP / LightGBM）彻底消除了 OOS 后半段（`Seg B`）衰减**（`0.203/0.203` 零衰减），但 G27 仅用了 `PT-Small + Base` 两家族，尚未纳入最强的 **`Kronos-mini (Tok-2k)`**（单头 `0.2031`）与零衰减的 **`C2 L=45`**（融合 `0.204/0.207`），也未探索 **跨股潜因子注意力（Cross-Stock Latent Factor Attention）** 与 **平滑上尾秩加权 Cosine-IC**（兼顾全截面稳定与 `k=10/20` 实盘收益）。
2. **`Kronos-mini (4.1M, Tok-2k)` 与 `C2` 构成了极致性价比的生产级双核**，但我们尚未：
   - 探索 `Mini` 的全谱短中窗（`L=15, 22, 30, 45, 60, 90`）与高阶时序池化（`m3, m5, m10, diff5, diff10, std5`）；
   - 首次提取 Kronos 内部的 **`dep_layer` (`s2` 条件隐状态)** 与 **`s1` LM 头确定性 1 步解析期望/熵/不确定度**（零采样成本）；
   - 对 `Kronos-mini (4.1M)` 执行类似 G28 的 **极温和顶层 Cosine-IC 微调 + 微调后全密度池化岭回归**；
   - 对 `C2` 做零新权重的 **`L=30/45/60` 多尺度时序池化 + 衰减/体制/波动率标准化目标 (`volnorm10`)** 闭式拟合。

---

## 2. 实验矩阵（并行双账号 2× T4）

| 实验 | 账号 / 算力 | 核心设计 | 参评候选头（65d OOS） |
|---|---|---|---|
| **G30** | `luckfu` (2× T4) | **四家族 (`Mini-2k + Base + PT-Small + C2_L45`, `8,896d`)** 统一 `L=45`（+ `Mini L=30`）提取；50 日 PCA-192（`48×4`）+ 8 路严格 2 折 OOF 岭回归锚 + 12 维严格因果截面/市场体制特征（`212d`） | 6 路闭式 Ridge（含首次四家族联合 `multi / decay / regime / ret5`）+ 6 路非线性/跨股头（`Quad-Resid-CosIC`, `Quad-FiLM-CosIC`, `Quad-FiLM-RankW`, `Quad-XStock-Attn`, `Quad-LGBM-Multi`, `Quad-LGBM-Ret5`），共 **12** 路 |
| **G31** | `smmt315` (2× T4) | **Part 1 (`stride=1`, `5.83M` 行)**：`Mini-2k` 6 窗（`L=15..90`）+ 8 维高阶池化 + `dep_layer(s2)` + `s1` 1 步解析分布特征 + `C2` 多尺度池化（`L=30/45/60`），覆盖 `std / decay730 / decay365 / regime` × `raw / multi / ret5 / volnorm10`；**Part 2**：`Kronos-mini` 双臂（GPU0 `L=45` / GPU1 `L=30`）极温和顶层 Cosine-IC 微调 + 全密度池化岭回归 | `Mini` 深层/多窗/`dep_layer`/`s1` 解析特征 + `C2` 多尺度池化 + `Mini+C2` 联合 + `Mini FT (L=45 / L=30)` 池化 Ridge，共 **20** 路 |

---

## 3. 成功判据（65 天 OOS `2026-07-17 .. 2026-10-23`）

1. **G30**：
   - 四家族非线性头（`Quad-FiLM` / `Quad-XStock-Attn` / `Quad-LGBM`）单头突破 **`Rank IC > 0.215`** 且保持 **`Seg B` 零显著衰减**；
   - `Quad-FiLM-RankW` 相比纯 `CosIC` 显著抬升单头 `k=10` 净收益与 Calmar；
   - 纳入总集成后突破 Wave 2 纪录（`IC > 0.2245` / `NIC > 0.1464`）。
2. **G31**：
   - 验证 `dep_layer (s2)`、`s1` 1 步解析特征、`volnorm10` 与 `Mini` 极温和微调是否带来正向增量（无论正负均如实记录科学结论）；
   - 仅用轻量 `C2 + Mini (4.1M)`（零 `Base` / 零 `PT-Small`）冲击 **`IC >= 0.220` 与 `k=10 > +30%`**。
