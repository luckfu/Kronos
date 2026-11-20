# GPU Alpha Wave 2C（G29）事前登记（2026-11-19）

> **严格防泄漏铁律**：
> 1. 所有拟合严格截止 **`2026-07-02`**（`2021-01-01..2026-04-30 stride=1` 全密度 `5,828,028` 行 + `val_data.pkl`）；
> 2. 所有正则强度仅在 **`2026-05-18 .. 2026-07-02`** 未见面验证段选型；
> 3. 开发期 OOS 对齐 **`2026-07-17 .. 2026-10-23`（65 天、`335,312` 行）**；**严禁触碰 `2026-10-24+` 封存验收窗**。

---

## 1. G29 填补的 Wave 2 最后三块空白

基于 **G26（`Kronos-mini + Tok-2k` 爆发）** 与 **Wave 2 三家族集成（`IC 0.2166`）** 的直接启示，`G29` 在单次 `L=45` 前向（加极轻 `Mini L=30`）下一次性验证：
1. **`Kronos-mini (4.1M, Tok-2k)` 未测增强组合**：
   - 全 5 层完整深度谱 `[L0, L1, L2, L3_pre, L4_last]`（`1,280d`）；
   - `Mini L=45` 时序池化 `[L1, L4_last, L4_m5, L4_diff5]`（`1,024d`）及四路加权/周期对照（`std / multi / decay / regime / ret5`）；
   - `Mini L=30 + L=45` 联合池化（`2,048d`）。
2. **`Mini-2k + Base` 跨分词器双家族联合头（`4,544d`）**：此前 G26 仅测过 `Mini + PT-Small`，尚未在单一线性头中联合最强的 `Mini-2k` 与 `Base`。
3. **三家族单头大一统（`Mini-2k + Base + PT-Small`，`6,592d`）**：检验在单个闭式岭回归（`alpha ∈ {1e7..1e10}`）中联合三大零微调异构家族能否超越后期秩投票集成。

- 脚本与打包：[`g29_mini_pool_trifamily_entry.py`](../gpu_alpha/g29_mini_pool_trifamily_entry.py) / [`push_g29.py`](../gpu_alpha/push_g29.py)
- Kaggle 运行目标：`user281434/kronos-alpha-g29-mini-pool-trifamily`
