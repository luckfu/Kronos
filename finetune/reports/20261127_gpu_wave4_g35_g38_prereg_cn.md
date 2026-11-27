# GPU Alpha Wave 4（G35–G38）与自适应执行护盾（E1）事前登记方案（2026-11-27）

> **严格防泄漏与生产纪律声明**：
> - 本协议在任何 Wave 4 训练与 `E1` 评测运行前冻结；
> - 所有底座特征、PCA 基、2-fold OOF 锚点、岭回归闭式拟合、LambdaRank/LightGBM、神经网络与市场崩盘危害度（Hazard）模型严格限制在 **`train <= 2026-04-30`** 拟合，超参/Epoch/正则/门控阈值仅在 **`2026-05-18 .. 2026-07-02`** 验证选型，线性闭式头最终重拟合严格截止 **`<= 2026-07-02`**；
> - 所有 65 天 OOS 评测统一对齐 **`2026-07-17 .. 2026-10-23`（`335,312` 行）**；
> - **`2026-10-24+` 封存验收窗 100% 保持封存不触碰**；生产线上 [`deploy/oracle-kronos/head_val_alpha1e4.npy`](../../deploy/oracle-kronos/head_val_alpha1e4.npy) 与 `kronos-web` 服务原封不动。

---

## 1. 立项动机：精准直击前几轮发现的四大结构性盲区

Wave 1–3 已将 65 天 OOS 全市场 `Rank IC` 从生产基线 `0.1428` 推至 `0.2436`，但结合首轮封存窗（`10-26..11-20`）和交易执行回测，真正制约实盘收益-风险比的剩余痛点集中在四处：

1. **满仓绿灯撞暴跌 + 个股左尾踩踏**（`11-04..11-12` 宽度 `30%..57%` 开仓撞 V 型闪崩）：常规仅预测点估计 `r10` 会选入高贝塔/高拥挤度易崩个股；
2. **高全市场 IC ≠ 流动池 `Top-10/20` 头部锋利**：G30 `LGBM_RET5` 全截面 `IC=0.2343` 但 `k=10` 仅 `+11.03%`，而实盘人工只买前 10–20 只高流动性标的；
3. **重型四家族旗舰（`Base 102M + PTS + Mini + C2`）线上部署重**：若能将四家族旗舰的横截面交互与体制感知能力蒸馏回 **零新权重 `C2-Only`** 或 **极简 `C2 + 4.1M Mini`**，Modal T4 延迟可降到最低；
4. **D5/D10 期限结构反转与事前风控前瞻性不足**：`MA20` 宽度属于滞后指标（`11-11` 仍为 `57%`），且需把“前 5 日涨”与“第 6–10 日延续”拆开，同时配合更短的自适应调仓节奏（`step=5`）与宽度速度门控。

---

## 2. Wave 4 五路协同实验设计

### 2.1 `G35` — 暴跌护盾：左尾下界、CVaR 与防 V 型反噬专精（Kaggle `wynstonliu` Slot 1）
- **特征与训练**：`2021-01-01..2026-04-30 stride=2`，L=45 四家族高信噪比表征（`Mini + PTS + C2 + Base` 末层/均值/`dep_s2` + 40 维 `s1/s2` 1 步解析统计，`6,248d`）。
- **四项路径依赖抗跌目标**（严格仅读取 `[ti, ti+10]`）：
  1. `floor10`: `min_{k=1..10}(close[t+k]/close[t] - 1)`（10 日路径最低谷）；
  2. `cvar10`: `r10 - 0.75 * max(0, -min_ret_10d)`（下行惩罚收益）；
  3. `nowhipsaw`: `r10 - 0.60 * max(0, r5 - r10)`（专惩“前 5 日急涨、后 5 日砸盘”V 型反噬股）；
  4. `calmar10`: `r10 / max(max_dd_10d, 0.03)`（10 日路径卡玛）。
- **体制样本加权**：新增 `crashentry` 方案——训练期对 `breadth_ma20 >= 0.30` 且随后发生 `mean(r10) <= -3.5%` 或 `mean(r10 - r5) <= -3%` 的“绿灯崩盘前夜”赋予 `4×` 权重。
- **模型家族**：块落盘 Ridge + PCA-192 + 2-fold OOF 锚点 + 3-seed `DualRewardTailShieldMLP`（联合预测 `mu_pred` 与 `floor_pred`，输出 `gamma ∈ {0.35, 0.70}` 风险调整合成） + 2 路 LightGBM（`cvar10`, `nowhipsaw`）。

### 2.2 `G36` — 流动池 `Top-10/20` `LambdaRank` 与 `Focal-ListNet` 头部锐化引擎（Kaggle `user281434` Slot 1）
- **特征与尺度不变流动池掩码**：L=45 四家族 + `Mini L=30` + `dep_s2` + 14 维因果上下文（含日内横截面 `med_amt_20d` 百分位 `amt_pct` 与 `is_liq_top65 = (amt_pct >= 0.35)`，完全跨数据集尺度不变）。
- **头部精确排序模型**：
  1. `LGBMRanker(objective="lambdarank", ndcg_eval_at=[10, 20], lambdarank_truncation_level=30)`（`r10` 与 `multi` 双目标）：日内按截面百分位赋予 `0..15` 指数分级标签（重点拉开前 0.5% / 1% / 2% / 5% / 10% 差距），并将流动性后 35% 个股封顶在 `<= 3` 级，迫使树分裂把高分集中给真正可交易的大容量强股；
  2. `Focal-ListNet MLP`（3-seed）：交叉熵目标分布 `softmax(2.5 * y_z + log_liq_mask)`，直接优化全池头部概率质量；
  3. `CrossStockFactorAttentionNet-TopK`（3-seed）：`0.5 * Focal-ListNet + 0.5 * multi_cosic`；
  4. 线性头部锚点 + `LambdaRank`/`ListNet` 混合打分。

### 2.3 `G37` — 四家族旗舰向“零新增权重 `C2` 单核 / 极简 `C2+Mini`”蒸馏（Kaggle `wynstonliu` Slot 2）
- **教师（严格 `<= 2026-07-02`）**：2-fold OOF 四家族超级教师 `T_super`（四家族 Ridge + 3-seed Residual MLP + LightGBM-Ret5 等权融合）。
- **双学生分支（OOS 打分物理不加载 `Base` / `PT-Small`）**：
  - **学生 A（`C2ONLY_*` 零新增权重）**：仅用 `C2` 多窗（`L=20, 45, 90, 120` + `dep_c` + 10 维 `s1/s2` 统计）+ 12 维因果上下文，训练蒸馏 Ridge、3-seed Residual MLP、3-seed CrossStock-Attn 与 LightGBM；
  - **学生 B（`MINIC2_*` 极简双核，仅 `C2 + 4.1M Mini`）**：`C2` + `Mini (L=30, 45, dep_m, s1/s2 stats)` 同构蒸馏。

### 2.4 `G38` — 多周期期限结构（`r3/r5/r10` 第 6–10 日延续）+ GPU 前瞻崩盘危害度网络（Kaggle `user281434` Slot 2）
- **期限结构延续模型（`2019-01-01..2026-07-02`）**：显式将 `r10` 拆解为 `r5`（前 5 日）与 **`r_fwd5_10 = close[t+10]/close[t+5] - 1`（第 6–10 日纯延续）**；训练 `TermStructContNet` 与 LightGBM 同时预测 `[r3, r5, r_fwd5_10, r10]`，构造一致性选股分 `z(hat_r5) + z(hat_r_fwd5_10) - 0.4 * relu(z(hat_r5) - z(hat_r_fwd5_10))` 过滤短命脉冲股。
- **因果市场崩盘前瞻预警器 `P_crash(t)`**：在长周期（`2019..2026-07-02`）提取日度市场宏观与 Kronos 不确定性特征（`breadth_ma20`、`breadth_ma5`、`Δbreadth_3d/5d`、`s1` 平均熵/分歧度、`s1` 预期收益离散度、隐状态余弦离散度、市场实现波动比、高位滞涨占比等），用 L2 逻辑回归与岭回归预测“活跃日（`breadth >= 0.30`）未来 10 日发生暴跌/宽度坍塌（`univ_ret_d10 <= -4%` 或 `min Δbreadth_5d <= -15%`）”的因果概率 `P_crash(t)`，供交易层做降仓对比。

### 2.5 `E1` — 自适应执行护盾 CPU 专项（零 GPU，65d OOS）
- 在 `C2`、`ENS_W2F_C2_MINI29`、`W3_LIGHT_C2AR_MINIC2_5WAY`、`W3_FLAGSHIP_5PILLAR_WITH_S1S2`、`W3_SUPREME_AR_W2_LGBMRET5` 上对比：
  1. **调仓节奏**：`step=5`（5 日换仓 + 30% 缓冲，全 5 相位均值）vs `step=10`（10 相位均值）；
  2. **因果门控与尾部过滤**：`RAW` vs `G_BREADTH` vs `G_BREADTH_VEL`（在 `breadth >= 30%` 但 `Δbreadth_5d <= -0.12` 或 `breadth_ma5 < breadth_ma20 - 0.10` 时自动降档半仓/防御）vs `LowVol-85%`（剔除 20 日实现波动率前 15% 的极端高波股）。

---

## 3. 成功检验标准（65 天 OOS `2026-07-17 .. 2026-10-23`）

1. **赛道 A（零新增权重 `C2-Only`）**：`G37 C2ONLY` 集成超越 `W3_C2ONLY_FULL4`（`IC 0.2277`, `k=10 +19.77%`, `k=20 +17.49%`）；
2. **赛道 B（极简轻量双核 `C2+Mini`）**：`G37 MINIC2 + G35/G36` 超越 `ENS_W2F_C2_MINI29`（`k=10 +29.03%`, `IC 0.2113`）或 `W3_LIGHT_C2AR_MINIC2_5WAY`（`IC 0.2310`, `Calmar 2.252`）；
3. **赛道 C（流动池头部锐化旗舰）**：融合 `G36 LambdaRank/Focal-ListNet` 后显著拉高 `liq10/liq20`（目标 `>= +4.3%`）与 `k=10/k=20` 扣费净收益（目标 `k=10 >= +28%..+30%+`）；
4. **赛道 D（防暴跌与高卡玛旗舰）**：`G35 + G38` 在保持高 `IC` 的同时压低最大回撤（目标 `DD <= -8.5%`, `Calmar >= 2.6..3.0+`），并在 `E1` 验证速度门控/高频换仓对 V 型反转的真实减损效果。
