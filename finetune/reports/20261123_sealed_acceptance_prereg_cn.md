# Wave 2 / Wave 3 封存窗一次性正式验收事前登记（`2026-10-26 .. 2026-11-20`，2026-11-23）

> **铁律声明**：本文件在任何 Wave 2 / Wave 3 模型触碰 `2026-10-26 .. 2026-11-20` 封存数据**之前**冻结。所有参评权重、PCA 基、岭回归头、3-seed 神经网络与 LightGBM 均严格拟合自 **`<= 2026-07-02`**（选型验证 `<= 2026-07-02`），65 天 OOS 筛选截止 **`2026-10-23`**。本次验收全程**零重训、零调参、零权重再拟合、零事后筛选**，在 Kaggle 2× T4 GPU 上执行纯前向推理并一次性产出验收结论。

---

## 1. 封存验收数据与窗口定义

- **封存数据源**：Oracle 生产 PostgreSQL（`public.ashare_daily_k` × `public.ashare_adj_factor`）+ 线上 `/opt/kronos-web/data/prediction_shadow/2026-10-26 .. 2026-11-20/full_market/ranking.csv`
- **完整推理面板**：`20` 个交易日（`2026-10-26 .. 2026-11-20`），`86,195` 行，`4,324` 只标的；每条样本严格具备完整 `120` 个交易日后复权历史窗（`2026-04-28 .. asof`，缺失数 `0`）。
- **两层到期评估窗口 + 门控分体制拆解**：
  1. **主窗口 `D10`**：`10` 个到期信号日（`2026-10-26 .. 2026-11-06` → 目标日 `2026-11-09 .. 2026-11-20`），包含 `7` 天弱势防御期（`breadth_ma20 = 14.5%..23.1%`）与 **`3` 天 `G_BREADTH >= 30%` 活跃期（`11-04 30.3%`, `11-05 42.8%`, `11-06 34.3%`）**。
  2. **扩展窗口 `D5`**：`15` 个到期信号日（`2026-10-26 .. 2026-11-13` → 目标日 `2026-11-02 .. 2026-11-20`），包含 `8` 天弱势防御期与 **`7` 天 `G_BREADTH >= 30%` 活跃期（`11-04..11-12: 30.3%..57.2%`）**。
  3. **分体制拆解**：对 `D10` 与 `D5` 分别报告全量、**`breadth >= 30%` 实盘开仓期**与 **`breadth < 30%` 空仓防御期**。

---

## 2. 生产一致性门禁（Parity Gate）

在统计任何封存指标之前，必须先通过线上一致性校验：
- 在 Kaggle 推理脚本中同时计算现有线上旧头 `deploy/oracle-kronos/head_val_alpha1e4.npy` 对 `2026-11-20` 全市场（`4,318` 股）的打分；
- 与 `/opt/kronos-web/data/prediction_shadow/2026-11-20/full_market/ranking.csv` 逐股比对：要求 `score_hidden` Pearson 相关 `> 0.9999`，`score_fused` 相关 `> 0.9999`。

---

## 3. 事前锁定的 15 路参评体系（等权横截面秩平均，零调参）

| 梯队 | 方案名 | 组成成员（全部冻结自 `<= 2026-07-02`） |
|---|---|---|
| **基线 0** | `C2` | 线上原版 `C2 L=120` 16 路径 AR (`predicted_return_d10`) |
| **基线 1** | `FUSE_VAL_1E4` | 当前线上生产融合 (`C2 + head_val_alpha1e4.npy`) |
| **Wave 2 轻量代表** | `ENS_W2F_C2_MINI29` | `C2 + G29_MINI_L45_POOL_REGIME` |
| **Wave 2 全能冠军** | `W2_CHAMP_FILM27` | `C2 + G29_MINI_L45_POOL_REGIME + G25_BASE_L52_POOL + G27_FILM_REGIME_MLP` |
| **Wave 3 零拟合 AR** | `G32_C2_L45_AR16_D10_MEAN` / `W3_PURE_AR_DUALWIN` | 单独 `C2 L=45 AR-16` / `C2 L=120 AR + C2 L=45 AR` |
| **Wave 3 零新权重 `C2` 单核** | `W3_C2ONLY_AR_RICH` | `C2 + G32_C2_L45_AR16_D10_MEAN + G31_C2_L45_RICH_REGIME` |
| **Wave 3 零新权重 `C2` 单核** | `W3_C2ONLY_FULL4` | `C2 + G32_C2_L45_AR16_D10_MEAN + G31_C2_L45_RICH_REGIME + G34_DEP_C2_RAW` |
| **Wave 3 极简双核 (`C2+Mini`)** | `W3_LIGHT_C2AR_MINIC2_5WAY` | `C2 + G32_C2_L45_AR16_D10_MEAN + G31_MINIC2_JOINT_REGIME + G31_MINIC2_JOINT_RET5 + G31_MINI_L45_DEP_S2` |
| **Wave 3 极简双核 (`C2+Mini`)** | `W3_LIGHT_C2AR_MINI29_G31` | `C2 + G32_C2_L45_AR16_D10_MEAN + G29_MINI_L45_POOL_REGIME + G31_MINIC2_JOINT_RET5 + G31_MINI_L30_45_RICH_VOLNORM` |
| **Wave 3 旗舰 `100%` 正** | `W3_FLAGSHIP_AR_PLUS_W2CHAMP` | `C2 + G32_C2_L45_AR16_D10_MEAN + G29_MINI_L45_POOL_REGIME + G25_BASE_L52_POOL + G27_FILM_REGIME_MLP` |
| **Wave 3 旗舰 IC 总冠军** | `W3_SUPREME_AR_W2_LGBMRET5` | `C2 + G32_C2_L45_AR16_D10_MEAN + G29_MINI_L45_POOL_REGIME + G25_BASE_L52_POOL + G27_FILM_REGIME_MLP + G30_QUAD_LGBM_RET5` |
| **Wave 3 旗舰 5 支柱全量** | `W3_SUPREME_5PILLAR_LGBMRET5` | `C2 + G32_C2_L45_AR16_D10_MEAN + G31_MINIC2_JOINT_REGIME + G33_BASE_L52_RICH_RET20 + G34_DEP_QUAD_REGIME + G30_QUAD_LGBM_RET5 + G30_QUAD_XSTOCK_ATTN` |
| **Wave 3 旗舰低回撤王** | `W3_FLAGSHIP_5PILLAR_WITH_S1S2` | `C2 + G32_C2_L45_AR16_D10_MEAN + G31_MINIC2_JOINT_REGIME + G33_BASE_L52_RICH_RET20 + G34_DEP_PLUS_STATS_REGIME + G34_S1S2_STATS_QUAD40 + G30_QUAD_FILM_RANKW + G30_QUAD_XSTOCK_ATTN + G27_FILM_REGIME_MLP` |
| **Wave 3 旗舰 Calmar 王** | `W3_FLAGSHIP_HIGH_CALMAR` | `C2 + G32_C2_L45_AR16_D10_PROBUP + G29_MINI_L30_45_RAW + G31_MINIC2_JOINT_DECAY365 + G33_BASE_L45_ALL_DEP_REGIME + G34_DEP_QUAD_DECAY365 + G30_QUAD_FILM_RANKW + G27_RESID_MLP_COSIC` |
| **纯前向诊断（零 AR）** | `W3_NO_AR_LGBMRET5_CORE` | `G30_QUAD_LGBM_RET5 + G30_QUAD_XSTOCK_ATTN + G31_MINIC2_JOINT_REGIME + G33_BASE_L52_RICH_RET20` |

> 同时输出全部单头组件诊断，以便精确归因每一根支柱（`C2 L=45 AR` / `Mini` / `Base ret20` / `dep_layer(s2)` / `G30 LGBM & XStock` / `G27 FiLM`）在极端弱市与反弹切换期中的独立贡献。

---

## 4. 验收通过标准（客观决策门槛）

候选方案达到“建议进入生产切换/灰度升级”需同时满足：
1. **生产一致性校验通过**（`score_hidden` & `score_fused` 相关 `> 0.9999`）；
2. **主窗口 `D10`（10 日）及扩展窗口 `D5`（15 日）均严格击败当前线上 `FUSE_VAL_1E4`**（Rank IC、中性化 IC、流动性 `liq10/liq20` 综合占优）；
3. **在 `G_BREADTH >= 30%` 实盘开仓期（`D10` 3 日 & `D5` 7 日）均跑赢原版 `C2` 且 Rank IC 为正、流动池超额为正**；
4. **若全量 `D10`（含 7 天 `breadth < 30%` 极弱期）亦稳定跑赢原版 `C2`，则直接给出“通过，推荐生产升级”结论**；若仅在 `breadth >= 30%` 开仓期大胜而在极弱期持平/略弱，则明确给出“配合 `G_BREADTH` 门控有条件通过”的实事求是结论。
