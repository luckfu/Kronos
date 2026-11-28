# GPU Alpha Wave 5（G39–G40）、Study E2（无 `G32` 与持仓日度熔断）与 12 月平行封存观察轨事前登记方案（2026-11-28）

> **严格防泄漏与生产纪律声明**：
> - 本方案在运行任何 `E2`、`G39`、`G40` 实验或构建平行封存脚本前冻结；
> - 所有底座微调（含可微 Straight-Through `dep_layer(s2)`）、PCA 基、2-fold OOF 超级教师与学生锚点、闭式岭回归、3-seed 神经网络与 LightGBM 严格限定在 **`train <= 2026-04-30`** 拟合，超参/正则/Epoch/模型选择严格限定在 **`VAL_SEL = 2026-05-18 .. 2026-07-02`** 选型，闭式回归最终重拟合严格截止 **`<= 2026-07-02`**；
> - Clean 65 天 OOS 评测窗统一为 **`2026-07-17 .. 2026-10-23`（`335,312` 行，`65` 个交易信号日）**，跨实验集成全程采用**零 OOS 调权等权横截面秩平均**；
> - **`2026-10-24+` 封存验收窗 100% 保持未触碰**（主轨与平行轨脚本仅做代码编写与合成夹具单元测试，不提前对 `scratch/sealed_20261026_20261120` 调参或筛选）；
> - **主封存协议零改动**：[`20261125_sealed_review_dec_prereg_cn.md`](finetune/reports/20261125_sealed_review_dec_prereg_cn.md)、[`eval_sealed_review_dec.py`](finetune/gpu_alpha/eval_sealed_review_dec.py)、线上生产权重 [`head_val_alpha1e4.npy`](deploy/oracle-kronos/head_val_alpha1e4.npy) 与 `kronos-web` 服务原封不动。

---

## 1. 四线协同立项动机

1. **线路 1（Study `E2` — 本地零 GPU）**：首轮封存复核已证伪 `G32_C2_L45_AR16`（5 日强但第 6–10 日均值回归剧烈反噬），且 Wave 4 单头冠军 `G35_LGBM_CVAR10 (IC=0.2448, Seg B=0.318)` 尚未进入总集成，现有风控门控也仅在换仓日触发、持仓途中遇到连续 3–5 天崩盘无日度保护；
2. **线路 2（实验 `G39` — Kaggle `wynstonliu` 2×T4）**：`G35` 抗跌强但依赖重型四家族，`G37` 极简双核仅蒸馏了普通 `multi/ret5` 教师导致后半段衰减（`Seg A/B = 0.244/0.181`）。将 **`G35`（`cvar10` / `nowhipsaw` / `floor10`）+ `G38`（`fwd5_10`）** 合成 2-fold OOF 3-Target 超级教师跨架构蒸馏进 **物理零 `Base/PTS` 的 `C2-Only` 与 `C2 + 4.1M Mini` 多头学生**；
3. **线路 3（实验 `G40` — Kaggle `user281434` 2×T4）**：突破过去仅靠冻结底座抽 `L=45` 特征的天花板，首创 **`4.1M Kronos-mini` 可微 Straight-Through Estimator（STE）`dep_layer(s2)` + 顶层 `Tail-Aware` 温和微调**（前向与 `argmax` 严格等价，反向可微穿透至 `dep_layer` 与 `proj_s1`），随后联合 `Frozen C2` 做 2-fold OOF 抗跌延续精通（物理零 `Base/PTS`）；
4. **线路 4（Track 4 — 12 月平行封存观察轨）**：独立于主轨 10 路协议，提前固化并单测 Wave 4/5 冻结候选的封存推理核与 `eval_sealed_secondary_w4w5_dec.py`，待 12 月样本门槛达标后一次性客观验真。

---

## 2. 事前锁定的实验设计与公式口径

### 2.1 Study `E2`：无 `G32` 重构集成 + 混合门控 + 持仓中日度因果熔断（`MCB`）
- **核心重构候选（均严格采用等权秩平均）**：
  - `NOG32_C2ONLY_DISTILL5`：`C2 + G31_c2_l45_rich_regime + G34_dep_c2_raw + G37_c2only_distill_resid + G37_c2only_distill_xstock`；
  - `NOG32_LIGHT_MINIC2_DISTILL6`：`C2 + G29_mini + G31_minic2_joint_regime + G31_mini_dep_s2 + G37_minic2_distill_resid + G37_minic2_distill_xstock`；
  - `NOG32_CRASH_SHIELD_CORE`：`C2 + G31_minic2_joint_regime + G33_base_ret20 + G30_quad_resid_cosic + G35_lgbm_cvar10 + G35_tailshield_cvar_shield35 + G35_tailshield_nowhipsaw_shield35 + G35_ridge_crashentry + G38_ridge_fwd5_10`；
  - `NOG32_TAIL_CONT_APEX`：`C2 + G27_resid_cosic + G30_quad_resid_cosic + G35_lgbm_cvar10 + G35_lgbm_nowhipsaw + G35_ridge_nowhipsaw + G36_quad_liq_ridge + G38_ridge_fwd5_10 + G38_lgbm_fwd5_10`；
  - `WITHG32_CRASH_SHIELD_CVAR10`（含 `G32` 对照）：与 `W4_CRASH_SHIELD_5PILLAR` 对比剔除 `G32` 净效应与期限匹配（`step=5` vs `step=10`）。
- **混合门控与日度熔断公式（严格因果 `<= asof_date`）**：
  - `G_HYBRID_MIN(d) = min(G_BREADTH_VEL(d), G_HAZARD_G38(d))`；
  - 日度熔断目标档位 `mcb_panic_tier(d)`：若 `breadth_ma20(d) < 0.28` 或 `Δbreadth_3d(d) <= -0.15` 或 `Δbreadth_5d(d) <= -0.18` 则为 `0`，否则为 `min(2, G_BREADTH_VEL(d))`；
  - 持仓中非换仓日（仅当 `decision_day <= max_signal_day`）只允许**单向向下棘轮降档** `active_tier = min(active_tier, mcb_tier[decision_day])`，若当前持股数超过 `target_k`，按最近一次换仓日排序次日开盘卖出末尾持仓（含完整双边费率 + 涨跌停约束），本轮周期内不盘中复仓，同时增设 `+10bp` 额外单边费率压力测试（`25bp` + 滑点）。

### 2.2 实验 `G39`：四家族 3-Target 左尾/延续超级教师向极简双核蒸馏（`wynstonliu`）
- **学生抽取器 `LightStudentCompactExtractor`（`STUDENT_DIM = 7188`，OOS 物理零 `Base/PTS`）**：
  - `C2 (5130d)`：`L=20 last`, `L=90 last`, 压轴 `L=45 [l3, l5, l6, last, m5, diff5, dep_c, st_c(10)]`；
  - `Mini (2058d)`：`L=30 [last, m5]`, 压轴 `L=45 [l1, last, m5, m10, diff5, dep_m, st_m(10)]`。
- **2-fold OOF 3-Target 超级教师（`<= 2026-07-02`）**：
  - 拟合 `T_cvar`、`T_fwd`（`r_fwd5_10 = (1+r10)/max(1+r5,1e-4) - 1`）、`T_floor`（线性教师 Ridge + `run_in_isolated_process` 拟合 2-fold OOF `LightGBM`）；
  - 双臂学生训练（`C2-Only` `IN_DIM=82` 与 `C2+Mini` `IN_DIM=124`）：3-seed `LightDualTailDistillMLP`（联合 `mu_cvar, mu_fwd, fl_tail`）+ 3-seed `LightCrossStockTailNet` + 4 路 `fit_lgbm_regressor_isolated`，并结合线性流动池软降权对照（`amt20_pct < 0.35 -> 0.25×` 样本权）。

### 2.3 实验 `G40`：`4.1M Kronos-mini` 可微 STE `dep_layer` + `Tail-Aware` 温和微调 & `FT-Mini + C2`（`user281434`）
- **Part 1**：`TailAwareMiniDepAdapter`（`freeze_until=2`，解冻 `transformer[2..3] + norm + dep_layer + head.proj_s1`）在双 GPU 并行（`L=45` & `L=30`）上用 Straight-Through Estimator 穿透 `dep_layer(s2)`，`1.5` epochs 优化 `0.70 * tail_penalized_cosic(p_cont, y_blend, y_floor) + 0.30 * cosic(p_floor, y_floor) + 0.08 * KL + 2500 * L2_anchor`，于 `VAL_SEL` 每 `0.25 epoch` 择优；
- **Part 2**：提取微调后 `FT-Mini (L=30 + L=45)` + 冻结 `C2 (L=30/45/90)` 联合表征（`5918d`，物理零 `Base/PTS`），拟合 2-fold OOF 路径风控与延续 Ridge、3-seed `DualRewardTailShieldMLP` / `CrossStockTailNet` 与 2 路隔离 LightGBM。

---

## 3. 成功检验标准（65 天 OOS `2026-07-17 .. 2026-10-23`）

1. **Study `E2`**：剔除 `G32` 并纳入 `G35_LGBM_CVAR10 + G38_FWD5_10` 后，`Rank IC`、后半段稳定性 `Seg B` 与实盘净收益/卡玛比全面不弱于（且结构上优于）含 `G32` 版本；`MCB` 在突发急跌周期显著降低回撤，且经受住 `25bp + 滑点` 压力测试；
2. **`G39`（极简双核蒸馏迭代）**：`W5_LIGHT_NOG32_G39_TAILFWD`（零 `G32`、零 `Base/PTS`）较 `W4_LIGHT_MINIC2_DISTILL` 显著抬升 `Seg B IC`（缓解 `0.244 → 0.181` 衰减），同时保持顶级 `liq10/20` 与 Calmar；
3. **`G40`（`Mini STE dep_layer Tail-Aware FT`）**：验证微调显著抬升 `VAL_SEL` 抗跌延续指标（优于 `step 0`），并在 `C2 + FT-Mini` 联合体系上贡献相对冻结表征的正交增量；
4. **Track 4**：主轨与平行轨单元测试全绿，不引入任何 `2026-10-24+` 泄漏与主协议改动。
