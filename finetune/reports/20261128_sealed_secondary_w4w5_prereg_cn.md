# 12 月平行封存观察轨（Wave 4 + Wave 5 `No-G32` 冻结候选与执行护盾）事前登记协议（2026-11-28）

> **与主封存轨严格物理隔离声明**：
> - **主封存轨（原封不动）**：[`20261125_sealed_review_dec_prereg_cn.md`](finetune/reports/20261125_sealed_review_dec_prereg_cn.md) 与 [`eval_sealed_review_dec.py`](finetune/gpu_alpha/eval_sealed_review_dec.py) 仍专门仲裁 `Wave 2 / Wave 3` 10 路候选与当前线上 [`head_val_alpha1e4.npy`](deploy/oracle-kronos/head_val_alpha1e4.npy) 是否执行首次切换；两份主轨文件及 `kronos-web` 服务 **100% 原封不动**；
> - **平行观察轨（本协议）**：仅在 12 月份样本窗同步验证 **`Wave 4`（`G35/G37/G38`）+ `Wave 5`（`G39/G40` + `E2 No-G32`）** 冻结模型及因果执行护盾（`G_BREADTH_VEL` / `G_HAZARD_G38` / `MCB`），不干扰主轨裁定，所有模型参数冻结自 `<= 2026-07-02`，所有候选名单仅从 Clean 65d OOS（`<= 2026-10-23`）事前选定，**对 `2026-10-24+` 封存验收窗 100% 未提前运行或调参**。

---

## 1. 复核口径与共有裁定标尺（100% 复用主轨硬门槛与 100 分计分）

[eval_sealed_secondary_w4w5_dec.py](finetune/gpu_alpha/eval_sealed_secondary_w4w5_dec.py) 直接导入并复用主轨 [eval_sealed_review_dec.py](finetune/gpu_alpha/eval_sealed_review_dec.py) 中的 `ProtocolConfig`、`compute_breadth_map`、`classify_dates_by_regime`、`check_prerequisites`、`evaluate_horizon_scope` 与 `score_candidate`，保证两轨采用逐比特一致的客观标尺：

1. **前置硬门槛（`check_prerequisites`，一票否决，否则输出 `DEFER_AUTO_EXTEND`）**：
   - 线上复现一致性门禁：`score_hidden_corr > 0.9999` 且 `score_fused_corr > 0.9999`；
   - 口径 B 累计 `D10` 到期信号日 **`>= 20`** 天（`cumulative_start = 2026-10-26`）；
   - 口径 A 纯增量未拆封 `D10` 到期信号日 **`>= 10`** 天（`strict_d10_start = 2026-11-09`；`D5` 为 `2026-11-16`）；
   - 口径 B 归属 `NORMAL_ACTIVE`（`breadth_ma20 >= 0.30`, `univ_ret_d10 ∈ (-0.05, +0.08]`, `min_delta_breadth_5d > -0.20`）的 `D10` 信号日 **`>= 8`** 天。
2. **分体制 100 分计分与红线否决**：
   - 常态开仓期 `65%`（`IC/ICIR` 25 + `NIC` 15 + `liq10/20` 15 + `D10/D5` 胜率 10）+ 极端开仓期 `30%`（极端 `liq10/20` 15 + `NIC` 韧性 10 + `D5` 抗反噬 5）+ 空仓防御期 `5%`；
   - 红线否决：`<= 0.010` 增量常态退化、极端 `liq10 < -1.5%` 或 `NIC <= 0`、极端 `D5 IC <= -0.02`；
   - 达标阈值：`A_PROMOTE >= 80`（且口径 A 均不弱于 `FUSE_VAL_1E4` 且至少两项严格胜出），`B_SHADOW_CANARY 65..79.5`，`<= 5` 分差距内优先部署复杂度更低档位（`SECONDARY_COMPLEXITY_RANK`）。

---

## 2. 事前冻结参评名单（全部 `No-G32`、全部剔除 `G36 LambdaRank`）

所有组合均严格采用**零 OOS 调权等权横截面秩平均**（成员来自 [eval_wave5_master_ensembles.py](finetune/gpu_alpha/eval_wave5_master_ensembles.py) 的 65d OOS 终版定型清单）：

| 类别 | 候选 / 基准名称 | 核心架构与推理依赖 | 65d OOS（`<=10-23`）关键参照 | 复杂度排序 |
|---|---|---|---|---:|
| **生产对照** | `FUSE_VAL_1E4` | `r_c2 + r_p_val_1e4`（当前线上） | 平行轨基准对照 | — |
| **基线 0** | `C2` | 原版 `C2 L=120` 16 路 AR | `IC 0.1428, k10 +11.88% (-22.09%)` | — |
| **赛道 A · 纯 `C2`** | `NOG32_C2ONLY_DISTILL5` | 仅 `C2` 底座（W3/G37，**零 `G32/Base/PTS/Mini`**） | `IC 0.2018, k10 +13.59%` | `1` |
| **赛道 A · W5 纯 `C2`** | `W5_C2ONLY_TAILFWD_NOG32` | 仅 `C2` 底座融合 `G37 + G39 fwd5_10/cont_shield` | `IC 0.2035, NIC 0.1310, s5 +14.40%` | `1` |
| **赛道 B · E2 轻量** | `NOG32_LIGHT_MINIC2_DISTILL6` | 冻结 `C2 + 4.1M Mini`（零 `Base/PTS/G32`） | `k10 +25.13% (-10.06% / Cal 2.497), s5 HYB Cal 3.801` | `2` |
| **赛道 B · G39 轻量** | `W5_LIGHT_NOG32_G39_TAILFWD` | `C2 + 4.1M Mini` + `G39` 三目标抗跌延续蒸馏 | `IC 0.2145, Seg B 0.186, s5 HYB Cal 2.777` | `2` |
| **赛道 B · G40 `FT-Mini`** | `W5_LIGHT_NOG32_G40_FTMINI` | `C2 + 4.1M STE FT-Mini`（零 `Base/PTS`） | `IC 0.2230, NIC 0.1482, s5 RAW +25.12% (-7.73% / Cal 3.249)` | `3` |
| **赛道 B · 极简合体** | `W5_LIGHT_NOG32_SUPER_MINIC2` | W2–W5 `Mini + FT-Mini + C2` 总集成（零 `Base/PTS`） | `IC 0.2182, liq10 +3.68%, s5 G_HAZARD Cal 3.095` | `3` |
| **赛道 D · E2 重型** | `NOG32_CRASH_SHIELD_CORE` | 4 家族不含 `G32` + `G35_LGBM_CVAR10 + G38` | `IC 0.2315 (ICIR 2.14, 100%, Seg A/B 0.227/0.239)` | `4` |
| **赛道 D · W5 重型** | `W5_NOG32_CRASH_SHIELD_APEX` | W2–W5 四家族 + `G39/G40` 终极抗跌组合 | `IC 0.2268, Seg A/B 0.228/0.225, s5 DD -6.98%, MCB -3.37%` | `5` |
| **单头参照 1** | `SINGLE_G35_LGBM_CVAR10` | 四家族 `G35_lgbm_cvar10` | `IC 0.2448, Seg B 0.318` | `4` |
| **单头参照 2** | `SINGLE_G40_FTMINI_L45_E2E` | 端到端单核 `4.1M FT-Mini L=45` | `IC 0.1927, k10 +24.79% (-10.09% / Cal 2.457)` | `2` |

### 同期因果执行护盾旁路对照（`eval_sealed_execution_overlays`）
在满足门槛前提下，同步回放 6 套因果护盾在封存 K 线上的 `k=10` 扣费实盘对比（严格 `max_decision_day <= last(asof)`）：
- `RAW`、`G_BREADTH`、`G_BREADTH_VEL`、`G_HAZARD_G38`（来自 `sealed_g38_regime_hazard.csv`）、`RAW_MCB`、`G_HAZARD_G38_MCB`。

---

## 3. 12 月达标后一键复核标准流程（未到时候严禁提前执行）

1. 按主轨拉取最新 `sealed_kline_150d.parquet`、`sealed_shadow_rankings.parquet` 与 `sealed_w2w3_hidden_scores.parquet`，连同冻结的模型权重集合（`g35..g40` 产物与 `head_val_alpha1e4.npy`）打包更新 Kaggle 数据集 `wynstonliu/kronos-sealed-dec-w4w5-bundle`；
2. 运行推理推流脚本上传并执行纯前向打分核 [`sealed_w4w5_secondary_score_entry.py`](finetune/gpu_alpha/sealed_w4w5_secondary_score_entry.py)：
   ```bash
   python3 finetune/gpu_alpha/push_sealed_w4w5_secondary.py --push
   ```
3. 将 `sealed_w4w5_out/` 下载至 `scratch/sealed_dec_secondary/`，一次性运行本地判决脚本输出 100 分榜、红线检查与执行护盾矩阵：
   ```bash
   python3 finetune/gpu_alpha/eval_sealed_secondary_w4w5_dec.py --sealed-dir scratch/sealed_dec_secondary --output finetune/reports/202612_sealed_secondary_w4w5_results.json
   ```
