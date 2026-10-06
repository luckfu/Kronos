# Beta v2.1 C1：val 生成式 IC 重新选 checkpoint（Step 1 结果）

更新：2026-10-06 CST

## 一句话结论

在验证集上按**生成式 return10d Rank IC**（而不是 WFL）重新打分后，排第一的是 **Beta v2.1 发布版 Best@475**（C1 之前的父本），
不是 C1 forecast 训练出的 Seg155。C1 forecast 训练**降低了 WFL，却让 val 生成式 IC 小幅下降**。
所以 WFL 不能用来代替生成式 IC 选 checkpoint。原计划从 Seg155 出发的 12 段 forecast-only 余弦 pilot（`771d09b`）
**暂缓，等父 checkpoint 定下来再说**。注意：Best@475 **还没有在密封 OOS 上评估过**，它的 OOS 生成式 IC 未知。

## Kernel

- Slug：`luckfu/kronos-beta-v21-c1-val-gen-ic`，**COMPLETE**，耗时 4514 s（约 1.25 h），双 T4
- 代码：`c7b107b`（`finetune/kaggle_beta_v21_c1_val_gen_ic.py`、`finetune/evaluate_beta_v21_val_gen_ic.py`）
- `training=false`（零训练），`sealed_oos_read=false`（**没有读密封 OOS**）
- 原始输出（box）：`/workspace/kronos_patrol_out/valgenic_final_1006_1425/output/beta_v2_1_c1_val_gen_ic/results/`
  （`comparison.json` + 3 个 `*_val_summary.json` + `*_val_predictions.csv.gz`；`plan.json` 在上一级）
- 机器可读汇总：`finetune/reports/beta_v21_c1_val_gen_ic_reselection.json`

## 评估合同

| 项 | 值 |
|---|---|
| 数据 | `temporal_symbol_validation_v1` `val_data.pkl`（sha `4cce31bc…`），signal 2025-07-01 → 2026-07-02，全量 242 日 / 123,836 窗口 |
| 子样本 | 排序后的 signal date 上 `np.linspace(0, 241, 24).round()`，每日全部股票 → **24 日 / 12,256 窗口**（相邻日期约隔 10.5 个交易日） |
| Decode | 生产配方 prod T0.65 / top_p 0.8 / N5，seed **20260906**（单 seed） |
| 分数 | `predicted_return_d10` = N5 条 AR 采样的 `denorm_close_d10 / last_close − 1` 均值 |
| 标签 | `return_10d = close[asof+10] / close[asof] − 1` |
| 主指标 | return10d Rank IC 日均（每日 Spearman 再平均） |
| WFL | 同一子样本上的 teacher-forcing CE，C1 horizon 权重 1.364…0.455，逐样本平均 |

## 三个 checkpoint

| 标签 | 说明 | SHA |
|---|---|---|
| `beta_v21_release_best475` | Beta v2.1 发布版 Best@475，即 C1 forecast 微调之前的父本（含 aux 头） | `e1bd5584…` |
| `seg8_rank_unfreeze_best` | v20：从 Seg19 解冻 trunk（1e-7）+ ranking aux，forecast 路径有变化 | `8a29277a…` |
| `seg155_forecast_best` | C1 wc forecast 最低 WFL（全局 Seg155 = local segment 26），forecast-only，无 aux | `8b11a759…` |

## 主表（24 日子样本）

| checkpoint | ret10d IC 日均 | SE | pooled | ICIR | IC+ 日 | Top-Bottom 十分位 | 五分位 | utility IC 日均 | WFL（子样本） | WFL（全量 val，训练日志） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Best@475（发布版）** | **0.3145** | 0.0252 | **0.4574** | **2.544** | 24/24 | **+7.17%** | +6.06% | 0.1455 | 2.3205 | —（未记录） |
| Seg8 rank-unfreeze | 0.2989 | 0.0260 | 0.4375 | 2.344 | 24/24 | +6.03% | +5.67% | 0.1462 | 2.3221 | 2.3255 |
| Seg155 forecast | 0.2954 | 0.0251 | 0.4312 | 2.399 | 24/24 | +6.24% | +5.85% | 0.1438 | **2.3091** | **2.3124** |

Top-Bottom = 每日按分数最高组减最低组的平均 return10d，再取日均，不含换手和成本。
utility IC 是纯收益分数接 utility 标签的 ablation，三者几乎一样（0.144–0.146）。

### 排序

| 依据 | 排序 |
|---|---|
| WFL（子样本，越低越好） | Seg155（2.3091） < Best@475（2.3205） < Seg8（2.3221） |
| 生成式 ret10d IC 日均 | Best@475（0.3145） > Seg8（0.2989） > Seg155（0.2954） |

`divergence_wfl_subsample_vs_ic = true`，`divergence_wfl_full_vs_ic = true`。WFL 最好的 Seg155 在 IC 上排最后。

## 配对比较（逐日配对，24 日）

由 `by_signal_date` 重新计算；t 按 iid、df = 23。

| 对比 | 指标 | 日均差 | SE | t | p（双侧） | 胜日 | 符号检验 p | 最小 / 最大日差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Best@475 − Seg155 | ret10d IC | **+0.0191** | 0.0052 | **3.64** | 0.0014 | **19/24** | 0.0066 | −0.043 / +0.068 |
| Best@475 − Seg155 | TB 十分位 | +0.93% | 0.34% | 2.76 | 0.011 | 17/24 | 0.064 | −1.62% / +5.76% |
| Best@475 − Seg8 | ret10d IC | **+0.0155** | 0.0044 | **3.56** | 0.0017 | **20/24** | 0.0015 | −0.038 / +0.064 |
| Best@475 − Seg8 | TB 十分位 | +1.14% | 0.26% | 4.40 | 0.0002 | 19/24 | 0.0066 | −1.00% / +3.97% |
| Seg8 − Seg155 | ret10d IC | +0.0036 | 0.0045 | 0.79 | 0.44 | 13/24 | 0.84 | −0.058 / +0.056 |

怎么读：

- 差值本身不大：+0.019 约为单个 checkpoint 日均 IC 的 SE（~0.025）的 0.75 倍，相对提升约 6%。
  配对 t 之所以高，是因为三个 checkpoint 每天的 IC 同涨同跌，配对后把共同波动抵掉了。
- 方向一致：Best@475 在 19/24（对 Seg155）、20/24（对 Seg8）天更高。
- Seg8 和 Seg155 之间没有可分辨的差别。

## 逐日 return10d Rank IC

| asof | 样本 | Best@475 | Seg8 | Seg155 | Best@475 − Seg155 | 全市场 ret10d 均值 |
|---|---:|---:|---:|---:|---:|---:|
| 2025-07-03 | 477 | 0.285 | 0.268 | 0.246 | +0.039 | +2.36% |
| 2025-07-17 | 510 | 0.187 | 0.177 | 0.176 | +0.011 | +1.40% |
| 2025-08-01 | 511 | 0.254 | 0.239 | 0.208 | +0.045 | +4.57% |
| 2025-08-15 | 510 | 0.380 | 0.348 | 0.360 | +0.020 | +2.37% |
| 2025-09-01 | 511 | 0.301 | 0.326 | 0.309 | -0.008 | +0.13% |
| 2025-09-15 | 512 | 0.266 | 0.256 | 0.253 | +0.013 | -0.56% |
| 2025-09-30 | 512 | 0.599 | 0.565 | 0.572 | +0.027 | +0.41% |
| 2025-10-22 | 511 | 0.192 | 0.190 | 0.198 | -0.006 | +1.80% |
| 2025-11-06 | 512 | 0.349 | 0.346 | 0.351 | -0.003 | -1.43% |
| 2025-11-20 | 512 | 0.194 | 0.185 | 0.190 | +0.004 | -1.47% |
| 2025-12-05 | 511 | 0.173 | 0.149 | 0.170 | +0.004 | -0.43% |
| 2025-12-19 | 513 | 0.420 | 0.421 | 0.413 | +0.008 | +4.16% |
| 2026-01-07 | 512 | 0.248 | 0.227 | 0.215 | +0.034 | +5.76% |
| 2026-01-21 | 514 | 0.257 | 0.227 | 0.224 | +0.033 | -0.65% |
| 2026-02-05 | 511 | 0.403 | 0.441 | 0.445 | -0.043 | +4.86% |
| 2026-02-27 | 514 | 0.388 | 0.374 | 0.384 | +0.004 | -3.49% |
| 2026-03-16 | 514 | 0.194 | 0.191 | 0.149 | +0.045 | -5.00% |
| 2026-03-30 | 513 | 0.339 | 0.340 | 0.347 | -0.008 | +2.11% |
| 2026-04-15 | 514 | 0.276 | 0.246 | 0.240 | +0.036 | +2.39% |
| 2026-04-29 | 508 | 0.430 | 0.405 | 0.411 | +0.019 | +2.79% |
| 2026-05-19 | 515 | 0.206 | 0.142 | 0.200 | +0.006 | -5.71% |
| 2026-06-02 | 514 | 0.296 | 0.258 | 0.246 | +0.049 | -1.28% |
| 2026-06-17 | 512 | 0.257 | 0.212 | 0.194 | +0.063 | -2.45% |
| 2026-07-02 | 513 | 0.653 | 0.641 | 0.586 | +0.068 | -4.78% |

三个 checkpoint 每天 IC 都是正的。Best@475 的优势没有集中在某一种市场状态：涨市日（如 08-01、01-07）和跌市日（如 03-16、06-17、07-02）都有较大正差；
落后的 5 天（09-01、10-22、11-06、02-05、03-30）里只有 02-05（−0.043）差得明显。

## 解读

1. **C1 forecast 训练对生成式 IC 没有帮助，反而略有损害。** 从 Best@475 到 Seg155，子样本 WFL 从 2.3205 降到 2.3091，
   但 val 生成式 return10d IC 日均从 0.3145 降到 0.2954（配对 −0.019，19/24 天更差），十分位价差从 +7.17% 降到 +6.24%。
2. **WFL 不是选 checkpoint 的有效代理指标。** 三个 checkpoint 的 WFL 排序和 IC 排序完全不一致；WFL 最好的那个 IC 最差。
   以后按生成式用途选 checkpoint，要直接测 val 生成式 IC，不能用 WFL 选。
3. **所以 Seg155 作为余弦 pilot 的父本不再有充分理由。** `771d09b` 的 12 段 forecast-only 余弦 pilot
   （`finetune/docs/beta_v21_c1_forecast_cosine_pilot_cn.md`，**没有在 Kaggle 上启动**）的父本是按 WFL 选出来的 Seg155。
   在父本定下来之前，**这个 pilot 暂缓**。
4. **不要夸大。** 这只说明在 val 上 Best@475 的生成式 IC 比 Seg155 高约 0.02，**不说明** Best@475 在密封 OOS 上能追上 C2（0.18）。

## 注意事项

- **子样本**：只用了 242 个 val 日期中的 24 个（linspace），每个 checkpoint 12,256 窗口。
- **单 decode seed**（20260906，N5）。三个 checkpoint 用同一 seed，差值不受 seed 偏向影响，但没有测 seed 方差。
- **配对 t 偏乐观**：相邻日期约隔 10.5 个交易日，10 日标签窗口基本首尾相接、少量重叠；市场状态也有自相关。p 值只作参考，结论写成「差约 0.02、方向稳定」。
- **val 生成式 IC 相对密封 OOS 偏乐观**：Seg155 在 val 上是 0.295，在密封 18d OOS 上只有 **0.117（prod）/ 0.124（rank）**。
  Best@475 的 val 优势能不能保留到 OOS 是未知数。
- **Best@475 还没有在密封 OOS 上评估过。** 如果用密封 OOS 来决定选哪个 checkpoint，
  这份 OOS 就部分失去「只用于评估」的地位（等于用它做了一次模型选择）。
- Best@475 的全量 val WFL 本 kernel 没有记录（只有子样本 WFL 2.3205）。

## 下一步选项

> **2026-10-06 已选 A**：余弦 pilot 父本换成 Best@475，新 slug `luckfu/kronos-beta-v21-c1-forecast-cosine-pilot-best475`，
> 见 [`beta_v21_c1_forecast_cosine_pilot_cn.md`](beta_v21_c1_forecast_cosine_pilot_cn.md)。B、C 未执行。
>
> **2026-10-06 pilot 结果（COMPLETE）**：同合同 val 生成式 IC 日均 Seg0（Best@475）0.3141 → Seg9 **0.3254**（最高）/ Seg12 0.3234 / Seg6 0.3245；
> Seg9 − Seg0 配对 +0.0113（t 1.80，15/24），Seg6–12 一致高约 +0.01 但不确定；WFL 最好的 Seg10（2.3178）IC 0.3196，又不是 IC 最好；
> Seg9 − Seg155 +0.030（t 4.72，21/24）。详见 [`beta_v21_c1_forecast_cosine_pilot_cn.md`](beta_v21_c1_forecast_cosine_pilot_cn.md)、
> `finetune/reports/beta_v21_c1_forecast_cosine_pilot_best475.json`。下一步（GPU 配额恢复后）：Seg9 + Seg0 一次性密封 OOS 最终评估（kernel 已备好未推送）。


| 选项 | 做法 | 优点 | 代价 / 风险 |
|---|---|---|---|
| **A** | 余弦 pilot 父本换成 **Best@475**，其余配方不变（12 段 forecast-only，uniform_cosine 1e-5→1e-6，段内 snapshot + val 生成式 IC） | 直接在 val IC 最好的起点上测退火；不碰密封 OOS | 需要改 kernel 的父本数据集（Best@475 带 aux 头，forecast-only 时需确认 aux 关闭后加载正常）；判定基线换成 Best@475 的 0.3145 |
| **B** | 先让 **Best@475 跑一遍密封 18d OOS 生成式收益**（Baseline 2 同配方，prod N5，约 2.7 h） | 直接知道 Best@475 在 OOS 上是否比 Seg155 的 0.117 更好、离 C2 0.18 多远 | 用 OOS 做 checkpoint 选择，部分消耗它「只评估」的地位；之后再做的 OOS 比较要注明这一点 |
| **C** | 按原计划跑 **Seg155 父本**的余弦 pilot（`771d09b`） | 不用改代码，可直接启动 | 父本是按 WFL 选的，而 WFL 已被证明不能代表生成式 IC；val 上起点低约 0.02 |

## 相关路径

- 本文 JSON：`finetune/reports/beta_v21_c1_val_gen_ic_reselection.json`
- 余弦 pilot（Best@475 父本，已完成）：`finetune/docs/beta_v21_c1_forecast_cosine_pilot_cn.md`；JSON `finetune/reports/beta_v21_c1_forecast_cosine_pilot_best475.json`
- 密封 OOS 对比 / Baseline 2：`finetune/docs/beta_v21_c1_rank_oos_vs_small_c2_cn.md`
- Kernel 代码：`finetune/kaggle_beta_v21_c1_val_gen_ic.py`、`finetune/evaluate_beta_v21_val_gen_ic.py`
