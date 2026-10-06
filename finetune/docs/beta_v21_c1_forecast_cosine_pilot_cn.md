# Beta v2.1 C1 forecast-only 余弦退火 pilot（12 seg，双 T4）

更新：2026-10-06 CST（kernel **COMPLETE**，结果见下方「结果」各节）

> **状态：COMPLETE（2026-10-06，耗时 15,236 s ≈ 4.2 h）。父本 = Best@475。**
> 结论：从发布版 Best@475 做余弦退火，val 生成式 return10d IC 有**小幅、方向一致但不确定**的提升
> （Seg6–12 比 Seg0 高约 +0.01，配对 t 1.7–2.0）。val IC 最好的是 **Seg9 0.3254**（Seg0 0.3141）；
> WFL 最好的 Seg10（2.3178）**又一次不是** IC 最好的；所有 pilot snapshot 在 val 生成式 IC 上都明显高于 Seg155（0.2954）。
> 按事先写好的判定规则（需高出 ≥ 1 个 SE ≈ 0.025），本结果属于 **「持平」**，退火解释不了与 C2 的差距。
> Seg9 权重已存为 Kaggle 私有数据集 `luckfu/kronos-beta-v21-c1-cosine-pilot-best475-seg9`。
> 下一步：GPU 配额恢复后，对 **Seg9 与 Seg0（Best@475）一起做一次性密封 OOS 生成式收益评估**，
> 作为最终评估，**不是调参循环**（kernel 已备好未推送，见「建议下一步」）。
>
> 背景（2026-10-06 启动时）：Step 1 val 生成式 IC 重选（`luckfu/kronos-beta-v21-c1-val-gen-ic`，见
> [`beta_v21_c1_val_gen_ic_reselection_cn.md`](beta_v21_c1_val_gen_ic_reselection_cn.md)）显示
> Best@475（0.3145）> Seg8（0.2989）> Seg155（0.2954），而 WFL 排序正好相反，所以父本从 Seg155 换成 Best@475。
> Seg155 版本仍可构建（`--parent seg155`），但**未启动**。

## 要回答的问题

密封 18d OOS 上，我们 Seg155 的生成式 return10d 日均 Rank IC 是 **0.117**（prod arm），
C2（small stage2 cosine refinement）是 **0.177 / 0.180**（prod / rank）。
C2 与 C1 的最大配方差异之一是 C2 最后走了 **uniform_cosine 1e-5 → 1e-6 退火**，
C1 wc 线一直是 warmup_constant hold。本 pilot 只验证一件事：

> 从 val 生成式 IC 最好的 Best@475 出发做 C2 式余弦退火，val 生成式 IC 会不会随退火上升？

## Kernel

| 项 | Best@475 版（本次启动） | Seg155 版（保留，未启动） |
|---|---|---|
| Slug | `luckfu/kronos-beta-v21-c1-forecast-cosine-pilot-best475` | `luckfu/kronos-beta-v21-c1-forecast-cosine-pilot` |
| 构建 | `python finetune/build_kaggle_beta_v21_c1_forecast_cosine_pilot_kernel.py`（默认 `--parent best475`） | 同上加 `--parent seg155` |
| Staging | `finetune/kaggle_beta_v21_c1_forecast_cosine_pilot_best475_kernel/` | `finetune/kaggle_beta_v21_c1_forecast_cosine_pilot_kernel/` |
| 输出目录 | `/kaggle/working/kronos_beta_v21_c1_forecast_cosine_pilot_best475/outputs/models/beta_v2_1_c1_forecast_cosine_pilot_best475/` | `…/kronos_beta_v21_c1_forecast_cosine_pilot/outputs/models/beta_v2_1_c1_forecast_cosine_pilot/` |
| SwanLab | <https://swanlab.cn/@roc_fu/finance/runs/beta_v2_1_c1_forecast_cosine_pilot_best475> | `…/runs/beta_v2_1_c1_forecast_cosine_pilot` |
| 数据集 | `luckfu/a-share-120d-temporal-symbol-holdout` | 再加 `luckfu/kronos-beta-v21-c1-seg155-forecast-best` |

- Runner：`finetune/kaggle_beta_v21_c1_forecast_cosine_pilot.py`（`PILOT_PARENT` 由 builder 改写，源文件默认 `best475`）
- 双 T4，docker pin `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c5…d461`，无 TPU
- SwanLab：project `finance` / workspace `roc_fu`，run id = 输出名；API key 依次取环境变量 →
  Kaggle secret `SWANLAB_API_KEY` → 代码内 fallback（与 C1 dual-T4 训练 kernel 相同）

## 父本：Best@475

| 项 | 值 |
|---|---|
| 来源 | ModelScope `luckfu/Kronos-A-Share-Beta-V2-1` 根目录 `model.safetensors`，SHA `e1bd55842996…2f97`；定位方式与 val-gen-ic kernel 的 `beta_v21_release_best475` 完全相同（按 SHA 在 snapshot 里找） |
| config | `num_sectors=86`、`context_layer=10`、`use_size_percentile=true`、`size_mlp_hidden_dim=64`，与训练环境一致；`use_beta_v21_auxiliary=true`（**带 aux 头**） |
| aux 头处理 | pilot 是 forecast-only：模型按 `use_beta_v21_auxiliary=False` 构建，只加载 trunk + forecast head（197 个张量），**丢弃** `return_head.*` / `barrier_head.*` 共 4 个张量，aux 头不参与训练 |
| 加载检查 | 训练前 runner 用同样参数构建模型，与 safetensors header 比对：只允许 aux 头张量多出/缺失；任何其它多出、缺失或形状不符的 key 都直接中止。`train_predictor.py` 加载后另打一行 `Parent weights: …` 日志（只记录，不报错），两个方向（带 aux 父本 → 无 aux 模型、无 aux 父本 → 带 aux 模型）都不会崩 |
| 初始化 | 仅权重；**新 AdamW**；不加载任何 last_state |
| Best 阈值 | `KEEP_EXISTING_BEST=0`：Best@475 没有 C1 全量 val WFL 记录，训练内 best_model 从第 1 段起按 WFL 重新记（只作参考，不用于选择） |
| Step 1 参照 | val 生成式 IC 日均 0.31447（SE 0.0252），子样本 WFL 2.3205 |

## 配方

| 项 | 值 |
|---|---|
| 损失 | forecast-only（`PREDICTOR_LOSS_MODE=forecast`，history 0.02，C1 horizon 权重 1.364…0.455） |
| aux / ranking | **关**（`USE_BETA_V21_AUXILIARY=0`；ranking weight 0.05 仅为 config 解析需要，aux 关时不生效） |
| 喂数 | `SAME_DAY_RANKING_BATCHES=0`：每段取 seed 20261002 随机置换后的 coverage 切片，DistributedSampler `shuffle=False` 按该乱序读取，**段内不按 signal_date 排序**（`dataset.py` `set_epoch_seed` 明确不 argsort 日期） |
| 可训练 | 全部参数（`TRAINABLE_TRANSFORMER_LAYERS=-1`），单 LR 族（`SPLIT_TRUNK_HEAD_LR=0`），同 C2 |
| 调度 | `uniform_cosine`，1e-5 → 1e-6，warmup 0，**恰好 12 段**（`EPOCHS=12`，一个全局计划） |
| 段末 LR（预期） | Seg3 8.68e-6 / Seg6 5.50e-6 / Seg9 2.32e-6 / Seg12 1.00e-6 |
| 分块训练 | 4 块 × 3 段：`MAX_SEGMENTS_PER_RUN=3`，第 2 块起 `RESUME_TRAINING=1` 从 `last_state.pt` 续（optimizer、scheduler、AMP scaler、RNG 全部恢复；单测验证 2+2 分块与一次跑 4 段的权重和 LR 一致） |
| Coverage | seed 20261002，offset 0，20000 样本/段；batch 32 × 2 GPU = 64，AMP fp16 |
| 验证 | 每段 full val（123,836 窗口）记 WFL |
| Snapshot | 每 3 段（Seg3/6/9/12）→ `snapshots/segNNN`（权重 + `snapshot_metric.json`） |
| WFL 红线 | C1 线红线 2.32736787（Seg155 全量 WFL 2.31236787 + 0.015），只打标记，不停训、不参与选择 |

## In-kernel val 生成式 IC（与 Step 1 同合同）

每块训练结束、GPU 空出来后，**立刻**用 `finetune/val_gen_ic_driver.py` 给新产生的 checkpoint 打分：

| 时点 | 打分对象 |
|---|---|
| 第 1 块（Seg1–3）后 | **Seg0 = 父本 Best@475**（`seg000_beta_v21_release_best475`，先打），然后 Seg3 |
| 第 2 / 3 / 4 块后 | Seg6 / Seg9 / Seg12 |
| 训练结束后 | 训练内 best_model（按 WFL）若与所有 snapshot 和父本 SHA 都不同，再加打一个 |

- 每个 checkpoint 的 24 个日期 shard 一到齐就落盘：`val_gen_ic/results/<label>_val_summary.json`、
  `<label>_val_predictions.csv.gz`，并更新累计的 `val_gen_ic/comparison.json`（每次都含之前所有 checkpoint）
- 每次打分后重写 `pilot_selection.json`：**按 val 生成式 return10d IC 日均选最优**，同时列出子样本 WFL、
  全量 val WFL、是否超 WFL 红线、相对 Seg0 的 IC 差；WFL 只记录不参与选择
- SwanLab：`train/loss`、`train/forecast_loss`、`train/learning_rate` 每 100 step；每段 `validation/weighted_forecast_loss`；
  每个 checkpoint `valgenic/return10d_ic_daily`、`valgenic/wfl_subsample`、`valgenic/delta_ic_vs_seg0`（x 轴 = 段号 × 每段 step）
- 合同：`temporal_symbol_validation_v1` `val_data.pkl`（sha `4cce31bc…`），242 日中 `np.linspace(0, 241, 24).round()` 取 24 日、
  每日全部股票 → 12,256 窗口；prod decode T0.65 / p0.8 / N5，seed 20260906；分数 = mean(close_d10 / last_close − 1)；
  label = raw close return_10d；同窗口 teacher-forcing WFL；不读密封 OOS
- 父本 Seg0 在本 kernel 内重打一次，可与 Step 1 的 0.3145 对照，检验复现性

## 时间估算

| 阶段 | 估计 |
|---|---|
| 下载 / 安装 | ~6 min |
| 训练 | 每段 ~7.3 min（C1 wc forecast-only 实测 7:15–7:17，含 full val）× 12 ≈ 1.5 h；每块重启 ~2.5 min × 4 |
| 打分 | Step 1 实测每个 checkpoint ~24–27 min；Seg0 + Seg3/6/9/12 = 5 次 ≈ 2.2 h，best_model 若不同再 +27 min |
| 合计 | **约 4.0–4.5 h**；最坏（每段 13.5 min）约 5.6 h，远低于 12 h |

所以**不削减打分**：5 个点全打，每个都是完整 24 日 / N5。保护：9.5 h 后不再开新训练块；11.5 h 后打分 worker 不再接新 shard。

## 启动记录（2026-10-06）

- Kaggle：`luckfu/kronos-beta-v21-c1-forecast-cosine-pilot-best475` **version 1**，14:38 CST push，状态 RUNNING；
  kernel 内 `github_ready` commit `8ec7d0b`
- SwanLab：<https://swanlab.cn/@roc_fu/finance/runs/beta_v2_1_c1_forecast_cosine_pilot_best475>（key 来源 fallback；Kaggle secret 未挂到该 kernel）
- 加载检查通过：父本 201 个张量，模型 197 个全部加载，丢弃 `barrier_head.{bias,weight}`、`return_head.{bias,weight}`；
  可训练参数 102,437,248 / 102,437,248；LR 计划 3,756 步（313 步/段 × 12），warmup 0
- 首批训练日志：

  ```
  [Rank 0, Segment 1/12, Step 100/313] Adaptation LR 9.9842682191e-06, ..., Loss: 2.1733, Forecast: 2.1205, History: 2.4591
  [Rank 0, Segment 1/12, Step 200/313] Adaptation LR 9.9371828716e-06, ..., Loss: 2.2328, Forecast: 2.1893, History: 2.4598
  [Rank 0, Segment 1/12, Step 300/313] Adaptation LR 9.8590731735e-06, ..., Loss: 2.2947, Forecast: 2.2768, History: 2.2898
  ```

- 第 1 块：每段 7:46–7:52（含 full val）；全量 val WFL Seg1 2.32170 / Seg2 2.32015 / Seg3 2.32431（均低于红线 2.32737）；
  Seg3 snapshot 落盘后训练进程正常退出（`Chunk limit reached after 3 segment(s)`），随即开始打 Seg0 + Seg3
- Seg0 前 4 个日期的逐日 IC（0.289 / 0.187 / 0.253 / 0.381）与 Step 1 的 Best@475（0.285 / 0.187 / 0.254 / 0.380）基本一致；
  每个 shard ~110–118 s，即每个 checkpoint ~23 min
- 预计：Seg0/Seg3 结果约 16:00 CST，之后每 ~50 min 一个 snapshot（Seg6 ~16:50、Seg9 ~17:40、Seg12 ~18:30），
  全部结束约 **18:30–19:00 CST**（总 ~4 h）

## 判定规则

以本 kernel 内 Seg0（Best@475）的 val 生成式 IC 日均为基线（Step 1 为 0.3145，SE 0.025），看 Seg3 → 6 → 9 → 12：

1. **随退火上升**（后段 snapshot 明显高于 Seg0，例如 Seg9/12 高出 ≥ 1 个 SE，且趋势单调或接近单调）→
   拿 val IC 最好的 snapshot 去密封 18d OOS，用 Baseline 2 同配方测试，对照 **0.117**（Seg155）/ **0.180**（C2）。
2. **持平**（都在 Seg0 ± 1 SE 内，无趋势）→ **停**：差距不来自退火。
3. **下降** → 同样停，记录退火对该父本有害（与 Step 1 观察到的「C1 forecast 训练降 WFL 但降 IC」一致）。

## 结果（2026-10-06，kernel COMPLETE）

- Kaggle：`luckfu/kronos-beta-v21-c1-forecast-cosine-pilot-best475` version 1，**COMPLETE**，总耗时 15,235.6 s（≈ 4.2 h），12/12 段全部完成
- SwanLab：<https://swanlab.cn/@roc_fu/finance/runs/beta_v2_1_c1_forecast_cosine_pilot_best475>（28,261 条记录上传完成）
- 原始输出（box）：`/workspace/kronos_patrol_out/pilot475_final_1006_2116/output/`
  （`pilot_selection.json`、`summary.json`、`val_gen_ic/results/*_val_summary.json`、`comparison.json`）
- 机器可读结果：`finetune/reports/beta_v21_c1_forecast_cosine_pilot_best475.json`
- `sealed_oos_read=false`：本 kernel **没有读密封 OOS**
- 下面的配对统计是在 box 上根据各 `*_val_summary.json` 的逐日 IC **重新计算**的，与 kernel 内 `pilot_selection.json` 的数字一致

### 主表（val 24 日子样本，prod T0.65 / p0.8 / N5，seed 20260906）

| Checkpoint | 段末 LR | ret10d IC 日均 | SE | ICIR | pooled | Top-Bottom 十分位 | 子样本 WFL | 全量 val WFL | 相对 Seg0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Seg0 = Best@475（父本） | — | 0.3141 | 0.0252 | 2.54 | 0.4571 | +7.16% | 2.3205 | —（未记录） | 0 |
| Seg3 | 8.68e-6 | 0.3140 | 0.0247 | 2.59 | 0.4561 | +6.62% | 2.3208 | 2.3243 | −0.0000 |
| Seg6 | 5.50e-6 | 0.3245 | 0.0254 | 2.60 | 0.4704 | +6.97% | 2.3182 | 2.3220 | +0.0104 |
| **Seg9** | 2.32e-6 | **0.3254** | 0.0266 | 2.50 | **0.4728** | **+7.21%** | 2.3182 | 2.3215 | **+0.0113** |
| Seg10（训练内 WFL best） | 1.60e-6 | 0.3196 | 0.0256 | 2.55 | 0.4677 | +6.87% | **2.3144** | **2.3178** | +0.0055 |
| Seg12 | 1.00e-6 | 0.3234 | 0.0259 | 2.55 | 0.4687 | +7.15% | 2.3163 | 2.3198 | +0.0094 |
| *参照（Step 1 同合同）* Seg155 | — | 0.2954 | 0.0251 | 2.40 | 0.4312 | +6.24% | 2.3091 | 2.3124 | −0.0187 |
| *参照* Seg8 rank-unfreeze | — | 0.2989 | 0.0260 | 2.34 | 0.4375 | +6.03% | 2.3221 | 2.3255 | −0.0152 |

- 24 天全部 IC > 0（所有 checkpoint 正 IC 日占比 100%）
- 全量 val WFL 逐段：2.3217 / 2.3201 / 2.3243 / 2.3218 / 2.3217 / 2.3220 / 2.3251 / 2.3179 / 2.3215 / 2.3178 / 2.3209 / 2.3198；
  **红线 2.32737 从未越过**（最高 Seg7 2.3251）。最低 WFL 是 Seg10 2.3178（Seg8 2.3179 几乎相同），比 Seg155（2.3124）仍高
- **按 IC 最好：Seg9；按 WFL 最好：Seg10**——WFL 再次不能代表生成式 IC
- 复现性：本 kernel 内 Seg0 = 0.31407，Step 1 的 Best@475 = 0.31447，配对差 −0.0004（SE 0.0005，t −0.89），
  同一 checkpoint / decode / seed 的重跑噪声远小于下面 ~0.01 的差值

### 配对检验（逐日配对，n = 24，t 自由度 23）

| 比较 | 均差 | SE | t | p（双侧） | 95% CI | 胜出天数 | 符号检验 p | Wilcoxon p |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| Seg3 − Seg0 | −0.0000 | 0.0044 | −0.01 | 0.99 | [−0.0092, +0.0091] | 10/24 | 0.54 | 0.81 |
| Seg6 − Seg0 | +0.0104 | 0.0051 | 2.03 | 0.055 | [−0.0002, +0.0211] | 16/24 | 0.15 | 0.049 |
| **Seg9 − Seg0** | **+0.0113** | 0.0063 | **1.80** | 0.085 | [−0.0017, +0.0243] | 15/24 | 0.31 | 0.095 |
| Seg10 − Seg0 | +0.0055 | 0.0050 | 1.09 | 0.29 | [−0.0049, +0.0159] | 13/24 | 0.84 | 0.38 |
| Seg12 − Seg0 | +0.0094 | 0.0057 | 1.66 | 0.11 | [−0.0023, +0.0211] | 14/24 | 0.54 | 0.11 |
| (Seg6+Seg9+Seg12)/3 − Seg0 | +0.0104 | 0.0055 | 1.89 | — | — | 15/24 | — | — |
| Seg9 − Seg10 | +0.0058 | 0.0023 | 2.53 | 0.019 | [+0.0010, +0.0106] | 17/24 | 0.064 | 0.021 |
| Seg9 − Seg6 | +0.0009 | 0.0030 | 0.29 | 0.77 | [−0.0054, +0.0071] | 13/24 | 0.84 | 0.66 |
| Seg9 − Seg12 | +0.0019 | 0.0025 | 0.77 | 0.45 | [−0.0032, +0.0071] | 10/24 | 0.54 | 0.94 |
| Seg0 − Seg155 | +0.0187 | 0.0052 | 3.59 | 0.002 | [+0.0079, +0.0295] | 19/24 | 0.007 | 0.002 |
| Seg6 − Seg155 | +0.0291 | 0.0059 | 4.94 | <0.001 | [+0.0169, +0.0413] | 20/24 | 0.002 | <0.001 |
| **Seg9 − Seg155** | **+0.0300** | 0.0064 | **4.72** | <0.001 | [+0.0169, +0.0431] | **21/24** | <0.001 | <0.001 |
| Seg10 − Seg155 | +0.0242 | 0.0060 | 4.05 | <0.001 | [+0.0118, +0.0365] | 20/24 | 0.002 | <0.001 |
| **Seg12 − Seg155** | **+0.0281** | 0.0052 | **5.39** | <0.001 | [+0.0173, +0.0388] | **22/24** | <0.001 | <0.001 |

Seg155 的逐日 IC 来自 Step 1 kernel（同 24 个日期、同 decode、同 seed，已核对 `subsample_dates` 与 `decode` 完全一致）。

分半看 Seg9 − Seg0：前 12 日（2025H2）**+0.019**（8/12）；后 12 日（2026H1，离密封 OOS 窗口最近）只有 **+0.004**（7/12）。

### 逐日 return10d Rank IC（加粗 = 当日 pilot 六个点中最高）

| asof | 样本 | Seg0 | Seg3 | Seg6 | Seg9 | Seg10 | Seg12 | Seg155 | Seg9−Seg0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-07-03 | 477 | 0.289 | 0.268 | **0.291** | 0.250 | 0.256 | 0.260 | 0.246 | −0.039 |
| 2025-07-17 | 510 | 0.187 | 0.203 | 0.216 | **0.249** | 0.232 | 0.222 | 0.176 | +0.062 |
| 2025-08-01 | 511 | 0.253 | 0.249 | 0.280 | 0.281 | 0.268 | **0.284** | 0.208 | +0.028 |
| 2025-08-15 | 510 | **0.381** | 0.362 | 0.354 | 0.360 | 0.360 | 0.360 | 0.360 | −0.021 |
| 2025-09-01 | 511 | 0.297 | **0.329** | 0.309 | 0.326 | 0.325 | 0.326 | 0.309 | +0.029 |
| 2025-09-15 | 512 | 0.264 | 0.305 | 0.306 | **0.319** | 0.298 | 0.306 | 0.253 | +0.054 |
| 2025-09-30 | 512 | **0.598** | 0.589 | 0.582 | 0.597 | 0.589 | 0.586 | 0.572 | −0.001 |
| 2025-10-22 | 511 | 0.191 | 0.225 | 0.224 | **0.225** | 0.214 | 0.215 | 0.198 | +0.034 |
| 2025-11-06 | 512 | 0.349 | 0.350 | 0.352 | 0.355 | 0.351 | **0.359** | 0.351 | +0.006 |
| 2025-11-20 | 512 | 0.194 | 0.213 | **0.233** | 0.232 | 0.228 | 0.223 | 0.190 | +0.038 |
| 2025-12-05 | 511 | **0.176** | 0.175 | 0.175 | 0.163 | 0.164 | 0.164 | 0.170 | −0.013 |
| 2025-12-19 | 513 | 0.417 | 0.423 | 0.460 | 0.468 | 0.460 | **0.475** | 0.413 | +0.051 |
| 2026-01-07 | 512 | 0.246 | **0.273** | 0.272 | 0.263 | 0.244 | 0.243 | 0.215 | +0.017 |
| 2026-01-21 | 514 | 0.255 | 0.253 | **0.256** | 0.254 | 0.255 | 0.254 | 0.224 | −0.001 |
| 2026-02-05 | 511 | 0.406 | 0.403 | 0.436 | 0.428 | 0.403 | **0.438** | 0.445 | +0.022 |
| 2026-02-27 | 514 | 0.387 | 0.405 | 0.402 | 0.400 | 0.389 | **0.409** | 0.384 | +0.013 |
| 2026-03-16 | 514 | **0.188** | 0.167 | 0.174 | 0.157 | 0.168 | 0.156 | 0.149 | −0.032 |
| 2026-03-30 | 513 | 0.340 | 0.331 | 0.375 | 0.376 | 0.374 | **0.382** | 0.348 | +0.036 |
| 2026-04-15 | 514 | **0.278** | 0.254 | 0.252 | 0.248 | 0.259 | 0.262 | 0.240 | −0.030 |
| 2026-04-29 | 508 | 0.428 | **0.441** | 0.419 | 0.438 | 0.422 | 0.434 | 0.411 | +0.010 |
| 2026-05-19 | 515 | 0.207 | 0.165 | 0.194 | 0.200 | 0.189 | **0.213** | 0.200 | −0.007 |
| 2026-06-02 | 514 | 0.298 | 0.271 | **0.343** | 0.331 | 0.339 | 0.337 | 0.246 | +0.033 |
| 2026-06-17 | 512 | **0.255** | 0.254 | 0.212 | 0.205 | 0.220 | 0.205 | 0.194 | −0.051 |
| 2026-07-02 | 513 | 0.653 | 0.628 | 0.670 | **0.684** | 0.663 | 0.651 | 0.585 | +0.032 |
| **均值** | 12,256 | 0.3141 | 0.3140 | 0.3245 | **0.3254** | 0.3196 | 0.3234 | 0.2954 | +0.0113 |

### 对照判定规则

| 规则 | 本次 |
|---|---|
| 「上升」：后段 snapshot 比 Seg0 高 ≥ 1 个 SE（≈ 0.025，不配对）且大致单调 | 最大提升 +0.0113 ≈ 0.45 个不配对 SE；轨迹为 Seg3 持平 → Seg6 上升 → Seg9/12 平台，**不满足** |
| 「持平」：都在 Seg0 ± 1 SE 内 | **字面上属于这一档** |
| 「下降」 | 否 |

配对看（同一天比较，噪声小得多）：Seg6/9/12 一致比 Seg0 高约 +0.01（t 1.7–2.0，14–16/24 天），
所以准确说法是「**小幅、方向一致、但不确定**」，而不是「完全没效果」。

### 解读

1. **退火有小幅、一致但不确定的收益。** Seg6–12 比父本 Seg0 高约 +0.01（配对 t 1.7–2.0）；Seg3（LR 仍在 8.7e-6）没有提升，
   收益出现在 LR 降到 ~5.5e-6 以下之后，之后进入平台（Seg9 与 Seg6 / Seg12 差别在噪声内）。
2. **WFL 最好的又不是 IC 最好的。** 训练内按 WFL 选出的 Seg10（2.3178）IC 0.3196，比 Seg9 低 0.0058（t 2.53，17/24 天）。
   和 Step 1 的结论一致：选生成式 checkpoint 要直接测 val 生成式 IC，不能用 WFL。
3. **所有 pilot snapshot 都明显好于 Seg155。** Seg9 +0.030（t 4.72，21/24）、Seg12 +0.028（t 5.39，22/24）。
   其中大部分（+0.019）来自父本选择本身（Seg0 − Seg155 +0.0187，t 3.59），退火只多贡献约 +0.01。
4. **退火不是与 C2 差距的主要来源。** 密封 OOS 上 C2 − Seg155 ≈ 0.060；val 上退火只带来 ~0.01，即使全部保留到 OOS 也补不上差距。
   本 pilot 回答的问题（「C2 式退火能不能显著提高生成式 IC」）的答案是：**不能显著提高**。
5. **Seg9 仍是目前 val 生成式 IC 最好的冻结候选**，值得做一次密封 OOS 最终评估；但这是对候选的最终评估，不是对退火假设的继续追加。

### 注意事项

- **val 生成式 IC 相对密封 OOS 偏乐观**：Seg155 val 0.295，密封 18d OOS 只有 0.117（prod）/ 0.124（rank）。Seg9 和 Best@475 的 OOS 都**未知**。
- **选择偏差**：Seg9 是 5 个 pilot snapshot（Seg3/6/9/10/12）里 val IC 最高的，+0.0113 偏高估；Seg6/9/12 平均 +0.0104（t 1.89）更公允。
- **分半不稳**：Seg9 − Seg0 的收益集中在前半（2025H2 +0.019，8/12）；离 OOS 最近的后半（2026H1）只有 +0.004（7/12）。
- **子样本 + 单 seed**：242 个 val 日期只用了 24 个；decode 只用 seed 20260906、N5，未测 seed 方差（同 seed 重跑噪声 ~0.0005）。
- **配对 t 偏乐观**：相邻日期约隔 10.5 个交易日，10 日标签窗口基本首尾相接、少量重叠，市场状态也有自相关；p 值只作参考。
- Seg0（Best@475）没有全量 val WFL；Seg0 带 aux 头（AR decode 不用），pilot snapshot 全部 forecast-only（无 aux 头）。

### 权重持久化

| 项 | 值 |
|---|---|
| Kaggle 数据集 | `luckfu/kronos-beta-v21-c1-cosine-pilot-best475-seg9`（**私有**，status `ready`，id 12404852） |
| 内容 | `best_metric.json`、`README.md`、`checkpoints/best_model/{model.safetensors, config.json, best_metric.json, snapshot_metric.json, README.md}`，布局与 `luckfu/kronos-beta-v21-c1-seg155-forecast-best` 相同 |
| 权重 SHA256 | `f9d3da03f8e5b55824bff28e31e00cc039ee021126d71891bda7b5daeb76e3c8`（409,771,056 字节；从 Kaggle 重新下载核对一致） |
| config | `use_beta_v21_auxiliary=false`、`num_sectors=86`、`context_layer=10`、`use_size_percentile=true`、d_model 832 / 12 层 |
| `best_metric.json` | 与 Seg155 数据集同字段（large_metrics 等），另含 val 生成式 IC（0.3254，SE 0.0266）、全量 val WFL 2.3215、SHA、父本、来源 kernel、`sealed_oos_read=false` |

### 建议下一步：Seg9 + Seg0 一次性密封 OOS 最终评估（已备好，未推送）

GPU 配额恢复后，用 Baseline 2 同合同在密封 18d 包上评估 **Seg9（先）和 Seg0 = Best@475（后）**，**只跑一次，作为最终评估**：
不根据结果再换 checkpoint、改 decode 或再训练（事先登记写在 kernel 的 `PREREGISTRATION` 里）。主比较 = Seg9 − Seg0 逐日配对；
次要比较 = 各自对 Seg155 prod（0.1170）和 C2 prod（0.1769）逐日配对。

| 项 | 值 |
|---|---|
| Slug | `luckfu/kronos-beta-v21-c1-gen-return-oos-pilot-seg9`（新，私有，2× T4，docker pin `37c64f7dd9…`） |
| 代码 commit | `98f3d0b`（runner / builder / worker / staging / 单测 9 项通过，含 CPU 端到端：Seg9 先打分、逐日 shard 即时落盘） |
| 构建 | `python3 finetune/build_kaggle_beta_v21_c1_gen_return_oos_kernel.py --variant pilot_seg9` |
| Staging | `finetune/kaggle_beta_v21_c1_gen_return_oos_pilot_seg9_kernel/` |
| 推送（配额恢复后） | `kaggle kernels push -p finetune/kaggle_beta_v21_c1_gen_return_oos_pilot_seg9_kernel` |
| 数据集 | `luckfu/a-share-120d-temporal-symbol-holdout`（密封包 `kronos_beta_v2_time_oos_through_20260903`，18 日 / 92,751）+ `luckfu/kronos-beta-v21-c1-cosine-pilot-best475-seg9`；Seg0 与 tokenizer 从 ModelScope 按 SHA 取 |
| Decode | prod T0.65 / top_p 0.8 / N5 / seed 20260906（只跑这一臂） |
| 输出目录 | `/kaggle/working/beta_v2_1_c1_gen_return_oos_pilot_seg9/`（`shards/`、`results/`、`comparison.json`、`run.log`） |
| SwanLab | <https://swanlab.cn/@roc_fu/finance/runs/beta_v2_1_c1_gen_return_oos_pilot_seg9>（每个日期 shard 记 `oos/<seg>/ret10d_ic_day` / 累计均值 / 对 Seg155 差；每个 checkpoint 完成记 `oos_final/*`） |

**保护措施**

- 顺序 Seg9 → Seg0：两个 worker 按计划顺序**动态认领**（原子 claim 文件）(checkpoint, arm, date) 任务，所有 Seg9 shard 被领完之后才开始 Seg0
- 每个日期 shard 一完成就落盘 `shards/<label>__prod_t065_p80_n5_<date>.csv.gz`（逐日预测），父进程下次轮询立即打分，
  写 `results/progress.json`、`run.log` 的 `shard_scored` 行、SwanLab
- 每个 checkpoint 的 18 个 shard 到齐**立刻**打分并保存 `results/<label>_prod_t065_p80_n5_{summary.json, predictions.csv.gz, by_date.csv}` 和 `comparison.json`（`final=false`），最后再写 `final=true`
- 11 h 后 worker 不再领新 shard；启动时校验 tokenizer / Seg9 / Best@475 SHA、18 日 / 92,751 样本、2× T4

**运行时间估计**（Baseline 2 prod 臂实测每个日期 shard 973–1,124 s / T4：kernel A 1,063–1,124，均值 1,092；prod-only kernel 973–1,036，均值 1,005）

| 阶段 | 估计 |
|---|---|
| 启动（解包、ModelScope、SHA、读包、SwanLab） | ~3–5 min |
| Seg9：18 shard / 2 卡 = 9 轮 | ≈ 2.4–2.8 h → **约 2.5–2.9 h 时 Seg9 结果落盘** |
| Seg0：再 9 轮 | ≈ 2.4–2.8 h |
| 合计 | **≈ 5.0–5.7 h**，远低于 12 h 上限（11 h 保护线前有 5 h 以上余量）；消耗 GPU 配额约 5–6 h |


## 相关文件

- Runner / builder：`finetune/kaggle_beta_v21_c1_forecast_cosine_pilot.py`、`finetune/build_kaggle_beta_v21_c1_forecast_cosine_pilot_kernel.py`
- 打分：`finetune/val_gen_ic_driver.py`、`finetune/evaluate_beta_v21_val_gen_ic.py`
- 训练：`finetune/train_predictor.py`（snapshot 钩子 + 父本 key 日志）
- 单测：`tests/test_forecast_cosine_pilot.py`
- 父本选择依据：[`beta_v21_c1_val_gen_ic_reselection_cn.md`](beta_v21_c1_val_gen_ic_reselection_cn.md)
- 结果 JSON：`finetune/reports/beta_v21_c1_forecast_cosine_pilot_best475.json`
- 原始输出（box）：`/workspace/kronos_patrol_out/pilot475_final_1006_2116/output/`；Seg9 权重下载：`/workspace/kronos_patrol_out/pilot475_seg9_weights/`
- 密封 OOS 最终评估 kernel（未推送）：runner `finetune/kaggle_beta_v21_c1_gen_return_oos_pilot.py`、staging `finetune/kaggle_beta_v21_c1_gen_return_oos_pilot_seg9_kernel/`、
  builder `finetune/build_kaggle_beta_v21_c1_gen_return_oos_kernel.py --variant pilot_seg9`、worker `finetune/evaluate_beta_v21_generative_return_oos.py`、单测 `tests/test_gen_return_oos_pilot_seg9.py`
