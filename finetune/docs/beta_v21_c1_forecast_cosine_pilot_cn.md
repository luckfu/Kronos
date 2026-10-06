# Beta v2.1 C1 forecast-only 余弦退火 pilot（12 seg，双 T4）

更新：2026-10-06 CST

> **状态：已启动，父本 = Best@475。**（2026-10-06）
> Step 1 val 生成式 IC 重选（`luckfu/kronos-beta-v21-c1-val-gen-ic`，见
> [`beta_v21_c1_val_gen_ic_reselection_cn.md`](beta_v21_c1_val_gen_ic_reselection_cn.md)）显示：
> val 生成式 return10d IC 日均 **Best@475（发布版）0.3145** > Seg8 0.2989 > **Seg155 0.2954**
> （Best@475 − Seg155 配对 +0.019，19/24 天更高），而 WFL 排序正好相反（Seg155 最低）。
> 原父本 Seg155 是按 WFL 选的，WFL 已被证明不能代表生成式 IC，所以按选项 **A** 把父本换成
> val IC 最好的 **Best@475**，其余配方不变。Seg155 版本仍可构建（`--parent seg155`），但**未启动**。

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

## 判定规则

以本 kernel 内 Seg0（Best@475）的 val 生成式 IC 日均为基线（Step 1 为 0.3145，SE 0.025），看 Seg3 → 6 → 9 → 12：

1. **随退火上升**（后段 snapshot 明显高于 Seg0，例如 Seg9/12 高出 ≥ 1 个 SE，且趋势单调或接近单调）→
   拿 val IC 最好的 snapshot 去密封 18d OOS，用 Baseline 2 同配方测试，对照 **0.117**（Seg155）/ **0.180**（C2）。
2. **持平**（都在 Seg0 ± 1 SE 内，无趋势）→ **停**：差距不来自退火。
3. **下降** → 同样停，记录退火对该父本有害（与 Step 1 观察到的「C1 forecast 训练降 WFL 但降 IC」一致）。

## 相关文件

- Runner / builder：`finetune/kaggle_beta_v21_c1_forecast_cosine_pilot.py`、`finetune/build_kaggle_beta_v21_c1_forecast_cosine_pilot_kernel.py`
- 打分：`finetune/val_gen_ic_driver.py`、`finetune/evaluate_beta_v21_val_gen_ic.py`
- 训练：`finetune/train_predictor.py`（snapshot 钩子 + 父本 key 日志）
- 单测：`tests/test_forecast_cosine_pilot.py`
- 父本选择依据：[`beta_v21_c1_val_gen_ic_reselection_cn.md`](beta_v21_c1_val_gen_ic_reselection_cn.md)
