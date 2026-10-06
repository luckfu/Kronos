# Beta v2.1 C1 forecast-only 余弦退火 pilot（12 seg，双 T4）

更新：2026-10-06 CST

## 要回答的问题

密封 18d OOS 上，我们 Seg155 的生成式 return10d 日均 Rank IC 是 **0.117**（prod arm），
C2（small stage2 cosine refinement）是 **0.177 / 0.180**（prod / rank）。差距约 2–3 SE。
C2 与 C1 的最大配方差异之一是 C2 最后走了 **uniform_cosine 1e-5 → 1e-6 退火**，
C1 wc 线一直是 warmup_constant 1e-5 hold。本 pilot 只验证一件事：

> 在 Seg155 上做 C2 式余弦退火，val 生成式 IC 会不会随退火上升？

## Kernel

- Slug：`luckfu/kronos-beta-v21-c1-forecast-cosine-pilot`（新 slug；不碰 C1 训练 notebook）
- Runner：`finetune/kaggle_beta_v21_c1_forecast_cosine_pilot.py`
- Builder：`finetune/build_kaggle_beta_v21_c1_forecast_cosine_pilot_kernel.py`
- Staging：`finetune/kaggle_beta_v21_c1_forecast_cosine_pilot_kernel/`
- 双 T4，docker pin `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c5…d461`，无 TPU
- 数据集：`luckfu/a-share-120d-temporal-symbol-holdout`、`luckfu/kronos-beta-v21-c1-seg155-forecast-best`

## 配方

| 项 | 值 |
|---|---|
| 父本 | Seg155 forecast-best（全局 Seg155 = wc local segment 26），`checkpoints/best_model`，SHA `8b11a759e72d…`，WFL 2.312367872672933 |
| 初始化 | 仅权重；**新 AdamW**；不加载 last_state |
| 损失 | forecast-only（`PREDICTOR_LOSS_MODE=forecast`，history 0.02，C1 horizon 权重 1.364…0.455） |
| aux / ranking | **关**（`USE_BETA_V21_AUXILIARY=0`；ranking weight 0.05 仅为 config 解析需要，aux 关时不生效） |
| 喂数 | `SAME_DAY_RANKING_BATCHES=0`：shuffled coverage 顺序，**段内不按 signal_date 排序** |
| 可训练 | 全部参数（`TRAINABLE_TRANSFORMER_LAYERS=-1`，heads-only 关），单 LR 族（`SPLIT_TRUNK_HEAD_LR=0`），同 C2 |
| 调度 | `uniform_cosine`，peak 1e-5 → min 1e-6，warmup ratio 0（start = peak = 1e-5），**恰好 12 段**（`EPOCHS=12`，`REQUIRE_FULL_COVERAGE=0`） |
| 段末 LR（预期） | Seg3 8.68e-6 / Seg6 5.50e-6 / Seg9 2.32e-6 / Seg12 1.00e-6 |
| Coverage | seed 20261002，offset 0（与 C1 rank 线 v16/v20 同一 coverage 序列），20000 样本/段 |
| Batch | 32 × 2 GPU = 64，AMP fp16 |
| 验证 | 每段 full val（123,836 窗口），best = weighted forecast loss，阈值保留 Seg155 的 2.31236787 |
| Snapshot | `KRONOS_SNAPSHOT_EVERY_SEGMENTS=3` → `outputs/models/beta_v2_1_c1_forecast_cosine_pilot/snapshots/seg003/006/009/012`（权重 + `snapshot_metric.json`：段号、全量 WFL、段末 LR） |
| SwanLab | `beta_v2_1_c1_forecast_cosine_pilot` |
| Soft-stop | 12 段 / 30000s；之后留 ≥3h 给 in-kernel 评估 |

`train_predictor.py` 新增的 snapshot 钩子由环境变量控制（默认关），写在 best 保存之后、
resume state 之前，只存权重，不影响已有训练 notebook。单测：`tests/test_forecast_cosine_pilot.py`。

## In-kernel val 生成式 IC（与 Step 1 同合同）

训练结束后，kernel 用 `finetune/val_gen_ic_driver.py` 依次打分
Seg3 / Seg6 / Seg9 / Seg12 snapshot；若 best_model 与所有 snapshot 及父本 SHA 都不同，
再加打 best_model。**每个 checkpoint 24 个日期 shard 一到齐就立刻打分落盘**
（`val_gen_ic/results/<label>_val_summary.json`、`comparison.json`）。

合同与 Step 1（`luckfu/kronos-beta-v21-c1-val-gen-ic`）完全相同：

- `temporal_symbol_validation_v1` `val_data.pkl`（sha `4cce31bc…`，242 日 / 123,836 窗口）
- 子样本：排序后的 242 个 signal date 上 `np.linspace(0, 241, 24).round()`，每日全部股票 → 12,256 窗口
- decode：Baseline 2 prod arm T0.65 / p0.8 / N5，seed 20260906；分数 = mean(close_d10 / last_close − 1)
- label：raw close return_10d；同窗口 teacher-forcing WFL
- 不读密封 OOS

Seg155 父本的 val 生成式 IC 取自 Step 1 kernel（同合同同 seed），作为趋势起点。

## 时间估算

- 安装 / 下载：~10 min
- 训练：8.5–13 min/段（含 full val）× 12 ≈ 1.7–2.6 h
- 评估：每个 checkpoint ~20 min（24 shard × ~100 s / 2 GPU），4–5 个 ≈ 1.3–1.7 h
- 合计 ≈ **3.3–4.5 h**，远低于 12 h

## 判定规则

以 Step 1 的 Seg155 val 生成式 IC（daily）为基线，看 Seg3 → 6 → 9 → 12：

1. **随退火上升**（后段 snapshot 的 daily IC 明显高于 Seg155，例如 Seg9/12 高出 ≥ 1 个 Step-1 SE，
   且趋势单调或接近单调）→ 拿 val IC 最好的 snapshot 去密封 18d OOS，用与 Baseline 2 相同的生成式
   prod 配方测试，对照 **0.117**（Seg155）/ **0.180**（C2）。
2. **持平**（全部落在 Seg155 ± 1 SE 内，无趋势）→ **停**：差距不来自退火，不再追这条线。
3. **下降** → 同样停，记录退火对 C1 有害。

WFL 与 IC 的排序分歧单独标记（`divergence_wfl_*_vs_ic`），但判定只看 val 生成式 IC。
