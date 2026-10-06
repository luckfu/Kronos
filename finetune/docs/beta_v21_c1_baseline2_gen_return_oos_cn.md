# Baseline 2：Beta Seg155 生成式 OHLC → 派生 10 日收益（密封 18d OOS）

更新：2026-10-05 CST

## 假设

C2 的 return10d Rank IC ~0.18 来自 **生成式 AR decode** 路径
（`auto_regressive_inference` 多样本均值），而不是训练过的 `return_head`。
Beta 的 `expected_utility_score` / `return_head` 被 ranking 污染，对 raw 10d 收益
Rank IC 接近 0 / 负。

本 baseline **零训练**：在密封包上用 **我们自己的** Seg155 forecast-best
走与 C2 同精神的生成式预报，派生 `predicted_return_d10` 再打分。

## 精确 decode 配方（beta-base / Seg155）

| 项 | 值 |
|---|---|
| 对象 | 真 AR 采样 `model.kronos.auto_regressive_inference` |
| 非 | teacher-forcing CE；非 Stage3 soft top-N stitch |
| `return_samples` | True |
| 分数定义 | `mean_N ( denorm_close[d10] / last_close - 1 )` |
| ranking arm | T=0.60 / top_p=0.9 / N=16 / seed=20260906 |
| production arm | T=0.65 / top_p=0.8 / N=5 / seed=20260906 |
| 其它 | top_k=0, clip=5, max_context=512, pred_len=10, lookback=120 |
| 条件 | sector_id + size_percentile（Beta 合同） |
| 权重 | `luckfu/kronos-beta-v21-c1-seg155-forecast-best` SHA `8b11a759e72d…` |
| aux | Seg155 `use_beta_v21_auxiliary=False`（干净预报；AR 不用 heads） |

Seg19 trunk 自 Seg155 冻结，生成式预报路径应近似相同；本 kernel **只跑 Seg155**
以省 GPU，与 Seg19 return_head 数字对照用已密封表。

## 评分合同（与 C2 vs Beta 表同包）

包：`kronos_beta_v2_time_oos_through_20260903`（08-11→09-03 / 92751）

| 指标 | 定义 |
|---|---|
| return10d Rank IC | 日均 Spearman(`predicted_return_d10`, `return_10d`) |
| utility Rank IC | 日均 Spearman(`predicted_return_d10`, package utility) — ablation |
| Pairwise | 同日 `\|Δutility\|≥0.005`，score=`predicted_return_d10`，平局算错 |

对照：C2 rank decode ~0.18；Seg19 `expected_utility_score`↔return10d ~-0.007。

## Kernel

- slug：`luckfu/kronos-beta-v21-c1-gen-return-oos`（**eval-only**，不覆盖 dual-T4 训练）
- docker pin：`sha256:37c64f7dd9…`（与 dual-T4 / rank-oos 相同）
- machine：2× T4，(arm, date) round-robin
- 重建：

```bash
python3 finetune/build_kaggle_beta_v21_c1_gen_return_oos_kernel.py
kaggle kernels push -p finetune/kaggle_beta_v21_c1_gen_return_oos_kernel
```

## 代码

- 核心：`finetune/evaluate_beta_v21_generative_return_oos.py`
- Runner：`finetune/kaggle_beta_v21_c1_gen_return_oos.py`
- Builder：`finetune/build_kaggle_beta_v21_c1_gen_return_oos_kernel.py`

## 运行记录（2026-10-05 CST）

- Commit `78b3bda`；`kaggle kernels push` → version 1，**RUNNING**（~23:11 CST 启动）
- URL：https://www.kaggle.com/code/luckfu/kronos-beta-v21-c1-gen-return-oos
- 校验：tokenizer SHA `59d85f6a…` ✓；Seg155 SHA `8b11a759e72d…` ✓；18 日 / 92751 ✓
- 实测吞吐：prod N=5 每个日期 shard ≈ 1,060–1,110 s / T4（effective batch 256）
- ETA：prod 臂 18 shard / 2 卡 ≈ 9 轮 ≈ 2.7 h → 约 **10-06 02:00 CST**；
  rank N=16 每 shard ≈ 3.5k s → 9 轮 ≈ 8.7 h，会碰到 39,600 s 截止（约 10-06 10:10 CST），
  **rank 臂可能只完成 ~16/18 日**（可容忍部分完成：prod 先完整打分；rank 不完整则记为 incomplete）。
  如 rank 被截断，再单独推 rank-only 补跑。

## 保险：prod-only kernel（2026-10-06 CST）

担心原 kernel（两臂、末尾才打分，预计 ~10:55–11:00 收尾 vs 12h 硬杀 ~11:11）被杀，另起：

- slug：`luckfu/kronos-beta-v21-c1-gen-return-oos-prod`（eval-only，不动正在跑的原 kernel）
- 只跑 production 臂 T=0.65 / top_p=0.8 / N=5 / seed 20260906；同 Seg155 SHA、同密封包、dual T4、docker `37c64f7dd9…`
- 构建：`python3 finetune/build_kaggle_beta_v21_c1_gen_return_oos_kernel.py --variant prod`

### Runner 教训（共享 runner 已改）

- 父进程轮询 shards，**每个臂 decode 完立即打分**并写 summary / predictions / comparison.json（`final=false`），最后再写 `final=true`
- 每个日期 shard 落盘即保留（`shards/<arm>_<date>.csv.gz`），`shard_done` 日志行带当日 return10d IC / utility IC / pairwise，kill 后日志里仍有部分结果
- `arm_scored` 行含 daily/pooled return10d IC、ICIR、正 IC 日占比、utility IC、pairwise 及对 C2 / Seg19 差值

## 结果（2026-10-06，kernel A COMPLETE）

原 kernel `luckfu/kronos-beta-v21-c1-gen-return-oos` 在 42,221 s（~10:55 CST）完成，两臂都已打分，未被 12h 硬杀：

| 臂 | ret10d IC 日均 / pooled | ICIR | IC+ 日 | utility IC | Pairwise |
|---|---|---:|---:|---:|---:|
| prod T0.65/p0.8/N5 | **0.1170** / 0.1631 | 1.25 | 15/18 | 0.0757 | 54.06% |
| rank T0.6/p0.9/N16 | **0.1242** / 0.1764 | 1.19 | 15/18 | 0.0773 | 54.16% |

C2 0.1796 / Seg19 return_head −0.0073。最后 3 日（09-01..09-03）转负。
完整表、逐日 IC、结论见 `finetune/docs/beta_v21_c1_rank_oos_vs_small_c2_cn.md`；JSON `finetune/reports/beta_v21_c1_baseline2_gen_return_oos_18d.json`。
保险 kernel `-prod` 逐日 IC 与 A 的 prod 臂一致（同 seed 确定性），属冗余。
