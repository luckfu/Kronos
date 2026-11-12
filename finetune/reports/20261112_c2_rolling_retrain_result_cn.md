# C2 滚动续训实验结果（2026-11-12）

事前登记：[`20261111_c2_rolling_retrain_prereg_cn.md`](20261111_c2_rolling_retrain_prereg_cn.md)（2026-11-11 21:33 写定，早于任何评测结果）。
机器可读结果：[`20261112_c2_rolling_retrain_compare.json`](20261112_c2_rolling_retrain_compare.json)。
比较脚本：[`rolling_compare.py`](../oos_eval/base_clean_sealed_oos/rolling_compare.py)。

## 1. 结论

**按事前登记的标准，滚动续训没有通过，结果是显著变差。**

- 主候选（学习率 1e-5，跑完一遍新数据）的日度 Rank IC 比原版 C2 低 0.044，95% 区间 [−0.074, −0.028]，上界小于 0。
- 5 个候选全部判为“显著变差”，没有一个通过。
- 预测准确度也变差：第 10 日平均绝对误差多 0.6 个百分点，方向正确率低 7.7 个百分点。
- 把学习率降到十分之一没有用。

所以“用截止日之后的新数据把 C2 续训一小段”这个做法，至少在这套训练配方下不成立。原版 C2 应继续留在生产上。

## 2. 实验做了什么

| 项 | 取值 |
|---|---|
| 起点 | C2 best（SHA `4ee469d4…`），全新优化器 |
| 训练数据 | 只含新数据：信号日 2026-07-03 → 08-28（41 天），训练股票 4,677 只，190,084 个窗口 |
| 训练量 | 一遍覆盖 = 10 段，每段存一个检查点 |
| 配方 | 与 C2 Stage2 相同：加权 forecast + 0.02 history，双 T4，每卡 32，fp16，1% 预热后常数学习率 |
| 评测 | 信号日 2026-09-15 → 10-23，23 天，118,764 条；16 路径取中位数，seed 20261104 |
| 对照 | 同一窗口、同一口径重跑的原版 C2，按 `identity` 逐条配对，标签逐位一致 |

训练数据包的行情止于 2026-09-14，评测窗口从 09-15 开始，训练标签与评测标签不重叠。

| 臂 | 学习率 | 评测的检查点 | 权重相对 C2 的偏移（第 3 → 10 段） |
|---|---|---|---|
| A（主） | 1e-5 | 第 3、6、10 段 | 0.80% → 1.60% |
| B | 1e-6 | 第 3、10 段 | 0.18% → 0.41% |

训练每臂约 15 分钟。验证损失（训练包里 520 只验证股票，同期）一路下降：臂 A 2.144 → 2.033，臂 B 2.307 → 2.141。

## 3. 结果

评测窗口上的第 10 日指标，16 路径：

| 模型 | Rank IC | 剥离风格后的 IC | 平均绝对误差 | 方向正确率 | IC 高于 C2 的天数 |
|---|---:|---:|---:|---:|---:|
| 原版 C2 | **0.158** | **0.119** | **7.19%** | **63.3%** | — |
| A 第 3 段 | 0.109 | 0.092 | 7.84% | 55.0% | 5/23 |
| A 第 6 段 | 0.113 | 0.091 | 7.86% | 54.9% | 3/23 |
| **A 第 10 段（主候选）** | 0.114 | 0.089 | 7.82% | 55.6% | 3/23 |
| B 第 3 段 | 0.119 | 0.105 | 7.75% | 56.3% | 5/23 |
| B 第 10 段 | 0.105 | 0.092 | 7.87% | 54.8% | 4/23 |

参照：永远预测 0 的平均绝对误差是 7.95%；永远猜跌的方向正确率是 72.0%。
“剥离风格”指对市值、行业、10 日反转做逐日回归后的残差 IC。

候选减 C2 的配对差（5 日块 bootstrap，2000 次）：

| 候选 | Rank IC 差 | 95% 区间 | 判定 | 误差差 | 方向差 |
|---|---:|---|---|---:|---:|
| **A 第 10 段** | −0.044 | [−0.074, −0.028] | 显著变差 | +0.63 个百分点 | −7.7 个百分点 |
| A 第 6 段 | −0.045 | [−0.079, −0.025] | 显著变差 | +0.67 | −8.5 |
| A 第 3 段 | −0.049 | [−0.096, −0.016] | 显著变差 | +0.66 | −8.3 |
| B 第 10 段 | −0.053 | [−0.105, −0.013] | 显著变差 | +0.68 | −8.5 |
| B 第 3 段 | −0.039 | [−0.075, −0.004] | 显著变差 | +0.56 | −7.0 |

误差差和方向差的 95% 区间也都不跨 0（见 JSON）。
剥离风格后的 IC 差：主候选 −0.030 [−0.064, −0.011]；臂 B 两个候选和臂 A 第 3 段的区间跨 0。

## 4. 这不是评测噪声

同一份 C2 权重做过两次独立的 16 路径评测（下午的 22 天子集和本次的对照），两者在重合的 8 天、41,308 条上：
IC 分别为 0.152 和 0.146，差 0.006；逐日 IC 的平均绝对差 0.006。
续训造成的 IC 差距（0.039–0.053）是它的 6–9 倍。

## 5. 变差的样子（事后诊断，不参与判定）

**损伤在前 3 段就已经发生，之后基本不再变化。** 臂 A 第 3、6、10 段的 IC 是 0.109、0.113、0.114。
这与 9 月的续训实验、Base 的探针一致：破坏是即时的。

**降低学习率没有保护作用。** 臂 B 第 3 段的权重只偏移了 0.18%，IC 已经掉了 0.039。

**模型变得不那么看空，而评测期市场在跌。**

| | 平均预测（第 10 日） | 预测上涨的比例 |
|---|---:|---:|
| 原版 C2 | −2.8% | 20% |
| 5 个候选 | −0.8% 到 −1.2% | 37% 到 41% |

训练用的 41 天里，全市场 10 日收益平均 +0.96%，25 天上涨。评测的 23 天里平均 −4.43%，只有 2 天上涨。
续训把模型的整体水平往训练期的行情拉了过去，到了评测期正好反向。方向正确率和绝对误差的恶化主要来自这里。

**排序能力也变差了，不只是整体偏移。** Rank IC 不受整体水平影响，仍然掉了 0.04–0.05。
只看“个股相对当日平均”的误差，候选是 6.70%–6.72%，原版 C2 是 6.59%，零预测是 6.58%。

**差距集中在评测窗口的前半段。** 9 月的 11 天里，C2 的 IC 是 0.120，候选是 0.028–0.064；10 月的 12 天里，C2 是 0.193，候选是 0.160–0.180。
两个半段各只有十来天，这个差别不足以下结论。

## 6. 这个结果说明什么，不说明什么

说明的：

- 在现有的 CE 训练配方下，用最近两个月的数据续训 C2 会让它变差，学习率 1e-5 和 1e-6 都一样。
- 这与此前从 C2 续训的每一次尝试方向一致。区别在于这次用的是它没见过的新数据，结果仍然如此，所以问题不在“数据旧”，而在续训这个动作本身。
- 验证损失再次与样本外表现反向：损失一路下降，样本外变差。

不说明的：

- **只测了一个时间点。** 评测期恰好是下跌行情，训练期是上涨行情。如果两段行情方向一致，绝对误差和方向上的恶化可能小一些；排序上的恶化是否也会小，不知道。
- **没有测“新旧数据混合全量重训”。** 那需要约 129 个 GPU 小时，而且新数据只占不到 4%。
- **没有测其他训练目标或冻结主干的做法。**
- 23 天只相当于 2–3 个不重叠的十日周期；评测窗口此前被用来比较过别的模型（本实验的候选从未在它上面挑选过）。

## 7. 与登记不一致的地方

- 任务描述里的训练信号日是 07-03 → 08-31，实际到 08-28。训练器每个窗口需要 131 行，为了让行情止于 09-14 只能到这里。这一点在登记文件里已写明。
- 无其他偏离：评测的检查点、口径、判定规则、bootstrap 参数都按登记执行。

## 8. 配额与产物

GPU 用量：训练两臂合计约 0.5 小时；6 个 16 路径评测合计约 15 小时（每个约 2.5 小时）。
结束时 smmt315 剩 16.0 小时，wynstonliu 剩 9.6 小时（2026-11-14 刷新）。

| 产物 | 位置 |
|---|---|
| 训练数据包（私有） | `smmt315/c2-roll-newdata-20260703-20260914`、`wynstonliu/c2-roll-newdata-20260703-20260914` |
| 训练 kernel（含 10 个检查点） | `smmt315/c2-roll-lr1e5`、`wynstonliu/c2-roll-lr1e6` |
| 评测 kernel | `smmt315/kronos-c2-roll-base-p16`、`smmt315/c2-roll-a-seg{03,06,10}-p16`、`wynstonliu/c2-roll-b-seg{03,10}-p16` |
| 预测分片（本机，不入库） | `scratch/c2_roll/eval/` |
| 代码 | `finetune/c2_rolling/`（数据包、面板、启动器、推送器），`rolling_compare.py` |

没有触碰 2026-10-24 之后的数据，没有改动生产环境，没有提交 git。

## 9. 复现

```bash
# 数据包（本机，零 GPU）
python3 finetune/c2_rolling/build_roll_package.py \
  --raw /tmp/oos_raw/full_20250701_20261106.csv \
  --sector-source data/temporal_holdout/20260717_20261023/evaluation_samples.jsonl \
  --symbol-split data/a_share_full_market_v1_beta_symbol_holdout_90_10_v1/symbol_split.csv \
  --signal-start 2026-07-03 --last-bar 2026-09-14 --output-dir scratch/c2_roll/package

# 训练（每臂一个 kernel）
KAGGLE_API_TOKEN=<token> KAGGLE_USERNAME=smmt315 \
ROLL_ARM=lr1e5 ROLL_LR=1e-5 ROLL_WARMUP_START_LR=1e-6 python3 finetune/c2_rolling/push_roll.py --push

# 评测（每个检查点一个 kernel）。对照：去掉 KRONOS_OOS_PARENTS，改用
#   KRONOS_OOS_EXTRA_DATASETS=luckfu/kronos-small-cosine-c2-best-weights
#   BASE_OOS_WEIGHTS_GLOB='**/kronos-small-cosine-c2-best-weights/model.safetensors'
KAGGLE_API_TOKEN=<token> KAGGLE_USERNAME=smmt315 \
KRONOS_OOS_OWNER=smmt315 KRONOS_OOS_SLUG=c2-roll-a-seg10-p16 KRONOS_OOS_TITLE="C2 roll a seg10 p16" \
KRONOS_OOS_DATASET=luckfu/a-share-120d-oos-20260717-20261023-v2 KRONOS_OOS_PARENTS=smmt315/c2-roll-lr1e5 \
BASE_OOS_LABEL=roll_a_seg10_p16 BASE_OOS_WEIGHTS_GLOB='**/roll_ckpt/seg010/model.safetensors' \
BASE_OOS_HF_TOKENIZER=NeoQuasar/Kronos-Tokenizer-base BASE_OOS_CONTEXT_LAYER=6 BASE_OOS_SEED=20261104 \
BASE_OOS_PATHS=16 BASE_OOS_PATH_AGG=median BASE_OOS_DATE_FROM=2026-09-15 BASE_OOS_DATE_TO=2026-10-23 \
BASE_OOS_EXPECT_SAMPLES=335312 BASE_OOS_EXPECT_DATES=65 \
BASE_OOS_EXPECT_FIRST=2026-07-17 BASE_OOS_EXPECT_LAST=2026-10-23 BASE_OOS_EXPECT_TARGET_END=2026-11-06 \
python3 finetune/oos_eval/base_clean_sealed_oos/push_oos_bundled.py --push

# 下载分片
python3 finetune/oos_eval/base_clean_sealed_oos/fetch_kernel_shards.py \
  --kernel smmt315/c2-roll-a-seg10-p16 --out scratch/c2_roll/eval/a_seg10

# 比较
python3 finetune/oos_eval/base_clean_sealed_oos/rolling_compare.py \
  --base c2=scratch/c2_roll/eval/c2 \
  --cand a_seg10=scratch/c2_roll/eval/a_seg10 --cand a_seg06=scratch/c2_roll/eval/a_seg06 \
  --cand a_seg03=scratch/c2_roll/eval/a_seg03 --cand b_seg10=scratch/c2_roll/eval/b_seg10 \
  --cand b_seg03=scratch/c2_roll/eval/b_seg03 \
  --raw scratch/oos65/combined_raw.csv --date-from 2026-09-15 --date-to 2026-10-23 \
  --out finetune/reports/20261112_c2_rolling_retrain_compare.json
```
