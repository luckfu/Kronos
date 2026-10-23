# 接手文档：换电脑继续（2026-10-23）

**仓库**：<https://github.com/luckfu/Kronos>
**撰写时 HEAD**：`ebbf4d4`（= `origin/master`；本文档随后以一个 docs 提交追加）
**工作区**：干净
**用途**：换到另一台电脑后，10 分钟内接上；知道现在做到哪、下一步做什么、别踩哪些坑。

---

## 0. 一句话现状

> **C2 仍是唯一基线；epoch2 / bigbatch e03b 两条续训线已关闭；sealed 8 天窗口已用掉；验证集 Rank IC（16 条路径）排序方向对，但 20 天样本下还不显著。§5.1 的 60 天验证 IC 代码已就绪（`N_DATES=60`，`N_SYMBOLS=200`，16 条路径中位数不变）；kernel `wynstonliu/kronos-small-validation-rank-ic-20261023` 已另行推送，状态 RUNNING，结果未出。然后写 C2 的 8b 执行约束回测。不要开新训练，不要再推这个 kernel。**

---

## 1. 已完成 / 未完成

| 任务 | 状态 | 结论 |
|---|---|---|
| epoch2 训练（C03 续训） | 完成 520/520 | Best 2.274381@501；OOS 排序退化；**线已关** |
| bigbatch e03b 训练 | 完成 520/520 | Best 2.261617@457；OOS 更差；**线已关，不推 e04b** |
| sealed 8 天 OOS（C2 vs e03b） | 完成 | C2 0.0652 > e03b 0.0525；e03b 不晋级；C2 弱 alpha |
| 验证集 Rank IC（1 条路径） | 完成 | C2 0.2204 / C03 0.2184 / e03b 0.2174 / epoch2 0.1914；C2/C03/e03b 分不开 |
| **验证集 Rank IC（16 条，生产口径）** | **完成** | C2 0.2535 / C03 0.2434 / e03b 0.2112 / epoch2 0.2028；**方向对，20 天仍不显著** |
| 60 天验证 IC | **代码已就绪；kernel RUNNING** | `N_DATES=60`、`N_SYMBOLS=200`、16 条路径中位数未改。Kernel `wynstonliu/kronos-small-validation-rank-ic-20261023` 已由另一代理推送，结果未出 |
| C2 8b 执行约束回测 | **代码已落地，§5.2 数字未出** | 只做 `\|pred\|≥5%`。行情要到 2026-11-06，见 [`C2_EXECUTION_CONSTRAINED_BACKTEST_8B_CN.md`](C2_EXECUTION_CONSTRAINED_BACKTEST_8B_CN.md) |
| TimesFM sealed §3.3 | **未做** | SweetSpot / C07 SWA 尚未在 sealed 窗口评测 |

---

## 2. 关键结果

### 2.1 Sealed 8 天（2026-09-18 → 09-30，38,100 窗口，`sealed_first_use=true`）

| 指标 | C2 | e03b | 结论 |
|---|---:|---:|---|
| D10 日均 Rank IC | **0.0652** | 0.0525 | e03b 配对 -0.0128，CI [-0.0393, -0.0042]；关线 |
| D10 方向 | **53.77%** | 51.87% | e03b 更差 |
| 全样本扣费净 PnL/笔（0.3%） | **+0.80%** | +0.59% | 8 天窗口偏“好做” |
| `\|pred_d10\| ≥5%` 子集净 PnL | **+2.92%**（覆盖 15.7%） | +2.39% | C2 的 8b 口径 |
| `\|pred_d10\| ≥8%` 子集 | **+3.81%**（覆盖 5.2%） | +2.90% | 更窄 |
| `halt_gap`（35 样本） | 48.6% / **-5.70%** | 57.1% / -3.60% | 必须事前剔除 |

全 42 天 C2 IC 0.1116，但其中 34 天是已污染的旧窗口，**不能用于选型**。

报告：[`SEALED_OOS_RESULT_20260918_20260930_CN.md`](SEALED_OOS_RESULT_20260918_20260930_CN.md)；JSON：[`small_sealed_oos_20260918_20260930.json`](../finetune/reports/small_sealed_oos_20260918_20260930.json)。

### 2.2 验证集 Rank IC（16 条路径，生产口径）

20 个 signal dates × 200 只 = 4,000 窗口；每窗口 16 条路径，取 D10 中位数；2025-07-03 → 2026-07-02。

| 模型 | 16 条平均 IC | 中位数 | 正 IC 天数 | 1 条平均 IC |
|---|---:|---:|---:|---:|
| C2 | **0.2535** | 0.2372 | 20/20 | 0.2204 |
| C03 | 0.2434 | 0.2300 | 20/20 | 0.2184 |
| e03b | 0.2112 | 0.2137 | 17/20 | 0.2174 |
| epoch2 | 0.2028 | 0.2362 | 16/20 | 0.1914 |

排序：**C2 > C03 > e03b > epoch2**，和 sealed OOS 的方向一致。

配对差（同一天相减，20 天，block=5）：

| 对比 | 16 条平均差 | 95% CI | 前者胜天数 |
|---|---:|---|---:|
| C2 − C03 | +0.0101 | [-0.0113, +0.0313] | 13/20 |
| C2 − e03b | +0.0422 | [-0.0069, +0.0963] | 12/20 |
| C2 − epoch2 | +0.0506 | [-0.0101, +0.1089] | 13/20 |
| C03 − e03b | +0.0322 | [-0.0006, +0.0625] | 12/20 |

**所有 CI 都跨 0。** 16 条比 1 条差距更大、方向更清楚，但 20 天还不够显著；**不能直接接进训练当唯一选模指标**。

报告：[`SMALL_VALIDATION_RANK_IC_20261023_CN.md`](reports/SMALL_VALIDATION_RANK_IC_20261023_CN.md)；16 条 JSON：[`small_validation_rank_ic_20261023_16path.json`](../finetune/reports/small_validation_rank_ic_20261023_16path.json)。

### 2.3 训练侧时间线

| 模型 | val selection loss | Best 段 | sealed/34 日 OOS |
|---|---:|---:|---|
| clean C03 | 2.280612 | 518 | 34 日 IC 0.1200 |
| epoch2 | 2.274381 | 501 | 34 日 IC 0.0843 |
| bigbatch e03b | **2.261617** | 457 | sealed IC 0.0525 |

结论：**验证 loss 越低，OOS 排序越差**，已三次独立出现；不要再开同合同训练。

---

## 3. 资产与账号

### Kaggle 数据集

| 用途 | 数据集 |
|---|---|
| 训练/验证面板（520 只验证股票） | `smmt315/a-share-120d-temporal-symbol-holdout` |
| C2 best 权重 | `wynstonliu/kronos-small-c2-best-checkpoint` |
| epoch2 best 权重（e05 私有 kernel 不可共享，已上传） | `wynstonliu/kronos-small-epoch2-best-checkpoint` |
| sealed 42 天 OOS（CC0，冻结） | `wynstonliu/a-share-120d-oos-20260803-20260930` |

### Kaggle Kernel

| Kernel | 作用 | 状态 |
|---|---|---|
| `wynstonliu/kronos-small-validation-rank-ic-20261023` | 验证集 Rank IC。已完成的是 20 日 × 16 条；60 日版本已另行推送 | **RUNNING**（60 日；不要重复 push） |
| `wynstonliu/kronos-small-sealed-oos-20260803-20260930` | sealed 42 天 OOS（C2 + e03b） | COMPLETE |
| `wynstonliu/kronos-small-clean-bigbatch-e03b` | e03b 训练（520/520） | COMPLETE |
| `luckfu/kronos-small-clean-wc-dual-t4-c03` | C03 best 权重 | COMPLETE |
| `luckfu/kronos-small-clean-epoch2-from-c03-e05` | epoch2 原始 kernel | **私有，wynstonliu 不可访问**；用上面的 epoch2 dataset |

### 本地产物（`artifacts/` 被 gitignore）

- `artifacts/small_validation_rank_ic_16path/validation_rank_ic/summary.json`
- `artifacts/small_sealed_oos/small_sealed_oos/shards/`（84 个）
- `artifacts/small_clean_epoch2_oos/`（68 个）

新电脑没有这些；需要时从对应 kernel output 重新下载。

---

## 4. 换电脑怎么开始

```bash
git clone https://github.com/luckfu/Kronos.git
cd Kronos
git pull --ff-only origin master
git log -1 --oneline        # 应为本文档之后的 docs 提交
```

Kaggle CLI 登录（当前主账号 `wynstonliu`）：

```bash
kaggle auth login
kaggle config view          # username: wynstonliu
```

若遇到 `SSLError: EOF occurred in violation of protocol`：

```bash
export NO_PROXY="api.kaggle.com,kaggle.com,www.kaggle.com,storage.googleapis.com"
export no_proxy="$NO_PROXY"
```

下载已有结果（可选）：

```bash
kaggle kernels output wynstonliu/kronos-small-validation-rank-ic-20261023 \
  -p artifacts/small_validation_rank_ic_16path -o

kaggle kernels output wynstonliu/kronos-small-sealed-oos-20260803-20260930 \
  -p artifacts/small_sealed_oos -o
```

---

## 5. 下一步（按优先级）

### 5.1 验证集 Rank IC：20 → 60 个日期

目的：看 16 条路径下，C2 对 C03/e03b 的差距能不能统计上分开。

**代码已就绪。** [`finetune/small_validation_rank_ic/evaluate_validation_rank_ic.py`](../finetune/small_validation_rank_ic/evaluate_validation_rank_ic.py) 现在是：

```python
N_DATES, N_SYMBOLS = 60, 200
SAMPLE_COUNT = 16  # 未改：生产口径，按 D10 路径中位数排序
```

`build/` 被 gitignore。需要本地包时可以重建，但 **不要再 push**：

```bash
python3 finetune/small_validation_rank_ic/build_kernel.py
# 不要执行 kaggle kernels push。60 日 kernel 已由另一代理推送。
```

- Kernel：`wynstonliu/kronos-small-validation-rank-ic-20261023`，状态 **RUNNING**。
- 规模：60 × 200 = 12,000 窗口；16 条路径；预计 **1.5 小时**（30h GPU 额度足够）。
- 结果出来之前不要填写 IC / CI，也不要把上面 20 日已完成的表改成 60 日。
- 验收（等 RUNNING 结束、下载 summary 之后再写）：C2−C03、C2−e03b 的配对 CI 是否仍然跨 0。
  - 如果 C2 显著高于 C03/e03b：可以考虑把 `validation_rank_ic` 接进训练，作为选模指标（先小规模 pilot）。
  - 如果仍然跨 0：**放弃验证集 Rank IC 选模**，不要改训练。
- 不要用 sealed 8 天来调这个阈值；它已经用掉。

### 5.2 C2 的 8b 执行约束回测

按 [`NEXT_STEP_SEALED_WINDOW_PLAN_20261022_CN.md`](NEXT_STEP_SEALED_WINDOW_PLAN_20261022_CN.md) §3.1/§4.3：

- 只做 C2；
- 只做 `|pred_d10| ≥ 5%` + 事前流动性/停牌过滤；
- 计入涨跌停、停牌、滑点、换手；
- 成本 0.3% 双边；
- 目标窗口：**2026-10-09 → 10-23**（D10 标签 2026-11-06 成熟）；
- 不要用 sealed 8 天（已消费）。

引擎和驱动已写好（[`execution_constrained.py`](../finetune/small_clean_oos/execution_constrained.py)、[`run_c2_execution_backtest.py`](../finetune/small_clean_oos/run_c2_execution_backtest.py)）。跑法、输入和「现在为什么没有净值」见 [`C2_EXECUTION_CONSTRAINED_BACKTEST_8B_CN.md`](C2_EXECUTION_CONSTRAINED_BACKTEST_8B_CN.md)。本仓库没有 10/09–10/23 的 C2 shard，也没有到 2026-11-06 的行情，所以还没有 §5.2 的 PnL / 夏普。34 日和已消费的 8 日都不能填这个数字。

### 5.3 TimesFM sealed §3.3

SweetSpot native / C07 SWA 尚未在 sealed 窗口评测；先做 `analyze_predictions.py` 的 `horizon=10` 不重叠 cohort 修正，再按 §3.3 判定。不要和 Small 的结论混在一起。

### 5.4 不做的事

- 不开新训练；不推 e04b / epoch3。
- 不在 34 日旧窗口上再评任何东西。
- 不为任何结果调整已写死的 §3 阈值或口径。
- 不动线上 Beta V1.2 / Web UI / Modal。

---

## 6. 已知坑

1. **路径数口径不一致**：生产 16 条；34 日 OOS 5 条；sealed OOS **1 条**；验证 IC 现在 16 条。sealed 的 1 条结果不能直接代表生产；跨口径不要横比。
2. **sealed 8 天已用掉**（`sealed_first_use=true`），不能再用于选型/调参。
3. **42 天 full 包含旧 34 天**（已污染），只能参考，不能用于晋级；只看 first-use 8 天。
4. **20 天验证 IC 不显著**：C2/C03/e03b 的配对 CI 全跨 0；不要把它当唯一选模指标。
5. **e05 kernel 私有**：`luckfu/kronos-small-clean-epoch2-from-c03-e05` 对 `wynstonliu` 不可访问；epoch2 best 已上传为 dataset `wynstonliu/kronos-small-epoch2-best-checkpoint`。
6. **SwanLab key 仍是仓库字面量**（`finetune/kaggle_small_0_1_stage2_wc_dual_t4_kernel/...`），仓库公开；应轮换并改 Kaggle Secrets。
7. **`artifacts/`、`build/` 被 gitignore**：新电脑没有分片/权重/build manifest；从 Kaggle 重新下载。
8. **`push_chunk.py` 的 source_commit 旧一格**：bigbatch 续跑时 `source_sha256` 是新的，但 `source_commit` 可能仍是上一块；不影响权重，只影响血缘。当前不再续 bigbatch，不用管。
9. **kaggle CLI 版本警告**可忽略；认证失败时先 `kaggle auth login`。

---

## 7. 当前 Git 状态

```text
ebbf4d4 docs: record 16-path validation Rank IC (production parity)
fc32425 feat(eval): use production 16-path median in validation Rank IC
40ac963 feat(eval): measure validation Rank IC for C2/C03/epoch2/e03b
69b4528 docs: add sealed OOS cost/amplitude panel supplement
d6aaf68 docs: record sealed OOS judgment (C2 weak alpha, bigbatch closed)
```

本轮新增/修改的关键文件：

- `finetune/small_validation_rank_ic/evaluate_validation_rank_ic.py`（16 条 + 中位数）
- `finetune/small_validation_rank_ic/build_kernel.py`
- `finetune/reports/small_validation_rank_ic_20261023.json`（1 条对照）
- `finetune/reports/small_validation_rank_ic_20261023_16path.json`（16 条）
- `docs/reports/SMALL_VALIDATION_RANK_IC_20261023_CN.md`
- `docs/SEALED_OOS_RESULT_20260918_20260930_CN.md`
- `docs/HANDOVER_20261023_SEALED_OOS_CN.md`

---

## 8. 一句话交接

> **训练线已经关了，别再开新训练。§5.1 的 60 天验证 Rank IC 代码已经落地（16 条路径中位数不变），kernel `wynstonliu/kronos-small-validation-rank-ic-20261023` 正在跑，结果未出，不要重复 push。等它结束再看能不能分开 C2/C03/e03b；如果能，再小规模接入选模。然后写 C2 的 8b 执行约束回测（|pred|≥5% + 停牌/流动性/涨跌停/滑点），用 10/09–10/23 新窗口验证。TimesFM 的 sealed 评测另开一条线，不要混。**
