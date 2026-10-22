# 下一步方案：密封新窗口一次性判定（2026-10-22）

**状态**：方案已定，等待 `wynstonliu/kronos-small-clean-bigbatch-e03b` 训练完成后启动。
**适用**：Kronos-small `small_0.1` 与 TimesFM 3.0 V11 两条线。
**前置文档**：[`HANDOVER_20261021_CN.md`](HANDOVER_20261021_CN.md)（Small epoch2 关线）、[`sweetspot_native_oos_analysis_cn.md`](../finetune/timesfm3_v11_trajectory/docs/sweetspot_native_oos_analysis_cn.md) §7/§9、[`s02_training_and_oos_report_cn.md`](../finetune/timesfm3_v11_trajectory/docs/s02_training_and_oos_report_cn.md) §3–4。

---

## 0. 一句话

**两条线都不再开新训练；用一个没人看过的时间窗（signal 2026-09-18 → 09-30，之后追加 10-09 → 10-23），按本文 §3 事先写死的规则，一次性判完全部冻结候选。** 所需脚本都已在仓库里，不新写模型代码。

---

## 1. 为什么是这一步

9 月以来两条线合计 20 多个候选，没有一个晋级。弯路只有一种形态：

| 环节 | 事实 |
|---|---|
| 评测窗口 | 所有候选都在同一批窗口上判（13 日 → 18 日 → 19 日 → 34 日）。34 日窗口已被 C2/C03/C04/epoch2/C07/C08/SweetSpot/S01/S02 反复使用，文档自己已注明 winner's curse。 |
| 选型指标 | 与 OOS 脱钩已三次独立证实：Small weighted forecast loss（epoch2 -0.27% ↔ OOS Rank IC -0.058）、Stage 3 CE（alpha-optimal ≠ CE-optimal）、TimesFM `trajectory_score`（SweetSpot→S01→S02 验证单调升、OOS 单调降）。 |
| 后处理 | 18 日 rule sweep 已做：扣费夏普最高 0.77；`\|pred_d10\| ≥ 3%` 门槛净收益 +0.12%/笔。后处理不是免费空间。 |

循环是：新训练 → 同一窗口判 → 不如基线 → 关线 → 再开新训练。再加一个训练合同不会打破它；能打破的只有「新窗口 + 事先写死的规则」。

---

## 2. 候选（冻结，不增不减）

| 线 | 候选 | 权重来源 | 角色 |
|---|---|---|---|
| Small | **C2 best**（段 179） | `smmt315/kronos-small-0-1-stage2-cosine-refinement-c2`；SHA `4ee469d4…` | 基线 |
| Small | **bigbatch e03b best** | `wynstonliu/kronos-small-clean-bigbatch-e03b` output（训练完成后冻结 SHA） | 唯一在跑的新权重 |
| TimesFM | **SweetSpot native SWA** | `ibcaoo/kronos-timesfm3-v11-sweetspot-merged-model-export` | 当前最优交付物 |
| TimesFM | **C07 SWA** | C07 best+last 预测空间 50/50 | 对照 |

不加 epoch2、C03、C04、S01、S02（已关）。e03b 若 `summary.json` 不是 `completed`，则 Small 只评 C2 一个。

**bigbatch 现状**（2026-10-22 16:10 +08 核过）：e01b、e02b 已 `COMPLETE`（`luckfu`）；e03b 在 `wynstonliu` 账号，当前 `luckfu` 凭据不可查。**e03b 之后不推 e04b。**

---

## 3. 判定规则（出结果前不改）

统一口径：D10 单点方向、D10 日均 Rank IC（Spearman，按 signal date 平均）、成本 0.3% 双边、10 层 overlay 回测（[`neutralized_ls_backtest.py`](../finetune/neutralized_ls_backtest.py)）。配对差用连续 10 日 block bootstrap，2000 次，seed 20261022。

### 3.1 Small：C2 在新窗口上还有没有 alpha

| C2 D10 日均 Rank IC | 结论 | 后续 |
|---|---|---|
| ≥ 0.10 | alpha 在 9 月后仍在 | 写第 8b 步执行约束回测；论文主结果可引用 |
| 0.05 – 0.10 | 弱于 34 日窗口（0.1424），但非零 | 写第 8b 步，但只针对 `\|pred_d10\| ≥ 5%` 子集与事前流动性过滤；不做全样本 TopK |
| < 0.05 | 9 月后无可用截面 alpha | Small 作为选股信号的路到此为止；封存 C2，不再训，**不写第 8b 步** |

第 8a 步的固定规则回测在三种情形下都出，与 IC 同报。

### 3.2 Small：bigbatch e03b 是否晋级

晋级条件**同时**满足：D10 日均 Rank IC 高于 C2，且配对 bootstrap 95% CI 不含 0。否则关线，记入 §4.4。方向差不作为晋级依据（34 日上已证明方向可升 IC 可降）。

### 3.3 TimesFM：能否当方向 overlay

主指标是 **DirAcc 相对多数类基线的超额**（不是绝对胜率；sweetspot 分析已指出 9 月段绝对 57.49% 相对基线只有 +1.32pp）。

| 条件 | 结论 |
|---|---|
| 超额 ≥ +3pp 且日均 IC ≥ 0.05 | 保留为方向/仓位 overlay；不单独做 TopK 多空 |
| 否则 | 封存。SweetSpot native 仍是工程交付物，但不再当研究主线 |

### 3.4 无论结果如何

- 新窗口第一片（09-18 → 09-30）只看一次。第二片（10-09 → 10-23）等第一片结论写完再构造。
- 这两片在报告中标 `sealed_first_use=true`，之后任何候选再用它们就按「已污染」记。
- 不为任何结果追加「再多看几天」。

---

## 4. 执行步骤

全部用现成脚本；每一步的输入输出接口已核对。

### 4.1 本地：构造新窗口包（CPU，≈ 1 天）

**第 1 步：拿 34 日包的原始数据。** 本机没有 `data/a_share_full_market_v1_beta_temporal_symbol_validation_no_suspension_v1/`（在另一台机器）。从 Kaggle 数据集下载：

```bash
export NO_PROXY="api.kaggle.com,kaggle.com,www.kaggle.com,storage.googleapis.com"; export no_proxy="$NO_PROXY"
kaggle datasets download smmt315/a-share-120d-temporal-symbol-holdout -p artifacts/holdout_34d --unzip
unzip -o artifacts/holdout_34d/evaluation_oos.zip -d artifacts/holdout_34d
ls artifacts/holdout_34d/evaluation_oos/   # evaluation_manifest.json evaluation_panel.pkl evaluation_samples.jsonl combined_raw.csv
```

`combined_raw.csv` 到 2026-10-09；`evaluation_panel.pkl` 每只股票带 `sector` 列。

**第 2 步：增量下载行情。** 从 2026-08-01 开始下（与旧 raw 重叠两个月，用来检测前复权口径漂移）：

```bash
python3 - <<'PY'
import pandas as pd
raw = pd.read_csv("artifacts/holdout_34d/evaluation_oos/combined_raw.csv", usecols=["symbol"])
raw.drop_duplicates().to_csv("artifacts/holdout_34d/symbols.csv", index=False)
PY
python3 finetune/download_a_share_parallel.py \
  --symbols-file artifacts/holdout_34d/symbols.csv \
  --start 2026-08-01 --end 2026-10-22 \
  --output artifacts/raw_incremental_20261022/a_share_incremental.csv \
  --chunk-dir artifacts/raw_incremental_20261022/chunks --workers 4
```

下载器输出含 `market_cap`（amount/turn 代理），满足 `prepare_actual_traded_oos.py` 的列要求。

**第 3 步：构造包。**

```bash
python3 finetune/prepare_actual_traded_oos.py \
  --base-panel artifacts/holdout_34d/evaluation_oos/evaluation_panel.pkl \
  --previous-raw artifacts/holdout_34d/evaluation_oos/combined_raw.csv \
  --incremental-raw artifacts/raw_incremental_20261022/a_share_incremental.csv \
  --previous-package artifacts/holdout_34d/evaluation_oos \
  --signal-start 2026-09-18 --signal-end 2026-09-30 \
  --output-dir data/sealed_oos_20260918_20260930
```

接口说明：`--base-panel` 只用于继承每只股票最新的 `sector` 标签（`append_actual_traded` 第 70–74 行），`evaluation_panel.pkl` 满足；`--previous-package` 用于断言新样本 identity 与 34 日包无重叠。

**若报 `Overlapping raw rows disagree between source downloads`**：说明 8–10 月间有股票复权基准变了。处理：全量重下 `--start 2026-01-01`，加 `--refresh-history` 重跑第 3 步。不要用 `verify=False` 式的绕过。

预期：9 个 signal dates（09-18、21–25、28–30；若 09-25 中秋休市则 8 个），每日约 5,100 只，约 45,000 窗口。标签最晚 10-22。

**第 4 步：发布为 Kaggle 数据集**，`luckfu` 账号，名字 `kronos-sealed-oos-20260918-20260930`，manifest 里的 SHA 原样写入下面两个 kernel 的 gate。

### 4.2 Kaggle：推理（GPU，≈ 1 天）

**第 5 步：Small。** 复制 `finetune/small_clean_oos/evaluate_epoch2.py` → `finetune/small_sealed_oos/evaluate_sealed.py`，只改：

- `EXPECTED_SAMPLES / EXPECTED_DATES / PANEL_SHA / SAMPLES_SHA / IDENTITIES_SHA` → 第 3 步 manifest 的值
- 第 79–80 行日期断言 → `("2026-09-18", "2026-09-30", "2026-10-22")`
- 第 102 行训练数据隔离 → `>= "2026-09-18"`
- `patterns` → `c2_best` + `bigbatch_best`（glob `**/small_0.1_clean_bigbatch_20261022/checkpoints/best_model/model.safetensors`），`expected_segment` 以 e03b 的 `best_metric.json` 为准
- bootstrap seed → 20261022

`kernel_sources` 挂 C2 dataset + e03b output；双 T4；两模型同 run 配对。9 日 × 2 模型约 2.5 小时。

**第 6 步：TimesFM。** `finetune/timesfm3_v11_trajectory/eval_oos/eval_native_merged.py` 直接 glob `**/evaluation_panel.pkl` 与 `**/evaluation_samples.jsonl`，挂新数据集即可，代码不改；用 `push_eval_native_merged.py` 提交。C07 SWA 走 `push_eval.py --chunk c07 --tags best last`，本地做 50/50 融合。单权重约 25 分钟。

### 4.3 本地：出报告（CPU，≈ 半天）

**第 7 步。** Small 用 `finetune/small_clean_oos/panel.py`（L1/L2/L3 + `halt_gap` + 配对 CI）。TimesFM 用 `eval_oos/analyze_predictions.py`，**先把 TopK 年化改成 `horizon=10` 不重叠 cohort**（sweetspot 分析 §9 第 4 条待办），再出表。

**第 8a 步：固定规则回测（无条件，与 IC 同报）。** 在新窗口的 C2 预测上跑下面三条**事先定死**的规则，不开 `neutralized_ls_backtest.py --sweep`，不新增规则：

| 规则 | 来源 | 18 日窗口扣费夏普（参考） |
|---|---|---|
| `overlay_1d_tb10` | `neutralized_ls_backtest.py` 冻结基线：10 层日 overlay，Top/Bottom 10% | 0.15 |
| `overlay_5d_tb10` | 18 日 rule sweep 最优：每 5 日开一个 10 日袖套 | 0.77 |
| `panel.py` L1 幅度门槛 ≥3% / ≥5%，剔除 `halt_gap` | 34 日 panel 的 L1 层 | +0.12% / +0.65% 每笔 |
| **幅度标定后的门槛**：在 34 日 C2 预测上拟合单调映射（isotonic，`pred_d10` → 实际 `\|ret_d10\|` 分位），应用到新窗口后再套 ≥3% / ≥5% | 34 日 L2：中位幅度比 0.856，Top Decile 预测 14.5% vs 实际 10.1% | 新增，无参考 |
| **双模型组合**：C2 `pred_d10 ≥ 5%` **且** SweetSpot native D10 方向为涨，多头，剔除 `halt_gap`；对照 C2-only ≥5% | 两条线首次同窗口配对 | 新增，无参考 |

成本 0.3% 双边。输出换手、净值、回撤、每笔净收益。这一步回答的是「在旧窗口上选出来的规则，到 9 月还成不成立」，不是选新规则。

后两条的约束：标定映射只在 34 日上拟合，参数先落盘再碰新窗口；组合规则只有这一条，阈值 5% 与 panel 一致，不扫 3%/8%，不扫 SweetSpot 的置信度。单调标定不改排序，所以 Rank IC 不受影响，只改门槛的覆盖率与每笔净收益。组合规则可行的前提是第 5、6 步的预测都来自同一份 `evaluation_samples.jsonl`，按 `(symbol, asof_date)` 配对；配对率低于 99% 则组合规则不出数。

**第 8b 步：执行约束回测（有条件）。** 现有回测代码（`neutralized_ls_backtest.py`、`backtest_august_costs.py`、`backtest_august_path.py`、`panel.py`）都不处理涨跌停、停牌、滑点，这部分是新代码。只在 §3.1 命中 ≥ 0.05 时写；< 0.05 不写。写时的约束：信号日次日开盘成交；一字涨停不能买、一字跌停不能卖；停牌顺延到复牌日；滑点按成交额分层固定值；仓位按 10 日持有分层。规则沿用 8a 的三条，仍不扫描。

**第 9 步。** 按 §3 写结论，一页，放 `docs/SEALED_OOS_RESULT_20260918_20260930_CN.md`；报告 JSON 放 `finetune/reports/`。写完再构造第二片。

---

## 5. 等 e03b 期间可以做的事（都不碰模型）

1. 第 1–2 步可以现在就跑（下载不依赖 e03b）。
2. 第 3 步构造第一片。构造完**不要**先拿 C2 跑一次「看看」——那会把这片用掉。
3. 轮换 `SWANLAB_API_KEY`：`finetune/kaggle_small_0_1_stage2_wc_dual_t4_kernel/kaggle_small_0_1_stage2_wc_dual_t4.py` 里仍是字面量，仓库公开。改为 Kaggle Secrets，并从 git 历史角度视为已泄露。
4. `analyze_predictions.py` 的 `horizon=10` 不重叠 cohort 修正（第 7 步前置）。
5. 把第 8a 步的五条规则写成一个不带 `--sweep` 的驱动脚本，接 `evaluate_sealed.py` 的 shard 输出格式和 TimesFM 的 `npz` 输出；可以用 34 日的旧 shard 做接口冒烟，但不把 34 日的数字写进新报告。
7. 在 34 日 C2 shard 上拟合幅度标定映射，参数写入 `finetune/reports/c2_amplitude_calibration_fit_on_34d.json`，先提交再构造新窗口。这是第 8a 步标定规则的唯一允许来源。
6. 把 `wynstonliu` 凭据配进本机 Kaggle CLI 的第二个 profile，否则 e03b 跑完也查不到。

---

## 6. 不做的事

- 不推 bigbatch e04b；不开 Rank Loss / ListNet / 换手惩罚训练。这些是 §3.1 命中 ≥ 0.10 之后才有资格讨论的方向；若 C2 在新窗口上 IC 不在，换目标也救不回来。届时若开，选型指标必须是验证集上的同日截面 Rank IC，不再用 forecast loss / CE / `trajectory_score`；同日 panel 必须是真实按交易日组 batch，不是 batch 内偶然同日的 pair（Stage 3 P1 的教训）。
- 不把 C06 加进候选。它在 Day 2–5 的短线方向（约 57.3%–57.6%）只对短持有策略有意义，而本轮的持有期是 10 日；加进来只增加多重比较。若将来做短持有线，另开窗口。
- 不在 34 日窗口上再评任何东西。
- 不动 webui / Beta V1.2 / Modal 线上。
- 不为任何候选的结果调整 §3 的阈值或口径。

---

## 7. 账号与路径

| 项 | 值 |
|---|---|
| 34 日包 | Kaggle `smmt315/a-share-120d-temporal-symbol-holdout` |
| C2 best | `smmt315/kronos-small-0-1-stage2-cosine-refinement-c2`；SHA `4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a` |
| Tokenizer | SHA `59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee` |
| bigbatch 链 | `luckfu/…-e01b` ✅ → `luckfu/…-e02b` ✅ → `wynstonliu/…-e03b`（跑中） |
| SweetSpot native | `ibcaoo/kronos-timesfm3-v11-sweetspot-merged-model-export` |
| 新窗口数据集（待建） | `luckfu/kronos-sealed-oos-20260918-20260930` |
| 新窗口 Small kernel（待建） | `luckfu/kronos-small-sealed-oos-20260918` |
| 本地包 | `data/sealed_oos_20260918_20260930/` |
| 本地 raw 增量 | `artifacts/raw_incremental_20261022/`（.gitignore 忽略） |

Kaggle CLI 若遇 `SSLError: EOF occurred in violation of protocol`，按 HANDOVER §5.1 设 `NO_PROXY`，不要关证书校验。

---

## 8. 一句话交接

> 两条线的同目标续训都已到头，再开训练只会重复「同一窗口判、不如基线、关线」。下一步是构造 signal 2026-09-18 → 09-30 的密封窗口（脚本 `prepare_actual_traded_oos.py`，输入来自 Kaggle 34 日包 + BaoStock 增量），等 e03b 跑完后把 C2 / e03b / SweetSpot native / C07 SWA 四个冻结候选各推一次，按 §3 事先写死的阈值出结论。第一片只看一次；e03b 之后不推 e04b。
