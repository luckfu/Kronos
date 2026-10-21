# Small Clean Epoch2 接手文档（2026-10-21）

**仓库**：<https://github.com/luckfu/Kronos>
**交接时 HEAD**：`c71ce03`（= `origin/master`；本文档的最终状态更新随后以一个 docs 提交追加）
**工作区**：干净
**用途**：换机器 / 隔天继续时，10 分钟内恢复上下文，知道做了什么、结论是什么、下一步怎么跑。

> **状态（2026-10-21 14:11 UTC 更新）**：`luckfu/kronos-small-clean-epoch2-vs-c2-full-oos` 已 **COMPLETE**，output 已下载校验，三层报告已生成，**epoch2 线已按 §6.3 判定关闭**。结论见 §4.2b，执行流水见 §6.5.1。

---

## 0. 先看这四件事

1. **epoch2 已训练完成 520/520 段，但验证 loss 已饱和。**
   Best 在 **Segment 501**，weighted forecast selection loss **2.2743811**；C03 父 Best 为 2.2806120。改善只有 0.27%，而且第 242 段之后 259 段只再降 **5.7e-5**。

2. **✅ epoch2 的 OOS 已跑完并判定：关闭这条线。**
   同 run 配对、同 34 日合同：D10 方向 **53.82% vs C2 54.59%（-0.77pp，越 0.3pp 关闭门槛）**；幅度 Rank IC **0.2103 vs 0.2470（-0.0368，CI 不含 0）**；D10 日均 Rank IC **0.0843 vs 0.1424（-0.058，CI 不含 0）**。**验证集 -0.27% 的 loss 改善没有转化成 OOS 排序能力。** 详见 §4.2b。

3. **冻结 34 日 OOS 上，C2 仍优于 clean C03。**
   D10 方向 **54.59% vs 53.86%**（Δ **-0.73pp**，block bootstrap CI 不含 0）；幅度 Rank IC **0.247 vs 0.235**（Δ **-0.0116**，CI 不含 0）。MAE、D10 Rank IC、净 PnL 的差异在 34 日口径下**不显著**。

4. **真正能覆盖 0.4% 成本的是高预测幅度样本。**
   C2 全样本扣费净 PnL **-0.35%/笔**；`|pred_d10| ≥ 3%` 才转正（**+0.12%**），≥5% 为 **+0.65%**，≥8% 为 **+1.38%**；预测幅度 Top Decile 为 **+1.47%**。**既然三轮续训（C03/C04/epoch2）都没超过 C2，下一步应转向 §6.4 的条件化交易验证，而不是继续堆训练。**

---

## 1. Git 状态

| 项目 | 值 |
|---|---|
| 分支 | `master` |
| 交接时 HEAD | `c71ce03`；**OOS 收尾 HEAD `6cd6d44`**；本文档的 OOS 结果更新随后追加 |
| origin | `https://github.com/luckfu/Kronos.git` |
| 评测代码 | `976819f` panel + probes + baseline report |
|  | `563272d` epoch2-vs-C2 OOS kernel |
|  | `dff8b20` epoch2 panel report builder |
| relay 修复 | `d4df68e` chunk owner / run_id |
| 文档 | `8422d72` 初版接手文档；`5518c09` C05 命名消歧；`c71ce03` OOS 状态 |
| 未跟踪但应保留 | 无（`artifacts/`、`build_epoch2/` 被 .gitignore 忽略，见 §7） |

```bash
git log -8 --oneline
# (待追加) docs: archive epoch2 vs C2 OOS results
# 6cd6d44 docs: finalize epoch2 OOS handover for home handoff
# c71ce03 docs: record submitted epoch2-vs-C2 OOS kernel
# dff8b20 feat(eval): add epoch2-vs-C2 panel report builder
# 563272d feat(eval): add epoch2-vs-C2 full OOS kernel
# 5518c09 docs: disambiguate Clean C05 from TimesFM C05 in handover
# 8422d72 docs: add Small Clean epoch2 handover for 2026-10-21
# d4df68e fix(small-clean): record relay chunk owner and chunk run id
```

---

## 2. 模型血缘与当前状态

| 阶段 | 位置 | 段 | 状态 | 关键指标 |
|---|---|---:|---|---|
| **C2 Best**（线上/基线） | `smmt315/kronos-small-0-1-stage2-cosine-refinement-c2`（Kaggle dataset）；ModelScope `luckfu/Kronos-small-0.1-Cosine-C2-Best` | 179 | 已发布 | SHA `4ee469d4…`；D10 方向 54.59%，Rank IC 0.1424 |
| **Clean C03 Best** | `luckfu/kronos-small-clean-wc-dual-t4-c03` | 518 | COMPLETE | SHA `0837035e…`；selection loss 2.2806120 |
| **C04 Best** | `user281434/kronos-small-clean-wc-dual-t4-c04` | 715（last 720） | COMPLETE | best SHA `81d73521…`；selection loss 2.2818275 |
| **Clean C05**（第三轮幂律采样） | `user281434/kronos-small-clean-wc-dual-t4-c05` | 721→920 | **状态未知** | 当前 `luckfu` 凭据查询被拒，见 §7 |
| **Epoch2 Best** | `luckfu/kronos-small-clean-epoch2-from-c03-e05` | 501（last 520） | COMPLETE | best SHA `68feb21b…`；selection loss **2.2743811** |
| Epoch2 前序 chunk | `user281434/…e02`（291 段）、`luckfu/…e04`（498 段） |  | COMPLETE | e02 当前账号不可查；e04 可查 |

> **命名冲突提醒**：本表里的 **Clean C05** 是 **Kronos-small clean 的第三轮幂律采样接力**（721→920 段），**不是** TimesFM 3.0 V11 的 C05。TimesFM 线也有自己的 C02–C08 编号；其中 C05 是 Step 111k 的旧 checkpoint，见 [`c06_training_and_gradient_anomaly_technical_report_cn.md`](../finetune/timesfm3_v11_trajectory/docs/c06_training_and_gradient_anomaly_technical_report_cn.md)。两条线的 C0x 编号互不相干。

**Epoch2 训练合同**（来自 `experiment_manifest.json`）：

- 实验：`small_0.1_clean_epoch2_from_c03_20261020`
- 父模型：C03 best（SHA `0837035e…`）
- 数据：`smmt315/a-share-120d-temporal-symbol-holdout`，manifest SHA `b85ccc45…`
- 520 段 × 每段约 20,000 次抽样，1 遍覆盖，共 **10,397,679** 个训练窗口
- 双 T4，batch 32/GPU（global 64），AMP float16，`lr=1e-5`，`warmup_constant`，loss=forecast，history_weight=0.02
- 验证：固定 **123,836** 个窗口，242 个 signal days（2025-07-03 至 2026-07-02）
- 终态：`status=completed`，`completed_segments=520`，global step **162,724**
- 最后一轮验证（seg 520）：weighted forecast **2.2790106**，unweighted forecast **2.2692410**

> 注意：e05 打包源码的 `source_commit` 是 `620e8d6`，source bundle SHA `375ddf…` 与 e04 相同。它**不包含**后来提交的 owner/run_id 日志修复；这只影响日志字段，不影响权重和训练合同。下一次 chunk 才会用到新修复。

---

## 3. 本轮产出（代码地图）

| 文件 | 类型 | 作用 |
|---|---|---|
| [`finetune/small_clean_oos/panel.py`](../finetune/small_clean_oos/panel.py) | 新增 | 三层评测：L1 成本后 PnL（按预测幅度分层）、L2 幅度 Rank IC/标定、L3 D10 Rank IC/Top-Bottom；含 `halt_gap` 事前子集和 moving-block bootstrap |
| [`finetune/small_clean_oos/baseline_panel.py`](../finetune/small_clean_oos/baseline_panel.py) | 新增 | 从已发布 shard 重建 C2 vs clean C03 基线；输出可逐字节复现 |
| [`finetune/reports/small_clean_direction_amplitude_baseline_20261021.json`](../finetune/reports/small_clean_direction_amplitude_baseline_20261021.json) | 新增 | C2 vs C03 的完整三层结果、paired delta、bootstrap CI |
| [`finetune/small_clean_epoch2_relay/probe_direction_amplitude.py`](../finetune/small_clean_epoch2_relay/probe_direction_amplitude.py) | 新增 | 在 C2 的 18 日 consensus 预测包上做方向/幅度/成本探针 |
| [`finetune/small_clean_epoch2_relay/probe_direction_amplitude_oos.py`](../finetune/small_clean_epoch2_relay/probe_direction_amplitude_oos.py) | 新增 | 在 34 日 OOS 分片上做方向/幅度/停牌子集探针 |
| [`finetune/small_clean_epoch2_relay/probe_swanlab_loss.py`](../finetune/small_clean_epoch2_relay/probe_swanlab_loss.py) | 新增 | 从 SwanLab 拉 validation/train loss 并回归，判断平台期 |
| [`finetune/small_clean_epoch2_relay/launch.py`](../finetune/small_clean_epoch2_relay/launch.py) | 修改 | 从 chunk config 读取 owner；dashboard 事件使用 chunk 级 run_id |
| [`finetune/small_clean_epoch2_relay/push_chunk.py`](../finetune/small_clean_epoch2_relay/push_chunk.py) | 修改 | 在 build config 中写入 `owner` |

---

## 4. 关键结果

### 4.1 epoch2 训练：完成，但饱和

| 指标 | 值 |
|---|---:|
| 完成段数 | 520 / 520 |
| 唯一训练窗口 | 10,397,679 |
| Best 段 | **501** |
| Best weighted forecast | **2.2743811** |
| C03 父 Best | 2.2806120 |
| 相对改善 | **-0.00623（-0.27%）** |
| e02 Best @242 | 2.2744378 |
| 242→501 段改善 | **-0.0000566** |
| 验证曲线（469 点） | 2.2651–2.2783；斜率约 **-1.3e-5/段**；首 2.2721，末 2.2708 |
| seg 520 最后验证 | weighted 2.2790106 / unweighted 2.2692410 |

**解释**：`lr=1e-5 + warmup_constant` 在这一轮已到噪声地板。`best` 主要出现在前 242 段；后 259 段几乎不再改善。继续 epoch3 的期望收益极低。若要继续训练，应改变学习率/优化器/采样分布或训练目标，而不是重复同一合同。

**边界**：以上是固定验证集上的 loss，不是 OOS；epoch2 还没有 OOS 结论。

### 4.2 冻结 34 日 OOS：C2 vs clean C03

OOS 合同：**2026-08-03 至 2026-09-17，34 个 signal dates，175,428 个窗口，5,173 只股票**；标签最晚 2026-10-09；最晚训练/验证记录 2026-07-17，早于首个 OOS signal。

| 指标 | C2 Best | C03 clean Best | Δ（clean−C2） | 34 日 bootstrap 95% CI |
|---|---:|---:|---:|---|
| D10 方向 | **54.59%** | 53.86% | **-0.73pp** | [-1.11, -0.09]（不含 0） |
| 10 日全路径方向 | 54.02% | 53.78% | -0.24pp | [-0.55, +0.23] |
| D10 MAE | **5.11%** | 5.42% | +0.59pp | [-0.10, +1.44] |
| D10 RMSE | **8.11%** | 8.61% | +0.83pp | [-0.26, +2.01] |
| D10 日均 Rank IC | **0.1424** | 0.1200 | -0.0224 | [-0.0449, +0.0148] |
| 幅度 Rank IC | **0.2470** | 0.2354 | **-0.0116** | [-0.0185, -0.0052]（不含 0） |
| 全样本扣费净 PnL/笔 | **-0.35%** | -0.52% | -0.17pp | [-0.29, +0.04] |
| 预测幅度 Top Decile 净 PnL | **+1.47%** | +0.65% | -0.82pp | 点估计 |

**统计上稳健的只有两条**：D10 方向退化和幅度排序退化。**MAE、D10 Rank IC、净 PnL 的 CI 都跨 0**，不要用它们下“显著变差”的结论。

> **口径提醒**：旧报告里的 `path_direction` 是 10 个 horizon 的平均（-0.24pp，不显著）；新 panel 的 `direction` 是 D10 单点（-0.73pp，显著）。两者不矛盾，但不能混用。

### 4.2b epoch2 Best（段 501）同窗配对 OOS：已完成，判定关闭

同一次 kernel run 内配对解码，复用同一 34 日冻结合同、同一 175,428 窗口，`label_date_d*` 与 identity 逐项一致，因此是**同 run 配对**比较，不存在跨 run 合并误差。来源 kernel `luckfu/kronos-small-clean-epoch2-vs-c2-full-oos`：`summary.json` 为 `complete`，68 个 shard 齐全（c2/epoch2 各 34 日），`plan.json` 中 epoch2 SHA 为 `68feb21b…`。

| 指标 | C2 Best | epoch2 Best | Δ（epoch2−C2） | 34 日配对 bootstrap 95% CI |
|---|---:|---:|---:|---|
| D10 方向 | **54.59%** | 53.82% | **-0.77pp** | [-3.01, +2.97] |
| 10-horizon 平均方向 | 54.02% | 53.80% | -0.22pp | [-2.34, +2.38] |
| D10 MAE | **7.01%** | 7.70% | +0.70pp | [-0.02, +1.50] |
| D10 RMSE | **10.78%** | 11.83% | +1.05pp | [-0.25, +2.35] |
| D10 日均 Rank IC | **0.1424** | 0.0843 | **-0.0581** | **[-0.0939, -0.0025]（不含 0）** |
| 幅度 Rank IC | **0.2470** | 0.2103 | **-0.0368** | **[-0.0467, -0.0227]（不含 0）** |
| Top-Bottom 10% spread | **2.86%** | 1.00% | -1.86pp | [-3.20, +0.66] |
| 全样本扣费净 PnL/笔 | -0.35% | -0.32% | +0.03pp | [-0.24, +0.54] |
| 预测幅度 Top Decile 净 PnL | **+1.47%** | +0.58% | -0.89pp | 点估计 |
| `halt_gap` 方向（235 样本） | 36.6% | **46.8%** | +10.2pp | 点估计 |

**判定（按 §6.3 事先写死的规则）：关闭 epoch2 线。**

- epoch2 的 D10 方向比 C2 低 **0.77pp**，已越过 **0.3pp** 关闭门槛；幅度 Rank IC 同时低于 C2，且 CI 不含 0。两条关闭条件**同时命中**，不需要再做 forward test。
- 退化最重的是排序层：D10 日均 Rank IC 0.1424 → 0.0843（-0.058，CI 不含 0），Top-Bottom 10% spread 2.86% → 1.00%，正 IC 天数 29/34 → 26/34。§4.1 那 **-0.27%** 的验证集 loss 改善**没有转化到 OOS 排序能力**，反而伴随确定性退化。
- 全样本净 PnL 名义 +0.03pp 改善，但 CI 跨 0，且两者都是负值（-0.35% / -0.32%），不构成收益改善证据。Top Decile 净 PnL 从 +1.47% 掉到 +0.58%，与幅度 IC 退化方向一致。
- 唯一“改善”是 `halt_gap` 方向 36.6% → 46.8%，但该子集只有 235 样本（0.13%），方向仍远低于 50%，且两个模型的 `halt_gap` 扣费 PnL 依然是灾难（-7.95% / -6.74%）。不构成晋级理由。
- **C2 继续作为唯一线上基线，当前无晋级候选。** epoch2 / epoch3 同合同续训线到此终止（见 §4.4 与 §8）。

> **口径提醒**：① 本表 `direction` 是 D10 单点，`10-horizon 平均方向` 才是旧报告的 `path_direction`，不要混读。② 配对 bootstrap CI 明显宽于 §4.2 的独立 bootstrap，因为差值序列的逐日波动更大；方向差在本口径下**不显著**，但方向绝对值已越阈值、幅度与排序退化显著，判定成立。③ 成本 0.4% round trip，与 panel 一致。

> 报告：[`finetune/reports/small_clean_epoch2_vs_c2_full_oos_20261021.json`](../finetune/reports/small_clean_epoch2_vs_c2_full_oos_20261021.json)

### 4.3 三层评测的交易含义

**L1 成本门槛（C2）**：

| 预测 \|D10\| 阈值 | 覆盖率 | 方向 | 扣 0.4% 成本后净 PnL/笔 |
|---|---:|---:|---:|
| ≥1% | 87.0% | 55.5% | -0.29% |
| ≥2% | 72.3% | 57.0% | -0.12% |
| ≥3% | 57.3% | 58.6% | **+0.12%** |
| ≥5% | 33.8% | 61.0% | **+0.65%** |
| ≥8% | 15.8% | 62.6% | **+1.38%** |

方向随幅度单调上升：最低 decile 51.5% → 最高 decile **60.9%**；最高 decile 净 PnL **+1.47%/笔**。

**L2 幅度**：日均幅度 Rank IC **0.247**，ICIR **4.13**，**34/34 天为正**。但标定非线性：中位幅度比 0.856（整体低估），Top Decile 预测 14.5% vs 实际 10.1%（尾部高估）。C03 把中位幅度比推到 0.921，却把幅度排序 IC 降到 0.235，尾部高估更严重（17.0% vs 9.65%）。

**L3 排序**：C2 D10 Rank IC 日均 0.1424、ICIR 1.20、29/34 天为正；Top-Bottom 10% spread 日均 2.86%。

**两个陷阱子集**：

| 子集 | 样本 | C2 方向 | C2 扣费 PnL/笔 | 说明 |
|---|---:|---:|---:|---|
| `halt_gap`（停牌/标签滞后） | 235（0.13%） | **36.6%** | **-7.95%** | 必须做事前停牌/流动性过滤 |
| 事后“已实现高波动 Top Decile” | 17,559 | 35.5% | **-6.64%** | 事后条件化，不可交易 |

反过来，**事前**波动/成交额 selector 才有正收益（0.3% 成本口径）：`log_amt20` **+1.63%**、`abs_mom20` **+1.48%**、`range20` **+0.87%**、`vol20` **+0.74%**、`atr20` **+0.60%**。它们与“事后已实现高波动”的重叠只有 10%–18%。

> 来源：[ex-ante 尾部报告](reports/small_clean_exante_tail_20261019.json)（0.3% 成本）与 [panel 基线报告](../finetune/reports/small_clean_direction_amplitude_baseline_20261021.json)（0.4% 成本）。引用时不要混口径。

### 4.4 其他线

- **C04 vs C03**：整体方向 -0.03pp（不显著），整体 MAE **变差 +0.13pp**（CI 不含 0）；但“已实现大幅波动 Top Decile”方向 **+3.65pp**（CI [+2.55, +4.76]）。幂律采样只改善尾部方向，没有改善整体校准。
- **epoch2 vs C2**：配对 OOS，D10 方向 **-0.77pp**、幅度 Rank IC **-0.0368**（CI 不含 0）、D10 Rank IC **-0.058**（CI 不含 0）。**线已关闭**，详见 §4.2b。
- **Clean C05**（Kronos-small 第三轮）：10-18 16:45 已确认真实启动（721/920 段），但当前 `luckfu` 凭据查询 `user281434` 的 kernel 返回 permission denied，**没有终态归档**。
- **TimesFM 3.0 V11**：C07 best OOS 方向 **58.42%**、Pooled IC **0.1557**；C08 best **56.84%**、**0.1503**。同一 175,428 窗口上方向强于 Kronos C2，但日度 IC/ICIR、轨迹相关度和评测口径不同，不能直接横比。见 [C07/C08 复盘](../finetune/timesfm3_v11_trajectory/docs/c08_training_and_oos_technical_report_cn.md)。

---

## 5. 在家复现

### 5.1 环境

```bash
git clone https://github.com/luckfu/Kronos.git
cd Kronos
git switch master
git pull --ff-only origin master

python3 -c "import sys, numpy, pandas; print(sys.version.split()[0], numpy.__version__, pandas.__version__)"
# 本次验证：3.12.12 / 1.26.4 / 2.3.3
```

`panel.py` / `baseline_panel.py` / `report_epoch2.py` 需要 **numpy + pandas + scipy**；训练/推理才需要 torch。SwanLab 探针需要 `swanlab`，可用 `python3 -m venv .venv-swanlab-api && .venv-swanlab-api/bin/pip install swanlab pandas numpy scipy`。

> **2026-10-21 实测坑**：`pandas.Series.corr(method="spearman")` 在 pandas 2.3.x 内部 `from scipy.stats import spearmanr`，**没有 scipy 会直接 `ModuleNotFoundError`**，而报错位置在 `panel.py:266` 的 `official_stats`，看起来像代码 bug。本次在 homebrew `python3`（无 scipy）上踩到，改用带 scipy 的解释器（如 `/opt/miniconda3/bin/python3`）即可。若要在 homebrew python 上跑：`python3 -m pip install scipy`。

> **Kaggle CLI + 系统代理的坑**：macOS 开了 Clash/Verge 之类的 7890 端口代理时，`requests` 会读取系统代理并对 `api.kaggle.com` 做 TLS 拦截式转发，多次报 `SSLError: EOF occurred in violation of protocol` / `record layer failure`，而 `curl` 与直连都正常。绕过办法（本次使用）：
>
> ```bash
> export NO_PROXY="api.kaggle.com,kaggle.com,www.kaggle.com,storage.googleapis.com"
> export no_proxy="$NO_PROXY"
> ```
>
> 不要为此关闭证书校验，也不要 `verify=False`。

### 5.2 拿 OOS 分片（仓库里没有，`artifacts/` 被忽略）

```bash
kaggle config view   # 应为 username: luckfu
kaggle kernels output luckfu/kronos-small-clean-vs-c2-full-oos \
  -p artifacts/small_clean_oos_shards -o

ls artifacts/small_clean_oos_shards/small_clean_oos/shards | wc -l   # 期望 68
```

目录应为 `artifacts/small_clean_oos_shards/small_clean_oos/shards/c2_best_YYYY-MM-DD.csv.gz` 与 `clean_best_YYYY-MM-DD.csv.gz`。每个模型 34 天，共 68 个分片，合计约 71MB。

### 5.3 重算基线（应逐字节一致）

```bash
python3 finetune/small_clean_oos/baseline_panel.py
```

期望尾部：

```text
c2_best: 175428 rows, 34 dates
clean_best: 175428 rows, 34 dates
wrote finetune/reports/small_clean_direction_amplitude_baseline_20261021.json
sanity: c2 path_direction=0.540222 mae=0.051143 rmse=0.081114
        c3 path_direction=0.537800 mae=0.054180 rmse=0.086150
  direction          delta=-0.00732  95%CI [-0.01110,-0.00095]*
  amplitude_rank_ic  delta=-0.01164  95%CI [-0.01848,-0.00522]*
```

`git status` 应该在**数值上**没有 diff。实践提醒：换 numpy/BLAS 构建后，报告里的浮点末位可能变化（本次在 homebrew numpy 2.3.5 上重算，只有 `x.xxxxxx…8044` vs `…8348` 这类 1e-16 级差异，93 行全为纯浮点末位，不影响任何结论）。**只看数值差异，不要把这类末位波动当成回归**；确认无实质差异后 `git checkout -- finetune/reports/small_clean_direction_amplitude_baseline_20261021.json` 还原即可，避免噪声提交。

### 5.4 跑探针

```bash
# 34 日 OOS 的方向/幅度/停牌子集
python3 finetune/small_clean_epoch2_relay/probe_direction_amplitude_oos.py

# C2 的 18 日 consensus 预测包
python3 finetune/small_clean_epoch2_relay/probe_direction_amplitude.py

# SwanLab loss 平台期（需要 swanlab；凭据从既有 kernel 源码读取，不打印）
.venv-swanlab-api/bin/python finetune/small_clean_epoch2_relay/probe_swanlab_loss.py smce2c3_long
```

### 5.5 下载 e05 权重并校验

```bash
kaggle kernels output luckfu/kronos-small-clean-epoch2-from-c03-e05 \
  -p artifacts/kernels/e05 -o
```

约 220MB（best/last 各 99MB + tokenizer 16MB + 日志/指标）。

```bash
BASE=artifacts/kernels/e05/clean_small_runtime/outputs/models/small_0.1_clean_epoch2_from_c03_20261020

shasum -a 256 "$BASE/checkpoints/best_model/model.safetensors"
# 68feb21bb0d2d6b385e69e95765536b25f2cad07a8efb011cd1bcdea878fda5d
shasum -a 256 "$BASE/checkpoints/last_model/model.safetensors"
# 2f6bd405841a6b8a62eb1aa8e1327a5e7d15f05706eda088e460cf7c660e2368

python3 - <<'PY'
import json
base = "artifacts/kernels/e05/clean_small_runtime/outputs/models/small_0.1_clean_epoch2_from_c03_20261020"
s = json.load(open(f"{base}/summary.json"))["final_result"]
b = json.load(open(f"{base}/checkpoints/best_model/best_metric.json"))
print(s["status"], s["completed_segments"], s["best_val_loss"])
print("best segment", b["segment"], "selection_loss", b["selection_loss"])
PY
# 期望：completed 520 2.2743811252686346
#       best segment 501 selection_loss 2.2743811252686346
```

`best_model/` 是完整的 `from_pretrained` 目录：`model.safetensors`（F32，141 个 tensor，约 99MB）、`config.json`（86 sectors、context_layer=6、8 层、d_model=512）、`best_metric.json`。

### 5.6 快速自检清单

- [ ] `git log -1` 是本文之后的 docs 提交，工作区干净
- [ ] `shards` 有 68 个文件，两个模型各 34 天
- [ ] `baseline_panel.py` 输出与提交报告一致
- [ ] e05 `summary.json` 为 `completed / 520`
- [ ] e05 best SHA = `68feb21b…`
- [ ] 没有把 `artifacts/`、`build/`、`*.safetensors` 加进 Git

---

## 6. epoch2 OOS（已执行完毕，结论：关闭）

### 6.1 C2 + epoch2 配对：**COMPLETE（2026-10-21 14:11 UTC 收尾）**

**状态**：`luckfu/kronos-small-clean-epoch2-vs-c2-full-oos` 已 **COMPLETE**，source SHA `1ef4094a…`。复用同一 34 日冻结合同，一次产出 `c2_best_*` 和 `epoch2_best_*` 两组 shard，C2 在同 run 内重新解码，换来配对一致性。output 已下载，68 个 shard 齐全，`summary.json` 为 `complete`。

验收（§6.2）逐条通过：两模型各 175,428 窗口 × 34 dates；identity 与 date 集合完全一致；`inputs_verified` 含 epoch2 SHA `68feb21b…`；`plan.json` 的 epoch2 SHA 亦为 `68feb21b…`。三层报告已生成，含 L1/L2/L3 + `halt_gap` + 配对 bootstrap CI。

**结论：按 §6.3 判定规则关闭 epoch2 线**（D10 方向 -0.77pp 越 0.3pp 门槛、幅度 Rank IC -0.0368 CI 不含 0）。完整数据表与判读见 **§4.2b**；执行流水见 §6.5。

实际代码：

- [`finetune/small_clean_oos/evaluate_epoch2.py`](../finetune/small_clean_oos/evaluate_epoch2.py)：`patterns` 为 c2/epoch2，epoch2 `expected_segment=501`，SHA gate `68feb21b…`，校验 `summary.json` 为 completed/520；`worker` 按 rank 映射两个 label。
- [`finetune/small_clean_oos/submit_epoch2.py`](../finetune/small_clean_oos/submit_epoch2.py)：kernel id、C2 + e05 `kernel_sources`、source bundle。
- [`finetune/small_clean_oos/report_epoch2.py`](../finetune/small_clean_oos/report_epoch2.py)：下载 shard 后生成三层面板报告，并复用旧 C03 shard 做三方对比。

**如果重跑/只补 epoch2**：只推理 epoch2 更省 GPU（单卡约 2.3h）；届时把 `evaluate_epoch2.py` 的 `worker` 限制为 rank 0、去掉 C2 label，复用已有 C2/C03 shard 做合并。但本次已经选了配对方案，不要再改。

### 6.2 验收标准（本次已全部通过 ✅）

- [x] 每个模型 **175,428** 个窗口 × **34** 个 dates
- [x] identity 集合与 C2/C03 完全一致；`label_date_d*` 逐项一致
- [x] `inputs_verified` 日志包含 epoch2 的 SHA `68feb21b…`
- [x] `summary.json` 为 `complete`，不是 `partial`
- [x] 三方 `panel()` 与 `paired_deltas()` 输出齐全，包含 L1/L2/L3 + halt_gap

### 6.3 判定规则（建议先写死再跑）

- 若 epoch2 的 **D10 方向 < C2 - 0.3pp**，或 **幅度 Rank IC < C2**：**关闭 epoch2 线**，不再追加训练。
- 若方向/幅度至少一项显著优于 C2：仍不直接上线；先在**未参与本轮设计的日期窗口**做一次 forward test。
- 无论结果如何，**C2 继续作为线上基线**；当前没有晋级候选。

**实际判定结果（2026-10-21 执行）**：

| 检查项 | 实测 | 是否触发关闭 |
|---|---|---|
| D10 方向 < C2 − 0.3pp | 53.82% vs 54.59%，**-0.77pp** | **是** |
| 幅度 Rank IC < C2 | 0.2103 vs 0.2470，**-0.0368**（CI 不含 0） | **是** |

→ **两条关闭条件同时命中，epoch2 线关闭。** 不进入 forward test，不上线，C2 保持唯一基线。详见 §4.2b。

> 判定前置提醒（下次仍适用）：本规则必须在看到 OOS 结果**之前**写死，否则容易事后挑选口径。本次是在 kernel 提交前就写入 §6.3 的。

### 6.4 之后：条件化交易验证

把评测从“全样本 Rank IC”推进到可交易的规则：

1. `|pred_d10| ≥ 3%`（或 5%）作为入场门槛。
2. 事前剔除 `halt_gap`（停牌/标签滞后）、低流动性、涨跌停无法成交。
3. 用真实日度组合回测：换手、滑点、涨跌停、停牌执行约束都要计入，而不是等权单笔平均。
4. 在**新日期窗口**验证，不能再用这 34 天（已参与设计讨论）。

### 6.5 回家后的执行清单

**0）确认状态。**（原始交接时 kernel 仍为 `RUNNING`，日志接口为空是已知延迟。本次已执行完毕，结果见 §6.5.1；以下命令保留作参考/复跑用）

```bash
kaggle kernels status luckfu/kronos-small-clean-epoch2-vs-c2-full-oos
kaggle kernels logs   luckfu/kronos-small-clean-epoch2-vs-c2-full-oos
```

若报 `Permission 'kernels.get' was denied`，先重新登录或临时生成 token：

```bash
kaggle auth login
# 或
KAGGLE_API_TOKEN="$(kaggle auth print-access-token)" kaggle kernels status luckfu/kronos-small-clean-epoch2-vs-c2-full-oos
```

> 本会话的沙箱曾因 `~/.kaggle` 只读而用 `HOME=/tmp/khome` 复制凭据；家里正常登录即可，不需要这一步。

**1）下载 output。** 完成后执行：

```bash
kaggle kernels output luckfu/kronos-small-clean-epoch2-vs-c2-full-oos -p artifacts -o
```

应得到：

```text
artifacts/small_clean_epoch2_oos/shards/c2_best_YYYY-MM-DD.csv.gz        # 34 个
artifacts/small_clean_epoch2_oos/shards/epoch2_best_YYYY-MM-DD.csv.gz   # 34 个
artifacts/small_clean_epoch2_oos/summary.json
artifacts/small_clean_epoch2_oos/plan.json
artifacts/small_clean_epoch2_oos/run.log
```

**2）核对完整性。** 68 个 shard；`summary.json` 应为 `status=complete`，且 `plan.json` 里 epoch2 的 SHA 是 `68feb21b…`：

```bash
ls artifacts/small_clean_epoch2_oos/shards | wc -l   # 期望 68
python3 - <<'PY'
import json
base = "artifacts/small_clean_epoch2_oos"
s = json.load(open(f"{base}/summary.json"))
p = json.load(open(f"{base}/plan.json"))
print("status:", s.get("status"))
print("models:", {k: v["sha256"][:12] for k, v in p["models"].items()})
for k in ("c2_best", "epoch2_best"):
    print(k, s[k]["path_direction"], s[k]["path_return_mae"])
print("paired:", s.get("paired_delta_epoch2_minus_c2"))
PY
```

**3）生成三层报告。** 脚本默认读 `artifacts/small_clean_epoch2_oos/shards`，并自动复用旧 C03 shard：

```bash
python3 finetune/small_clean_oos/report_epoch2.py
```

输出 `finetune/reports/small_clean_epoch2_vs_c2_full_oos_20261021.json`。快速看关键量：

```bash
python3 - <<'PY'
import json
d = json.load(open("finetune/reports/small_clean_epoch2_vs_c2_full_oos_20261021.json"))
for label in ("c2_best", "epoch2_best", "clean_best"):
    s = d["scores"].get(label)
    if not s:
        continue
    print(label,
          "d10_dir=", round(s["official"]["horizons"]["10"]["direction"], 5),
          "rank_ic=", round(s["L3_ranking"]["d10_rank_ic_daily_mean"], 5),
          "amp_ic=", round(s["L2_amplitude"]["amplitude_rank_ic_daily_mean"], 5),
          "net_pnl=", round(s["L1_money"]["net_pnl"], 5),
          "top_decile_pnl=", round(s["L1_money"]["top_amplitude_decile"]["net_pnl"], 5))
print("epoch2_minus_c2:", d["paired_deltas"]["epoch2_best_minus_c2_best"])
PY
```

**4）按 §6.3 判定。** 重点看：

- `D10 方向` 是否低于 C2 0.3pp 以上
- `幅度 Rank IC` 是否低于 C2
- `d10_rank_ic_daily_mean` 是否至少不退化
- `net_pnl` / `top_amplitude_decile.net_pnl` 是否改善
- `halt_gap` 子集是否仍为灾难

若三项里有两项明显退化，直接关闭 epoch2 线；若改善，也先做新日期窗口的 forward test，不要直接上线。

**5）归档并提交。**

```bash
git add docs/HANDOVER_20261021_CN.md README.md \
        finetune/reports/small_clean_epoch2_vs_c2_full_oos_20261021.json
git commit -m "docs: archive epoch2 vs C2 OOS results"
git push origin master
```

建议同时更新：本文 §4.2/§4.3 增加 epoch2 行、§6.1 状态改为 COMPLETE、README 的“当前状态”表加一行。

**如果 `summary.json` 是 `partial` 或 kernel 报错：**

- `partial` 通常是 11h15m 预算在 34 天里没跑完。已完成的 `{label}_{date}.csv.gz` 可以复用：新开一个 resume kernel，把上一次 output 挂到 `/kaggle/input`，启动前复制到 `OUTPUT/shards/`，`worker` 会跳过已存在的日期。
- `ERROR` 先看 `run.log` 和 `kaggle kernels logs`；常见原因是输入 SHA、checkpoint 段号、tokenizer 唯一性。不要在没有日志证据时重复提交。

#### 6.5.1 本次执行流水（2026-10-21 22:00–22:20 +08）

上述 5 步已全部跑完，实际结果：

| 步骤 | 结果 |
|---|---|
| 0）状态 | `KernelWorkerStatus.COMPLETE`；日志 1,017 条，无 Traceback |
| 1）下载 | 68 shard + `summary.json` + `plan.json` + `run.log`；另附 `swanlog/` 与 source bundle |
| 2）完整性 | `status=complete`；c2/epoch2 各 34 日 × 175,428 行；epoch2 SHA `68feb21bb0d2…` ✅ |
| 3）报告 | 生成 `finetune/reports/small_clean_epoch2_vs_c2_full_oos_20261021.json`（三方 + 配对 CI） |
| 4）判定 | **关闭 epoch2 线**（见 §4.2b / §6.3） |
| 5）归档 | 本文 + README + 报告 JSON 已提交 |

本次踩到并已在上文 §5.1 记录的两个环境坑：**系统代理导致 Kaggle CLI SSL 失败**（用 `NO_PROXY` 绕过）、**缺 scipy 导致 spearman 报错**（换带 scipy 的解释器）。

C03 归档分片不在仓库里，需按 §5.2 单独下载 `luckfu/kronos-small-clean-vs-c2-full-oos`，`report_epoch2.py` 会自动复用做三方对比。


---

## 7. 已知风险与口径陷阱

1. **34 个 signal dates + 10 日重叠标签**：有效独立观测很少；block bootstrap CI 只能当参考，不是 sealed 盲测。不要把“CI 不含 0”当成生产级显著。
2. **成本口径不统一**：panel 用 0.4%（round trip），ex-ante 尾部报告用 0.3%，TopK 回测另有一套。跨报告引用时必须注明。
3. **指标口径不统一**：`path_direction`（10 horizon 平均）≠ `d10_direction`；`daily IC` ≠ `pooled IC`；19 日 ≠ 34 日；裁剪还原 ≠ 原始价格。不能跨报告直接比大小。
4. **账号血缘断裂**：e02、Clean C05 在 `user281434`；e04/e05 在 `luckfu`。本机 Kaggle CLI 在 10-21 15:37 切到 `luckfu` OAuth，所以 `user281434` 的 kernel 查询被拒。需要旧账号凭据或在 Kaggle 网页切换账号。
5. **凭据泄露风险（重要）**：`finetune/kaggle_small_0_1_stage2_wc_dual_t4_kernel/kaggle_small_0_1_stage2_wc_dual_t4.py` 中仍有字面量 `SWANLAB_API_KEY`，`push_chunk.py` 和 SwanLab 探针会读取它。该文件当前仍在 Git 跟踪中；如果仓库是公开的，应视为已泄露。建议尽快轮换 key，并改为 Kaggle Secrets / 环境变量；不要把该值写进文档或新提交。
6. **`artifacts/` 与 `build/` 被 .gitignore 忽略**：新克隆没有 OOS 分片、e05 权重、chunk build manifest。e04 的 build manifest 是后来根据 published manifest 重建的；e05 的 `build/` 只在原机器上。需要时从 Kaggle output 重新下载，或从 `experiment_manifest.json` 重建。
7. **e05 的 source bundle 不含 owner 修复**：`source_sha256=375ddf…` 与 e04 相同，`source_commit=620e8d6`。权重没问题；但下一次 chunk 若沿用旧 bundle，日志里的 owner 仍可能是旧值。要更新就重新打包并显式允许 source change。
8. **技术总报告未回填**：~~README 仍写“当前状态 2026-09”~~ → **已更新为 2026-10，并加入“续训支线（已关闭）”一行与 epoch2 结论**；但 [`finetune/SMALL_0_1_MODEL_AND_TRAINING_TECHNICAL_REPORT_CN.md`](../finetune/SMALL_0_1_MODEL_AND_TRAINING_TECHNICAL_REPORT_CN.md) 仍停在 C03/C04，没有 epoch2 和三层评测正文。需要有人做一次系统性回填。
9. **选型指标可能与 OOS 脱钩**：epoch2 用 weighted forecast loss 选 best（seg 501，-0.27% 改善），OOS 排序 IC 反而退 0.058。三轮续训（C03/C04/epoch2）都是这个模式。下次训练前应先确认“验证 loss 改善 → OOS 排序改善”这条链路是否还成立，否则选型信号本身是噪声。

---

## 8. 待决问题

1. epoch2 是否值得继续？**✅ 已决：不继续，线已关闭。** OOS 完成，D10 方向 -0.77pp 越阈值、幅度 Rank IC -0.0368（CI 不含 0），按 §6.3 事先规则关闭（§4.2b）。**不要再提交同合同的 epoch3。**
2. 下一阶段是“继续训模型”还是“做条件化交易”？**建议后者（现在更明确）**：三轮 clean 续训（C03/C04/epoch2）在 OOS 上都没有超过 C2，方向稳定在 53.8%–54.0% 区间，已到该合同的噪声地板；继续同配方训练的期望收益接近零。真正有空间的是 §6.4 的成本门槛 + 事前过滤 + 组合级回测。
3. Clean C05（721→920）是否已完成？需要旧账号或网页确认；不要在未确认前重复提交。
4. 是否要把 C2 换成 clean/epoch2？**已决：不换。** 两个候选在方向和幅度排序上均劣于 C2（§4.2、§4.2b）。
5. 是否轮换 SwanLab key 并移除仓库中的字面量？**建议尽快做**，这是独立于模型进展的安全债。
6. **新问题**：既然三轮续训都退化，是否该回头质疑“验证集 loss 作为选型指标”的有效性？本次 epoch2 的 best 段（501）loss 改善 -0.27%，OOS 排序 IC 却退 0.058。建议下一轮若要训，**直接用 OOS 排序 IC 类指标做选型**，而不是 weighted forecast loss。


---

## 9. 附录：路径、哈希、Kernel

### 本地路径

| 内容 | 路径 |
|---|---|
| C2/C03 OOS 分片 | `artifacts/small_clean_oos_shards/small_clean_oos/shards/`（68 个，约 71MB） |
| epoch2 OOS 输出（待下载） | `artifacts/small_clean_epoch2_oos/`（shards + summary/plan/run.log，见 §6.5） |
| e05 权重 | `artifacts/kernels/e05/clean_small_runtime/outputs/models/small_0.1_clean_epoch2_from_c03_20261020/` |
| C2/C03 基线报告 | `finetune/reports/small_clean_direction_amplitude_baseline_20261021.json` |
| epoch2 报告（待生成） | `finetune/reports/small_clean_epoch2_vs_c2_full_oos_20261021.json` |
| 三层评测 | `finetune/small_clean_oos/panel.py` |
| epoch2 OOS 评测 | `finetune/small_clean_oos/evaluate_epoch2.py` |
| epoch2 报告生成 | `finetune/small_clean_oos/report_epoch2.py` |
| 接力入口 | `finetune/small_clean_epoch2_relay/launch.py` |
| 打包/提交 | `finetune/small_clean_epoch2_relay/push_chunk.py` |

### 哈希

| 对象 | SHA-256 |
|---|---|
| C2 best | `4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a` |
| C03 best | `0837035e4cde286240b196cb2c7fda1dc671cacc2b501b00ab7f0ec38d887c9f` |
| C04 best | `81d735211a6cda10da6562ecb83ee6fb98ce1e7a0daa2b834242179905b59b5a` |
| Epoch2 best | `68feb21bb0d2d6b385e69e95765536b25f2cad07a8efb011cd1bcdea878fda5d` |
| Epoch2 last | `2f6bd405841a6b8a62eb1aa8e1327a5e7d15f05706eda088e460cf7c660e2368` |
| Tokenizer | `59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee` |
| 数据 manifest | `b85ccc45f7cd0e895e607e6e75c7a603427bd62a353a016742a52b2943d652cd` |
| OOS panel | `be159cd89474a22dc000403137c7d7647cd2719eb325631a9f5ba701defa8118` |
| OOS identities | `057f6912f783483ad43914c90d83ea4a54d299ef7400831c0a0dbf814fd665d1` |

### Kaggle Kernel

| Kernel | 作用 | 状态 |
|---|---|---|
| `luckfu/kronos-small-clean-epoch2-vs-c2-full-oos` | C2 + epoch2 配对 OOS，产出 shards/summary | **COMPLETE（2026-10-21 14:11 UTC），output 已下载校验** |
| `luckfu/kronos-small-clean-vs-c2-full-oos` | C2 vs C03 的 34 日 OOS，产出 shards | COMPLETE |
| `luckfu/kronos-small-clean-epoch2-from-c03-e05` | epoch2 最后一个 chunk（499–520） | COMPLETE |
| `luckfu/kronos-small-clean-epoch2-from-c03-e04` | epoch2 chunk 292–498 | COMPLETE |
| `user281434/kronos-small-clean-epoch2-from-c03-e02` | epoch2 chunk 1–291 | COMPLETE，当前账号不可查 |
| `user281434/kronos-small-clean-wc-dual-t4-c05` | Clean C05 第三轮 721→920 | 启动已确认，终态未知 |

### 一句话交接

> **epoch2 的 34 日配对 OOS 已跑完并归档：D10 方向 53.82% vs C2 54.59%（-0.77pp，越 0.3pp 关闭门槛），幅度 Rank IC 0.2103 vs 0.2470（-0.0368，CI 不含 0），D10 日均 Rank IC 0.0843 vs 0.1424（-0.058，CI 不含 0）。按事先写死的 §6.3 规则，epoch2 线已关闭，C2 仍是唯一线上基线，不再追加同合同训练。下一阶段的重点是“高预测幅度（≥3%/5%）+ 事前停牌与流动性过滤 + 组合级回测”的条件化交易验证（§6.4），而不是继续堆训练。**
