# 冻结协议：训练中途按时间切分 Rank IC 选 checkpoint

状态：**冻结（2026-10-23）**。群聊约定；改任何一条须显式修订本文并另开 commit，禁止口头变口径。

本文只冻结**中途选模**规则。不推 Kaggle、不替换 C2。尺子以本文为准；代码是否已接线见 §9。

相关：

- 诊断证据（非本协议）：[`reports/SMALL_VALIDATION_RANK_IC_20261023_60DATES_CN.md`](reports/SMALL_VALIDATION_RANK_IC_20261023_60DATES_CN.md)
- 晋级门（Musk / §5.2）：[`C2_EXECUTION_CONSTRAINED_BACKTEST_8B_CN.md`](C2_EXECUTION_CONSTRAINED_BACKTEST_8B_CN.md)
- 当前续训：[`BIGBATCH2_FROM_E03B_20261023_CN.md`](BIGBATCH2_FROM_E03B_20261023_CN.md)

---

## 1. 两层尺子（不得混用）

| 层 | 用途 | 尺子 | 谁定 |
|---|---|---|---|
| **A. 中途选 checkpoint** | 同一训练 run 内谁当 `best_model` | **时间切分**日频 D10 Rank IC（本文） | 训练侧 |
| **B. 晋级 / 替换 C2 基线** | 能不能换线上基线 | §5.2 执行约束净收益/回撤 | Musk；见 8b 文档 |

A 的赢家只进**短名单**。未过 B，**不得**宣称替换 C2，也不得把 A 的 IC 写成「可上线」。

---

## 2. 主尺子（层 A）定义

### 2.1 指标

- 预测目标：**D10** 单点收益（与生产 / sealed / 34 日报告同口径）。
- 截面：每个 **signal date** 内，预测 D10 vs 实际 D10 的 **Spearman Rank IC**。
- 汇总：验证窗内各交易日 IC 的**算术平均**（日均 Rank IC）。
- 路径：**16 条**采样路径；窗口内对 16 条 D10 取 **中位数** 再参与当日截面排序（生产口径；`SAMPLE_COUNT=16`）。
- 生成超参冻结对齐现有评测：`T=0.65`，`top_p=0.8`，路径 seed 与现网评测脚本一致（当前诊断用 `20260906`）；改 seed 须修订本文。

### 2.2 必须是时间切分，禁止 symbol holdout 选型

- **选型窗** = 按**交易日时间**切出的连续（或预先写死的）signal date 集合。
- 训练样本的时间上界必须 **严格早于** 选型窗最早 signal date（含其 D10 标签所需行情都不进入训练覆盖）。本版冻结字段取 clean split 的 `train_cutoff`，见 §7。训练股票的全历史仍覆盖这个窗口，边界说明见 §9。
- **禁止**用 `a-share-120d-temporal-symbol-holdout` 的 **symbol holdout 验证集**（520 只 holdout 股、2025-07-03→2026-07-02 那套）作为层 A 的正式选模协议。该面板上的 20/60 日 Rank IC 只作 **pilot 可行性旁证**，见 §6。
- 选型窗日历必须在接入训练前落盘为只读 manifest（日期列表 + SHA）；开跑后不得增删日期「好看一点」。

### 2.3 验收窗（永远不参与选型）

下列窗口 **只验收、不选模、不调参、不扫阈值**：

| 窗口 | 角色 |
|---|---|
| sealed 8 日（2026-09-18→09-30） | 已消费；只读历史 |
| §5.2 目标窗（2026-10-09→10-23，D10 约至 2026-11-06） | 执行约束验收 |
| 34 日包（信号至 2026-09-17） | 历史对照，不进层 A |

任何把上述窗口 IC/净值写进 `best_metric.json` 或用来改 early-stop 的行为，视为违约。

---

## 3. 评测节奏（稀疏；冻结默认）

### 3.1 依据

Clean bigbatch / e03b 线：`samples_per_segment = 20000`，一轮覆盖 `TOTAL_SEGMENTS = 520`。  
`finetune/train_predictor.py` 里昂贵校验的默认间隔是：

```text
validation_large_interval_segments = 10
```

（`should_run_large_validation`：第 1 段、每 `interval` 段、以及最后一段。）

日频 Rank IC（现为单路径）仍远重于 forecast val loss（60 日×全截面单 ckpt 仍可能数十分钟量级），**不得**每 segment 都跑。

### 3.2 冻结默认

| 项 | 值 |
|---|---|
| 评测单位 | **segment**（不是 micro-batch / 不是 wall-clock 分钟） |
| 默认间隔 **N** | **每 20 个 segment**（2026-10-26 起；原 N=10） |
| 强制点 | 本 run/本 epoch 的 **segment 1**、每个满足 `segment % 20 == 0` 的点、以及 **最后一个完成的 segment**（含墙钟提前停） |
| 与现网对齐 | 原 N=10 ≡ `validation_large_interval_segments`；现因墙钟启用协议预留的 N=20 逃生门 |

可选加速（**仅 pilot 日志**，不改变「谁是 best」的定义）：同一套日期协议下可先用更少日期做冒烟；**写入 `best_model` 的正式分数**必须用冻结满规格窗 + 当前 `n_paths`（现为 1）与 `path_agg=median`。

**N=20 修订（2026-10-26）**：双 T4 实测 N=10 时单次 Temporal Rank IC（60 日 panel、`N_PATHS=1`、日期分片）约 **87 分钟**，每 10 段训练仅约 25 分钟，校验占墙钟约 **78%**，满足「双 T4 墙钟证明 N=10 不可承受」。按本文原逃生门改成 **N=20**；禁止改成「看 loss 曲线再手挑」。

### 3.3 val loss 的地位

- **每个** segment（或现有 trainer 已有的 quick/full 节奏）继续算并打 SwanLab：**保留**。
- val loss / `selection_loss`（forecast / weighted forecast / CE 等）**不得**再决定 `checkpoints/best_model`。
- `best_metric.json` 正式字段应以 Rank IC 为准（实现阶段再接线；本文件不改代码）。过渡期若磁盘上仍有旧 `selection_loss`，文档与看板须标明「非选模」。

---

## 4. 如何选定 best

1. 仅在 §3.2 触发的 sparse 评测点上计算日均 D10 Rank IC。
2. **更高更好**。严格大于当前 best 则原子替换 `checkpoints/best_model`，并写入至少：
   - `daily_rank_ic_mean`（选型分）
   - `segment` / `step`
   - `n_dates` / `n_paths` / `path_agg=median`
   - `selection_protocol=temporal_rank_ic_v1`
   - 选型窗 manifest SHA
3. IC **并列**（建议绝对值差 &lt; 1e-6）：保留**更早**的 segment（偏保守，少训一点）。
4. 未触发 Rank IC 评测的 segment：**不更新** best；只更新 `last_model` / `last_state`。
5. 短名单：每个训练 run 最多保留 **Rank IC best** 一个进 §5.2；不得用 val loss 第二名「也顺便晋级」。

---

## 5. 明确禁止

- 用 **symbol holdout** 验证 Rank IC 正式选 `best_model`。
- 用 **val loss**（含 weighted forecast / CE / path alignment）选 best。
- 用 sealed 8 日、§5.2 窗、或任何未冻结的未来窗做选型或调 N / 日期 / 路径数。
- 每 step / 每 batch 跑 Rank IC。
- 扫 TopK、扫 `|pred|` 阈值、扫温度 / top_p 来抬 IC。
- 把层 A 的 IC 提升写成「可替换 C2」或「开大规模新训」的许可。
- 在本协议未接线前，把正在跑的 bigbatch2 结果按 Rank IC **事后重选**冒充中途选模（只可做离线对照，须标注 `post_hoc`）。

---

## 6. 与 60 日 holdout 诊断的关系

[`SMALL_VALIDATION_RANK_IC_20261023_60DATES_CN.md`](reports/SMALL_VALIDATION_RANK_IC_20261023_60DATES_CN.md)：

- C2 &gt; C03 &gt; epoch2 ≈ e03b；C2−C03 / C2−e03b 配对 CI 不跨 0。
- 这是 **symbol holdout** 证据，支持「Rank IC 值得做选模 pilot」，**不是**本文的选型协议本身。
- Pilot 设计可引用该结论；正式层 A 必须换到 §2.2 时间切分窗。

---

## 7. 接入状态（日历已冻结，2026-10-23）

选型日历已落盘：`finetune/manifests/temporal_rank_ic_v1/selection_window.json`。日期与 `sha256` 以该文件为准。下面是同一份列表。

| 项 | 值 |
|---|---|
| n_dates | **60** |
| 区间 | `2026-04-01` → `2026-06-30`（含首尾） |
| 日历 | 上证公告〔2025〕45号（2025-12-22）。周末休市；这个窗口里没有周末开市。相对 pandas 工作日，只再去掉 `2026-04-06`、`2026-05-01`、`2026-05-04`、`2026-05-05`、`2026-06-19` |
| train_time_upper_bound | `2024-12-31`（clean `split.train_cutoff`） |
| sha256 | `0721b3defca10d2dab4dcaf433f1efab12304340298e6304fca6d3c9a9f3ce0d` |
| panel_sha256 | **空**。只许 `finetune/build_temporal_rank_ic_panel.py` 在写出面板后填入，禁止手填 |

```text
2026-04-01
2026-04-02
2026-04-03
2026-04-07
2026-04-08
2026-04-09
2026-04-10
2026-04-13
2026-04-14
2026-04-15
2026-04-16
2026-04-17
2026-04-20
2026-04-21
2026-04-22
2026-04-23
2026-04-24
2026-04-27
2026-04-28
2026-04-29
2026-04-30
2026-05-06
2026-05-07
2026-05-08
2026-05-11
2026-05-12
2026-05-13
2026-05-14
2026-05-15
2026-05-18
2026-05-19
2026-05-20
2026-05-21
2026-05-22
2026-05-25
2026-05-26
2026-05-27
2026-05-28
2026-05-29
2026-06-01
2026-06-02
2026-06-03
2026-06-04
2026-06-05
2026-06-08
2026-06-09
2026-06-10
2026-06-11
2026-06-12
2026-06-15
2026-06-16
2026-06-17
2026-06-18
2026-06-22
2026-06-23
2026-06-24
2026-06-25
2026-06-26
2026-06-29
2026-06-30
```

这些日期都不在 34 日包（`2026-08-03`→`2026-09-17`）、sealed 8 日（`2026-09-18`→`2026-09-30`）或 §5.2（`2026-10-09`→`2026-10-23`）里。信号日停在 6 月 30 日，不为了凑满 60 天伸进 7 月。`2026-06-30` 的 D10 标签是其后第 10 个上交所交易日 `2026-07-14`，仍早于 34 日包。

面板还没进仓库，现在也不能推 Kaggle：

1. `python finetune/build_temporal_rank_ic_panel.py --dataset-root <clean dataset> --output <dir>/temporal_rank_ic_panel.pkl`
2. 把这个 pkl 上传成**单独**的 Kaggle dataset。不要并进 `smmt315/a-share-120d-temporal-symbol-holdout`，文件名不要用 `val_data.pkl`。
3. 确认脚本已经把 manifest 里的 `panel_sha256` 写成该文件的 SHA，并提交这一行。
4. `export KRONOS_TEMPORAL_RANK_IC_PANEL_DATASET=<owner>/<panel-dataset>` 之后才可以 `python -u finetune/small_clean_bigbatch_relay/push_chunk.py 1`。

e01（按 val loss 选模）已取消。`panel_sha256` 还是空的时候，`push_chunk.py` 失败关闭，不要热重推。短名单要换 C2，仍然只走 §5.2。

---

## 8. 版本

| 字段 | 值 |
|---|---|
| protocol_id | `temporal_rank_ic_v1` |
| frozen_date | 2026-10-23（Asia/Shanghai） |
| sparse_N_segments | 20 |
| n_paths | 1 |
| path_aggregation | median |
| horizon | D10 |
| promotion_gate | section_5_2_execution_constrained |

---

## 9. 实现状态（2026-10-23）

选模规则已接到训练代码。默认**关**，旧 run 仍按 val loss 选 `best_model`。bigbatch2 启动脚本把协议设为 `temporal_rank_ic_v1`。

| 项 | 位置 |
|---|---|
| 协议判断、manifest SHA、日均 Rank IC、单路径 | `finetune/temporal_rank_ic_selection.py` |
| 稀疏触发（第 1 段、每 20 段、计划末段，外加本 run 首段与墙钟/chunk 末段）并写 `best_model` | `finetune/train_predictor.py` |
| 开关 `KRONOS_CHECKPOINT_SELECTION_PROTOCOL` | `finetune/config.py` |
| bigbatch2 默认打开 | `finetune/small_clean_bigbatch_relay/launch.py` |
| 选型窗日历与面板构建 | `finetune/manifests/temporal_rank_ic_v1/selection_window.json`、`finetune/build_temporal_rank_ic_panel.py`、同目录 README |
| 单测 | `tests/test_temporal_rank_ic_selection.py` |

选型日历已经提交：60 个信号日，`sha256` 见 §7。`panel_sha256` 仍是空字符串。`selection_window.json` 缺失、日历 SHA 不符、日期落进禁止窗口、`panel_sha256` 还不是 64 位十六进制、或面板落到 symbol-holdout `val_data.pkl` 时失败关闭，不会改用 val loss，也不会改用 symbol holdout。`push_chunk.py` 在日历和面板 SHA 都钉住之前拒绝打包；面板字节到启动时再对一次。val loss 仍每个 segment 记录。路径 seed 只作用于当次评分前向，评完恢复训练 RNG。

`train_time_upper_bound` 用的是 clean `data_manifest.split.train_cutoff` = `2024-12-31`。仓库里没有这份 manifest 原文。`launch.py` 把 `DATA_SHA` 钉在 `b85ccc45f7cd0e895e607e6e75c7a603427bd62a353a016742a52b2943d652cd`；`build_temporal_symbol_validation_dataset.py` 的默认 cutoff 是这一天，`remove_suspended_rows.py` 原样复制 `split`。预清洗 manifest 的训练股票最后一根 bar 在 beta v21 报告里是 `2026-07-17`，最后完整窗口 asof 是 `2026-07-02`（`build_time_disjoint_val_panel.py`）。所以这 60 天晚于 split cutoff，但仍落在训练股票 `train_data.pkl` 已经覆盖的日历里。本冻结不把信号日挪进 7 月或任何验收窗。构建脚本若发现挂载的 manifest `train_cutoff` 不是 `2024-12-31`，会直接停，不会沿用这份日期。

`push_chunk.py` 还要求环境变量 `KRONOS_TEMPORAL_RANK_IC_PANEL_DATASET=owner/slug`。这个 slug 不能是训练数据集 `smmt315/a-share-120d-temporal-symbol-holdout`。生成的 `kernel-metadata.json` 把训练数据集和面板数据集一起放进 `dataset_sources`。`launch.py` 在 `/kaggle/input` 里只接受一份名为 `temporal_rank_ic_panel.pkl`、且路径不含 symbol-holdout 的文件。

`best_metric.json` 在该模式下写入 `daily_rank_ic_mean`、`segment`、`step`、`n_dates`、`n_paths`、`n_paths=1`、`path_agg=median`、`selection_protocol=temporal_rank_ic_v1`、manifest SHA，并标记 `val_loss_selects_best=false`。IC 并列（绝对差 &lt; 1e-6）保留更早的 checkpoint。

e01（val loss 选模）已取消。面板 SHA 钉上之前不要热重推 bigbatch2。晋级替换 C2 仍只走 §5.2，本实现不改那道门。

**实现调参备注（2026-10-26，不改冻结字段）**

- 稀疏间隔已切到 **N=20**（见 §3.2 / §8）；`launch.py` 与 `DEFAULT_INTERVAL_SEGMENTS` 同步为 20。
- `INFERENCE_BATCH`（`finetune/temporal_rank_ic_selection.py`）从 **4 → 24**：不在协议冻结字段（冻结的是 n_paths / path_seed / T / top_p / top_k / lookback / predict / 日历 / panel SHA）。batch 改变每日期内 RNG 消耗顺序，**打破跨 checkpoint 的噪声配对**。切换后的**首次** Temporal Rank IC 分数作为新基准；晋级替换 C2 仍只认 §5.2 独立验证。
