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
- 训练样本的时间上界必须 **严格早于** 选型窗最早 signal date（含其 D10 标签所需行情都不进入训练覆盖）。
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

日频 16 路径 Rank IC 远重于 forecast val loss（60 日×16 路径单 ckpt 约十余分钟量级），**不得**每 segment 都跑。

### 3.2 冻结默认

| 项 | 值 |
|---|---|
| 评测单位 | **segment**（不是 micro-batch / 不是 wall-clock 分钟） |
| 默认间隔 **N** | **每 10 个 segment** |
| 强制点 | 本 run/本 epoch 的 **segment 1**、每个满足 `segment % 10 == 0` 的点、以及 **最后一个完成的 segment**（含墙钟提前停） |
| 与现网对齐 | N=10 ≡ 现有 `validation_large_interval_segments` 默认 |

可选加速（**仅 pilot 日志**，不改变「谁是 best」的定义）：同一套日期协议下可先用更少日期做冒烟；**写入 `best_model` 的正式分数**必须用冻结满规格窗 + 16 路径中位数。

若双 T4 墙钟证明 N=10 不可承受：只允许改成 **N=20**，且必须修订本文并留 commit；禁止改成「看 loss 曲线再手挑」。

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

## 7. 接入顺序（文档门禁；本 commit 不动代码）

1. 冻结并落盘时间切分选型窗 manifest。
2. 离线对 C2 / 一两个续训 ckpt 在该窗上复算 16 路径 IC（sanity）。
3. 再改 trainer：sparse 评测 + `best_model` 改挂 Rank IC。
4. 小规模 pilot run（短墙钟）；**并行**的 bigbatch2 仍按旧 val loss 跑完，不中途改规则。
5. Pilot 短名单 → 仅当 §5.2 标签齐且 `publishable_as_section_5_2=true` 时谈是否动 C2。

---

## 8. 版本

| 字段 | 值 |
|---|---|
| protocol_id | `temporal_rank_ic_v1` |
| frozen_date | 2026-10-23（Asia/Shanghai） |
| sparse_N_segments | 10 |
| n_paths | 16 |
| path_aggregation | median |
| horizon | D10 |
| promotion_gate | section_5_2_execution_constrained |

---

## 9. 实现状态（2026-10-23）

选模规则已接到训练代码。默认**关**，旧 run 仍按 val loss 选 `best_model`。bigbatch2 启动脚本把协议设为 `temporal_rank_ic_v1`。

| 项 | 位置 |
|---|---|
| 协议判断、manifest SHA、日均 Rank IC、16 路径中位数 | `finetune/temporal_rank_ic_selection.py` |
| 稀疏触发（第 1 段、每 10 段、计划末段，外加本 run 首段与墙钟/chunk 末段）并写 `best_model` | `finetune/train_predictor.py` |
| 开关 `KRONOS_CHECKPOINT_SELECTION_PROTOCOL` | `finetune/config.py` |
| bigbatch2 默认打开 | `finetune/small_clean_bigbatch_relay/launch.py` |
| 选型窗占位说明 | `finetune/manifests/temporal_rank_ic_v1/README.md` |
| 单测 | `tests/test_temporal_rank_ic_selection.py` |

选型日历 **尚未**随代码提交。`selection_window.json` 缺失、日历 SHA 不符、日期落进禁止窗口、或面板落到 symbol-holdout `val_data.pkl` 时失败关闭，不会改用 val loss，也不会改用 symbol holdout。推送前会检查日历本身；面板 SHA 在启动时核对。val loss 仍每个 segment 记录。路径 seed 只作用于当次评分前向，评完恢复训练 RNG。

`best_metric.json` 在该模式下写入 `daily_rank_ic_mean`、`segment`、`step`、`n_dates`、`n_paths`、`path_agg=median`、`selection_protocol=temporal_rank_ic_v1`、manifest SHA，并标记 `val_loss_selects_best=false`。IC 并列（绝对差 &lt; 1e-6）保留更早的 checkpoint。

e01（val loss 选模）已取消。下一轮 bigbatch2 按本实现启动；§7.4「本轮仍按旧 val loss」只描述取消前的那次运行。晋级替换 C2 仍只走 §5.2，本实现不改那道门。
