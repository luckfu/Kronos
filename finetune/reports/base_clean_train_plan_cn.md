# Base 干净重训计划（镜像 Small→C2）— APPROVED-FOR-M1

**状态：APPROVED-FOR-M1（§7.1 开训 checklist 已按群默认勾完；允许推 Bootstrap）**  
日期：2026-10-26（Asia/Singapore）  
本轮修订：纳入「量化交易模型大师」群评审；补 §5.2 晋级门槛 **PROPOSAL**；双 T4 smoke→scale / accum / 密封定性停训 / checklist。  
**2026-10-26 再修（Elon 群话）：** 写入 Base→Small 历史与 LR 经验；Main 峰值 **锚定 Small 1e-5**；**禁止**因尺寸≈4×自动 ÷4 到 1e-6；M1 短扫否决后才降。  
**2026-10-26 再修（Elon + 量化交易模型大师）：** **§5.2 数值门槛不再是开训 blocker**——仅 **promote 前勾选**；开训只卡 **M0/M1 可运行**（见 §7）。M1 短扫定为 **5e-6 vs 1e-5**。  
**2026-10-26 SwanLab：** 群确认对齐 Small——**阶段换新看板**（Bootstrap ≠ Main）；Base 用 `base_clean_0.1_*`，**不续** `smce2_*` / 旧 Beta 板。  
**2026-10-27 Bootstrap chunk2 推送：** resume_segment=22（chunk1 COMPLETE segs2→21）；同板 `base_clean_0.1_bootstrap` / self-parent slug；硬要求：QUICK=2048（无 fixed-manifest 真子集）、val+Full/None/Shuffled 每5 seg（含 chunk-end）、`KRONOS_DISABLE_BEST_UPDATES=1`（只续 last）；VAL_BATCH=64；跳过 micro 32×1×2（Main 再议）。  
**2026-10-28 Bootstrap chunk3 接力（luckfu）：** chunk2 结束于 coverage segment 317（log `resume from coverage segment 318`）；chunk3 改由 **luckfu** 推送，`kernel_sources` 挂 user281434 的共享 notebook（跨账号）；`resume_segment=318` / `completed=317` / `max_segments=36`；看板仍 `base_clean_0.1_bootstrap`（`resume=allow`，`replay_after=317`）；relay 与 runner 用 2026-10-27 runtime snapshot（Fix A/C 默认关），与 chunk1/2 同代码。**已确认 08:35 起从 segment 318 训练（SwanLab）；Kaggle logs 接口有延迟，进度以 SwanLab 的 `segment` 为准。**

**2026-10-28 chunk3 收尾 + chunk4 接力 + 日志整改：** chunk3（luckfu，父=user281434 chunk2）已跑完 **seg 318→353（36 段，2h23m，4m05s/seg）**，`stop_reason=segment_limit`，`resume_segment=354`，无 OOM/无重启；forecast **2.59314**（seg1 2.99579，−13.4%），条件消融 `Δfull-none −1.29e-4 / Δfull-shuf −4.64e-5`（仍在身份地板，符合"数据暴露差"结论）；峰值显存 2.55GB。**Best 冻结在 seg21（2.69015），落后当前 0.0965 → 评估/后续阶段只能用 `last_model`+`last_state`。** chunk4 已推（同板 `base_clean_0.1_bootstrap`，`resume=354` / `completed=353` / `max_segments=36`，self-parent luckfu），**仅观测改动**：runner 每行加 UTC 时间戳、每段打 `[PROGRESS]`（含 seg 秒数/ETA/消融/显存）、条件消融与条件范数上 SwanLab（曲线不再缺 ablation）、`push/monitor` 改走 REST（CLI 2.2.4 打 api.kaggle.com 会 `SSLEOFError`）。**训练途中取日志**：`python3 finetune/base_clean_bootstrap_relay/monitor_chunk.py --follow`（SSE）；`kaggle kernels output` 只在 session 结束后才有内容，别再拿它当实时通道。详见 `finetune/base_clean_bootstrap_relay/notes/20261028_chunk3_log_audit_cn.md`。

**2026-10-28 棒长改墙钟制 + Main 就绪：** 旧的一棒 36 段（2.5h）意味着一遍 520 段要 **14 次接力**，每次还有约 6 分钟启动/拷贝；改成 **墙钟预算制**（`KRONOS_BOOTSTRAP_RUNTIME_BUDGET_SECONDS`，默认 36000s＝10h；段数上限 150 仅作刹车），与 Small 后期 WC 的做法一致（Small 用 40,500s 预算、约 200 段/棒）。实测 Base 4m05s/段 → 10h ≈ 145 段，一遍 coverage 只需 **4 棒**。Main 的入口也已备好（`launch_main.py` / `push_main.py`）：只继承 Bootstrap `last_model`、**重建 optimizer/scheduler**、全参数峰值 1e-5、新板 `base_clean_0.1_main`。**有意偏离 Small 之处**：Small Main 每段跑全量 123,836；Base 实测全量 forward ≈8–10 min/段（1935 个 val batch @64），每段跑等于 12 min/段、520 段 ≈104h，不可行，故改为 **quick 2048 每 5 段（选 Best）+ 全量 123,836 每 20 段及首末段（里程碑）**，并明确记为"与 Small 选模口径不同，数值不可直接比"。切换与回滚步骤见 `finetune/base_clean_bootstrap_relay/notes/20261028_main_stage_plan_cn.md`。**注意：Bootstrap 的 Best 仍冻结在 seg21（2.69015），Main 起点必须用 last_model。**

**2026-10-26 开训绿灯：** Elon 绿推 Bootstrap seg1；§7.1 按群默认全部勾完；**Bootstrap timing seg1 已推送且 RUNNING**：`user281434/kronos-base-clean-0-1-bootstrap-timing`（SwanLab `base_clean_0.1_bootstrap_timing`；accum `4×8×2=64`）。整体计划仍为 **approved-for-M1**（非全线定版通过）。§5.2 仍为 promote-only。  
目标：清掉旧 Base/Beta 血缘文档后，用 **与 Small→C2 同一配方** 在 `NeoQuasar/Kronos-base` 上从零建 A 股血缘；**不以旧 V6/V1.2/V2.1 权重为起点**。

对照配方来源（只读，不继承其权重）：`finetune/KRONOS_SMALL_V21_FROM_SCRATCH_PLAN_CN.md`、`finetune/SMALL_0_1_MODEL_AND_TRAINING_TECHNICAL_REPORT_CN.md`；Small 定版史 = Cosine C2 Best @ Seg179（**仅历史参考，Base 不预锁同号**）。

---

## 0. 边界（评审先锁）

| 项 | 约定 |
|----|------|
| 起点 | **公开 HF `NeoQuasar/Kronos-base`** + 共用 `NeoQuasar/Kronos-Tokenizer-base`；禁止加载任何 Beta/V6/V1.x/V2.1 checkpoint |
| 条件 | 行业 embedding + 市值百分位 MLP；**输出层零初始化（no-op 起步）** |
| 注入 | `context_layer=10`（12 层里后 2 层吸收条件；对应 Small 的 6/8） |
| 定版损失 | **纯 forecast CE**（+ 小权重 history）；**晋级/定版时 aux 关闭** |
| 正式判决 | §5.2 信号窗 **2026-10-09～10-23**（未污染）；旧密封 8/03–9/30 **仅 REFERENCE** |
| 对照基线 | 执行侧对照 **冻结 Small C2**（同口径、同窗），不拿旧 Beta / 密封42 `+2.59%` 当门槛 |
| 硬件 | **锁定双 T4 only**；不以 A800 为主路径；策略 = **昂贵 smoke → scale**，非硬否决 |
| 段数 | **不预锁 Seg179 / 534**；先 timing seg → 按墙钟/配额反推本轮容量 |
| 数据合同 | **与 Small 完全同一合同**（holdout / `asset_metadata` SHA / 20k 窗/seg / lookback120·pred10）；**不**另切 Base panel、**不**重切 holdout |
| Base-only 差 | `context_layer=10`、更大骨干、更低 micro-batch（+accum） |
| 代码保留 | **暂不删 `finetune/beta_v21.py`**（Small/C2 与本线仍硬依赖） |
| 本草案 | **§7.1 已通过** → 允许推 **Bootstrap max_segments=1**；整体仍 **approved-for-M1**；§5.2 数字门槛仍为 promote-only |


---

## 0.05 Elon 历史说明（LR 权威来源）

群聊定调（2026-10-26 Asia/Singapore）：

1. **先训的是 Base**；觉得太慢才改训 **Small**。  
2. Small 上 **多次尝试** 后，`validation/forecast_loss` 下降较优的峰值 LR 是 **1e-5** —— 这是 **经验最优**，不是随手抄的。  
3. 旧 Base/Beta 正式线多用 **1e-6**：**未经多次调试，不是权威好值**；**不得**把「旧 Beta=1e-6」当作本干净重训的默认峰值。  
4. 本计划 Main 峰值因此 **锚定 Small 的 1e-5**；尺寸≈4× **不**自动 ÷4。可选在 M1 做短扫（见 §2 / §0.1），**仅当 M1 梯度/条件范数炸掉才降**。

---

## 0.1 群评审已同意 / 硬点（量化交易模型大师）

**已同意：** HF Base 干净起点；镜像 Small→C2；双 T4；先 timing seg；§5.2 仅 10/09–10/23 可发表；**Main 峰值锚定 Small 1e-5**（旧 Beta 1e-6 不权威；禁自动 ÷4；M1 否决才降）。

**硬点（必须进计划）：**

1. **Main 默认峰值 = Small 的 1e-5**（Elon：旧 Beta 1e-6 未经充分调试，**不权威**）。**禁止**因 Base≈4× 自动 ÷4 到 1e-6，也 **禁止**「偏烫所以先往 1e-6 靠」的默认叙事。M1 timing seg 做短扫（**5e-6 vs 1e-5**；可选再试 3e-6 / 3e-5）看 grads + val；**先验 1e-5，仅 M1 否决（炸梯度 / 条件范数失控）才降**。Bootstrap 仍用双 LR（主干可略低于条件）。  
2. **必须文档化** `micro_bs × accum × n_gpu` 与 **真实 update 数/seg**；同配额下 **不要期望** Small 量级覆盖（周配额可能只够 Bootstrap + 部分 Main）。  
3. **§5.2 数值晋级门槛**（见 §5.2）：**不是开训 blocker**。正式窗标签未齐前不定死；**仅 promote 前**由 Elon 勾选。开训只卡 §7「开训 blocker」。

**另：** promote 时 aux off；**不要删 `beta_v21.py`**；**§7.1 已勾完** → 允许推 Bootstrap seg1；§5.2 数字仍可后补（promote-only）。

---

## 1. 模型合同

```text
predictor: NeoQuasar/Kronos-base
tokenizer: NeoQuasar/Kronos-Tokenizer-base
n_layers: 12
d_model: 832
n_heads: 16
ff_dim: （随 HF config）
params: ~102.4M（约 Small 的 4×）
lookback / predict: 120 / 10
num_sectors: 86（含 unknown）
num_size_buckets: 0
use_size_percentile: True
size 输入: [percentile, is_known] → MLP → d_model
context_layer: 10
条件输出层: 全零 init
DualHead / path CE: 与 Small 共用契约
use_beta_v21_auxiliary: 训练中途可试验，但 **promote 定版必须 aux off**
```

数据合同与 Small **完全相同**：symbol-holdout 固定 val、`asset_metadata` 信号日行业 + 同日截面市值百分位、SHA 写入 manifest；每 seg ~20k 无放回窗；lookback 120 / predict 10。**不**建 Base 专用 panel、**不**重切 holdout。不采用 Beta v3 动态市值路径。

---

## 2. 训练阶段（严格镜像 Small→C2 顺序）

命名建议：`base_clean_0.1_*`（避免再出现 `beta_v*`）。

**SwanLab（对齐 Small，群已确认）：**
- project / workspace 仍 `finance` / `roc_fu`（账户已确认 **user281434**）。
- **每个训练阶段开新 run / 新看板**，同阶段多 chunk 用同一 `SWANLAB_RUN_ID` + `resume=allow` 接力。
- Small 证据：Bootstrap = `small_0.1_bootstrap_timing`（计时）→ 长跑 `small_0.1_bootstrap`（run id `n3sjweb7`）；Main = **另板** `small_0.1_main` / `small_0_1_main`；Cosine/WC 再各开新板（如 `small_0.1_stage2_cosine_refinement`、`small_0.1_stage2_wc_dual_t4`）。Clean 线另例：`smce2_e01e_rankic1p` 注明「新看板，不再续 smce2_big」。
- Base 推荐（勿续 smce2_* / beta_v*）：
  - 计时 / M1：`base_clean_0.1_bootstrap_timing`
  - Bootstrap 长跑：`base_clean_0.1_bootstrap`
  - Main：`base_clean_0.1_main`
  - 后续 WC / Cosine：`base_clean_0.1_wc_*` / `base_clean_0.1_cosine_*`（阶段到了再定）

### 阶段 0 — 零训基线 + 数据验收

1. 加载 HF Base + tokenizer；条件零 init。  
2. 固定 holdout 上跑 full / forecast loss、MAE、Rank IC、方向等。  
3. 落 `baseline_manifest.json`（数据 SHA、模型 SHA、context_layer=10）。  
4. 失败则不开 GPU 长训。

### 阶段 1 — Bootstrap（双 LR 启动，短）

目的：让条件分支拿到稳定梯度；**不以本阶段 Best 定版**。

```text
双 LR（保留）：主干 LR 可略低于条件（例 1e-6 / 1e-5）——仅 Bootstrap 启动，不定版
条件（行业/市值）LR: 1e-5
scheduler: warmup_cosine，warmup ≈1%
loss: forecast CE + 0.02×history；aux ramp 关
每 seg: ~20k 无放回窗（与 Small 同族）
验证: quick 仅遥测；完整 forecast 作里程碑
长度: **不预锁**「跟 Small 同数百 seg」；先健康检查若干 seg，再按 §3 墙钟容量反推本轮能跑多少
监控: 条件输出范数 / 主干范数；两族梯度 RMS（作 M1 abort 闸）
说明: Bootstrap 主干偏低 ≠ Main 应用旧 Beta 1e-6；Main 仍锚定 1e-5
```

若条件能量失控 → 降条件峰值或加长 warmup，**不**直接进 Main。

### 阶段 2 — Main（统一峰值；**锚定 Small 1e-5**）

- 权重继承 Bootstrap **末点或 Best**（manifest 写死选哪个）。  
- **重建** optimizer/scheduler（不继承 Bootstrap sched 状态）。  
- **默认全参数峰值 = 1e-5**（与 Small 经验最优对齐；**不以旧 Beta 1e-6 为默认**），`warmup_cosine`（或与 Small Main 一致的 cosine）。  
- **尺寸规则**：Base≈4× **不**自动 ÷4 LR。可选 size-aware 短扫带：**5e-6 与 1e-5**（及 M1 已列的 3e-6 / 可选 3e-5）——用于确认，**不是**强制把峰值压到 1e-6。  
- **M1 闸（唯一降 LR 理由）**：**先验 1e-5**；仅当 timing/短扫显示 grad RMS 炸掉或条件范数失控 → 再降峰值后进多 seg Main。**禁止**「偏烫所以先往 1e-6 靠」。  
- loss：纯 CE（forecast + 0.02×history）。  
- Best 选择：**完整隔离验证 forecast loss**（与 Small Stage2 一致；full-val 须分片/off-host，见 §3.3）。  
- 目标覆盖：**尽量扫全量数据窗**（同数据合同、同 seg 语义）；**不预锁** Small 的 ~534/遍 或 C2@179。实际本轮跑多少 seg = 双 T4 墙钟/配额反推（见 §3）。周配额可能 **只够 Bootstrap + 部分 Main** — 预期写进 session 计划，不强求 Small 覆盖。

### 阶段 3 — 延长 / WC（可选，跟 Small 实测路径）

Small 实际路径：Extend → warmup-constant **hold 1e-5** → dual-T4 DDP。  
Base 草案同样预留：

```text
若 Main 仍在改善：可 1 轮 Extend（新 seed）仍锁定峰值（经 M1 闸后的值）
随后 WC：warmup-constant hold 该峰值，目标有效 global batch 尽量对齐 64
硬件锁定双 T4：默认 micro-batch 1–4 + 梯度累积，凑有效 global batch≈64
```

### 阶段 4 — Cosine refinement → 候选定版

镜像 Small Cosine refinement：

```text
统一 LR: 约锁定峰值 → ~1e-6（uniform_cosine，无额外长 warmup）
覆盖: 目标约半遍量级，但 **实际 seg 数由双 T4 墙钟容量决定**（不预锁 267/534）
选点: 全量 val **forecast best** → 记为 Base-Clean C*（名称评审定，避免叫 Beta）
定版时: aux 必须 off；纯 CE
```

**不定版旁路（明确隔离）**：Rank-IC 选段、aux 开的试验、Clean 数据接力——不得污染定版名；密封窗只作 REFERENCE。

---

## 3. 硬件 / 配额 / 段数 / Accum（双 T4 smoke→scale）

| | Small→C2 实测 | Base-Clean（本草案） |
|--|--|--|
| 参数 | ~24.8M | ~102.4M（~4×） |
| **硬件** | Kaggle **双 T4** DDP | **锁定仅双 T4**；**非硬否决**，策略 = 昂贵 smoke → scale |
| 显存 / batch | global batch 64 可跑 | **micro-batch 1–4** + accum，凑 `micro_bs × accum × n_gpu ≈ 64` |
| 墙钟 | 已可接受 | 粗估 **≥3–4×** Small（同 seg、同有效 batch）；同配额下 seg **少于** Small |
| 段数政策 | C2 落在 Seg179（历史） | **不预锁 Seg179 / 534 / 267**；见下方「先计时再估」 |
| Host RAM | Small dual-rank 可扛 | full-val / Rank IC **易 host OOM** → 必须分片或 off-host（§3.3） |

### 3.1 Accum 表（开训前填死；群评审硬点）

目标有效 global batch ≈ 64：

| micro_bs / GPU | n_gpu | accum steps | 有效 global batch | 备注 |
|----------------|-------|-------------|-------------------|------|
| 4 | 2 | 8 | 64 | 首选试跑（若显存够） |
| 2 | 2 | 16 | 64 | 默认兜底 |
| 1 | 2 | 32 | 64 | OOM 时再降 |

每 seg **真实 optimizer update 数** = `ceil(windows_per_seg / 有效 global batch)`（约 `20000/64 ≈ 312` updates/seg，与 micro 切法无关）。Manifest / 训练日志必须写清：`micro_bs`、`accum`、`n_gpu`、`effective_batch`、`updates_per_seg`。**禁止**只报 step 数却把 accum 内 micro-step 当成 update。

### 3.2 段数政策（跟 Small 实测流程，不跟 Small 的段号）

1. **数据仍按合同全覆盖意图**（同 holdout / 同窗语义）；「段数」= 本轮配额墙内能完成的 checkpoint 步数，不是「用不用全市场」。  
2. **M1 先开 1 个实验 timing segment**（双 T4、定好的 micro-batch + accum）：必须记录 **peak allocated memory、step/s、OOM=no**、wall clock / step。  
3. 用该墙钟 **反推**：本次 session / 周配额内 Bootstrap、Main、WC、Cosine **各大约能跑多少 seg**。  
4. 再定本轮目标段数与是否拆多 session；**禁止**开训前写死「跟 C2 一样到 Seg179」或强追 534。  
5. 若 **单 seg 仅 train（不含 full-val）墙钟 > ~25–30 min**：缩小 Main 目标 / 拆 session，**不要**为对齐 Small 覆盖硬推。  
6. 若单 seg 过慢但未超阈：优先再降 micro-batch / 加 accum / 缩短 quick val 频率，而不是改配方 LR 或偷正式窗。

### 3.3 Val / Rank IC 策略（双 T4 host RAM）

- full-val gather、Rank IC 面板：**分片（sharded）或 off-host**，镜像 Small dual-rank 做法；禁止单进程把全市场 gather 进 host 导致 OOM。  
- quick val：训练环内低频遥测即可；完整 forecast Best 仍作里程碑，但跑法必须可在 T4 RAM 下复现。  
- M1 日志若出现 host OOM → 先修 val 策略，再开 Bootstrap 长跑。

### 3.4 节奏

① timing seg（含 §3.1 表 + peak mem / step/s / OOM + LR 短扫 **5e-6 vs 1e-5**，可选 3e-6/3e-5）→ ② 短 Bootstrap 健康检查（双 LR；grad RMS 闸）→ ③ Main（**默认峰值 1e-5**；仅 M1 否决才降；按容量尽量扫）→ ④ 再决定是否 WC/Cosine。

**§7.1 已通过；Bootstrap seg1 现正推送。** Kernel / SwanLab：project=`finance`、run=`base_clean_0.1_bootstrap_timing`、账户 **user281434**、`max_segments=1`；预计 GPU·小时用 timing seg 回填。**勿 resume 旧 smce2_* / Beta 板。**

---

## 4. Segment 里程碑（评审可改数字）

| 里程碑 | 含义 | 通过标准（草案） |
|--------|------|------------------|
| M0 | 零训基线 | manifest + SHA 齐全；无数据泄漏；Beta 权重目录不在 `PYTHONPATH` / 不被默认加载 |
| M1 | **计时 seg** + Bootstrap 健康 + LR 短扫 | 双 T4：peak allocated、step/s、OOM=no 落盘；accum 表填死；条件范数/grad RMS 可控；forecast 相对 M0 有方向性改善或不崩；**据此估本轮 seg 容量**；短扫 **5e-6 vs 1e-5**（可选 3e-6 / 3e-5）看 grads+val → **默认锁 Main=1e-5**；**仅炸掉才降**（禁自动 ÷4 / 禁默认压到 1e-6） |
| M2 | Main（容量内）结束 | 有完整-val forecast Best；相对 M1 明确下降；updates/seg 与有效 batch 可审计 |
| M3 | WC（可选） | 低 loss 区不发散；Best 刷新或持平可接受 |
| M4 | Cosine best | 退火后 forecast Best；**aux off**；记候选定版 |
| M5 | 执行 REFERENCE | 同口径 §5.2 规则 vs **Small C2**（密封窗仅参考；见 §5.1 定性停训） |
| M6 | 正式 §5.2 | **仅** 10/09–10/23 未污染窗；行情够 D10 后再跑；**promote 前**勾 §5.2 数字门槛（非开训 blocker） |

停训 / 不晋级（草案）：

- Bootstrap 条件能量失控且调参 1 次仍失控 → 停，回评审。  
- Main 容量内结束后 forecast 相对零训/Bootstrap **无实质改善** → 不进 WC，先查 LR/batch/数据。  
- 密封 REFERENCE 触发 §5.1 可选冻结线 → **暂停加训**，等 Elon；**不**据此 kill/promote。  
- Cosine 候选：正式窗就绪后按 §5.2 **当时已勾选**门槛判 promote；未过 → **不 promote**，不烧多遍硬推。开训本身 **不**要求事先锁死数字门槛。  
- 任何阶段 **碰 10/09–10-23 调参** → 违规，该窗作废。

晋级（promote）草案：

- 定版名只授予：**纯 CE + aux off + 完整-val forecast best** 的 Cosine（或评审指定）点。  
- 对外可发表 alpha：**仅** 正式 §5.2 窗，且满足 §5.2 **promote 时已勾选**门槛（相对 **同窗** Small C2）。开训不要求事先勾选。

---

## 5. §5.2 / OOS（不可破）

### 5.0 通用规则

1. **可发表**：信号 **2026-10-09～10-23**，持有到 D10（行情约至 11/06）。  
2. **8/03～9/30 密封 42 天**：只做 REFERENCE；可与 Small C2 并排，**不可单独晋级、不可据此改阈值刷窗**。  
3. Rank IC：只作 Layer A 选段遥测；Layer B 执行收益说话。  
4. 执行默认口径（与现 C2 锁）：`|pred|≥5%`、次日开、涨跌停/停牌、1/10 袖至 D10、分层滑点 + 0.3% 往返；探索口径（提前出、`>8%`）须双端重跑并标非协议。  
5. **注意**：C2 在 10/09–10-23 上的数字 **尚未跑出**（bars 未齐）。门槛是相对 **该窗同口径 C2**，**不是**相对密封42 的 `+2.59% / Sharpe 0.78`。

### 5.1 密封 REFERENCE — 定性停训（不 kill / 不 promote）

密封 8/03–9/30 仅参考：

- 若相对同口径 Small C2 在 **return + Sharpe + mean_net** 上 **明显更差** → **不要再烧更多 epoch** 硬推；回评审。  
- **可选冻结线（PROPOSAL，待 Elon）**：若 `total_return` 相对 C2 差距 **≤ −2pp** **且** `Sharpe ≤ C2_Sharpe − 0.3` → **冻结继续加训**，等 Elon 拍板。  
- 密封结果 **不能**单独 promote，也 **不能**单独永久 kill 该线。

### 5.2 晋级门槛（**仅 promote 前勾选**；**非开训 blocker**）— **PROPOSAL**

> 以下为 **PROPOSAL**，供 Elon 在 **promote 前**勾选/改数。  
> **Elon + 量化交易模型大师（2026-10-26）：数值门槛不再阻塞开训**——正式窗标签未齐前定太早也是空的；分段走到哪看到哪。开训只卡 §7 M0/M1 ops。  
> **未勾选前：可以开训 / 推 kernel（在 §7 开训 blocker 已勾完前提下）；不得宣称晋级 / promote。**  
> 比较对象：冻结 **Small C2**，在 **同一执行协议**、**同一正式窗 10/09–10/23** 上的结果（窗就绪后才跑）。  
> **任一条 Fail → 不 promote。**

| 指标 | PROPOSAL 门槛（相对同窗 C2） | Elon |
|------|------------------------------|------|
| `total_return` | **≥ C2_same_window − 0.5pp**（绝对百分点） | [ ] |
| `Sharpe` | **≥ C2_Sharpe − 0.15** | [ ] |
| `maxDD` | **不差于 \|C2_DD\| + 2pp**（绝对百分点；即回撤幅度最多再差 2pp） | [ ] |
| `mean_net` | **≥ 0 OR ≥ C2_mean_net**（满足其一即可） | [ ] |
| 辅：`win_rate` | **不要求**必须打赢 C2（若 return+Sharpe 已过） | [ ] 接受 / [ ] 改严 |
| 密封 8/03–9/30 | **不能单独 promote**；仅 REFERENCE + §5.1 定性停训 | [ ] |
| 执行协议 | 与 C2 锁相同（§5.0 第 4 条）；探索口径不计入 promote | [ ] |
| aux | promote 时 **必须 off** | [ ] |

**可选更严变体（备选，默不启用）：** 要求 `total_return` **严格 > C2** 且 `Sharpe` **严格 > C2**，且 `maxDD` 不差于 C2（无 +2pp 缓冲）。若启用，勾此处：[ ] 改用更严变体。

---

## 6. 与已清理血缘的关系

- 已删除旧 `BETA_V*` / `V1_BETA_*` / `FULL_MARKET_V1_BETA*` / Beta 报告文档与相关 beta kernel/tests（见本轮 staged 清理）。  
- **`finetune/beta_v21.py` 保留**（群评审硬点；Small/C2 与本线仍依赖）——**暂不删、不重命名**，除非评审另批。  
- **旧 Beta 权重目录未删**（体积大）；开训前确认：**不在 `PYTHONPATH`、不被默认 `KRONOS_PREDICTOR_PATH` 指向**；需要物理删除时另批。  
- 新跑必须用 **新 kernel 名 + 新输出目录**（`base_clean_0.1_*`），避免写入旧 `beta_v*` 路径。

---

## 7. Checklist（拆成开训 blocker / promote-only）

### 7.1 开训 blocker（全部勾完才允许推训 / 推 kernel）

> 只卡 **M0/M1 可运行**。**不含** §5.2 数值门槛。

- [x] 起点 = HF `NeoQuasar/Kronos-base` + Tokenizer-base；不用 V6/Beta 权重  
- [x] 镜像 Small→C2；**双 T4 only**；timing seg 先；段数由墙钟反推（不预锁 179/534）  
- [x] Main 峰值 **先验锚定 Small 1e-5**；禁自动 ÷4；禁默认压到旧 Beta 1e-6  
- [x] 正式可发表窗声明 = **仅 10/09–10/23**（规则不变；数字门槛另见 §7.2）  
- [x] **保留 `beta_v21.py`**（不删）；promote 时 **aux off**  
- [x] **Accum 表**：群默认锁定 **`4×8×2=64`**；updates/seg ≈ `20000/64 ≈ 312`（manifest 须落盘）  
- [x] **M1 LR 短扫**：**5e-6 vs 1e-5**（可选再试 3e-6 / 3e-5）；默认锁 Main=1e-5；grad/条件炸掉才降  
- [x] **Val / Rank IC 分片**（或 off-host），避免双 T4 host OOM  
- [x] **账户**：Kaggle / SwanLab 用 **user281434**
- [x] **SwanLab 命名（对齐 Small）**：Bootstrap **新开看板** `base_clean_0.1_bootstrap_timing`（勿续 Main/Cosine/smce2/Beta）；长跑再 `base_clean_0.1_bootstrap`；进 Main 再开 `base_clean_0.1_main`；同阶段 chunk 共用 run id + `resume=allow`；**禁止**写入 `smce2_*` / `beta_v*` / 旧 Small 板  
- [x] **旧 Beta 权重**：**隔离、不删**——不进 `PYTHONPATH` / 默认预测路径；M0 确认 HF 干净加载后再议物理删除  

**§7.1 已全部勾完（群默认）→ 开训 checklist PASSED。**  
**当前动作：** Bootstrap **`max_segments=1`** 现正推送（timing / M1；看板 `base_clean_0.1_bootstrap_timing`；账户 user281434）。  
整体计划状态：**APPROVED-FOR-M1**（非全阶段定版通过）。§5.2 仍见 §7.2（promote-only）。

### 7.2 Promote-only（**不阻塞开训**；正式窗就绪、要晋级时再勾）

- [ ] **§5.2 数值门槛**：§5.2 表由 Elon 勾选/改数（现为 PROPOSAL）  
- [ ] 同窗 Small C2 对照结果已落盘  
- [ ] 定版点 = 纯 CE + aux off + 完整-val forecast best  

> Elon 定调：门槛「训练了再说 / 走到哪一步看哪一步」；正式窗标签未齐前不定死数字。


## 2026-10-26 SwanLab 坑（Bootstrap 长跑）— Small 已踩过，Base 禁止再踩

新板回放 timing `metrics.jsonl` 时，seg1 validation 用 `total_steps=625` 打到高 step，而真训练 updates/seg≈**313**，导致 `segment` 曲线出现 1→2→1 假象。**训练未倒退。** 判读以 loss / LR / 日志 `Segment N` 为准。

**硬规则（下轮起强制）：**
- [ ] **新 SwanLab 板禁止回放 parent `metrics.jsonl`**（同板接力才允许 `resume=allow` + 续写）
- [ ] 若必须回放：用真实 `updates_per_seg`（≈313 / `len(train_loader)`）算 chart step，**禁止**默认 625
- [ ] 看板 `segment` 标量不作健康判据；以日志 `Segment N/M, Step a/b` + forecast 为准

## 2026-10-26 条件通路盯点（Base vs Small 结构差）

Validation `Condition Full/None/Shuffled`：Bootstrap 早期 Full−None / Full−Shuffled ≈ **−6e-5 / −1e-5 量级贴 0** 为预期（条件输出层零 init → no-op；双 LR 预热中）。

**硬盯：**
- [ ] Seg **50–100** 后若 Full−Shuffled delta 仍贴 0 → 警惕 `context_layer=10` 条件注入未生效（行业 emb + 市值 MLP 是 Base 相对 Small 的主结构差）
- [ ] 判读看日志 `Validation Condition Full/None/Shuffled Forecast`（当前未单独打 SwanLab key）
- [ ] 本信号**不**单独 abort 本 chunk；下 chunk 接力时复读，进入 Main 前复读

观测锚点（长跑 ~Seg20）：Full/None/Shuffled ≈ 2.68448 / 2.68454 / 2.68448；Δ Full−None ≈ −6.2e-5，Full−Shuffled ≈ −4.7e-6。

