# 条件身份：门禁协议与 P0 执行清单（导师批注稿）

日期：2026-10-27（Asia/Shanghai）
对应审阅稿：`finetune/reports/20261027_small_vs_base_cond_identity_mentor_review_cn.md`
状态：**协议稿**。未改任何训练/评测程序；live Bootstrap 保持无 γ 裸加；Small 生产冻结不变。

---

## 0. 裁定摘要

| 议题 | 裁定 | 关键条件 |
|---|---|---|
| 现行双门口径 | **不作论文口径**；作内部 triage 用，但能量门须降级为遥测/护栏 | 见 §2 三级阶梯 |
| Small quick n=2000 | **够判“数量级”，不够当证据** | 补 K=8 置换 + 配对 CI + 独立确认集 |
| 冷启真 A+C 长叉 | **同意方向，暂不开** | §7 三条前置全中才开 |
| 先无 margin | **是** | margin 版不得用 G1 判成功（循环论证） |

总纲：现有一组对照既不能证明“CE 在 Base 上学不出身份”，也不能支持反向结论。Base@Seg175 的观测落在**条件路径的噪声底**上，这是可外推的算术后果，不是需要解释的现象。判定的关键不是再跑多久，而是**先把测量口径、管线、零假设对齐，再用一个 1–2 小时的上界实验决定要不要花周级算力**。

---

## 1. 口径问题（四条，按杀伤力排序）

### 1.1 缺一条 0 成本的决定性测量：`sector_emb` 几何

审阅稿 §3.2 的“`sector ‖W‖_F ≈ 5.39 ≈ init`”跨模型不可读：Small 是 **zero-init**（`model/kronos.py:250`），从 0 长到 15.7；Base A+C 是 `Normal(0,0.02)`，init 本身就是 5.381。同一句“≈ init”在两边含义相反。

换成每元素 RMS（`‖W‖_F / √(87·d)`）：

| 模型 | 形状 | init | ‖W‖_F | 每元素 RMS | 相对 init |
|---|---|---|---|---|---|
| Small C2 best | 87×512 | 0（zero） | 15.7 | **0.074** | — |
| Base A+C Seg5 | 87×832 | 0.02 | 5.387 | **0.0201** | 1.00×（未动） |
| Base 裸 Boot@Seg175 | 87×832 | 0（zero） | **未测** | **≈0.002**（扩散估计） | — |

**Base@Seg175 的那一格是空的，而它恰恰是全部争议所在。** 权重已在 Kaggle Output 里，倒出来不需要训练。

### 1.2 `trunk_ratio` 不可移植

分子分母都随注入位点漂移：Small 在 6/8 层、512d；Base 在 10/12 层、832d，residual 范数不同量级。更严重的是**文档没写这个比值是在 γ 之前还是之后测的**——代码里 `condition_trunk_norm_ratio` 在 `_add_context` 内、`x + condition` 之前算（`model/kronos.py:300-318`），而 `condition_scale` 在 kernel 版里挂在哪一侧，master 上查不到。两种读法给出的“淹没倍数”差 20 倍以上。

→ **必须同时报 `condition_output_norm` 与 `trunk_input_norm` 两个绝对值**（stats 里本来就有），比值只作附注；口径写死为“同注入层、γ 之前”。

### 1.3 能量门会被常数偏置蒙过

`20261027_sector_emb_collapse_probe.md` 自证：B 把 `size_mlp` 放大成近常向量（size 128 vs sector 15）。一个全市场同向的市值偏置即可把 `trunk_ratio` 顶过 0.02，而身份为零。**能量门与内容门不是两份独立证据。**

### 1.4 两条读数不同源

- Small PASS 来自离线 kernel（`kronos-small-c2-cond-identity-ablation` + `cond_identity_ablation_result.json`），**代码与 JSON 均不在 master**；
- Base FAIL 来自训练内 ablation（`finetune/train_predictor.py:2068-2112`，teacher-forced、quick 2048，`launch_bootstrap.py:299`）。

“160 倍”里有多少是管线差，现在无法排除。另：`condition_scale` 在 master 上**不存在**（`git log -S condition_scale` 只命中文档），而 hub 加载默认 `strict=False` —— 把 A+C/γ ckpt 丢给 master 代码复评，γ 会被**静默丢弃**（25→1），结果必然错。

---

## 2. 门定义（替换审阅稿 §2）

能量门从“门”降为**遥测 + 护栏**；判定由 G1/G2 承担，G3 才是论文口径。

| 层 | 指标 | 判据 | 说明 |
|---|---|---|---|
| **G0 遥测/护栏** | `condition_output_norm`、`trunk_input_norm`、比值（γ 前、同层）；注入向量 between-sector 方差占比；`sector_emb` 每元素 RMS | **不设 pass/fail**。护栏：比值 ≤ 0.3（防爆）；between-sector 占比 ≥ 0.3（防恒等偏置顶包） | 只做归因 |
| **G1 CE 身份** | `FS = CE_full − CE_shuf`（K=8 置换均值）；零假设 `null = CE(perm_a) − CE(perm_b)` | `FS ≥ max(3·|null|, 1e-3)` **且** 连续 ≥3 次 eval 同号 **且** 95% CI 下界 > 3·|null| | 允许“可分辨” |
| **G2 输出身份** | 反事实换行业（保市值）→ 预测 10d 收益分位变化率 `r_swap`；换市值（保行业）→ `r_size` | `r_swap ≥ 5%` 且 `r_swap ≥ r_size` | 身份进入预测 |
| **G3 下游可用** | OOS Rank IC / 行业中性化后增量 | 正号且量级可报，单独出表 | 只有这层进论文 |

必报的分解行（G1 下挂）：`FS_sector`（换行业、保市值）与 `FS_size`（换市值、保行业）。现在的 roll-1 同时置换三者，无法归因到“行业身份”。

**为什么用 `perm_a vs perm_b` 作零假设**：1e-5 不是噪声底，它只是 Base 在 γ=1 下的历史巧合值。两个不同置换之差由构造决定其量级，且**自动随该模型自身的条件敏感度缩放**，因此跨架构可比、跨阶段可比。成本是每个 eval 多一次前向。

---

## 3. 对齐锚点（替换“对齐 Seg179”）

段号不可作锚点：Small C2 best 是**第 2505 累计段**（534+534+190+534+534+179），Base@Seg175 是**第 175 段**。四个可接受锚点，按优先级：

- **A1 CE 锚（主）**：判门时 Base 全量 val CE 与 Small C2（2.2944）差 ≤ 0.05。落不到，就写“两者拟合水平不同，身份对比不成立”，不得混用。
- **A2 暴露锚**：同时报 windows/param 比，不追求相等。
- **A3 LR 锚**：只在 `uniform_cosine` 退火到 ≤2e-6 的段判门，且该次 eval 必须打上 LR 标签。峰值 LR 下判门 = 判噪声。
- **A4 管线锚**：同一 harness、同一 2000 窗（index hash 落盘）、同一组 K 置换。

---

## 4. 预算算术（为什么“再跑长一点”不是主杠杆）

| | Small C2 best | Base 裸 Boot@Seg175 |
|---|---|---|
| 累计 segment | 2505 | 175 |
| 样本 | 50.1M（4.70 遍） | 3.5M（0.33 遍） |
| windows/param | **2.02** | **0.034**（差 59×） |
| `sector_emb` 每元素 RMS | 0.074 | ≈0.002 |

位移预算：条件 LR 1e-5 × 313 updates/seg × 175 seg = 5.5 万步。
- **相干上界** = 1e-5 × 5.5e4 = **0.55**/元素；
- **扩散底**（Adam 步长≈lr、符号随机）= 1e-5·√5.5e4 = **2.3e-3**/元素。

实测若落在 2e-3 附近，就是纯噪声端：到 Small 的 0.074 还需 32× 位移。外推：走扩散分支需 ~1000× 步数（≈17 万段，永不可达）；即使全程相干也需 ~5600 段。**结论：抬相对注入强度（γ）或条件 LR 才是杠杆；单纯加段数不可能收敛到可比状态。**

配套：脚本注释的 `~29 min/seg`（`launch_bootstrap.py:39`）与“10-26 开训、10-27 已到 Seg175”对不上，预算测算必须换成 SwanLab 实测墙钟。

---

## 5. P0 清单（开长叉前必须完成，全部不涉及新增长训）

> 落盘空表：**`finetune/reports/20261027_p0_cond_identity_fill_in_cn.md`**（另存为 `..._results_cn.md` 后填；空格视为未测）。

| 编号 | 内容 | 成本 | 产出 | 判定 |
|---|---|---|---|---|
| **P0-1** | 倒出 Seg175 ckpt 的 `sector_emb`：‖W‖_F、每元素 RMS、行范数分布、行间余弦（排除 unknown_sector_id） | ~0 | 一行数 | RMS ≤0.005 → “未启动”，不得引用 FS 作 CE 结论；0.005–0.02 → 起步；≥0.02 且 `FS < 3·null` → **此时才支持“CE 不驱动身份”** |
| **P0-2** | 同 harness、同 2000 窗、K=8 置换、含 null 与 2×2 分解，重打 Small C2 与 Base@Seg175 | 半天 | 两行表 + CI | 两边数字必须来自同一管线，否则后续一律标“历史读数” |
| **P0-3** | 冻结 trunk 的条件专训探针（§6） | 1–2 小时级 | `FS*(γ, lr)` 表 | 决定长叉开不开、γ 取多少 |
| **P0-4** | 实测单 seg 墙钟、updates/seg、val+ablation 占比 | ~0 | 预算表 | 反推本轮可承载段数 |

P0-2 的落盘要求：`fs_mean`、`fs_sd`（跨置换）、`null`、`fs_sector`、`fs_size`、`per_window_delta_sd`、`ci95`、`n`、`windows_index_sha256`、`lr_at_eval`、`segment`、`ckpt_sha256`。

---

## 6. 冻结 trunk 探针（P0-3 设计）

目的：在**不动主干**的前提下，给出“这套注入位点 + 这套 CE 最多能长出多少身份”的**上界** `FS*(γ, lr)`。这是唯一能在小时级回答“淹没 vs 饿着”的实验。

- 可训练：`sector_emb` / `size_mlp`（+γ 若启用），~78K 参数；trunk 全冻。
- 网格：γ ∈ {10, 30, 100} × 条件 LR ∈ {1e-4, 3e-4, 1e-3, 3e-3}（78K 参数、无主干干扰，可用远高于 1e-5 的 LR）。
- 步数 200–600，batch 64，取训练段；每个 cell 在 2000 窗上打 K=8 + null + 2×2。
- 落盘：`FS*`、`sector RMS`、between-sector 占比、`r_swap`。

判定：

| 观测 | 解释 | 动作 |
|---|---|---|
| max `FS*` < 1e-3 | 该注入位点/该损失**表达不出**身份 | 停长叉；问题改为“要不要 identity-sensitive 项 / 换注入位点” |
| `FS*` 随 γ 升后塌 | 淹没确认（结构性） | 取拐点 γ，长叉按拐点设 |
| `FS*` 在**高 LR 侧**才起来，且 ≥5e-3 | 主因是**条件 LR**，不是段数 | 长叉降级为“条件 LR 调参”，先跑 50–100 段验证 |
| `FS*` 与 γ、lr 都弱 | 上界本就低（信息论上限） | 2e-3 已是天花板，放弃“抬到 1e-2”的期待 |

---

## 7. 长叉开启条件与中期停看

**三条前置（全中才开）**：① P0-1/2 完成且几何与重评通过；② `FS*` 明显高于 G1 门；③ 预算能覆盖**退火尾**（必须含 `uniform_cosine` 到 ~1e-6 的收口段）。

**方案要点**：HF 冷启 + 真 A+C（γ 取 P0-3 拐点，非默认 25）+ 独立 kernel，不并进 live Bootstrap；判门用 A1/A3/A4 锚点。

| 检查点 | 主看 | 停/走 |
|---|---|---|
| 12.5% | sector RMS 斜率；`FS` vs `null` | 斜率≈0 且 `FS < null` → **停** |
| 25% | 同上 + γ 后比值（G0 护栏） | 比值 >0.3 → 降 γ 或停 |
| 50% | sector RMS 斜率是否 ≥ 首 1/4 的 20% | 否则 **停**（“该配方下 CE 不驱动身份”可结案） |
| 75% | 退火中段 G1 趋势（≥3 次同号） | 趋势不单调 → 记“未收敛”，不加段 |
| 100%（LR≤2e-6） | G1 + G2 | 终判；G1 过而 G2 不过 → 记为“CE 可分、预测不可用” |

必落盘的 5 个数：sector RMS、between-sector 占比、`FS`(K=8,mean±sd)、`null`、`r_swap`；每条都带 `lr_at_eval`。

---

## 8. margin 政策

- **先无 margin**：无 margin 是科学问题（CE 能否自行长出身份），margin 是工程手段（强行造出身份）。
- margin 直接优化 `CE_full − CE_shuf + m`，因此**必然把 G1 顶过去**——用 G1 判 margin 的成功是循环论证。margin 版只认 G3。
- `m` 必须锚在 `FS*` 上（建议 `m = 0.1–0.3 × FS*`）。P0-3 之前定 m 都是拍脑袋。
- margin 版单独开叉、单独命名，不并入 live Bootstrap，不与无 margin 版共用看板。

---

## 9. 对审阅稿 §4 的措辞修正

原文：“**不能推出**：「再裸跑久一点就会有」——裸 Boot 已至与 Small Bootstrap 同量级段数，FS 仍 ~1e-5。”

建议改为：

> 现有数据既不支持“会”，也不支持“不会”。Base@Seg175 的条件路径位移落在噪声端（≈0.002/元素，Small 为 0.074），离可比状态差 32×。按扩散外推，等段数等不出可比状态；要么抬相对注入强度（γ / 条件 LR），要么承认 Base 上不存在这套对照。

另：§3.1 与 §3.3 的 FS 来自两条不同管线，正文应标注为“历史读数（管线待对齐）”，P0-2 后替换。

---

## 10. 待补材料（5 项）

1. Seg175 ckpt 的 `sector_emb` 几何（P0-1）。
2. `cond_identity_ablation_result.json` 与对应离线 kernel 源码，入 `finetune/reports/`（否则 §3.1 那行数字无法复核）。
3. 同 harness / 同 2000 窗 / K=8 的重评结果（P0-2）。
4. `FS*(γ, lr)` 表（P0-3）。
5. 实测单 seg 墙钟与 updates/seg（P0-4）。

> 附注（仅作溯源提醒，不作指控）：`artifacts/swanlab_probe/smce2c3_long__train_forecast_loss.json` 中有两个 **train**-loss 点与审阅稿 Full/None 的 5 位小数重合（step 68300 / 34300）。概率上可能只是巧合，但这份报告的三元组目前**没有可复核的 JSON 落盘**，P0-2 正好一并清掉。

---

## 11. 事实核对锚点（供审计）

| 事实 | 位置 |
|---|---|
| `sector_emb` zero-init | `model/kronos.py:250` |
| 条件注入与 trunk 统计（γ 前） | `model/kronos.py:271`、`:300-318` |
| 训练内 ablation（roll-1、teacher-forced） | `finetune/train_predictor.py:2068-2112`、`:2262` |
| ablation 用 quick loader | `finetune/train_predictor.py:4167`、`:4244` |
| val 置换 seed 与 quick 子集 | `finetune/dataset.py:606`、`:620` |
| QUICK=2048 / ablation 每 5 seg | `finetune/base_clean_bootstrap_relay/launch_bootstrap.py:299-302` |
| `~29 min/seg` 假设 | `finetune/base_clean_bootstrap_relay/launch_bootstrap.py:39` |

---

## 12. 不变更声明

本协议**未改** `model/kronos.py`、`finetune/train_predictor.py`、`finetune/config.py`、`finetune/dataset.py`、任何 relay 脚本；未新增训练路径、未加 aux/margin；live Bootstrap 仍为无 γ 裸加；Small 生产冻结与 HOLD e01j 不变。
