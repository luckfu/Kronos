# P0 落盘结果草稿：条件身份门禁（部分填报）

日期：2026-10-27（Asia/Shanghai）
配套：`finetune/reports/20261027_cond_identity_gate_protocol_cn.md`（裁定与判定规则）
模板：`finetune/reports/20261027_p0_cond_identity_fill_in_cn.md`（空表未改）
状态：**P0-1 partial 复核 PASS（Monitor-only）→ 待合 master**。仅填 Base 裸 Boot Seg155/170/175 的 ‖W‖_F + 每元素 RMS（训练内 Condition Monitor）；行范数 / 行间余弦 / cos(row,mean) / between-sector **未测**；P0-2 / P0-3 / P0-4 **未测**。空格 = 未测，不得发明。

---

## 0. 填表约定（沿用模板）

1. 空格 = 未测，**不得**用估计值占位；估计值写进「注」列并标 `（估计）`。
2. 每个数带来源：`来源` 列写 kernel / 日志路径 / 命令，不写“见群”。
3. **两条管线不得混填**：训练内 telemetry 与离线重评分表；本草稿 **只填 P0-1 几何（训练内 `sector_embedding_weight_norm`）**，不据此下 P0-2 / G1 结论。
4. 涉及 γ 的 ckpt 必须在**带 `condition_scale` 的 kernel 代码**下评（本草稿未评 γ ckpt）。
5. 判定只认本表已填行；未填格子不得在结论页引用（§7 仅写 P0-1 第一条）。

---

## 1. 元信息

| 项 | 值 |
|---|---|
| 执行人 / 日期 | 量化交易模型大师 / 2026-10-27（Asia/Shanghai）；partial P0-1；Musk 复核 PASS |
| 仓库 commit（被测代码） | `b8166b7ee78a64d9ee5d5a85c4c1010543d5a44a`（Bootstrap v7 build_manifest `source_commit`；无 γ bare-add） |
| 数据 manifest SHA-256 |  |
| holdout / 训练窗合同 |  |
| tokenizer |  |
| sector 数（含 unknown） | 86（+1）→ 表形 87×832（Base） |
| device / 精度 |  |
| 环境变量快照 | `KRONOS_VALIDATION_QUICK_SAMPLES=` / `KRONOS_VALIDATION_SAMPLES=` / `KRONOS_CONDITION_ABLATION_INTERVAL_SEGMENTS=` |
| 几何换算 | 每元素 RMS = ‖W‖_F / √(87·832)；√(87·832)=269.0427475328038 |

---

## 2. P0-1 `sector_emb` 几何（排除 `unknown_sector_id=86` 的完整行几何 **未测**；下列 ‖W‖_F 为 Condition Monitor 上报的 `sector_embedding_weight_norm`＝‖W‖_F）

| run | ckpt SHA-256 | 形状 | init 类型 | ‖W‖_F | 每元素 RMS | 行范数 min / median / max | 行间余弦(off-diag) | cos(row, mean) | between-sector 占比 | 来源 | 注 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Small C2 best（Seg179） |  | 87×512 | zero | 15.7 | 0.074 |  |  |  |  |  | 已知参照（模板注）：15.7 / 0.074 / 0.13；ckpt SHA 未填 |
| Base A+C Seg5 |  | 87×832 | Normal(0,0.02) | 5.387 | 0.0201 |  |  |  |  |  | 已知参照（模板注）：5.387 / 0.0201；ckpt SHA 未填 |
| Base 裸 Boot Seg155 |  | 87×832 | zero | 1.5076783895492554 | 0.00560386 | 未测 | 未测 | 未测 | 未测 | `/tmp/base_bootstrap_long_logs_snap_fresh1004b.txt` Condition Monitor JSON at Seg155 Validation Condition；field `sector_embedding_weight_norm` | self-parent 只留 last，Seg155 ckpt 已被盖掉；完整行几何未测；**RMS 分母含 unknown** |
| Base 裸 Boot Seg170 |  | 87×832 | zero | 1.62124 | 0.00602596 | 未测 | 未测 | 未测 | 未测 | `/tmp/base_bootstrap_long_logs_snap_fresh1235.txt` Condition Monitor JSON at Seg170 Validation Condition；field `sector_embedding_weight_norm` | 同上，ckpt 已被盖掉；**RMS 分母含 unknown** |
| Base 裸 Boot Seg175 |  | 87×832 | zero | 1.65465 | 0.00615012 | 未测 | 未测 | 未测 | 未测 | `/tmp/base_bootstrap_long_logs_snap_fresh1235.txt` Condition Monitor JSON at Seg175 Validation Condition；field `sector_embedding_weight_norm` | **关键格**；RMS∈(0.005,0.02)→起步中；ckpt 已被 self-parent last_model 盖掉；**RMS 分母 √(87·832) 含 unknown 行**（与「排除 unknown」的完整行几何未测区分；Monitor ‖W‖_F 亦含 unknown） |

**速查判定（每元素 RMS）**

| 区间 | 解释 | 允许的表述 |
|---|---|---|
| ≤ 0.005 | 条件路径未启动 | 不得引用 FS 作“CE 学不出身份”的证据 |
| 0.005 – 0.02 | 起步中 | 只能说“未达可比状态” |
| ≥ 0.02 且 `FS < 3·null` | 表已长、FS 仍贴零 | **此时才支持“CE 对身份无压力”** |

**本草稿 P0-1 判定（仅 Base 裸 Boot@Seg175）**：RMS≈0.00615 → **起步中（0.005–0.02）**。只能说「未达可比状态」；不得引用 FS 作“CE 学不出身份”；也未达 ≥0.02 的「表已长」结案带。

---

## 3. P0-2 同管重评（离线 harness，同一 2000 窗，K=8）

表头：`管线＝离线重评`（与 §4 的训练内 telemetry **分表**）

| 字段 | 含义 | Small C2 best | Base 裸 Boot@Seg175 | Base A+C（若有） |
|---|---|---|---|---|
| `ckpt_sha256` | 权重指纹 |  |  |  |
| `segment` | 段号 | 179（cosine 内） |  |  |
| `lr_at_eval` | 该次 eval 的条件 LR |  |  |  |
| `n` | 评测窗数 |  |  |  |
| `windows_index_sha256` | 窗 index 指纹（证明同一批） |  |  |  |
| `K` | 置换数 | 8 | 8 | 8 |
| `fs_mean` | `CE_full − CE_shuf` 均值 |  |  |  |
| `fs_sd` | 跨置换 sd |  |  |  |
| `null` | `CE(perm_a) − CE(perm_b)` |  |  |  |
| `null_sd` | 跨置换 sd |  |  |  |
| `ci95` | 配对 bootstrap 95% CI |  |  |  |
| `per_window_delta_sd` | 逐窗 ΔCE 的 sd（算 SE 用） |  |  |  |
| `fs_sector` | 换行业、保市值 |  |  |  |
| `fs_size` | 换市值、保行业 |  |  |  |
| `r_swap` | 换行业后预测 10d 分位变化率 |  |  |  |
| `r_size` | 换市值后同上 |  |  |  |

**G1 判定行**（逐 ckpt 填）

| ckpt | `FS ≥ max(3·|null|, 1e-3)` | 连续 ≥3 次同号 | CI 下界 > 3·|null| | G1 |
|---|---|---|---|---|
| Small C2 best |  |  |  |  |
| Base 裸 Boot@Seg175 |  |  |  |  |

**G2 判定行**

| ckpt | `r_swap ≥ 5%` | `r_swap ≥ r_size` | G2 |
|---|---|---|---|
| Small C2 best |  |  |  |
| Base 裸 Boot@Seg175 |  |  |  |

---

## 4. 训练内 telemetry（与 §3 分表，不得混算）

| segment | `lr_at_eval` | `forecast_loss`(full) | `condition_none_forecast_loss` | `condition_shuffled_forecast_loss` | `full−none` | `full−shuffled` | `condition_trunk_norm_ratio`（γ 前） | `condition_output_norm` | `trunk_input_norm` |
|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |

> 提醒：现行 telemetry 只有 roll-1 单置换、无 null、无 `fs_sector`/`fs_size` 分解；**不可**直接当 P0-2 的结论用。本草稿未填本表（P0-1 几何单独在 §2）。

---

## 5. P0-3 冻结 trunk 探针：`FS*(γ, lr)`

trunk 全冻，仅训 `sector_emb` / `size_mlp`（+γ）；batch 64；步数 200–600；每个 cell 打 K=8 + null + 2×2。

| γ \ lr | 1e-4 | 3e-4 | 1e-3 | 3e-3 |
|---|---|---|---|---|
| 10 |  |  |  |  |
| 30 |  |  |  |  |
| 100 |  |  |  |  |

**结论行（四选一，勾一项）** — 未测，不勾。

- [ ] max `FS*` < 1e-3 → 该注入位点/损失**表达不出**身份 → **停长叉**，改“是否加 identity-sensitive 项 / 换注入位点”
- [ ] `FS*` 随 γ 升后塌 → **淹没确认**，取拐点 γ＝______
- [ ] 高 LR 侧才起且 `FS*` ≥ 5e-3 → 主因是**条件 LR** → 长叉降级为“条件 LR 调参”，先 50–100 段
- [ ] γ、lr 都弱 → 上界本就低 → 放弃“抬到 1e-2”，按天花板重设门

---

## 6. P0-4 预算实测

| 项 | 值 | 来源 |
|---|---|---|
| 单 seg 纯训练墙钟 |  |  |
| 单 seg 含 val + ablation 墙钟 |  |  |
| `updates/seg`（应 = ceil(20000/有效 batch)） |  |  |
| val 占比 / ablation 占比 |  |  |
| 本轮可用配额（GPU·h） |  |  |
| 可承载段数（含退火尾 ~200–267 段） |  | 反推 |

> 冲突待解：脚本注释 `~29 min/seg`（`launch_bootstrap.py:39`）与“10-26 开训、10-27 到 Seg175”不一致，以实测为准。本草稿未测。

---

## 7. 结论页（三条，仅填已有的第 1 条）

1. **P0-1**：Base@Seg175 `sector_emb` 每元素 RMS≈0.00615 → 判定：**起步中（0.005–0.02）**。不得引用 FS 作“CE 学不出身份”；也未达 ≥0.02 的“表已长”结案带。
2. **P0-3**：未测（空）。
3. **长叉**：未测 / 本草稿不下裁定。

**签署**：量化交易模型大师（执行，partial Monitor-only） / **Musk 复核 PASS（partial Monitor-only）**　日期：2026-10-27（Asia/Shanghai）

---

## 8. 禁止事项（复核时直接打回）

- 用训练内 telemetry 冒充实测重评，或把两条管线的数字放进同一张对比表。
- 用 master 的 hub 加载去评带 `condition_scale` 的 ckpt（γ 会被静默丢弃）。
- 用估计值填「实测」列，或用未填的格子支撑结论。
- 在峰值 LR 的段判 G1（未打 `lr_at_eval` 的判门视为无效）。
- 用 G1 判 margin 版的成功（margin 只认 G3）。
