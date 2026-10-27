# P0 落盘模板：条件身份门禁（填报复核用）

日期：2026-10-27（Asia/Shanghai）
配套：`finetune/reports/20261027_cond_identity_gate_protocol_cn.md`（裁定与判定规则）
用法：**另存为** `finetune/reports/20261027_p0_cond_identity_results_cn.md` 后再填；本文件保持空表，不要就地填。
状态：**空表，未测**。

---

## 0. 填表约定

1. 空格 = 未测，**不得**用估计值占位；估计值写进「注」列并标 `（估计）`。
2. 每个数带来源：`来源` 列写 kernel / 日志路径 / 命令，不写“见群”。
3. **两条管线不得混填**：训练内 telemetry（`train_predictor.py` 每 5 seg 的 ablation）与离线重评（P0-2）分表记录，且在表头标明属于哪条。
4. 涉及 γ 的 ckpt 必须在**带 `condition_scale` 的 kernel 代码**下评（master 的 hub 加载 `strict=False` 会静默丢 γ）。
5. 判定只认本表填完的行；未填的格子不得在结论页引用。
6. **每个判门 ckpt 必须挂不可覆写快照**（`ckpt_snapshot` 路径 + `ckpt_sha256`）；被后续 chunk 覆盖的行作废（协议 §4.1；Seg155/170/175 已发生一次）。
7. **锚点**：Base 侧锚 `durable last`，**Seg175 已退役**，不得再作为判门锚；判门同时报 A1（val CE）与 A3（`lr_at_eval`）。

---

## 1. 元信息

| 项 | 值 |
|---|---|
| 执行人 / 日期 | |
| 仓库 commit（被测代码） | |
| 数据 manifest SHA-256 | |
| holdout / 训练窗合同 | |
| tokenizer | |
| sector 数（含 unknown） | 86（+1） |
| device / 精度 | |
| 环境变量快照 | `KRONOS_VALIDATION_QUICK_SAMPLES=` / `KRONOS_VALIDATION_SAMPLES=` / `KRONOS_CONDITION_ABLATION_INTERVAL_SEGMENTS=` |

---

## 2. P0-1 `sector_emb` 几何（排除 `unknown_sector_id=86`）

| run | ckpt SHA-256 | 形状 | init 类型 | ‖W‖_F | 每元素 RMS | 行范数 min / median / max | 行间余弦(off-diag) | cos(row, mean) | between-sector 占比 | 来源 | 注 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Small C2 best（Seg179） | | 87×512 | zero | | | | | | | | 已知：15.7 / 0.074 / 0.13 |
| Base A+C Seg5 | | 87×832 | Normal(0,0.02) | | | | | | | | 已知：5.387 / 0.0201 |
| Base 裸 Boot Seg155 | | 87×832 | zero | | | | | | | | **待测** |
| Base 裸 Boot Seg170 | | 87×832 | zero | | | | | | | | **待测** |
| Base 裸 Boot Seg175 | | 87×832 | zero | 1.6547 | **0.00615** | | | | | Monitor（见 results） | **已测（Monitor-only）；行几何不可测：ckpt 已被覆盖** |

**速查判定（每元素 RMS）**

| 区间 | 解释 | 允许的表述 |
|---|---|---|
| ≤ 0.005 | 条件路径未启动 | 不得引用 FS 作“CE 学不出身份”的证据 |
| 0.005 – 0.02 | 起步中 | 只能说“未达可比状态” |
| ≥ 0.02 且 `FS < 3·null` | 表已长、FS 仍贴零 | **此时才支持“CE 对身份无压力”** |

---

## 3. P0-2 同管重评（离线 harness，同一 2000 窗，K=8）

表头：`管线＝离线重评`（与 §4 的训练内 telemetry **分表**）

| 字段 | 含义 | Small C2 best | Base **durable last**（Seg175 退役） | Base A+C（若有） |
|---|---|---|---|---|
| `ckpt_sha256` | 权重指纹 | | | |
| `ckpt_snapshot` | 不可覆写快照路径（空=作废） | | | |
| `segment` | 段号 | 179（cosine 内） | durable last | |
| `lr_at_eval` | 该次 eval 的条件 LR | | | |
| `n` | 评测窗数 | | | |
| `windows_index_sha256` | 窗 index 指纹（证明同一批） | | | |
| `K` | 置换数 | 8 | 8 | 8 |
| `fs_mean` | `CE_full − CE_shuf` 均值 | | | |
| `fs_sd` | 跨置换 sd | | | |
| `null` | `CE(perm_a) − CE(perm_b)` | | | |
| `null_sd` | 跨置换 sd | | | |
| `ci95` | 配对 bootstrap 95% CI | | | |
| `per_window_delta_sd` | 逐窗 ΔCE 的 sd（算 SE 用） | | | |
| `fs_sector` | 换行业、保市值 | | | |
| `fs_size` | 换市值、保行业 | | | |
| `r_swap` | 换行业后预测 10d 分位变化率 | | | |
| `r_size` | 换市值后同上 | | | |

**G1 判定行**（逐 ckpt 填）

| ckpt | `FS ≥ max(3·|null|, 1e-3)` | 连续 ≥3 次同号 | CI 下界 > 3·|null| | G1 |
|---|---|---|---|---|
| Small C2 best | | | | |
| Base durable last | | | | |

**G2 判定行**

| ckpt | `r_swap ≥ 5%` | `r_swap ≥ r_size` | G2 |
|---|---|---|---|
| Small C2 best | | | |
| Base durable last | | | |

---

## 4. 训练内 telemetry（与 §3 分表，不得混算）

| segment | `lr_at_eval` | `forecast_loss`(full) | `condition_none_forecast_loss` | `condition_shuffled_forecast_loss` | `full−none` | `full−shuffled` | `condition_trunk_norm_ratio`（γ 前） | `condition_output_norm` | `trunk_input_norm` |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

> 提醒：现行 telemetry 只有 roll-1 单置换、无 null、无 `fs_sector`/`fs_size` 分解；**不可**直接当 P0-2 的结论用。

---

## 5. P0-3 冻结 trunk 探针：`FS*(γ, lr)`

trunk 全冻，仅训 `sector_emb` / `size_mlp`（+γ）；batch 64；步数 200–600；每个 cell 打 K=8 + null + 2×2。

| γ \ lr | 1e-4 | 3e-4 | 1e-3 | 3e-3 |
|---|---|---|---|---|
| 10 | `FS*`= / RMS= / `r_swap`= | | | |
| 30 | | | | |
| 100 | | | | |

**结论行（四选一，勾一项）**

- [ ] max `FS*` < 1e-3 → 该注入位点/损失**表达不出**身份 → **停长叉**，改“是否加 identity-sensitive 项 / 换注入位点”
- [ ] `FS*` 随 γ 升后塌 → **淹没确认**，取拐点 γ＝______
- [ ] 高 LR 侧才起且 `FS*` ≥ 5e-3 → 主因是**条件 LR** → 长叉降级为“条件 LR 调参”，先 50–100 段
- [ ] γ、lr 都弱 → 上界本就低 → 放弃“抬到 1e-2”，按天花板重设门

---

## 6. P0-4 预算实测

| 项 | 值 | 来源 |
|---|---|---|
| 单 seg 纯训练墙钟 | | |
| 单 seg 含 val + ablation 墙钟 | | |
| `updates/seg`（应 = ceil(20000/有效 batch)） | | |
| val 占比 / ablation 占比 | | |
| 本轮可用配额（GPU·h） | | |
| 可承载段数（含退火尾 ~200–267 段） | | 反推 |

> 冲突待解：脚本注释 `~29 min/seg`（`launch_bootstrap.py:39`）与“10-26 开训、10-27 到 Seg175”不一致，以实测为准。

---

## 7. 结论页（三条，填完上面才写）

1. **P0-1**：Seg175 `sector_emb` 每元素 RMS＝**0.00615（实测，Monitor-only）** → 判定：**起步中**；durable last 的完整行几何待补。
2. **P0-3**：`FS*` 上界＝______（γ＝____，lr＝____）→ 瓶颈：表达 / 优化 / 条件 LR / 信息上限。
3. **长叉**：开 / 停；若开 → γ＝____、预算 ____ 段（含退火尾）、预注册停看表见协议 §7。

**签署**：______（执行） / ______（复核）　日期：______

---

## 8. 禁止事项（复核时直接打回）

- 用训练内 telemetry 冒充实测重评，或把两条管线的数字放进同一张对比表。
- 用 master 的 hub 加载去评带 `condition_scale` 的 ckpt（γ 会被静默丢弃）。
- 用估计值填「实测」列，或用未填的格子支撑结论。
- 在峰值 LR 的段判 G1（未打 `lr_at_eval` 的判门视为无效）。
- 用 G1 判 margin 版的成功（margin 只认 G3）。
