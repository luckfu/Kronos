# 条件身份：跨机交接（2026-10-27 收盘）

用途：换机器接手时只看这一页。执行口径以 `20261027_cond_identity_gate_protocol_cn.md` 为准，本文只是索引 + 待办。

**下一步的单独交接**：`20261027_cond_identity_next_step_handover_cn.md`（补 Small 低 Δ 参照点，含改哪 4 处、今天踩过的 6 个坑、payload 重建脚本）。

---

## 0. 一句话现状

新增的行业/市值条件层**两个都在工作**，但**身份只占条件收益的 25%，其中行业占 6%、市值占 18%**；剩下 75% 是一个全市场共享的常向量（对身份零贡献）。行业通道**不是没学到，是权重给得太小**——增益拉到 18× 仍不饱和，但纯 CE 的最优解就钉在 1 倍。

**2026-10-27 晚更新**：Base 与 Small 在低 Δ 端**都没有可测身份**（Small@Δ=0.0167 的 `fs_sector` = −9.2e-6，在 3e-5 地板以下）。**“Base 落后有结构”的假设被否，争论收口**，配方不动，继续裸跑。见 §7.2。

---

## 1. 今天落地的三个实测（都已进仓库）

| 实测 | 结论 | 关键数 | 产物 |
|---|---|---|---|
| **2×2 分解**（Small C2） | 行业真实存在，但比市值小 | `fs_sector −5.58e-4`(26%)、`fs_size −1.56e-3`(72%)、可加、行业 12σ、null ≤4.7e-5 | `20261027_cond_identity_2x2_small_c2.json` |
| **剂量反应**（Small C2） | 行业不饱和；CE 把 γ 钉在 1；市值已近最优 | γ=2 → 2.1×、CE+1.8e-4；γ=4 → 4.2×、CE+1.8e-3；γ=16 → 18.1×、CE+5.1e-2；β=0 → 行业仅 +30% | `20261027_cond_identity_sector_gain_sweep.json` |
| **轨迹**（Small 6 点 / Base 逐 10 段） | 每窗口位移率两边一样（1.56 vs 1.60e-9，差 2.5%），12× 差距 = 数据暴露差 | Small 51.8M 窗口 → Δ 0.0749；Base 4.8M → 0.0077 | `20261027_cond_delta_trajectory.json` |

对应 kernel（Kaggle，user281434）：

- `kronos-small-c2-cond-identity-2x2`（v1，COMPLETE，16 s）
- `kronos-small-c2-cond-sector-gain-sweep`（v1，状态 ERROR，但**8 个格子全部打印**，仅最后写 JSON 时用了作用域外变量；数据完整，重跑需把 `total` 提到函数外）

---

## 2. 数字速查（免得在新机器上重算）

| 量 | Small C2 | Base（bootstrap，采样时约 Seg240–250） |
|---|---|---|
| 累计段 | 2592 | 240 |
| 已见窗口 | 51.8M | 4.8M |
| Δ（每元素 RMS） | 0.0749 | 0.0077 |
| 每窗口位移 | 1.56e-9 | 1.60e-9 |
| `Full−None`（常数为主） | −8.76e-3 | −1.03e-4 |
| `Full−Shuffled`（身份） | −2.16e-3 | −3.62e-5 |
| `fs_sector` / `fs_size` | −5.58e-4 / −1.56e-3 | **未测** |
| 身份占条件收益 | 24.7% | 35%（同形状） |

注：Base 的 `fs_*` 未拆，因为 2×2 需要权重；权重现在只在 kernel output 里，且该 kernel 正在跑（RUNNING 时 output 读不到）。

---

## 3. 下一步（按成本，别跳步）

1. **零训练 · 今天就做**：C2 上推理侧 `γ_sector ∈ {1, 1.5, 2, 3, 4}`，**用下游指标选**（Rank IC / OOS / 行业中性化后增量），不用 CE 选。
   - 若 γ\* 显著优于 1 → 进第 3 步；若无响应 → **停**，行业通道不值得再投入。
2. ✅ **已完成**：Base 的 2×2（§7.1，Seg281）与 Small 低 Δ 的 2×2（§7.2 Δ=0.0167、§7.4 Δ=0.0269）。**判读：Base 落在低 Δ 噪声平台内，不显著低于曲线，争论收口，不动配方。**
3. **只在第 1 步通过后**：重训，γ 固定为 γ\*（**不要做成 learnable**，CE 会把它压回 1），行业 LR 可选 2–5× 仅作加速。
4. ✅ **P0-5 结项**（§7.4）：4 个 Δ 点，`Δ ≤ 0.027` 的 `|fs_sector|` 上界 1.2e-5、全部在地板下；C2（0.0749）是唯一可分点。**P0-3 不启用。**

---

## 4. 已被数据推翻 / 作废的说法

| 旧说法 | 现状 |
|---|---|
| “条件身份在 Base 上不生效” | 无依据；Base 的形状与 Small 同源，只是数据看少了 12× |
| “大 trunk 把条件信号淹掉” | 每窗口位移率两边差 2.5%，淹没在速率维度不成立 |
| “行业层白学” | **否**：行业 12σ、占身份 26% |
| “能量门 trunk_ratio 0.065 说明有身份” | **否**：其中 95% 是恒定 size bias，在 FS 里严格抵消 |
| “2048 个验证样本不够，要 3 万” | **否**：实测 `SE(FS)≈1e-5`，2048 够 |
| “A/C 两条修复（γ、非零 init）是默认方案” | **已砍**：零初始化被 Small 自己证否 |

---

## 5. 新机器上要准备什么

- **凭据（不进仓库）**：`KAGGLE_API_TOKEN`（user281434，用于 push/pull kernel 与读 output）、`SWANLAB_API_KEY`（拉日志）。本机放在 `.kaggle_home/.t_user281434` 与 `.kaggle_home/.swanlab_key`，该目录被 gitignore（`*`），**换机后需重新提供**。
- **Python**：仓库自带 `.venv-swanlab-api`（kaggle CLI 2.2.4 + swanlab 0.10.1），拉日志用：
  `KAGGLE_API_TOKEN=... .venv-swanlab-api/bin/kaggle kernels logs <ref>`
- **SwanLab 取数**：曲线里**没有** `sector_embedding_weight_norm` 与 ablation 指标，只有 **logs** 里有（`Condition Monitor JSON` / `Validation Condition Full/None/Shuffled`）。别再翻曲线。
- **改 kernel 的流程**（源码不在仓库，靠 kernel 自带的自解压 payload）：
  1. `kaggle kernels pull <ref>` 得到 wrapper；
  2. wrapper 里 `base64` 的 zip 解出仓库子集（`model/`、`finetune/`、`scratch/<实验>/launch_*.py`）；
  3. 改 `launch_*.py` → 重新 zip → base64 → 替换 wrapper 里的 payload **并同步更新那行 sha256 断言**；
  4. 写 `kernel-metadata.json`（`dataset_sources` 至少含 holdout + 权重数据集），`kaggle kernels push -p <dir>`。

---

## 6. 不要做的事

- 不要用 master 的 hub 加载去评带 `condition_scale` 的 ckpt（`strict=False` 会静默丢 γ）。
- 不要把 `lr_at_eval` 没打标签的 eval 当判门依据（§3 A3）。
- 不要用 G1 判 margin 版的成功（循环论证，只认 G3）。
- 不要在峰值 LR 段判门；不要用段号当锚（用已见窗口数，§4.2①）。
- 不要把 CE 份额当经济重要性读（须按注入变异归一，2×2 那一节）。

---

## 7. Base γ 扫描：**已跑完**（v2 COMPLETE）——结论在 §7.1

> 状态：`user281434/kronos-base-cond-sector-gain-sweep` **v2 COMPLETE**（v1 因 `SyntaxError` 立即失败，已修）。结果见 §7.1，原始数在 `20261027_cond_identity_2x2_base_seg281.json`。

**它做什么**：把 Base 那个 chunk 的 `last_model` 挂进来，在同一个 2000 窗验证集上做 3 件事——
1. Base 的 **2×2 分解**（`fs_sector` / `fs_size` / 置换零假设）；
2. **行业增益扫描** α ∈ {1, 1.5, 2, 3, 4, 8}（sector 向量乘 α）；
3. 一个 **β=0.5** 对照（压市值）。

每格 3 次前向 ≈ 6 s，总计算 ~1 分钟；大头是 torch 安装（~8 分钟）。

**取结果**

```bash
export KAGGLE_API_TOKEN=<user281434 的 token>   # 不在仓库里，需重新提供
.venv-swanlab-api/bin/kaggle kernels status user281434/kronos-base-cond-sector-gain-sweep
.venv-swanlab-api/bin/kaggle kernels logs   user281434/kronos-base-cond-sector-gain-sweep > /tmp/base_sweep.log
```

日志里找这几行（每格一行）：

```
{"phase": "decompose_done", "fs_sector": ..., "fs_size": ..., "null_sector": ...}
{"phase": "sweep_cell", "alpha_sector": 1.0, "beta_size": 1.0, "ce_full": ..., "fs_sector": ..., "fs_size": ...}
```

**怎么读（这才是重点）**

| 观测 | 含义 | 动作 |
|---|---|---|
| α=2~3 时 **`ce_full` 下降** | Base 现在**注入不足**，抬增益是白赚（既得身份又降 loss） | 直接给 Base 定 γ，并可进重训 |
| `ce_full` 随 α **单调上升** | 它已到自己的 CE 最优点 | 别动，等它自然长 |
| α=1 那行的 `fs_sector` 占 `fs_both` ≈ 26%/72% | 与 Small 同形状 → “同一条曲线上的不同位置” | 复刻可收口 |
| `fs_sector` 明显 < Small 的 26% | 行业那一路在 Base 上被额外压住 | 才值得查注入位点 |

**参考基线（Small C2）**：`fs_sector −5.58e-4`(26%)、`fs_size −1.56e-3`(72%)、`full−none −8.76e-3`。
**Base 当前态（Seg281）**：Δ 0.00844、`trunk_ratio` 5.19e-3、`full−shuffled −2.07e-5`（在噪声底附近，所以 **fs 只能当参考，主看 CE**）。

**若 v2 又失败**：先看 logs 里的最后一条 Traceback。已知的两处适配点（都在 `base_launch.py` 里）：
- `context_layer` 必须 **10**（Base 是 12 层注入第 10 层；Small 是 6）；
- 权重从 `kernel_sources` 挂载（`**/checkpoints/last_model/model.safetensors`），不走 dataset。

**这一跑不碰任何生产代码**：只读权重 + 只做前向，不改 `model/`、`finetune/`、relay。

### 7.1 结果（Base Seg281，2026-10-27）

| 量 | Base@Seg281 | Small C2（参照） |
|---|---|---|
| Δ（每元素 RMS） | 0.00843 | 0.0749 |
| 行业相对注入 | 0.017% | 0.332% |
| `Full−None` | −1.04e-4 | −8.76e-3 |
| `Full−Shuffled` | −4.0e-5（复测 −1.4e-5） | −2.16e-3 |
| `fs_sector` | **+1.2e-5**（符号反，且小于自身零假设 1.5e-5） | −5.58e-4（26%） |
| `fs_size` | −9.6e-6（约为零假设的 2 倍） | −1.56e-3（72%） |

**四条读数**

1. **Base 现在没有任何可测的身份效应**——sector 与 size 都在噪声里。
2. **“置换零假设”低估了真实噪声**：同一个 `both-shuffled` 量测两次差 **2.7e-5**（−4.0e-5 vs −1.4e-5），是置换零假设（1.8e-6）的 15 倍。所以 Base 的判读地板是 **~3e-5**，不是 1.8e-6。→ 协议 §2 的零假设口径需要按机型加一条“复测一致性”检查。
3. **8× 行业增益拉不动它**：`fs_sector` 仍在噪声；CE 只降 2.5e-5（α=4）。说明行业向量现在更像一个小的偏置修正，不是身份通道。
4. **按单位注入归一，Base 的行业通道比 Small 弱 ~11 倍**：α=8 时相对注入 1.34e-3（Small 3.32e-3），效率 0.015 vs 0.168。

**关键缺口**：Base 的 Δ=0.0084 **低于 Small 最早可测的点**（cum 516 时 0.0167）。所以“还没长成”和“长不成”现在分不开——**需要一个低 Δ 的 Small 参照点**。

> ✅ **已补齐（2026-10-27 晚）**：见 §7.2。结论是 **Base 正常，争论收口**。

---

### 7.2 Small @ Δ=0.0167 的 2×2：**命中“噪声”支，争论收口**（2026-10-27 晚）

**做了什么**：把 `user281434/kronos-base-cond-sector-gain-sweep` v2 的 kernel 源码解包，按 next-step 交接 §3 改 4 处（`kernel_sources` → Small Bootstrap 末态；glob → `best_model`；路径关键词 → `small_0.1_bootstrap`；`context_layer` 10 → 6），其余照抄（`num_sectors=86`、`use_size_percentile=True`、`use_condition_scale=False`、n=2000、batch 64、α 网格 {1,1.5,2,3,4,8}、β=0.5 对照）。

**跑在哪**：`wynstonliu/kronos-small-bootfinal-cond-2x2`（v1，COMPLETE，纯前向 ~32 s + 装 torch ~6 min）。

**权重**：`wynstonliu/kronos-small-0-1-bootstrap-final` → `checkpoints/best_model`，`sha256 f21eeed1…ae630`，`sector_emb ‖W‖_F = 3.5227194` → **Δ = 0.016691**（正是预注册的那个点）。

#### 数字

| 量 | Small@Δ=0.0167（本跑） | 地板 `max(3·null, 3e-5)` | Base@Δ=0.0084 | Small C2@Δ=0.0749 |
|---|---|---|---|---|
| `fs_sector` | **−9.23e-6** | 3e-5 | +1.21e-5 | −5.58e-4 |
| `fs_size` | **−3.64e-5** | 3e-5（自身 3·null 4.53e-5） | −9.55e-6 | −1.56e-3 |
| `fs_both` | −2.72e-5 | 3e-5 | −1.37e-5 | −2.16e-3 |
| null (sector/size/both) | −1.62e-6 / +1.51e-5 / −7.03e-6 | — | −1.52e-5 / −4.84e-6 / −1.83e-6 | ≤4.7e-5 |
| `Full−None` | −6.44e-4 | — | −1.04e-4 | −8.76e-3 |
| 身份占条件收益 | 4.2% | — | 13% | 24.7% |
| `trunk_ratio` | 1.04e-2 | — | 4.99e-3 | 6.5e-2 |

#### 判读（照 next-step 交接 §1/§6 的预注册规则，未自由发挥）

规则二选一：`fs_sector` 有 −2e-4 量级 → Base 落后有结构；`fs_sector` ≤3e-5 → Base 在正常轨道。

实测 `|fs_sector| = 9.2e-6`，**比 3e-5 地板低 3.3 倍，纯噪声**。→ **命中第二条**：

> **Base 不异常。** Small 自己在 Δ=0.0167 这个点也没有可测的行业身份。Base@Δ=0.0084 的 `+1.2e-5` 不是“被压住”，而是**同一条曲线更靠前的位置**。

四条读数：

1. **不要为“Base 落后”动 γ / 注入位点**——前提被否了。协议 §4.4 的决策路径停在第一步之前。
2. **`fs_size` 也不可分**：−3.64e-5，卡在自身 3·null（4.53e-5）与全局地板（3e-5）之间。这个 Δ 上**两条通道都没起来**，不像 C2 那样市值明显强于行业。
3. **α=8 仍拉不动**：`fs_sector` 只到 −3.32e-5（还在楼板），CE 降 8.1e-5。与 Base 同一签名——低 Δ 时行业向量只是个小偏置修正。
4. **身份对 Δ 是强超线性的**：Δ ×4.5（0.0167→0.0749）换来 `fs_sector` ×60（−9.2e-6→−5.58e-4），隐含指数 ≈2.7。**所以“同样的每窗口位移率”不能线性外推身份**——这正好解释了为什么 Base 现在什么都量不到。

#### 落盘

`finetune/reports/20261027_cond_identity_2x2_small_bootfinal.json`（含完整 ce 八元组、α 扫描 7 格、地板口径、三源交叉验证、以及下面这条更正）。

#### ⚠️ 更正：next-step 交接 §2 指错了 kernel

原文说 Bootstrap 末态在 `luckfu/kronos-small-0-1-bootstrap`。**实际那个 kernel 是个 1 段的 timing 探针**——SwanLab run 名 `small_0.1_bootstrap_timing`，日志写着 `Chunk limit reached after 1 segment(s)`。它的 `best_model` 只有 `‖W‖_F = 0.01878` → **Δ = 8.9e-5，小了约 187 倍**。照它跑会得到一个毫无意义的“Small 低 Δ 点”。

真实的 Bootstrap 链是：

```
luckfu/kronos-small-0-1-bootstrap        (1 段 timing, 不是末态)
  └─ luckfu/kronos-small-0-1-chunk-1c    → seg 241
     └─ luckfu/kronos-small-0-1-chunk-2  → seg 386
        └─ wynstonliu/kronos-small-0-1-chunk-3          → seg 531
           └─ wynstonliu/kronos-small-0-1-bootstrap-final → seg 532–534  ← 末态在这
```

**三源交叉验证**（都指向同一个权重）：
1. 本跑加载 `bootstrap-final/best_model`，读到 `‖W‖_F = 3.522719383`；
2. `wynstonliu/kronos-small-0-1-main-timing` 显式声明 `stage2_parent … /bootstrap-final/checkpoints/best_model`、`selection: bootstrap_best`，其第一个 Monitor 点 = **3.5229811668395996**；
3. `20261027_cond_delta_trajectory.json` 的 `bootstrap_end.W_F` = **3.5229811668395996**，与 2 逐位相同。

**作废产物**：`luckfu/kronos-small-bootstrap-end-cond-2x2` v1（COMPLETE）是照错权重跑的，全在 1e-6 噪声，**不要引用**。该 kernel 删除被拒（403），以私有状态留作痕迹。

**另一处小坑**：跨账号 `kernel_sources` 挂不上。用 `user281434` 推、挂 `luckfu/…` 时，Kaggle 静默把 `kernel_sources` 写成 `[""]`，kernel 照样起跑但一定在 `resolve_c2` 报 `no best_model found`。**挂载方必须与 kernel 属主同账号**（本次最终用 `wynstonliu` 推、挂 `wynstonliu` 的 output）。

---

### 7.3 收口之后该干什么

按协议 §4.2 的预算表：Base 到 Small Bootstrap-final 的 Δ（0.0167）还要 **~520 段**，到 Small C2 的 Δ（0.0749）要 **~2,325 段**。**继续裸跑，别动配方**（γ / 注入位点 / margin 都没有依据）。

---

### 7.4 P0-5 第三点（Δ=0.0269）与 `FS–Δ` 曲线现状（2026-10-27 深夜）

**跑在哪**：`luckfu/kronos-small-mainfinal-cond-2x2`（v1，COMPLETE，纯前向 ~16 s）。权重 `luckfu/kronos-small-0-1-main-final/checkpoints/best_model`，`sha256 8d0752f3…f674`，`‖W‖_F = 5.6877813` → **Δ = 0.026949**（cum 1066）。

> 注：`wynstonliu/kronos-small-0-1-main-timing` 只有 1 段，其 `best_model` 与 Bootstrap 末态几乎相同（3.5229→3.5239），**不能**当第三点。cum 1066 的末态在 `luckfu/kronos-small-0-1-main-final`。

#### 四点全表（同 harness / n=2000 / 同置换口径）

| 点 | Δ | `fs_sector` | 自身地板 `max(3·null, 3e-5)` | `fs_size` | 自身地板 | 段数 |
|---|---|---|---|---|---|---|
| Base Seg281 | 0.00843 | +1.21e-5 | 4.56e-5（**0.26×，噪声**） | −9.55e-6 | 3e-5（0.32×） | 281 |
| Small Bootstrap-final | 0.01669 | −9.23e-6 | 3e-5（**0.31×，噪声**） | −3.64e-5 | 4.53e-5（0.80×） | 534 |
| **Small Main-final** | **0.02695** | **−2.55e-6** | 5.89e-5（**0.04×，噪声**） | **−5.13e-5** | 4.36e-5（**1.18×，刚过线**） | 1066 |
| Small C2 | 0.07489 | −5.59e-4 | 1.40e-4（**4.0×，可分**） | −1.56e-3 | 9.02e-5（17×） | 2505 |

#### 结论（这一轮能给的与不能给的）

**能给的：**

1. **P0-5 的点数条件满足**：4 个不同 Δ，同一 harness、同一 n、同一置换口径。
2. **低 Δ 段是一个可引用的上界，不是一条拟合曲线**：`Δ ≤ 0.027` 时 `|fs_sector| < 1.21e-5`（三点包络），**始终在各自地板之下**。行业身份到 Δ≈0.03 仍不可测。
3. **市值通道先出来**：Δ=0.0269 时 `fs_size −5.13e-5` 首次越过自身地板（1.18×），且占 `fs_both` 的 94%。这与 C2（市值是行业的 3×）方向一致。
4. **可加性成立**：`fs_sector + fs_size = −5.39e-5` vs `fs_both = −5.46e-5`，残差 +6.6e-7。

**不能给的（重要，别外推）：**

- **不能拟合指数。** 三点全在噪声里、且 `|fs_sector|` 反而从 9.2e-6 降到 2.6e-6（都是散点，不是下降）。任何用这三点算出的 `p` 值都是拟合噪声。§7.2 里那个“指数 ≈2.7”只用了 Bootstrap-final 与 C2 两点，**标注为不稳**，正式结论只保留定性说法：**Δ≈0.03 以下无行业身份，上界 1.2e-5**。
- **不能声称曲线“已经画好”。** 要在 Δ<0.03 拿到可分点，需要更大的 n（当前 `SE(FS)≈1e-5`，而信号 <1e-5）或一个已经把通道长起来的模型。**这条路成本不低，且没有决策价值**——因为 Base 已经在噪声区的正中间，补不补低 Δ 点都不会改变“继续裸跑”的结论。

#### 所以 P0-5 的定性判法

协议 §5 要求 P0-5 回答“Base 是否显著低于 Small 曲线”。现在可以定答：

> **Base 不显著低于曲线，而是落在低 Δ 噪声平台内。** 低 Δ 平台（≤0.027）的实测包络是 1.2e-5，Base 的 +1.2e-5 正好就是这个包络的上沿。**P0-5 结项，判“落在曲线上”。** P0-3（冻结 trunk 探针）按 §6 的触发条件**不启用**。
