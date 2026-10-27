# Kronos 条件身份：Small PASS vs Base 未对齐对照（导师审阅稿）

日期：2026-10-27（Asia/Shanghai）  
作者侧：量化交易模型大师 / 群讨论纪要  
仓库：`luckfu/Kronos`  
状态：**已拍板 → 执行口径见 [`20261027_cond_identity_gate_protocol_cn.md`](20261027_cond_identity_gate_protocol_cn.md)**；未开新长叉；live Bootstrap 保持无 γ 裸加  
批注：本文为**审阅原稿**，保留以便对照；自 §8 起不再作为执行依据。

---

## 1. 问题

同一套行业 + 市值百分位条件注入、同一 forecast CE，为何：

- **Small Cosine C2 Best（Seg179）** 离线 Full / None / Shuffled 双门 **PASS**；
- **Base clean** 短 smoke（A+C±margin，~5 seg）与裸 Bootstrap（至 ~Seg170–175）仍 **Full−Shuffled ~1e-5 量级**，内容门不过？

用户直觉：Small 训练应是 Base 的复刻。需要判断是「Base 上 CE 学不出身份」还是「对照不齐 / 路径未走到」。

---

## 2. 双门定义（现行）

同时满足才算 Cond 有可用身份：

1. **能量门**：`trunk_ratio ≥ 0.02`（条件输出相对主干输入范数）；
2. **内容门**：`|Full − Shuffled|` 相对 ~1e-5 带抬升约一个数量级（勿称「噪声」；未过时写 **微弱且远未可用**）。

辅助几何：`sector_emb` 行是否离开 init（Frobenius / 行间余弦）；`size_mlp` 跨分位坍缩 **不是** 内容门失败的充分条件（Small PASS 时 size_mlp cos≈0.977 仍偏塌）。

---

## 3. 关键结果

### 3.1 Small C2 离线（PASS）

| 项 | 值 |
|---|---|
| Checkpoint | `user281434/kronos-small-0-1-cosine-c2-best-seg179` |
| SHA-256 | `4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a` |
| 配方 | bare-add，**无 γ**；血缘 Boot→Main→Extend→WC→WC→Cosine |
| Eval | n=2000 quick（默认 `KRONOS_VALIDATION_SAMPLES=2000`），非全量 123836 |
| Full / None / Shuffled | 2.28314 / 2.29189 / 2.28527 |
| Full−None / Full−Shuffled | ≈ −8.76e-3 / **−2.13e-3** |
| trunk_ratio median | **≈0.065** |
| dual-gate | **PASS** |
| sector ‖W‖_F / pairwise cos | ≈15.7 / ≈0.13（已散开） |
| size_mlp 跨分位 cos | ≈0.977（仍偏塌） |

Kernel：`user281434/kronos-small-c2-cond-identity-ablation`。

### 3.2 Base A+C(+margin) 短 smoke（内容 FAIL）

| 项 | 值 |
|---|---|
| 配方 | Fix **A**：γ₀≈25；Fix **C**：condition last-layer `Normal(0,0.02)`；±margin 烟另开 |
| 预算 | **~5 segments** |
| trunk_ratio | ≈0.0205（能量门刚过） |
| Full−Shuffled | ≈ **−3e-5**（微弱且远未可用） |
| dual-gate | **FAIL**（trunk PASS，内容 FAIL） |
| sector ‖W‖_F | ≈5.39 ≈ init（几乎未训动） |

结论：短 smoke **不能**证伪「长训后有身份」；只能说明开局能量可抬、内容未立。

### 3.3 Base 裸 Bootstrap live（至 Seg175，仍无内容）

| 项 | 值 |
|---|---|
| Kernel | `user281434/kronos-base-clean-0-1-bootstrap` v7 |
| 配方 | **无 γ / bare-add**（`use_condition_scale=False`）；双 LR |
| Seg170 | Full−None≈−1.18e-4，Full−Shuffled≈−5.90e-5；trunk≈4.06e-3 |
| Seg175 | Full−None≈−8.94e-5，Full−Shuffled≈−1.34e-5；trunk≈4.11e-3（~1.04× vs Seg155≈3.93e-3） |
| 判定 | 内容仍 **微弱且远未可用**；trunk 无物质跳升 |

SwanLab：`base_clean_0.1_bootstrap`。Bootstrap **继续裸跑、不并进 γ**。

---

## 4. 为何不是「复刻失败于 CE」

对照点不对齐（证据最强）：

| 维 | Small（PASS ckpt） | Base（当前对照） |
|---|---|---|
| 阶段 | Cosine-best **Seg179**（长血缘末端） | Boot **~Seg170–175** 或 A+C **Seg5 smoke** |
| 容量 | 8L / 512d / ~25M | 12L / 832d / ~102M（≈4×） |
| context_layer | 6 | 10 |
| 条件路径 | bare-add，无 γ | Boot 同；smoke 为 A+C（γ+非零 init） |
| 数据/损失 | 同 holdout SHA `b85ccc45…`、tokenizer-base、CE+0.02·history | 同 |
| Eval 量级 | quick n=2000 | Boot/smoke 条件消融多为 QUICK≈2048 |

**暂估（非定论）**：Base 尚未走到 Small 那种「Cond 有内容」的状态；更可能是 **规模淹没 + 缺足够长的 A+C 路径**，不是「CE 在 Base 上必然失效」。

**不能推出**：「再裸跑久一点就会有」——裸 Boot 已至与 Small Bootstrap 同量级段数，FS 仍 ~1e-5。

---

## 5. Fix A / Fix C 含义

- **C**：condition 最后一层 `Normal(0, 0.02)` 初始化（相对零初始化开局 no-op）。
- **A**：可学 `condition_scale`（γ），起步 γ₀≈25，减轻大 trunk 淹没。
- **margin / λ**：短烟内容已 FAIL；长叉 **默认不带**，除非另行点头。

---

## 6. 拟议下一实验（待拍板，**现在未开**）

开叉前二选一必须写死：

1. **Bootstrap durable 续跑 + 只加 γ**  
   - C 的 Normal init 会被 ckpt 盖掉 → 实际是「裸训中途加 γ」。  
   - 省算力，但与旧 Cond 纠缠，否了也说不清。

2. **HF Base 冷启（或显式重 init last-layer）真 A+C 长训**（**倾向锁此项**）  
   - 干净验「淹没 + 路径」。  
   - 独立 kernel，**不并进** live Bootstrap。  
   - 训到 sector 明显离开 init（Small ‖W‖_F≈15 量级），再同一套双门。  
   - 不对齐「再开 5seg smoke」。

群内倾向 **(2)**；**开不开、对齐多少 seg、是否带 margin → 等负责人拍板**。

---

## 7. 现行约束（执行侧）

- Small 生产冻结：C2 为固定 Small 基线；**HOLD e01j**；真 §5.2 等未用 2026-10-09→10-23 bars。
- Base live Bootstrap：**保持无 γ 旧配方**，禁止 smoke 环境变量污染 launch/push。
- 不再堆 λ / 条件 LR 烟雾作为主线。

---

## 8. 请导师审阅的点

> **导师批注（2026-10-27，四项均已裁）：** 双门口径、n=2000、冷启长叉、margin 的结论与**替换后的门定义/锚点/P0 清单**见 [`20261027_cond_identity_gate_protocol_cn.md`](20261027_cond_identity_gate_protocol_cn.md)；P0 结果按 [`20261027_p0_cond_identity_fill_in_cn.md`](20261027_p0_cond_identity_fill_in_cn.md) 填报复核。  
> 一句话：现行双门只能作内部 triage（能量门降为遥测/护栏）；长叉方向同意但**暂不开**，须先过 P0-1/2/3；先无 margin，且 margin 版不得用 G1 判成功。  
> 下文 1–4 为原提问，**保留不改**。

1. 双门与「微弱且远未可用」表述是否可接受为论文/实验协议口径。  
2. Small n=2000 quick PASS 是否足以支持「CE 可学出身份」的对称破缺论述（是否要求全量 holdout 复测）。  
3. 下一刀是否同意 **HF 冷启真 A+C 长叉**，以及建议最小段数 / 中期停看指标（sector ‖W‖_F、trunk_ratio、Full−Shuffled）。  
4. 是否允许在长叉中加入 margin，或必须先无 margin 跑通内容门。

---

## 9. 附件索引（同仓库 / 实验笔记）

- `finetune/reports/20261027_cond_identity_gate_protocol_cn.md` — **导师裁定 + 替换后的门定义 / 锚点 / P0 清单（执行依据）**  
- `finetune/reports/20261027_p0_cond_identity_fill_in_cn.md` — P0 落盘空表模板（填完提交复核）  
- `finetune/SMALL_0_1_MODEL_AND_TRAINING_TECHNICAL_REPORT_CN.md` — Small Bootstrap=1% warmup+cosine 双 LR 等血缘  
- `finetune/reports/base_clean_train_plan_cn.md` — Base 计划与 cl=10 等差异  
- `finetune/base_clean_bootstrap_relay/notes/20261027_sector_emb_collapse_probe.md`  
- `finetune/base_clean_bootstrap_relay/notes/20261027_condition_ce_identity_audit.md`  
- Small 离线结果 JSON（Kaggle output）：`cond_identity_ablation_result.json`（Seg179）

