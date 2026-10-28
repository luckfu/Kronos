# Base-Clean 阶段切换：Bootstrap → Main（2026-10-28 备妥）

用途：Bootstrap 到交接点时，照这一页切 Main。**当前状态：代码与推送件已备好，等交接点。**
（交接点原定 389 = chunk4 末段；若要更早/更晚，改 `KRONOS_MAIN_EXPECTED_BOOTSTRAP_COMPLETED` 即可。）

---

## 0. 一句话

Main = **只继承 Bootstrap 的 `last_model` 权重、重建 optimizer/scheduler、全参数峰值 1e-5、自己的 520 段 cosine**。
和小时候 Small 的 Main 同构；差别只在验证频率（见 §4）。

## 1. 和 Small 的 Main 对照

| 项 | Small Main | Base Main（本次） |
|---|---|---|
| 起点 | Bootstrap **best_model**（seg≈516） | Bootstrap **last_model**（Best 在 Bootstrap 被冻结，见 §3） |
| optimizer | 重建 | 重建（`last_state` 不继承） |
| LR | 全参数 **1e-5**，warmup 起点 1e-6，1% warmup + cosine | 相同 |
| batch | 64 | 64（micro4 × accum8 × 2） |
| 覆盖 | 534 段 × 20k | 520 段 × 20k（`ceil(10,397,679/20,000)`） |
| 验证 | **全量 123,836，每段** | quick **2048 每 5 段**（选 Best）+ 全量 **123,836 每 20 段** + 首/末段 |
| Best | 开，按全量 forecast | 开，按 quick-2048 forecast |
| 看板 | 新板 `small_0.1_main` | 新板 `base_clean_0.1_main` / run `base_clean_0_1_main` |

## 2. 切换命令（Bootstrap 那一棒 COMPLETE 之后）

```bash
# 1) 先 1 段 smoke（约 10 分钟，只验管线：父权重/新板/验证/首次 PROGRESS）
KRONOS_MAIN_MAX_SEGMENTS=1 KRONOS_MAIN_RUNTIME_BUDGET_SECONDS=1500 \
KRONOS_MAIN_EXPECTED_BOOTSTRAP_COMPLETED=389 \
python3 finetune/base_clean_bootstrap_relay/push_main.py --push --force
python3 finetune/base_clean_bootstrap_relay/monitor_chunk.py \
  --kernel luckfu/kronos-base-clean-0-1-main --follow --wait

# 2) smoke 通过后，推正式第一棒（10h 预算 / 上限 150 段）
KRONOS_MAIN_EXPECTED_BOOTSTRAP_COMPLETED=389 \
python3 finetune/base_clean_bootstrap_relay/push_main.py --push --force

# 3) 后续 Main 棒（同阶段接力，self-parent + last_state）
KRONOS_MAIN_CONTINUE=1 KRONOS_PARENT_KERNEL=luckfu/kronos-base-clean-0-1-main \
python3 finetune/base_clean_bootstrap_relay/push_main.py --push --force
```

Bootstrap 若还要续跑（比如交接点提前），棒长也改成墙钟制：

```bash
KRONOS_PARENT_KERNEL=luckfu/kronos-base-clean-0-1-bootstrap \
KRONOS_BOOTSTRAP_EXPECTED_RESUME=390 KRONOS_BOOTSTRAP_EXPECTED_COMPLETED=389 \
KRONOS_BOOTSTRAP_MAX_SEGMENTS=150 KRONOS_BOOTSTRAP_RUNTIME_BUDGET_SECONDS=36000 \
python3 finetune/base_clean_bootstrap_relay/push_bootstrap.py --push --force
```

## 3. 为什么 Main 从 last 而不是 best

Bootstrap 的 Best 被 `KRONOS_DISABLE_BEST_UPDATES=1` 冻结在 **seg21（2.69015）**，
而 seg353 实测已是 2.59366。用 best 等于退回 330 段前。
→ Main 必须从 `last_model` 起；Bootstrap 的 `best_model/` 在评估里一律不用。
（若要严格复刻 Small 的 "best→main"，得在 Bootstrap 末段开 Best 并挂 19,998 固定分层验证，
见 `20261028_chunk3_log_audit_cn.md` §6；当前决策是不做。）

## 4. 验证策略是**有意偏离** Small，必须写进结论

Small Main 每段跑全量 123,836。Base 上实测：1935 个 val batch（batch 64），
估算 forward-only ≈ 8–10 min/段；加上训练 ~4 min/段 → 12 min/段 → 520 段 ≈ 104h，**不可行**。

所以 Base Main：
- **quick 2048**（固定子集，seed=101 的前 2048）每 5 段一次 → Best 选择口径（配对比较，噪声远小于绝对 SE）；
- **全量 123,836** 每 20 段 + 首/末段 → 里程碑指标（可用于跨阶段对比，但**不是**选模口径）。
- 因此 **Base 的 "best" 与 Small 的 "best" 口径不同**，对外比较时不能直接比数值，只能比同口径曲线。

## 5. 检查点与回滚

- 每棒结束必落：`checkpoints/{best,last}_model`、`last_state.pt`、`metrics.jsonl`、`progress.json`、`summary.json`、`base_clean_manifest.json`。
- Main 第一棒若在 30 分钟内失败 → 直接重推（`--force`），不消耗有效配额太多。
- 若 Main 前 20 段的 forecast 明显**高于** Bootstrap 末段（比如 +0.02 以上）→ 先查 LR/验证口径，不要继续烧。
- 回滚：Bootstrap 的 Output 永远保留；Main 任何一棒都可丢弃重来（它只依赖 Bootstrap last_model）。

## 6. 代码落点

- 入口：[`../launch_main.py`](../launch_main.py)（父权重发现 + 权重重载 + 墙钟预算 + Main 验证默认）
- 推送：[`../push_main.py`](../push_main.py)（REST 优先；`--push --force` 才提交）
- runner 的 Main 验证默认：[`../../kaggle_kronos_base_clean.py`](../../kaggle_kronos_base_clean.py) `build_environment(..., stage="main")`
- 监控：[`../monitor_chunk.py`](../monitor_chunk.py)（`--kernel ... --follow` 对任何 slug 通用）

---

## 7. Main smoke 实测（2026-10-28 11:17Z 推送 v1，`max_segments=1`）

| 检查点 | 实测 |
|---|---|
| 父权重 | `parent_ready`: `/kaggle/input/notebooks/luckfu/kronos-base-clean-0-1-bootstrap/.../base_clean_0.1_bootstrap`，`approach=weights_only` |
| optimizer | `optimizer_state_restored=false`（未带 `last_state`，重建） |
| 学习率 | `Segment 1/520, Step 100/313`: **Adaptation LR = Condition LR = 1.553e-06**（统一峰值计划，非 Bootstrap 双 LR） |
| 数据 | 10,397,679 窗口 / 20000 per seg，`data_manifest` SHA 与 Bootstrap 一致 |
| 看板 | `base_clean_0.1_main`（新板，无 replay） |

即：**阶段切换的三件事（父权重=last、optimizer 重建、统一 LR）都按设计生效**。

实测耗时（修正 §4 的估算）：

| 量 | 实测 |
|---|---|
| seg1 总耗时 | **9m14s**（313 步训练 ≈3.5 min + 全量 123,836 验证 ≈5 min + 消融/保存） |
| 全量验证 | 968 个 batch（×2 rank），**≈5 min/次**（比先前估算的 8–10 min 快一倍） |
| Best | 已写入：`best=2.59125`（quick-2048 口径），Best 更新确认开启 |
| 衔接 | Bootstrap 末段 2.59049 → Main seg1 **2.59074**，同量级（LR 重启 warmup，未出现跳变） |
| 空转段 | 无全量验证的段预计 ~3.5–4 min/seg → 10h 预算 ≈ **145–150 段**，与设计一致 |

→ 每 20 段跑一次全量（约 5 min）相当于 +15s/段均摊，代价可以接受，暂不改频率。
