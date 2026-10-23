# Sealed OOS 交接（2026-10-23）

**用途**：换到办公室电脑继续处理时，从这份文档接上。

## 当前代码

```text
commit 4d48db8
feat(small-sealed-oos): add 42-day OOS eval kernel; sync local docs/console state
origin/master 已同步
```

办公室电脑：

```bash
git pull origin master
```

## 已完成的任务

```text
kernel: wynstonliu/kronos-small-sealed-oos-20260803-20260930
版本:   v3
状态:   COMPLETE（2026-10-23 09:14 最后一次运行）
配置:   42 天全量，C2 best + E03b best，sample_count=1，双 T4
```

**判定已完成**，见 [`SEALED_OOS_RESULT_20260918_20260930_CN.md`](SEALED_OOS_RESULT_20260918_20260930_CN.md)；
机器可读记录见 [`finetune/reports/small_sealed_oos_20260918_20260930.json`](../finetune/reports/small_sealed_oos_20260918_20260930.json)。

| §3 规则 | 结果 |
|---|---|
| §3.1 C2 alpha | IC **0.06523** → 档位 **0.05–0.10**（弱但非零）→ 写第 8b 步，只做 `\|pred_d10\| ≥ 5%` 子集 + 事前流动性过滤 |
| §3.2 e03b 晋级 | **不通过，关线**。e03b IC 0.05247 < C2 0.06523；配对 Δ=-0.01276，CI [-0.03926, -0.00420] |
| §3.3 TimesFM overlay | **未判定**（SweetSpot native / C07 SWA 的 sealed OOS 尚未评测） |

> `sealed_first_use=true`：该窗口已用掉一次，之后任何候选再用即视为已污染。

页面：

https://www.kaggle.com/code/wynstonliu/kronos-small-sealed-oos-20260803-20260930

跑完后 Output 里有：

```text
/kaggle/working/small_sealed_oos/summary.json
  full_42d          # 42 天全量 context
  first_use_9d      # 09-18 → 09-30 first-use 判定
  paired_full       # 42 天 C2 vs E03b 配对 bootstrap
  paired_first_use  # 9 天配对 bootstrap
```

用 CLI 下载（办公室电脑需有 `wynstonliu` 的 Kaggle token）：

```bash
kaggle kernels output wynstonliu/kronos-small-sealed-oos-20260803-20260930 -p ./sealed_out
```

## 相关公开数据集

| 用途 | 数据集 |
|---|---|
| OOS 42 天 | `wynstonliu/a-share-120d-oos-20260803-20260930` |
| C2 best checkpoint | `wynstonliu/kronos-small-c2-best-checkpoint`（SHA `4ee469d…`） |
| E03b 权重 | `wynstonliu/kronos-small-clean-bigbatch-e03b`（kernel output） |

## 判定依据（§3）

文档：[`NEXT_STEP_SEALED_WINDOW_PLAN_20261022_CN.md`](NEXT_STEP_SEALED_WINDOW_PLAN_20261022_CN.md)

- **§3.1 C2**：9 天 D10 日均 Rank IC ≥0.10 / 0.05–0.10 / <0.05；
- **§3.2 E03b 晋级**：E03b IC > C2 且配对 bootstrap 95% CI 不含 0；
- 结论写 `docs/SEALED_OOS_RESULT_20260918_20260930_CN.md`（**已写**）；
- 报告 JSON 放 `finetune/reports/`（**已写**：`small_sealed_oos_20260918_20260930.json`）；
- 写完再构造 10-09 → 10-23 第二片（等 11-06 标签成熟）。

## 已知问题

公开 OOS 数据集里的 `start_index` 存的是 **asof 行的索引**，不是旧约定的“窗口起点”。

- 本次 eval 已按 `asof_index - 119` 反推窗口起点，不影响结果；
- 以后别人直接用数据集时：**窗口起点 = start_index - 119**；
- 下次可出 dataset v2 修正该字段。

## 本地临时文件

`scratch/oos_public_20260803_20260930/` 里的 fetch/build 脚本**没有进 git**（`/scratch/` 被忽略）。

- 数据集已经公开，不需要重建；
- 如果确实要重建，脚本在本机这个目录；办公室电脑只靠公开数据集 + git 里的 eval kernel 就够了。
