# 交接：Base 线下一步规划已定，等执行（2026-11-11 11:00）

> 给接手的助手：读完本文就能开工，不用翻聊天记录。
> 阅读顺序：本文 → `docs/EXEC_20261111_NEXT_STEPS_CN.md`（任务单，照做）→ 需要背景时再查
> `docs/HANDOVER_20261111_BASE_CLEAN_COLLAPSE_CN.md`（上一棒交接，§2 事实、§4 踩坑）。

---

## 1. 一句话现状

上一棒确认了：预训练 `Kronos-base` 不微调时 OOS IC 为 0.0897（C2 的 79%），而我们的 CE 微调会在约 10 段内把它毁掉。
本棒**没有跑任何 GPU 任务**，只做了三件事：定方向、写好两个工具脚本、写好执行单。下一步是照执行单跑任务 1 和任务 2。

---

## 2. 本棒定下的方向（用户已同意，交由你执行）

1. **暂停交接文档 §5 的 B（冻主干 CE 微调）和 C（1e-7 拉长）**。理由：
   - Small 线 2026-09-17 的冻结探针已经回答过同类问题：只训 `forecast_head` 能守住 IC（0.184 → 0.182），但不涨；
     训 `dep_layer` 会慢慢漏（0.184 → 0.171）。见 `finetune/reports/stage3_freeze_probe_be_oos_results_20260917.json`。
   - Base 的 P1–P4 四个探针，没有一个让 IC 上升，只有"掉得快"和"掉得慢"。
   - 根本矛盾在目标函数：训练优化的是 token 交叉熵，考核看的是截面排序 IC。
2. **任务 1**：测未微调的 `NeoQuasar/Kronos-small`。C2 的 bootstrap 起点就是它（`finetune/kaggle_kronos_small_v21.py:419`），
   所以这个数字直接回答"C2 的微调到底带来了多少"。仓库里没有这个数。
3. **任务 2**：零 GPU，把预训练 Base 和 C2 的 65 天预测做等权融合和增量分析。
4. **任务 3**（冻结 Base + 排序头）：只有在任务 1、2 的结果满足执行单 §3 的条件时才做，且开工前要先写一页方案给用户确认。
5. **封存纪律**：2026-10-24 之后的数据不参与任何选择，留作最终验收窗口。

具体命令、判定表、配额分配都在执行单里，本文不重复。

---

## 3. 本棒产出（都未提交 git）

| 文件 | 说明 | 验证情况 |
|---|---|---|
| `docs/EXEC_20261111_NEXT_STEPS_CN.md` | 任务单：命令、自检点、判定表、交付物 | — |
| `finetune/oos_eval/base_clean_sealed_oos/fetch_kernel_shards.py` | 从评测 kernel 逐文件下载预测分片，绕开大 zip 截断 | 在 `luckfu/kronos-small-0-1-stage3-freeze-probe-be-oos` 上真实下载成功 |
| `finetune/oos_eval/base_clean_sealed_oos/fusion_analysis.py` | 对齐 + 单模型 IC / 配对差 / 相关 / 等权融合 / 中性化 / 增量 | 合成数据测过；**没有在真实分片上跑过** |
| `scratch/oos65/combined_raw.csv` | 65 天 OOS 包的原始面板，只用于算 10 日反转 | 88,801,267 字节，与服务器一致；SHA256 `28b76344…6111`，无清单可比对；`scratch/` 被 git 忽略 |
| `finetune/oos_eval/base_clean_sealed_oos/build/kronos-small-pretrained-oos65-s3/` | 任务 1 的干跑产物 | 干跑通过，`context_layer=6`、窗口检查参数正确，**未推送** |

**git 工作区说明**：除上表外，工作区里还有上一棒留下的大量未提交改动（`kaggle_kronos_base_clean.py`、
`evaluate_base_sealed.py`、`push_oos_bundled.py`、`push_main.py` 已修改；`push_probe.py`、
`build_temporal_holdout_package.py`、`push_kaggle_dataset.py`、四份 1108–1110 报告、两份交接文档为新文件）。
**不要擅自提交或回滚**，要提交先问用户。

---

## 4. 本棒新发现的坑（执行单里也写了）

- **Kaggle 新建 kernel 的 slug 由 title 生成**，不是由 `KRONOS_OOS_SLUG` 决定。title 转小写、空格换连字符后必须等于 slug。
  例：本地构建目录 `build/kronos-c2-oos65bb` 实际对应 kernel `user281434/kronos-c2-oos-65d-b`。
  `push_oos_bundled.py` 的默认 title 是 "Sealed OOS"，不设 `KRONOS_OOS_TITLE` 必然 409。
- **本机沙箱里 pandas/numpy 可能直接崩**（退出码 136，无任何输出），沙箱也不让写 `/tmp`。
  跑分析脚本时在沙箱外执行，临时文件放工作区内的 `scratch/`。
- **下载接口**：`ListKernelSessionOutput` 返回每个输出文件的签名 URL，支持分页（`pageToken`）；
  公开数据集单文件可走 `https://www.kaggle.com/api/v1/datasets/download/<owner>/<slug>/<file>`，会重定向到 GCS。
- **C2 历史评测的 tokenizer** 来自 user281434 的私有 kernel，其他账号挂不上。在别的账号重跑 C2 时，
  改用 `BASE_OOS_HF_TOKENIZER=NeoQuasar/Kronos-Tokenizer-base`（执行单 §2.1 已写好命令）。

---

## 5. 未验证 / 需要注意的事实

- C2 的 65 天分片在 `user281434/kronos-c2-oos-65d-b`，这个名字取自报告 `20261110_base_clean_65d_oos_vs_c2_cn.md:67`，
  **本棒没有实际列过它的输出**。如果列不出 65 个分片，按执行单 §2.1 的兜底处理。
- **魔搭上没有 C2 预测，只有权重**（2026-11-11 已查，不用再查）：`luckfu/Kronos-small-0.1-Cosine-C2-Best`
  只有 `model.safetensors` / `config.json` / `best_metric.json` / `README.md`；luckfu 名下的公开数据集只有无关的
  `kronos-v6-test-holdout-2026`。Kaggle 的 `luckfu/kronos-small-cosine-c2-best-weights` 就是这份魔搭权重的镜像，
  兜底重跑用的正是它。
- 预训练 Small 的 HF `config.json` 能不能直接喂给本地 `Kronos(**config)`，**没验证**。
  Base 走这条路是通的，Small 大概率也通；以 kernel 日志里 `hf_loaded` 的 missing / unexpected 为准（执行单 §1.2）。
- 融合脚本里的 10 日反转按 `volume > 0` 的交易行回看 10 行计算，和评测器的窗口口径一致，
  但不等于严格的 10 个自然交易日（停牌股会有差异）。

---

## 6. 凭据（不写值）

| 账号 | 用途 | 位置 |
|---|---|---|
| wynstonliu | 任务 1 推送；下载预训练 Base 分片 | 调用方环境变量 `KAGGLE_API_TOKEN` + `KAGGLE_USERNAME`，向用户要 |
| user281434 | 只用于下载 C2 分片（GPU 配额 0 不影响） | 同上，向用户要 |
| smmt315 | 兜底重跑 C2；任务 3 | 同上，向用户要 |
| luckfu | 本机默认账号 | `~/.kaggle/credentials.json`，`relay_auth.py` 自动刷新 |

Token 不写入任何文件或 git。

---

## 7. 第一步做什么

1. 向用户要 wynstonliu 和 user281434 的 token。
2. 同时开工：
   - 按执行单 §1.1 推送任务 1（先干跑确认，再加 `--push`）。
   - 按执行单 §2.1 下载两份 65 天分片，确认各 65 个文件，然后跑 §2.3 的分析。
3. 先核对 `single.base.ic ≈ 0.0897`、`single.c2.ic ≈ 0.1141`（偏差 > 0.002 就停下查），再读其他结果。
4. 两个任务的结论按执行单 §6 交付，写明各自落在判定表哪一行，以及任务 3 是否开工的建议。
