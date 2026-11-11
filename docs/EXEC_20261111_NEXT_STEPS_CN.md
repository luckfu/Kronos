# 执行单：Base 线下一步（2026-11-11）

> 给执行助手：这是可以直接照做的任务单。背景见 `docs/HANDOVER_20261111_BASE_CLEAN_COLLAPSE_CN.md`
> （先读它的 §2 和 §4）。本文只写"做什么、怎么做、看到什么数字就怎么判"。
> 任务 1 和任务 2 互不依赖，**并行做**。任务 3 只在任务 1、2 都出结论后才开始。

---

## 0. 为什么是这个顺序（30 秒）

- 交接文档的 B（冻主干 CE 微调）、C（1e-7 拉长）**暂不做**：Small 线 2026-09-17 的冻结探针
  （`finetune/reports/stage3_freeze_probe_be_oos_results_20260917.json`）已经证明，CE 继续训练
  只放 `forecast_head` 能守住 IC，但不会涨；放 `dep_layer` 会慢慢漏。Base 的 P1–P4 也没有一个让 IC 上升。
  根本矛盾是"训练目标 = token 交叉熵，考核 = 截面排序 IC"。
- 缺一个关键对照：**未微调的 `NeoQuasar/Kronos-small`** 在同一 OOS 上的 IC。
  C2 的 bootstrap 起点就是它（`finetune/kaggle_kronos_small_v21.py:419`），所以它直接回答
  "C2 的微调到底有没有带来收益"。
- 预训练 Base（0.0897）和 C2（0.1141）的预测如果相关性不高，**等权融合就可能超过 C2**，零训练成本。

---

## 1. 任务 1：预训练 Small 的 22 天快扫（wynstonliu，约 30–50 分钟 GPU）

### 1.1 推送

先不加 `--push` 干跑一次，确认打印的 summary 里 `id`、`context_layer=6`、`datasets` 正确，再加 `--push`。

```bash
cd /Users/fupengcheng/Documents/Kronos
export KAGGLE_API_TOKEN=<wynstonliu 的 token> KAGGLE_USERNAME=wynstonliu
KRONOS_OOS_OWNER=wynstonliu \
KRONOS_OOS_SLUG=kronos-small-pretrained-oos65-s3 \
KRONOS_OOS_TITLE="Kronos small pretrained OOS65 s3" \
KRONOS_OOS_DATASET=luckfu/a-share-120d-oos-20260717-20261023-v2 \
BASE_OOS_LABEL=kronos_small_pretrained \
BASE_OOS_HF_REPO=NeoQuasar/Kronos-small \
BASE_OOS_HF_TOKENIZER=NeoQuasar/Kronos-Tokenizer-base \
BASE_OOS_WEIGHTS_GLOB='**/NONE/model.safetensors' \
BASE_OOS_CONTEXT_LAYER=6 BASE_OOS_SEED=20261104 BASE_OOS_STRIDE=3 \
BASE_OOS_EXPECT_SAMPLES=335312 BASE_OOS_EXPECT_DATES=65 \
BASE_OOS_EXPECT_FIRST=2026-07-17 BASE_OOS_EXPECT_LAST=2026-10-23 BASE_OOS_EXPECT_TARGET_END=2026-11-06 \
python3 finetune/oos_eval/base_clean_sealed_oos/push_oos_bundled.py --push
```

注意：

- **`KRONOS_OOS_TITLE` 必须显式设置且全局唯一**。脚本默认 title 是 "Sealed OOS"，会 409。
- **Kaggle 新建 kernel 时 slug 由 title 生成**，不是由 `KRONOS_OOS_SLUG` 决定。
  上面的 title 转小写、空格换连字符后正好等于 slug，别改成不一致的形式。
  （历史教训：构建目录 `build/kronos-c2-oos65bb` 实际对应 kernel `user281434/kronos-c2-oos-65d-b`。）
- 满槽报 `Maximum batch GPU session count of 2 reached` 时等一会或换 smmt315（同样命令，改 owner 和 token）。

### 1.2 监控与自检

```bash
python3 finetune/base_clean_bootstrap_relay/monitor_chunk.py \
  --kernel wynstonliu/kronos-small-pretrained-oos65-s3 --follow --tail 50
```

日志里的 `hf_loaded` 阶段：`missing` 只能是 `sector_emb.*` / `size_mlp.*`，`unexpected` 必须为空。
否则说明 HF config 和本地 `Kronos` 结构不一致，**立刻停**，不要采信 IC。
`worker_dates` 应该是两张卡各 11 天（共 22 天）。

### 1.3 拉预测分片

```bash
python3 finetune/oos_eval/base_clean_sealed_oos/fetch_kernel_shards.py \
  --kernel wynstonliu/kronos-small-pretrained-oos65-s3 --out scratch/oos65/small_pretrained_s3
```

预期 22 个 `*.csv.gz` + `summary.json`。

### 1.4 判定（与 C2 在**同样 22 天**上配对比较）

C2 的 22 天数字没有现成值。等任务 2 拉到 C2 的 65 天分片后，用融合脚本做内连接，它会自动只保留共同的 22 天：

```bash
python3 finetune/oos_eval/base_clean_sealed_oos/fusion_analysis.py \
  --model small_pre=scratch/oos65/small_pretrained_s3 \
  --model c2=scratch/oos65/c2 \
  --raw scratch/oos65/combined_raw.csv \
  --out finetune/reports/20261111_small_pretrained_vs_c2_s3.json
```

看 `single.small_pre.ic`、`single.c2.ic`，以及 **`paired_single["c2-small_pre"]`**：
C2 减预训练 Small 的配对 IC 差和 95% CI（5 日块 bootstrap）。这就是"C2 微调到底带来多少"的答案。
顺带看 `prediction_corr`：C2 和它的起点相关有多高。

| 结果 | 结论 | 下一步 |
|---|---|---|
| small_pre ≥ C2 − 0.01 | C2 的"微调收益"基本不存在，CE 微调在两条线上都无效 | 停掉一切 CE 微调，资源全部给任务 3 |
| small_pre ≤ 0.08 | 微调确实帮到了 Small，Base 上的崩塌是 Base 特有问题 | 写报告，交给用户拍板是否查 Base/Small 差异 |
| 介于两者之间 | 不确定 | 补跑 65 天全量（去掉 `BASE_OOS_STRIDE`，slug/title 换 `…-full`），再判 |

参考数：预训练 Base 22 天 0.08924；C2 42 天旧子集 0.1114；C2 65 天 0.11405。

---

## 2. 任务 2：零 GPU 融合分析（本机 CPU，可立即开始）

### 2.1 拉两份 65 天分片

```bash
# 预训练 Base 65 天（wynstonliu 私有 kernel，需 wynstonliu 凭据）
KAGGLE_API_TOKEN=<wynstonliu> KAGGLE_USERNAME=wynstonliu \
python3 finetune/oos_eval/base_clean_sealed_oos/fetch_kernel_shards.py \
  --kernel wynstonliu/kronos-base-pretrained-oos65-full --out scratch/oos65/base_pretrained

# C2 65 天（user281434 私有 kernel，需 user281434 凭据；GPU 配额为 0 不影响下载）
KAGGLE_API_TOKEN=<user281434> KAGGLE_USERNAME=user281434 \
python3 finetune/oos_eval/base_clean_sealed_oos/fetch_kernel_shards.py \
  --kernel user281434/kronos-c2-oos-65d-b --out scratch/oos65/c2
```

每个目录预期 65 个分片。少于 65 个说明那次运行不完整，换同账号下的
`user281434/kronos-c2-oos-65d`（第一次运行）试试。

**如果拿不到 user281434 的凭据**：在 smmt315 上重跑 C2 65 天（约 2.5h GPU）。
历史 kernel 的 tokenizer 来自 user281434 的私有 kernel，跨账号挂不上，所以改用 HF tokenizer：

```bash
KAGGLE_API_TOKEN=<smmt315> KAGGLE_USERNAME=smmt315 \
KRONOS_OOS_OWNER=smmt315 KRONOS_OOS_SLUG=kronos-c2-oos65-rerun \
KRONOS_OOS_TITLE="Kronos C2 OOS65 rerun" \
KRONOS_OOS_DATASET=luckfu/a-share-120d-oos-20260717-20261023-v2 \
KRONOS_OOS_EXTRA_DATASETS=luckfu/kronos-small-cosine-c2-best-weights \
BASE_OOS_LABEL=c2_best \
BASE_OOS_WEIGHTS_GLOB='**/kronos-small-cosine-c2-best-weights/model.safetensors' \
BASE_OOS_HF_TOKENIZER=NeoQuasar/Kronos-Tokenizer-base \
BASE_OOS_CONTEXT_LAYER=6 BASE_OOS_SEED=20261104 \
BASE_OOS_EXPECT_SAMPLES=335312 BASE_OOS_EXPECT_DATES=65 \
BASE_OOS_EXPECT_FIRST=2026-07-17 BASE_OOS_EXPECT_LAST=2026-10-23 BASE_OOS_EXPECT_TARGET_END=2026-11-06 \
python3 finetune/oos_eval/base_clean_sealed_oos/push_oos_bundled.py --push
```

重跑后 IC 必须在 0.11405 ± 0.002 内（协议漂移校验），否则先查原因再用。

### 2.2 原始面板（已备好）

`scratch/oos65/combined_raw.csv` 已从 `luckfu/a-share-120d-oos-20260717-20261023-v2` 下载
（88,801,267 字节，与服务器 content-length 一致；SHA256 `28b76344…6111`，仓库内无清单可比对）。
只用于计算 10 日反转因子。`scratch/` 已被 `.gitignore` 忽略。

### 2.3 跑分析

```bash
python3 finetune/oos_eval/base_clean_sealed_oos/fusion_analysis.py \
  --model base=scratch/oos65/base_pretrained \
  --model c2=scratch/oos65/c2 \
  --raw scratch/oos65/combined_raw.csv \
  --out finetune/reports/20261111_oos65_base_c2_fusion.json
```

本机沙箱里跑 pandas/numpy 可能直接崩（退出码 136），遇到就在沙箱外跑。

脚本输出说明：

| 字段 | 含义 |
|---|---|
| `single.*` | 各模型 D10 日度 Rank IC 均值、ICIR、正 IC 天数比、Top-Bottom 10% 收益差。base 应 ≈ 0.0897、c2 ≈ 0.1141，偏离 > 0.002 先停下查 |
| `paired_single` | 单模型两两配对差 `b-a` 与 95% CI |
| `prediction_corr` | 两模型日度截面预测秩相关；以及两条日度 IC 序列的相关 |
| `fusion_equal_weight` | **预注册结果**：两模型日内百分位秩等权平均；`paired_vs` 是融合减单模型的配对差与 95% CI |
| `fusion_weight_grid_DIAGNOSTIC_ONLY` | 在 OOS 上扫权重，**只看形状，不能当结果报告** |
| `reversal_factor` | −ret10 自身的 IC，作参照 |
| `neutralized` | 对市值秩、反转秩、行业哑变量逐日回归后的残差 IC |
| `incremental` | `a|b` = a 对 b 的预测做逐日回归后的残差 IC，即"a 在 b 之外还有多少独立信号" |

### 2.4 判定

| 结果 | 结论 |
|---|---|
| `fusion_equal_weight.paired_vs.c2.ci95` 下界 > 0 | 融合显著优于 C2 → 当前最好的可用方案，写报告请用户拍板 |
| 融合 IC > C2 但 CI 跨 0 | 有迹象但未确认 → 留到新封存窗口（§4）再验，不要在本窗口上继续调权重 |
| 融合 ≤ C2 | Base 与 C2 信号高度重合，Base 没有增量价值；看 `incremental.base|c2` 确认 |

---

## 3. 任务 3：冻结 Base + 排序头（任务 1、2 出结论后再定，smmt315）

**只有在任务 1 判为"CE 微调无效"，或任务 2 显示 Base 有明显独立信号（`incremental.base|c2` ≥ 0.03）时才开工。**
开工前把方案写成一页给用户确认，不要直接烧配额。设计约束：

- Kronos 主干和 tokenizer **完全冻结**，不跑任何 CE / forecast loss。
  历史上"CE + 排序"混合目标已在 Small 上失败（`finetune/reports/stage3_p1_ce_rank_oos_results_20260916.json`，
  P1 last D10 IC 0.07）。
- 特征：预训练 Base 对每个窗口的末层隐状态（或 decode 出的 10 日路径），离线抽一次存盘，后续只训小头。
- 目标：按 `asof_date` 分组的 listwise / IC 型损失，标签为 D10 收益的日内百分位秩。
- 训练数据：`smmt315/a-share-120d-temporal-symbol-holdout`（私有，只有 smmt315 能挂），
  信号上界 2026-07-02，**不得触碰 2026-07-17 以后的任何数据**。
- 选型：`finetune/temporal_rank_ic_selection.py` 的 `temporal_rank_ic_v1`，禁止用 val loss 选 best。
- 进门门槛：22 天快扫稳定 > 0.089（预训练 Base）才跑 65 天全量。
- `finetune/kaggle_kairos_ranking_probe/` 是 Kairos 模型 + mfe10 目标的旧排序探针，目标和窗口不同，
  数字不可比，只能借鉴代码结构。

---

## 4. 封存窗口纪律（所有任务都适用）

65 天 OOS（2026-07-17 → 10-23）已经被拿来比较了十几个变体。从现在起：

- 任务 1、2 的结论可以用它来下，但**不在它上面挑融合权重、挑排序头超参**。
- **2026-10-24 之后的数据不参与任何选择**。等 D10 标签攒够 ≥ 20 个信号日（约 12 月中下旬），
  用 `finetune/build_temporal_holdout_package.py` 建新的验收包，对最终候选只测一次。

---

## 5. 配额（2026-11-14 刷新）

| 账号 | 剩余 | 本周用途 |
|---|---:|---|
| wynstonliu | ~17.6h | 任务 1（快扫 ~0.8h，必要时全量 ~2h） |
| smmt315 | ~26.4h | 任务 2 的 C2 重跑兜底（~2.5h）；任务 3 |
| luckfu | ~1h | 只推数据集，不跑 GPU |
| user281434 | 0h | 只用来下载 C2 分片 |

Token 不写入任何文件或 git。

---

## 6. 交付物

- `finetune/reports/20261111_small_pretrained_vs_c2_s3.json` + 一段中文结论（追加到本文末尾或新报告）。
- `finetune/reports/20261111_oos65_base_c2_fusion.json` + 一段中文结论。
- 两条判定表各落在哪一行，以及据此建议的任务 3 去留。

## 7. 本次新增的工具（已实测）

- `finetune/oos_eval/base_clean_sealed_oos/fetch_kernel_shards.py`：用 `ListKernelSessionOutput`
  逐文件签名链接下载，只拉 `shards/*.csv.gz` 和 `summary.json`，绕开大 zip 截断问题。
  已在 `luckfu/kronos-small-0-1-stage3-freeze-probe-be-oos` 上实测下载成功。
- `finetune/oos_eval/base_clean_sealed_oos/fusion_analysis.py`：按 `identity` 对齐、校验标签逐位一致后做
  单模型 / 相关 / 等权融合 / 配对 bootstrap / 中性化 / 增量分析。已用合成数据验证逻辑
  （两信号相关 0.4 时融合 IC 高于两者，符合预期）。
