# 交接：Base clean 复刻失败 & 崩塌定位（2026-11-11）

> 给下一个 AI / 下一个会话：不用翻聊天记录，这份文档自带全部上下文。
> 读完 §1–§2 就知道"我们现在在哪"；§3 是干活要用的全部资产；§4 是别再踩的坑；§5 是没定的方向。

---

## 1. 一句话现状

**预训练的 `NeoQuasar/Kronos-base` 本身在 A 股有真实的截面排序信号（OOS Rank IC ≈ 0.090），
但我们这两周按 C2 路径做的微调会在 ~10 个 segment 内把这个信号毁掉 70~80%。
Base 不是架构不行，是这套微调范式把它训坏的。**

之前"base_clean 复刻 C2"的所有 stage（Bootstrap→Main→Extend→WC→Cosine）OOS IC 只有 0.002~0.023，
约为 C2 的 1/20，且 validation loss 一路下降、OOS 一路下降（反相关）。

---

## 2. 已验证的事实（都有测量口径，别重新质疑）

### 2.1 测量基准（一切对比的前提）

- **干净 OOS 窗口**：`2026-07-17 → 2026-10-23`，65 个交易日，335,312 样本，5,180 只，标签齐到 2026-11-06。
  训练信号上界 2026-07-02，其 D10 标签最远到 2026-07-16，故 **2026-07-17 起严格干净**。
  数据集：`luckfu/a-share-120d-oos-20260717-20261023-v2`（Kaggle 公开，SHA 已回下载校验）。
- **快扫子集**：同一包按 `stride=3` 取 22 天（~113,480 样本），单点评测 ~50 分钟（65 天全量 ~2.5h）。
- **decode 协议**：seed 20261104，T=0.65，top_p=0.8，单路径，`evaluate_base_sealed.py`。
- **协议漂移校验已通过**：C2 重跑 42 天子集 IC=0.11141，历史值 0.1116，差 0.0002 → 跨运行可比。

### 2.2 核心数字（65 天全量）

| 模型 | OOS Rank IC | 相对 C2 |
|---|---:|---:|
| C2 best（Small，历史线产物） | 0.11405 | 100% |
| **预训练 base（0 段微调）** | **0.08973** | **79%** |
| base_clean 任一 stage | 0.002~0.023 | 2~20% |

预训练 base 剥离 10 日反转因子后 IC 仍 0.066（74%）→ 信号有独立内容，不是反转马甲。

### 2.3 崩塌速度探针（22 天子集，全部从同一预训练权重出发，各训 10 段 ≈ 3130 步）

| 探针 | LR（predictor/condition） | history 权重 | IC | 保留 |
|---|---|---|---:|---:|
| P1 原配方 | 1e-6 / 1e-5 | 0.02 | 0.02242 | 25% |
| P2 降 LR | 1e-7 / 1e-7 | 0.02 | 0.05495 | 62% |
| P3s 补目标 | 1e-6 / 1e-5 | 1.0 | 0.01668 | 19% |
| P4 双管 | 1e-7 / 1e-7 | 1.0 | 0.06140 | 69% |

### 2.4 根因判定

- ❌ 不是选型尺子（val 选 checkpoint）：换尺子救不回已被毁掉的权重。
- ❌ 不是数据版本（old vs clean）：此前受控 fork 已证不显著。
- ❌ 不是新增条件层（sector_emb + size_mlp）：开/关条件 IC 相同（0.0235 vs 0.0237），
  且它是零初始化、注入第 10/12 层、量级仅 hidden 的 2~3% —— 纯装饰，零贡献零负担。
- ❌ 不只是 LR：降到 1e-7 仍 10 段丢 38%。
- ✅ **是微调范式本身**：梯度方向与 OOS 排序对着干（与"val 降/OOS 降"反相关同源）。
  `加权 forecast(1.364…0.455) + 0.02 history` 的目标与 base 预训练表征不兼容。

### 2.5 当前唯一站得住的可行路径

**不微调，直接用预训练 base** → 拿到 C2 ~79% 的排序能力，干净样本外。
要继续微调则必须换范式（冻主干只训顶层 / 极短程极低 LR），属未验证的研究方向，需拍板。

---

## 3. 干活资产（全部可直接复用）

### 3.1 账号与配额（GPU 30h/周/账号，刷新于 2026-11-14）

| 账号 | 剩余（2026-11-11 早） | 用途 |
|---|---:|---|
| wynstonliu | ~17.6 h | 主力 GPU（训练/评测） |
| smmt315 | ~26.4 h | 主力 GPU + 私有训练数据集 owner |
| luckfu | ~1 h | 数据集 owner（OOS/验收包/C2 镜像） |
| user281434 | 0 h | 已耗尽，停用 |

**Token 不写入任何文档/git**（secret hygiene）。它们在调用方的环境变量里：
`KAGGLE_API_TOKEN` + `KAGGLE_USERNAME`，或 `~/.kaggle/credentials.json`（luckfu）。
REST 走 `finetune/base_clean_bootstrap_relay/relay_auth.py`（30 分钟 token 自刷、`trust_env=False` 防本机代理）。

查配额：
```python
s.post("https://api.kaggle.com/v1/kernels.KernelsApiService/GetAcceleratorQuotaStatistics", json={})
```

### 3.2 Kaggle 数据集（公开，SHA 已校验）

| 数据集 | 内容 |
|---|---|
| `luckfu/a-share-120d-oos-20260717-20261023-v2` | **65 天干净 OOS**（主测量集） |
| `luckfu/a-share-120d-acceptance-20261009-20261023` | 11 天验收窗（65 天子集） |
| `luckfu/kronos-small-cosine-c2-best-weights` | C2 best 权重镜像（SHA 4ee469d4…，源自魔搭） |
| `wynstonliu/a-share-120d-temporal-symbol-holdout-clean` | 训练数据集公开镜像（私有原版 `smmt315/…` 的副本，SHA 一致） |
| `wynstonliu/kronos-tokenizer-base-v1` | tokenizer（与 HF 版 SHA 一致 59d85f6a…） |
| `wynstonliu/bc-probe-p1-best` `…-p2-best` | 探针 checkpoint（注意：是初始快照，见 §4.2） |
| `smmt315/a-share-120d-temporal-symbol-holdout` | 训练数据集（smmt315 私有，仅 smmt315 可挂） |

历史已污染的（只作测量不作选型）：`wynstonliu/a-share-120d-oos-20260803-20260930`（42 天）。

### 3.3 关键脚本（repo 内，路径相对仓库根）

| 脚本 | 作用 | 驱动方式 |
|---|---|---|
| `finetune/base_clean_bootstrap_relay/push_probe.py` | 从预训练出发的短程探针训练 | env：`KRONOS_PROBE_SLUG` `KRONOS_PROBE_MAX_SEGMENTS` + 可选 `KRONOS_PREDICTOR_LEARNING_RATE` / `KRONOS_HISTORY_LOSS_WEIGHT` 等（`FORWARD_ENV` 白名单） |
| `finetune/oos_eval/base_clean_sealed_oos/push_oos_bundled.py` | 推送 OOS 评测 kernel | env：`KRONOS_OOS_*`（slug/dataset/parents）+ `BASE_OOS_*`（label/glob/stride/窗口契约） |
| `finetune/oos_eval/base_clean_sealed_oos/evaluate_base_sealed.py` | 评测器本体（被 bundle 进 kernel） | `BASE_OOS_*` env |
| `finetune/push_kaggle_dataset.py` | 推 Kaggle 数据集（REST，绕 CLI） | `--dir --owner --slug --title --public` |
| `finetune/build_temporal_holdout_package.py` | 从原始面板构建时间切分评测包 | `--raw --sector-source --signal-start/end --role` |
| `finetune/base_clean_bootstrap_relay/monitor_chunk.py` | kernel 状态/日志监控 | `--kernel owner/slug --follow --tail N` |
| `finetune/base_clean_bootstrap_relay/push_main.py` | 接力训练推送（含 cosine-a90 修复） | env |
| `/tmp/zipstream.py`（不在 repo） | 从截断 zip 抠完整条目（见 §4.3） | `ZIPOUT=dir python3 zipstream.py arch.raw "suffix"` |

### 3.4 评测器 env 速查（`BASE_OOS_*`）

```text
BASE_OOS_LABEL                 标签
BASE_OOS_WEIGHTS_GLOB          权重 glob（kernel_sources 挂载时）
BASE_OOS_HF_REPO               直读 HF（如 NeoQuasar/Kronos-base，strict=False）
BASE_OOS_HF_TOKENIZER          直读 HF tokenizer
BASE_OOS_TOKENIZER             显式 tokenizer 挂载目录（绕过 glob；失败按大小兜底）
BASE_OOS_CONTEXT_LAYER         10=Base / 6=Small
BASE_OOS_SEED                  20261104
BASE_OOS_STRIDE                3 → 65 天抽 22 天（快扫）
BASE_OOS_LIMIT                 只取最后 N 天
BASE_OOS_CONDITION             off → 冻结 sector/size 条件（消融用）
BASE_OOS_EXPECT_SAMPLES/DATES/FIRST/LAST/TARGET_END   窗口契约守卫
```

### 3.5 已写好的报告（细节都在这）

- `finetune/reports/20261110_base_clean_collapse_probe_cn.md` — **崩塌定位主报告**（探针矩阵+根因+基建修复）
- `finetune/reports/20261110_base_clean_65d_oos_vs_c2_cn.md` — base_clean vs C2 65 天对照
- `finetune/reports/20261109_cosine_schedule_horizon_cn.md` — cosine 调度表长度坑（`KRONOS_MAX_SEGMENTS` 不改变调度表长）
- `finetune/reports/20261108_temporal_holdout_windows_cn.md` — 时间切分窗口 + "训练污染≠选型污染"澄清

---

## 4. 踩过的坑（别再重踩）

### 4.1 Kaggle 平台

- **数据集 create 是瞬时会失败的**：`No gcs url found` / `Path must be unique` / title 重复。
  同参数重试一次常成功；**必须回下载并校验 SHA** 才算成功（否则可能拿到空壳）。
- **title 全局唯一**：dataset 和 kernel 都是。复用旧 title 会 409。
- **kernel 有 2 个 batch GPU 会话槽**：满时推送报 `Maximum batch GPU session count of 2 reached`；
  排队中的会话 `GetKernelSessionStatus` 返回 `{}`（看不到但占槽）。槽可能卡死很久，换账号或等。
- **`invalidDatasetSources` / `invalidKernelSources`**：挂载私有/不存在的资源时静默跳过，运行时才报"找不到"。
- **CLI 2.2.4 在本机代理下 `SSLEOFError`**：一切走 REST（`relay_auth.py` 已封装）。

### 4.2 模型 / 训练

- **bootstrap 的 `best_model` 只在全量验证时更新**；`large_interval=1,000,000` 时短程探针的
  `best_model` 是**初始快照**（主干与预训练逐位相同、条件头全零）。**要测训练产物请用 `last_model`。**
- **kernel 版本输出 zip 排布**：`best_model(~450MB) → last_state.pt(~1.2GB) → last_model`。
  下载在 ~0.5–1.1 GB 处被连接重置且**不支持 Range**，`last_model` 通常不可达；
  只有 `best_model` 能用 `/tmp/zipstream.py` 抠出。
- **cosine 阶段的调度表长度**由 `effective_epochs` 决定（`require_full_coverage=1` 时=520），
  `KRONOS_MAX_SEGMENTS` 只截断不缩短 —— 短跑 cosine 必须显式设 `KRONOS_EPOCHS=N` + `KRONOS_REQUIRE_FULL_COVERAGE=0`。
- **预训练权重无 `sector_emb/size_mlp`**，from_pretrained 会 strict 失败；用 `BASE_OOS_HF_REPO` 路径（strict=False）。

### 4.3 本机 / 数据

- **复权基准**：不同日期下载的 qfq 行情在分红送配后整段重算，直接拼接会在拼接点跳空。
  要统一就全量重下（如 `/tmp/oos_raw/full_20250701_20261106.csv`，329 交易日单一基准）。
- **数据集下载的 `IncompleteRead`**：`relay_auth` 的 `iter_content` 要在循环内捕获并收下 `e.partial`。
- **akshare 的 MiniRacer 非线程安全**：下载脚本必须 `--processes`，纯多线程会崩。

### 4.4 已修的代码 bug（HEAD 已改，快照只覆盖 config/train_predictor，不影响）

- `kaggle_kronos_base_clean.py`：`stop_gpu` 无 SwanLab key 时 `UnboundLocalError`（关机 ERROR）。
- 评测器 tokenizer 兜底：`weights` 变量名重用，日志把 model 路径打成 tokenizer 路径（显示问题）。

---

## 5. 未定的方向（要拍板的）

| 选项 | 含义 | 成本 | 我的倾向 |
|---|---|---|---|
| **A. 直接用预训练 base** | 上线/研究都以未微调的 base 为 base 线（C2 的 79%） | 0 | **推荐先做** |
| B. 冻主干微调 | `KRONOS_TRAINABLE_TRANSFORMER_LAYERS` 只放顶层，验证能否保住信号 | ~2h 探针 | 次选，预期增益存疑（条件头无贡献） |
| C. 极短程极低 LR | 1e-7 拉长到 30–50 段看信号是否稳定在 0.06 | ~4h | 可作为 B 的对照 |
| D. 放弃 base，回到 Small 线 | 接受 C2 即上限 | 0 | 若 A 不满足再考虑 |

**协议层面还有一个坑**：`docs/VALIDATION_RANK_IC_SELECTION_PROTOCOL_CN.md` §5 明文禁止用
symbol-holdout val loss 选 best，但 `kaggle_kronos_base_clean.py` 的 `STAGES[*].best_metric` 全是 `forecast`。
若未来重训，需先把选型接到 `temporal_rank_ic_v1`（代码已在 `finetune/temporal_rank_ic_selection.py`）。

---

## 6. 30 秒上手检查

```bash
cd /Users/fupengcheng/Documents/Kronos
# 1. 查配额（§3.1）
# 2. 跑一次预训练 base 的 22 天快扫（验证管线没坏）：
KAGGLE_API_TOKEN=… KAGGLE_USERNAME=wynstonliu \
KRONOS_OOS_OWNER=wynstonliu KRONOS_OOS_SLUG=handoff-smoke \
KRONOS_OOS_DATASET=luckfu/a-share-120d-oos-20260717-20261023-v2 \
BASE_OOS_LABEL=pretrained BASE_OOS_HF_REPO=NeoQuasar/Kronos-base \
BASE_OOS_HF_TOKENIZER=NeoQuasar/Kronos-Tokenizer-base \
BASE_OOS_WEIGHTS_GLOB='**/NONE/x' BASE_OOS_CONTEXT_LAYER=10 BASE_OOS_SEED=20261104 BASE_OOS_STRIDE=3 \
BASE_OOS_EXPECT_SAMPLES=335312 BASE_OOS_EXPECT_DATES=65 \
BASE_OOS_EXPECT_FIRST=2026-07-17 BASE_OOS_EXPECT_LAST=2026-10-23 BASE_OOS_EXPECT_TARGET_END=2026-11-06 \
python3 finetune/oos_eval/base_clean_sealed_oos/push_oos_bundled.py --push
# 预期：~50 分钟后 IC ≈ 0.089。若偏离很大，先查 tokenizer/config 是否被改动。
```
