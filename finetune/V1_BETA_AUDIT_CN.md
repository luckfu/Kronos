# A-share Full-Market v1-beta 代码与设计审核报告

审核对象：`FULL_MARKET_V1_BETA_PLAN_CN.md` 定义的实验口径，以及本地保存的
Kaggle 训练代码。本次审核为只读审核，**未修改任何代码**。

审核日期：2026-08-22

---

## 0. 先确认「哪份代码真正会跑」

仓库里有 4 份 v1-beta 代码副本，实际生效的只有一份：

| 副本 | 是否生效 | 说明 |
|---|---|---|
| `finetune/kaggle_v1_beta.py` 内嵌 base64 归档 | ✅ **线上真正生效** | Kaggle `kernel_type: script` 只上传单个 code_file，`SOURCE_ROOT`/tar.gz 在 Kaggle 上都不存在 → 落到内嵌 payload |
| `finetune/kaggle_v1_beta_minimal/Kronos/` | 与内嵌归档**逐字节相同** | 可作为可读副本 |
| `finetune/kaggle_v1_beta_kernel/Kronos/` | ❌ 陈旧 | `finetune/config.py` 硬编码 `use_size_features = True` |
| 仓库顶层 `finetune/config.py`、`finetune/train_predictor.py` | ❌ V6 时代版本 | `use_sector_features=False`、`num_size_buckets=10`、condition LR 1e-3、无 cosine 守卫 |

**下文所有行号均指内嵌归档（等价于 `kaggle_v1_beta_minimal/Kronos/`）。**
`finetune/kaggle_v1_beta.py` 的行号指仓库中该文件本身。

---

## 1. 已实测核对通过的部分

| 项 | 方案声称 | 实测结果 | 结论 |
|---|---|---|---|
| 监督窗口数 | 10,564,072 | **10,564,072** | ✅ 完全一致 |
| 段数 | 529/遍 × 2 = 1,058 | 529 × 2 = **1,058** | ✅ |
| Cosine `T_max` | 全局跨两遍 | **1,320,510** step（bs=16） | ✅ |
| 行业 embedding 行数 | 86 → 87 行 | `nn.Embedding(86+1, 832)`（`model/kronos.py:226`） | ✅ 满足 §8 |
| `num_size_buckets=0` | 不用离散桶 | `size_emb = None`（`model/kronos.py:227`） | ✅ |
| forecast loss 切片 | 只算未来 10 根 | `targets[119:129]` → 原始 bar 120–129 | ✅ 正好 10 根 |
| 归一化窗口 | 无未来泄露 | 均值/std 只用 `x[:120]`（`finetune/dataset.py:392-396`） | ✅ |
| 条件点时性 | 观察窗最后一天 | `start_idx + lookback - 1`（`dataset.py:407/412/416`） | ✅ |
| train/val 窗口重叠 | 应无 | **0 个重叠 asof** | ✅ 无同样本污染 |
| `size_percentile` 缺失处理 | — | `nan→0.5` + `known` flag 双通道（`kronos.py:290-293`） | ✅ |
| `best_model` 不被覆盖 | §7 硬要求 | `export_last_model.py` 只读 `best/`、只写 `last/` | ✅ |
| 条件注入层 | `context_layer=10` | 注入在 layer 10 之前，恰好被 `trainable_transformer_layers=2` 覆盖 | ✅ 设计自洽 |

**没有发现任何未来信息泄露。** 这是最重要的一条结论。

---

## 2. P0 — 首个正式 chunk 前必须解决

### P0-1 `run.log` 永远拿不到训练日志（直接违反 §7 与 §10）

```python
# finetune/kaggle_v1_beta.py
136: log_handle = open(output_root / "run.log", "a", buffering=1)
137: sys.stdout = Tee(sys.__stdout__, log_handle)     # 只改父进程 Python 层
138: sys.stderr = Tee(sys.__stderr__, log_handle)
...
186: subprocess.run(["python", "-u", str(SOURCE_ROOT / "finetune/train_predictor.py")], env=env, check=True)
```

子进程继承的是**操作系统级 fd 1/2**，不是父进程的 `Tee` 对象。
`train_predictor.py` 的全部输出（每段 loss、验证 loss、学习率、显存、checkpoint
记录）**只进 Kaggle 页面日志，一行都不会写进 `run.log`**。`run.log` 最终只有
父脚本那几行 JSON。同理第 89 行 P100 wheel 安装、第 97-98 行 ModelScope
下载的输出也全部丢失。

更糟的是两处守卫同时失效：

- 输出契约检查（191-197 行）只判断 `run.log` **存在**，而 136 行的
  `open(..., "a")` 保证它一定存在 → **恒真**。
- 接力校验（125-131 行）检查上游树里 `run.log` 存在 → 同样恒真。

结果是 §10 明令禁止的「只存在于页面日志的输出」会贯穿全部 1,058 段，而且没有
任何机制能发现。这会让整条 42–45 小时的接力链失去持久审计证据。

### P0-2 验证集塌缩成 9 个交易日，而文档声称 7 个月

`val_data.pkl` 只含 **2026-01-05 → 2026-07-31**（每股 139 根 bar），
**没有 2025 年的历史前缀**。而 `dataset.py:176`：

```python
self.window = self.config.lookback_window + self.config.predict_window + 1   # = 131
```

所以每股只有 `139 - 131 + 1 = 9` 个合法起点：

```text
验证窗口总数        45,409
验证 asof 日期范围  2026-07-06 → 2026-07-16
唯一 asof 交易日数  9          ← 方案 §3 写的是 2026-01-01 至 2026-07-31
每次实际使用        2,000 个（n_val_iter 默认值；runner 从不设置 KRONOS_VALIDATION_SAMPLES）
```

后果：

- **整个实验唯一的模型选择信号**（`best_model`）建立在约两周的横截面快照上，
  1,058 段全部用它打分。
- 45,409 个窗口里同一只股票出现 9 次、120 天回看窗重叠 >92%，
  **有效样本量远小于 45,409**，实质接近「5,201 只股票 × 1 个时点」。
- 选择方差极大，且完全绑定 2026 年 7 月上半月的单一市场状态。

这不是代码 bug（`val_signal_start/end` 被忠实执行了），而是**数据准备阶段
没给验证集留 120 天历史前缀**。要恢复文档声称的口径，`val_data.pkl` 需要包含
2025-07 起的行情作为回看上下文。

附带：train 标签最晚到 2026-07-17，val 输入/标签从 2026-07-06 起 →
**train 与 val 之间没有 embargo/purge 间隔**。样本本身不重叠（已实测 0 个），
但按金融时序惯例这里应留 10 个交易日隔离带。

---

## 3. P1 — 影响实验结论有效性

### P1-1 「零初始化 = no-op」的前提在 V6 上不成立

`model/kronos.py:246-252` 把 `sector_emb`、`size_mlp` 最后一层零初始化，注释说
「让预训练 checkpoint 保持原行为」。这对**没有条件分支的基座**成立，但
V6 Segment 568 **有** `size_emb`，而且量级很大：

```text
V6 size_emb.weight 逐桶 L2 范数（d_model=832）
  bucket 0..9:  10.45  9.89  7.79  5.91  4.25  3.89  5.28  7.31  9.58  14.38
  bucket 10 (unknown): 0.028      ← 未训练行，可作「零起点」对照
参照:  embedding.emb_s1 行范数均值 1.26
```

V6 在 layer 10 处往残差流注入的是一个**范数 3.9–14.4 的偏置**（token
embedding 的 3–11 倍），并呈明显 U 形（两端市值最强）。v1-beta 把 `size_emb`
整个丢弃（成为 `from_pretrained` 的 unexpected key），`size_mlp` 输出恰好为 0：

- **step 0 的 v1-beta ≠ V6**。V6 的后两层 + `dep_layer` + `head` 是在「有一个
  大偏置」的条件下训出来的，现在突然拿到 0。
- 初始 loss 会明显差于 V6，**这不能被解读为「新条件不好」**；也说明 1,058 段
  里有相当一部分算力是在补偿被丢弃的旧信号。
- §1 说「在保留 V6 通用 OHLCVA 能力基础上引入…」、§5 说「从 V6 继承
  Transformer 主体和预测头」——但丢掉这个偏置并不是中性操作。

可考虑的方向（本次不改代码）：用 V6 的 10 个桶向量对 `size_mlp` 做暖启动
（拟合 `bucket_percentile → embedding`），使 step 0 真正等价于 V6。

### P1-2 新条件的学习条件被同时削弱三次

```python
# finetune/train_predictor.py
379: optimizer_groups.append({'params': adaptation_params, 'lr': config['predictor_learning_rate']})
382: optimizer_groups.append({'params': condition_params, 'lr': config.get('condition_learning_rate', 1e-3)})
387:     weight_decay=config['adam_weight_decay']      # 0.1，无 no_decay 分组
```

1. **condition LR 从 V6 的 1e-3 降到 1e-4**：`finetune/config.py:140-142` 默认
   1e-4，且 `kaggle_v1_beta.py:167` 显式设 `KRONOS_CONDITION_LEARNING_RATE=1e-4`
   （所以 `train_predictor.py:382` 的 1e-3 fallback 永不生效）。V6 正是在 1e-3
   下把 `size_emb` 练到范数 4–14；v1-beta 要从**精确 0** 出发，却只给 1/10 学习率。
2. **`weight_decay=0.1` 施加到了 `sector_emb.weight` 和 `norm`**（RMSNorm，
   初值 1.0）。AdamW 的解耦衰减每步作用于全部 87 行，而只有当前 batch 里出现的
   ≤16 个行业行拿到梯度 → 衰减/信号步数比约 **87:16 ≈ 5.4:1**。对 embedding
   table 和 scale 参数施加 weight decay 是公认的反模式。
3. 两个 param group 的 LR 完全相同 → **分组本身失去意义**，新条件没有任何
   学习率优势。

条件分支只有 **126,656 个参数**（`sector_emb` 72,384 + `size_mlp` 54,272），
占模型 0.12%、占可训参数 0.62%。在上述三重削弱下，很可能训练结束时
`sector_emb`/`size_mlp` 的范数远达不到 V6 `size_emb` 的量级，
**「引入行业与连续市值条件」这个实验假设可能根本没被充分检验**。

### P1-3 主干冻结（仅 19.9% 可训）在方案里完全没写

`finetune/config.py:149` 硬编码 `trainable_transformer_layers = 2`，
`configure_trainable_parameters`（`train_predictor.py:178`）先冻结全部，再解冻
`sector_emb / size_emb / size_mlp / norm / dep_layer / head` + 最后 2 层：

```text
v1-beta 总参数     102,437,664
可训练              20,378,076   (19.9%)
冻结                embedding、time_emb、transformer.0 ~ transformer.9
新增条件参数        sector_emb 72,384 + size_mlp 54,272 = 126,656  (0.12%)
被丢弃的 V6 参数    size_emb 9,152
```

方案 §5 只说「继承 Transformer 主体」，从未提冻结；§8 preflight 也没有这一项。
这直接影响两件事：

- **§6 的 40 小时 / 1,058 段预算**。冻结确实省时间（最早的可训层是 layer 10，
  layer 0–9 无需 backward），但这个前提没写进方案，无法据此校核工时估算。
- **容量论证**。用 0.12% 的新参数 + 19.9% 的可训主干去学「全市场 + 行业条件」，
  这个选择应在方案里显式论证，而不是藏在 config 硬编码里。

### P1-4 推送目标存在歧义（RUNBOOK 明确点名过这个历史故障）

`finetune/kaggle_v1_beta_kernel/` 下同时存在两个 metadata，`code_file` 相同、
`id` 不同：

| 文件 | id | code_file |
|---|---|---|
| `finetune/kaggle_v1_beta-metadata.json` | `...v1-beta-120d-to-10d` | `kaggle_v1_beta.py`（`MAX_SEGMENTS_PER_RUN=1`） |
| `kaggle_v1_beta_kernel/kernel-metadata.json` | `...v1-beta-long-150` | `kaggle_v1_beta_long.py`（`=150`） |
| `kaggle_v1_beta_kernel/kernel-metadata-long.json` | `...v1-beta-long-150`（不同前缀） | `kaggle_v1_beta_long.py` |

`kaggle_v1_beta_long.py` 与主脚本**只差一行**（`KRONOS_MAX_SEGMENTS_PER_RUN`
默认 1 → 150）。`KAGGLE_RUNBOOK_CN.md` 把「metadata 与实际提交目标不一致」列为
V3–V6.01 的真实故障来源；这里正是同一形态。另外 `kernel-metadata-long.json`
用 `kernel_sources` 挂上游 kernel 输出，而 §7 步骤 3–4 要求用
**continuation Dataset**——两种机制都能工作，但方案和 metadata 说的不是同一件事。

同目录下的旧版 `kaggle_v1_beta_kernel/kaggle_v1_beta.py`（182 行）只
`copytree(resume_state.parent)`，**即只复制 `checkpoints/`**，直接违反 §7
「每次接力必须继承完整 output 树，而不是只复制 `checkpoints/`」，且没有
`required_history` 校验。它不是线上路径，但留在仓库里就是随时会被误用的陷阱。

### P1-5 §8 要求的「数据集 SHA-256 校验」没有实现

runner 只校验 V6 基座（`kaggle_v1_beta.py:106-109`，`69999253…`，✅ 这点很好），
**数据集 pkl / asset_metadata.csv 没有任何哈希校验**。同时数据发现逻辑不对称：

```python
58:  data_file = next(input_root.glob("**/processed_datasets/train_data.pkl"), None)   # 不查重
119: resume_candidates = list(input_root.glob("**/checkpoints/last_state.pt"))
120: if len(resume_candidates) > 1: raise SystemExit(...)                              # 查重
```

`KAGGLE_RUNBOOK_CN.md` 写的是「输入同名文件必须唯一；模型权重必须校验
SHA-256」。`next(glob)` 在挂载了两个含 `train_data.pkl` 的 Dataset 时会
**静默取任意一个**。

---

## 4. P2 — 会污染指标或让接力变脆

### P2-1 train / val 前向口径不一致，而 best_model 用 val 打分

```python
# train_predictor.py
~554  训练:  model(..., sector_id=..., size_percentile=...)          # 无 teacher forcing
 765  验证:  model(..., use_teacher_forcing=True, s1_targets=token_out[0], ...)
```

`kronos.py:339-344`：teacher forcing 时 `dep_layer` 拿到**真值 s1 embedding**；
训练时拿的是从自身 logits `multinomial` 采样的 s1。所以 **val loss 的 s2 分量
系统性偏乐观**，与 train loss 不同量纲，两条曲线不可直接比较。用于跨段排序尚可
（口径一致），但它低估了自回归误差，而 §C 要求「每段检查训练 loss、forecast
loss、验证 loss」时容易被误读。

### P2-2 首个 chunk 被 12 小时掐断 → 整个 chunk 报废

`train_predictor.py:631` 起的 `STOP_REQUESTED` 路径：保存 `last_state.pt`
（662 行）后 **`break` 跳过验证**，而 `# --- Validation Loop ---` 在 715 行。
因此不会写 `best_model`。而 `export_last_model.py:20` 要求 `best/config.json`
存在，否则 `SystemExit` → runner 的 `check=True`（`kaggle_v1_beta.py:187`）
抛错 → 输出契约检查（191 行）根本执行不到。

首个 chunk 若在段中被 SIGTERM，**有合法的 `last_state.pt` 却无法完成接力**。
第 2 个 chunk 起因为 `copytree` 已继承 `best_model` 而不受影响。

### P2-3 中断态 checkpoint 让漂移守卫恒真

段末保存（832 行）会写入 `scheduler_type` / `scheduler_min_learning_rate` /
两个 LR，但**中断态 `save_resume_state`（662 行）不写这些字段**。于是续训时：

```python
435: saved_scheduler_type = resume_state.get('scheduler_type', config.get('scheduler_type', 'cosine'))
438: if saved_scheduler_type != config.get('scheduler_type', 'cosine'): raise
```

取不到就回退成**当前 config 自己**，等于「拿自己比自己」，**守卫恒真**。
从中断态续训时，scheduler 类型 / min_lr 的一致性检查形同虚设。

### P2-4 `batch_size` 可被 env 覆盖，且不在漂移守卫内

`kaggle_v1_beta.py:170-172` 的 `KRONOS_BATCH_SIZE`、`KRONOS_NUM_WORKERS`、
`KRONOS_MAX_SEGMENTS_PER_RUN` 都是 `os.getenv(...)` 可覆盖的，而漂移守卫
（415-448 行）只覆盖 `effective_epochs / predictor_loss_mode /
history_loss_weight / scheduler_type / scheduler_min_learning_rate`。

`scheduler.load_state_dict`（454 行）会连 `T_max` 一起恢复，所以曲线形状不会
立刻被改坏（这点是好的）。但**实际执行的总步数会与 chunk 1 定下的
`T_max = 1,320,510` 失配**：若中途把 batch_size 调小，总步数 > `T_max`，
`CosineAnnealingLR` 越过最低点后会**重新上升**；反向若调大，则永远到不了
`eta_min`。§C 要求「同一全局 Cosine 学习率计划」，这里缺一道 `batch_size` 守卫。

附带：中途修改 `KRONOS_PREDICTOR_LEARNING_RATE` 会被**静默忽略**
（`base_lrs` 从 checkpoint 恢复），没有任何警告。

### P2-5 851 MB CSV 被解析两次，且本可完全避免

```text
data/a_share_full_market_v1_beta/asset_metadata.csv  =  851,634,832 bytes
```

`dataset.py:185-191`：

```python
185: self.has_inline_size = self.use_size_features and any('size_bucket' in frame.columns ...)
190: if self.has_inline_size and (...) and not self.use_sector_features:
191:     metadata_path = ''
```

抑制分支有**两个独立原因永不触发**：`use_size_features=0` → `has_inline_size`
为 False；且 `use_sector_features=True` → `not use_sector_features` 为 False。
于是 `AssetMetadata` 完整 `pd.read_csv` 851 MB，再 `groupby('symbol')` 切成
5,201 个 DataFrame。**train 和 val 各构造一个 `QlibDataset`，即解析两次。**

而 pickle 面板里本来就有这些列：

```text
train/val pkl 列: ['open','high','low','close','volume','amount','size_bucket','size_percentile','sector']
```

`size_percentile` 走的是 inline 路径（415-420 行，覆盖 CSV 值），
**只有 `sector` 仍依赖 CSV**——而 `dataset.py` 没有 inline sector 读取分支。
加上 `self.indices` 的 **10,564,072 个 Python tuple** 与 84 MB 的 int64
`coverage_order`，在 13 GB 的 Kaggle 实例上是实打实的 OOM 风险，且每段都要
重启 2 个 DataLoader worker。

### P2-6 第二遍逐字重放第一遍

```python
# dataset.py
356: segments_per_pass = math.ceil(self.total_samples / self.n_samples)
357: segment_in_pass = int(epoch) % segments_per_pass
```

`coverage_order` 只在 `__init__` 生成一次（315-322 行），两遍之间**不重新打乱**。
因此 pass 2 的段构成、段内顺序与 pass 1 完全相同（Segment 530 的样本集合与
次序 == Segment 1）。方案 §C 只说「直接进入第二遍」，没说是否有意如此。对
「两遍完整覆盖」的目标无害，但影响第二遍的正则化效果，值得在方案里明确。

### P2-7 每个 chunk 都重新从 ModelScope 下载 V6 基座

`kaggle_v1_beta.py:97-109` 每次都下载 + SHA 校验，**即使 `resume_training=1`
时权重来自 `last_state.pt`、根本不需要基座**。按 150 段/chunk 是 8 次外部依赖，
按 1 段/chunk 是 1,058 次。SHA pin 很好，但按 §7 精神（「数据集只上传一次」）
建议把本地已有的 `artifacts/kaggle_v6_segment568_model_dataset/` 发成 Kaggle
Dataset 挂载。

### P2-8 `experiment_manifest.json` 每 chunk 覆盖，且不在接力必需清单里

`kaggle_v1_beta.py:183` 每次重写，而 `required_history`（125-130 行）里没有它
→ **没有逐 chunk 的配置历史**。若某个 chunk 用了不同的 `KRONOS_BATCH_SIZE`，
事后无从查证。

---

## 5. P3 — 文档与代码不一致（不影响正确性，影响 preflight 可执行性）

| # | 方案原文 | 代码实际 |
|---|---|---|
| 1 | §2 数据目录 `data/a_share_full_market_2026/` | v1-beta 数据在 `data/a_share_full_market_v1_beta/`。两个目录**各有一份 851 MB `asset_metadata.csv`**（重复占约 1.7 GB），只有 `v1_beta` 那份有 `dataset-metadata.json` |
| 2 | §3「训练：截至 2026-07-17（完整 10 日标签边界）」 | 有效 asof 最晚 **2026-07-02**（共 2,792 个交易日，2015-01-05 起）。`window=131` 要求 asof 之后还有 11 根 bar，而面板止于 2026-07-17 → `KRONOS_TRAIN_SIGNAL_END=2026-07-17`（`kaggle_v1_beta.py:163`）**这个过滤器从不生效** |
| 3 | §4 batch 接口：`x(B,120,6)` / `x_stamp(B,120,5)` / `y_stamp(B,10,5)` / `sector_ids` / `size_percentiles` | 实际返回 `(x[B,131,6], x_stamp[B,131,5], sector_id, size_bucket, size_percentile)`（`dataset.py:425`）——历史与未来合成**一个** 131 长张量，没有独立的 `y_stamp`。§8 preflight「验证 120→10 窗口和 batch 字段形状」按字面**无法通过** |
| 4 | §5「重新初始化 `sector_emb`、`size_mlp`」 | `reset_size_conditioning`（`train_predictor.py:202`，1038 行调用）因 `KRONOS_RESET_SIZE_EMBEDDING` 从不设置而是 **no-op**。实际效果靠 `__init__` 零初始化 + `from_pretrained` 的 `strict=False` 默认值实现。**没有任何代码打印 missing/unexpected keys**，§8「验证 V6 主体权重加载，行业/市值分支为新初始化」缺少证据来源 |
| 5 | — | 上述行为依赖 huggingface_hub 默认 `strict=False`。`kaggle_v1_beta.py:115` pin 了 `huggingface_hub==0.33.1`，而 `tokenizer_root` 在 `/kaggle/working` 下每会话重建 → 该分支每次都执行 → **这个 pin 是「意外但关键」的依赖**。若该分支不跑或默认值变成 strict，V6 残留的 `size_emb.weight` 会让加载直接抛异常 |
| 6 | manifest 记录 `KRONOS_CONTEXT_LAYER=10` | `config.py:150` 硬编码 `self.context_layer = 10`，**从不读该环境变量**。manifest 把 inert 参数记成生效参数（`KRONOS_NUM_SECTORS`、`KRONOS_USE_SIZE_PERCENTILE` 等是真生效的） |
| 7 | §C「每段结束检查…」 | `KRONOS_EARLY_STOPPING_PATIENCE=0` → early stopping **完全关闭**（分支要求 `patience > 0`）。这应该是刻意的，但语义要写进文档 |
| 8 | — | `history_replay_ratio=0`、`balance_size_buckets=0`、`reset_size_embedding=0` → `build_balanced_coverage_order` / `select_stratified_replay` / `reset_size_conditioning` 全部是死代码（约 130 行） |

---

## 6. 总体判断

**设计骨架是合理的。** V6 → v1-beta 的两阶段思路、点时行业与市值条件、
`amount / turnover` 市值代理、forecast-only loss、无放回覆盖分段、全局 cosine、
一个 active kernel 的服务端接力，这些都站得住，而且**核心数值全部实测复现**
（10,564,072 / 529 / 1,058 / 1,320,510），**没有未来信息泄露**。

**但有两个问题会让 42–45 小时的投入拿不到可信结论：**

1. **`run.log` 拿不到训练日志**（P0-1）——1,058 段全程无持久审计证据，
   而两处守卫都恒真，不会报警。
2. **验证集只有 9 个交易日**（P0-2）——`best_model` 这个实验唯一的选择信号
   建立在 2026 年 7 月上半月的单一横截面上，与 §3 声称的 7 个月差距巨大。
   这需要重做 `val_data.pkl`（补 120 天历史前缀），不是改训练代码能解决的。

**还有一组问题会让「引入行业 + 连续市值条件」这个实验假设本身检验不充分**
（P1-1 / P1-2）：条件分支从精确 0 起步、只有 0.12% 参数、LR 比 V6 当年低 10 倍、
还要承受施加在 embedding 上的 `weight_decay=0.1`；同时 V6 那个范数 3.9–14.4 的
`size_emb` 被整体丢弃，使 step 0 并不等价于 V6。即使最终 loss 下降，也难以判断
到底是新条件起了作用，还是后两层在补偿被丢掉的旧偏置。

---

## 7. 建议补入阶段 A preflight 的可执行验收项

当前 §8 的 8 条里有 3 条按字面无法执行（batch 字段形状、tee 到 run.log、
V6 主体加载验证）。建议改为：

- 首段跑完后 `wc -l <output>/run.log`，确认包含 step 级日志而不只有几行 JSON；
- 打印 `from_pretrained` 的 missing / unexpected keys，确认恰好是
  `{sector_emb.weight, size_mlp.*}` 缺失 + `{size_emb.weight}` 多余；
- 每段记录 `sector_emb` 与 `size_mlp` 输出的 L2 范数，与 V6 `size_emb` 的
  3.9–14.4 对照，确认条件分支真的在学；
- 打印 `val_dataset` 的窗口数与唯一 asof 交易日数，作为验证集口径的硬校验；
- 打印可训练/总参数比（当前应为 20,378,076 / 102,437,664 = 19.9%），
  并把主干冻结策略写进方案 §5。

---

## 附录：本次审核使用的实测数据

```text
V6 Segment 568 (基座)
  总参数                        102,320,160
  n_layers=12, d_model=832, num_sectors=0, num_size_buckets=10
  context_layer=10, use_size_percentile=False, s1_bits=10, s2_bits=10
  size_emb.weight               (11, 832)
  size_emb 逐桶 L2 范数          10.45 9.89 7.79 5.91 4.25 3.89 5.28 7.31 9.58 14.38 / 0.028
  embedding.emb_s1 行范数均值    1.2584

v1-beta
  总参数                        102,437,664
  可训练                         20,378,076  (19.9%)
  新增 sector_emb                    72,384
  新增 size_mlp                      54,272
  丢弃的 V6 size_emb                  9,152
  逐层参数 transformer.0..11     7,885,722 / 层
  dep_layer 2,773,160 | embedding 3,089,216 | head 1,705,984 | norm 832 | time_emb 113,152

训练集
  监督窗口                      10,564,072   (方案 §6 声称值一致)
  有效 asof 范围                 2015-01-05 → 2026-07-02   (2,792 个交易日)
  段数                          529 / 遍 ×2 = 1,058
  Cosine T_max (bs=16)          1,320,510

验证集
  窗口                          45,409
  有效 asof 范围                 2026-07-06 → 2026-07-16   (9 个交易日)
  每次验证实际采样                2,000

train / val 窗口重叠            0
asset_metadata.csv 大小         851,634,832 bytes (被解析 2 次)
```

**本报告为只读审核产出，未修改任何训练代码或配置。**
