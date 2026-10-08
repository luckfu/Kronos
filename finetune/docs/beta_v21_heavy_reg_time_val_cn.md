# Beta v2.1 102M 重正则重训 + 时间隔离验证集（预注册）

> **状态：已准备，未推送（2026-10-07 CST）。** Kaggle kernel 已在本地构建并通过测试，
> 推送前需用户明确批准。本文件在任何 GPU 运行前提交，规则一经提交不再修改；
> 运行后只追加「结果」一节。

## 1. 假设

- 事实：Small C2（`c2_small_best_seg179`，d_model 512 / 8 层，resid/ffn dropout 0.25、attn 0.1、token 0.1，
  无 aux）密封 OOS IC **0.177**；Beta v2.1 Best@475（102M，resid/ffn 0.2、attn 0、token 0）**0.155**；
  C1 系列在 Best@475 上继续训练都更差。
- 现有 `temporal_symbol_validation_v1` 只做了**股票隔离**（520 只 holdout），**没有时间隔离**：
  val 信号 2025-07-01..2026-07-02 与训练期（train_data.pkl 截止 2026-07-17）完全重叠。
  val→OOS 的 IC 跌幅：C2 −0.058，Best@475 −0.160，Seg155 −0.178 —— 越大的模型跌得越多。
- 假设：102M 模型在记忆训练期的「市场状态」，旧 val 奖励这种记忆。把正则加到 C2 水平
  （dropout 0.25/0.25/0.1/0.1）并用**时间隔离**的验证集选 checkpoint，能得到外推更好的 102M 模型。

## 2. 数据设计

### 2.1 Best@475 的训练数据（不变）

- 数据集 `luckfu/a-share-120d-temporal-symbol-holdout` →
  `data/a_share_full_market_v1_beta_temporal_symbol_validation_v1/`（data_manifest sha `17afbeed…`）。
- 训练股票 4,678 只，holdout 520 只（90/10，`symbol_split.csv`）；train_data.pkl 最后一根 bar 2026-07-17，
  最后一个完整窗口的信号日 **2026-07-02**（120 lookback + 10 预测 + 1）。
- 祖先线：v2.0 Best@687 ← … ← V5/V6，其中 V5/V6 训练信号到 2026-07-16、**标签到 2026-07-31**；v1.1+ 祖先用过全股票池。

### 2.2 时间隔离验证集（本实验的选择集）

| 项目 | 内容 |
|---|---|
| 股票 | 同一批 520 只 holdout（相对本次重训的训练股票是隔离的） |
| 信号日 | **2026-07-17 .. 2026-08-10，共 17 个交易日** |
| 窗口数 | **8,784**（每日 515–517；identities sha `3c629d8b…3f9c`） |
| 10 日标签 | target 2026-07-31 .. 2026-08-24（raw close return_10d） |
| bar 来源 | **旧包** `kronos_small_0_1_time_oos_evaluation_20260826/evaluation/evaluation_panel.pkl`（manifest `kronos_v1_beta_checkpoint_evaluation_20260826`，sha `6827d0eb…0bb6`），截到 2026-08-25（08-10 的窗口需要 08-25 这根 bar） |
| 一致性校验 | 与 holdout `val_data.pkl`（sha `4cce31bc…`）在 ≤2026-07-17 的 131,472 行 OHLCVA/size_percentile/sector 完全一致 |
| 构建 | `finetune/build_time_disjoint_val_panel.py`（kernel 内现场构建，按 SHA 校验，显式拒绝密封 manifest 名与 `august_raw.csv`） |

**绝不读取**数据集根目录的密封包（`evaluation_manifest.json` = `kronos_beta_v2_time_oos_through_20260903`、
其 `evaluation_panel.pkl` / `evaluation_samples.jsonl` / `august_raw.csv`）。密封窗口 2026-08-11..09-03
已退役，不用于任何选择。

### 2.3 诚实说明（局限）

1. **时间隔离只对本次重训严格成立**：本次训练信号 ≤ 07-02（bar ≤ 07-17），val 信号 ≥ 07-17。
   但祖先 V5/V6 的训练标签到 07-31：07-17..07-30 这 13 天信号的标签路径部分落在祖先见过的区间。
   标签路径完全在 07-31 之后的「严格子集」只有 **7 天**（07-31、08-03..08-10），只作诊断报告。
2. 08-03..08-10 这 6 天曾用于 Best@475 / C2 的事后审计（release README 的 August audit）。
3. 07-29..08-10 的标签用到了 08-11..08-24 的 bar（密封期内的价格，但来自 2026-08-26 旧包，不是密封包）；
   这些 bar 只作为本 val 的标签，与密封集的信号/样本无关，密封集本身已退役。
4. 17 个日期的 10 日标签高度重叠，独立期大约只有 2 个：配对 t 偏乐观；在 12 个 snapshot 中取最大值还会再抬高。
   **val 关只是「准入」，真正的检验是下一个密封窗口的一次性确认。**
5. 更干净的备选（未采用，待决）：下载 2026-09-03 之后的新数据切出一段做 val，但会消耗下一个密封窗口。

## 3. 训练配置（相对 Best@475 的精确差异）

| 参数 | Best@475（release config.json） | 本实验 |
|---|---|---|
| resid_dropout_p | 0.2 | **0.25** |
| ffn_dropout_p | 0.2 | **0.25** |
| attn_dropout_p | 0.0 | **0.1** |
| token_dropout_p | 0.0 | **0.1** |
| use_beta_v21_auxiliary | true | **false**（forecast-only；return_head/barrier_head 4 个张量丢弃） |
| AdamW weight decay | 0.1 | 0.1（不变；与谱系和 C2 相同） |
| 其余结构 | d_model 832、12 层、16 头、ff 2048、context_layer 10、86 行业、size_percentile MLP 64、s1/s2 10 bit | 不变 |

- dropout 通过新增的 `KRONOS_RESID/FFN/ATTN/TOKEN_DROPOUT_P` 覆盖父本 config.json 再 `from_pretrained`，
  权重逐位不变。kernel 训练前在 CPU 上重建模型核对：197 个张量与父本完全相等、只丢 4 个 aux 张量、
  每个 Dropout 模块的 p 与 SDPA `attn_dropout_p` 都是新值（本地已用真实 Best@475 验证）；
  训练日志 `Predictor dropout (modules): resid=0.2500 ffn=0.2500 attn=0.1000 token=0.1000` 不符即中止。
  导出的 snapshot config.json 也带新 dropout（推理时 eval 模式，dropout 不生效）。
- `dep_layer`（DependencyAwareLayer）内部 dropout 一直为 0，与 C2 相同，不受配置影响。
- 全部参数可训练、单一 LR 族（条件层已训练过，C2 的双 LR 只用于 bootstrap，这里不需要）；
  fresh AdamW；不读父本 last_state。
- 样本顺序：coverage 置换（随机打乱，`KRONOS_SAME_DAY_RANKING_BATCHES=0`，段内**不按时间排序**），
  coverage seed 20261007，每段 20,000 样本，batch 32 × 2 GPU（有效 64），AMP fp16。
- 训练期 val（WFL，仅记录）也指向时间隔离 panel（8,784 窗口，每段 ~25 s，替代旧 val 的 ~320 s）。

## 4. 学习率调度（C2 式 WSD）

- 新调度器 `warmup_constant_cosine`：48 段 × 313 步 = **15,024 步**。
  - 第 1 段（313 步）线性 warmup 1e-6 → 1e-5；
  - 保持 1e-5 到第 32 段末（第 10,016 步）；
  - 第 33–48 段 cosine 1e-5 → 1e-6。
  - 段末 LR：Seg4–32 = 1e-5，Seg36 8.68e-6，Seg40 5.5e-6，Seg44 2.32e-6，Seg48 1e-6。
- 12 个可续训块 × 4 段（last_state.pt 续训，全局一个 48 段计划；测试验证 2+2 块续训与一次跑完的 LR/权重一致），
  每 4 段存 snapshot。新的 resume guard 字段（decay start、dropout 覆盖）只在使用时加入，旧 checkpoint 不受影响。
- 总样本 960k ≈ 训练窗口（9,457,646）的 10%。

## 5. 评估协议

- 驱动：`finetune/val_gen_ic_driver.py` 新增 `val_contract` 覆盖（只换验证集；解码 / 打分 / 标签 / WFL 与 Step 1 完全相同）。
- 解码：生产臂 `prod_t065_p80_n5`（T 0.65、top_p 0.8、N 5、seed 20260906）；score = N5 均值 close_d10/last_close − 1；
  label = raw close return_10d；**全部 17 天、全部股票**（不抽样）。
- 顺序：第 1 块训完后先打 **Seg0（Best@475）**、**参考 Seg9**（余弦 pilot Seg9，数据集
  `luckfu/kronos-beta-v21-c1-cosine-pilot-best475-seg9`，sha `f9d3da03…`），再打 seg004；之后每块训完立刻打新 snapshot。
  最后若「按 val WFL 的 best_model」与已打 snapshot 不同，作为诊断再打一次（不参与选择）。
- 每个 checkpoint 的 `*_val_summary.json`、累计 `comparison.json`、`heavy_reg_selection.json`（含逐日配对统计）
  在其 17 个 shard 落盘时立即写出；全部指标同步到 SwanLab。

## 6. 选择规则与成功标准（预注册）

- 候选：周期 snapshot `heavyreg_seg004 … heavyreg_seg048`（12 个）。Seg0、参考 Seg9、best_model 都不是候选。
- **C\*** = 时间隔离 val 上 return10d rank IC 日均最高的候选。
- **val 关（全部满足才通过）：**
  1. C\* − Seg0 逐日配对 t ≥ **2.0**（n = 17）；
  2. C\* 在 ≥ **10/17** 天上胜过 Seg0；
  3. 终点 Seg48 − Seg0 日均 IC 差 **> 0**（防止只靠挑最大值）；
  4. Seg0 与 Seg48 都已打完、配对 n = 17。
- **最终确认（一次性）：** 通过 val 关后冻结 C\*，在 **2026-09-03 之后的下一个密封窗口**上与 Best@475（以及 C2）
  用同一生产配方对比，C\* − Best@475 配对 t ≥ 2 才算成功。该窗口只用一次。
- **失败即停：** val 关不通过或密封确认不通过 → 停止该方向，不调参重跑、不换选择规则。
- 诊断（不影响判定）：严格子集 7 天上的差值；参考 Seg9 vs Seg0 在新 val 上的方向是否与密封 OOS 一致
  （密封 OOS：Seg0 0.1546 > Seg9 0.1455；旧 val：Seg9 0.3254 > Seg0 0.3141）——一致说明新 val 更可信。

## 7. 资源与 Kaggle

| 项目 | 内容 |
|---|---|
| 将推送的 kernel | `wynstonliu/kronos-beta-v21-heavy-reg-time-val`（私有，2× T4，联网，docker pin `sha256:37c64f7d…d461`） |
| 账号 | wynstonliu（`export KAGGLE_API_TOKEN=$(kaggle auth print-access-token)`）；luckfu GPU 配额已用完 |
| 数据集 | `luckfu/a-share-120d-temporal-symbol-holdout`（公开，可读）；`luckfu/kronos-beta-v21-c1-cosine-pilot-best475-seg9`（私有，组共享，wynstonliu 可读；缺失时自动跳过参考，不影响主流程） |
| 模型 | Best@475 与 tokenizer 从 ModelScope `luckfu/Kronos-A-Share-Beta-V2-1` 按 SHA 获取 |
| 输出 / SwanLab | `beta_v2_1_heavy_reg_time_val_best475`；<https://swanlab.cn/@roc_fu/finance/runs/beta_v2_1_heavy_reg_time_val_best475> |
| GPU 时间估算 | 启动 ~10 min；训练 48 段 × ~3 min（~150 s 训练 + ~25 s val + 存盘）≈ 2.4 h；块重启 12 × ~2.5 min ≈ 0.5 h；打分 14 个 checkpoint × 17 shard × ~115 s ÷ 2 GPU ≈ 3.8 h（+ 每轮启动 ~0.2 h）→ **合计约 7.1 h**，最坏约 9.5 h |
| 时间保护 | 9.5 h 后不再开新训练块；11.5 h 后 eval worker 停止领取 shard；剩余 < 20 min 不开新一轮打分 |
| 周配额 | wynstonliu 本周已用约 6.9 h（OOS seg9 ~5.1 h + val-C2 1.7 h）；剩余配额与重置时间 API 不可见（若为标准 30 h/周，约剩 23 h，可容纳一次 12 h 会话） |

## 8. 产物

- `finetune/build_time_disjoint_val_panel.py`：时间隔离 panel 构建（含密封包拒绝与 SHA 固定）。
- `finetune/kaggle_beta_v21_heavy_reg_time_val.py`：kernel runner；
  `finetune/build_kaggle_beta_v21_heavy_reg_time_val_kernel.py`：构建器（复用 cosine pilot 构建器的 docker pin 与源码清单）；
  `finetune/kaggle_beta_v21_heavy_reg_time_val_kernel/`：已构建的 staging（未推送）。
- `finetune/train_predictor.py` / `finetune/config.py`：`warmup_constant_cosine` 调度器、dropout 覆盖与日志、opt-in resume guard。
- `finetune/val_gen_ic_driver.py` / `finetune/evaluate_beta_v21_val_gen_ic.py`：`val_contract` 覆盖、可选信号区间、
  本地 CPU 冒烟开关 `KRONOS_VALGENIC_ALLOW_CPU=1`（默认行为不变）。
- `tests/test_heavy_reg_time_val.py`：配方、WSD 形状、续训等价、dropout 覆盖、panel 守卫、真实数据 8,784/17、
  QlibDataset 窗口一致、driver CPU 端到端、选择关、staging。
- 本地 CPU 冒烟（真实 Best@475 + 真实 panel 子集）：dropout/父本校验通过（10 s）；driver 在 2 天 × 6 只上完成解码打分（161 s）；
  `train_predictor.py` 主入口用 kernel 的完整环境变量（仅缩小段长/批量）跑 2 段、1+1 块续训：dropout 日志正确、WSD 计划打印、
  训练期 val 正好取到 6 只 × 17 天 = 102 个窗口、snapshot config.json 带新 dropout 且无 aux、段末 LR 1e-5 → 1e-6。
  （真实 train_data.pkl 875 MB 未在本地下载，训练集规模只在 Kaggle 上验证。）

## 9. 待决事项

1. 是否加一个 weight decay 臂（如 0.2）——本预注册只跑 WD 0.1 单臂。
2. 是否改用 2026-09-03 之后的新数据做更干净的 val（会占用下一个密封窗口）。
3. C2 在新 val 上的参考值需单独的低成本 eval-only kernel（C2 需在 e4b92bb 模型代码上解码，不放进本 kernel）。
4. Seg155 数据集对 wynstonliu 403，故参考用 Seg9。
5. wynstonliu 剩余周配额未知。

## 结果

### 中间结果（2026-10-07 19:14 CST，训练仍在进行）

> **中间结果，不是最终结论。** kernel `wynstonliu/kronos-beta-v21-heavy-reg-time-val` 已在 2026-10-07 下午获批推送，
> 19:13 拉取日志时仍为 RUNNING（文首「未推送」状态行是预注册时的原文，按规则不改）。已训到第 40 段，
> Seg40 snapshot 已存（sha `5c9f08fd…4112`），正在打分（17 个 shard 完成 2 个）；Seg44、Seg48 尚未训练/打分，
> 三者都在余弦退火段（段末 LR：Seg40 5.5e-6，Seg44 2.32e-6，Seg48 1e-6）。
> 第 6 节预注册规则保持原样；用户认为该关口定得过死，本节**不写通过/不通过判定**，只记录数据与走势。
> 数据来源：19:13 CST `kaggle kernels logs -f` 实时日志中的 `checkpoint_scored` / `checkpoint_val_gen_ic` 记录；
> 机读版 `finetune/reports/beta_v21_heavy_reg_time_val_interim.json`。

时间隔离 val（520 只 holdout × 2026-07-17..08-10 共 17 个信号日，8,784 窗口），生产解码 T0.65 / top_p 0.8 / N5，
标签 raw close return_10d。配对差均为「该 checkpoint − Seg0」逐日 IC 差；严格子集为标签路径完全在 07-31 之后的 7 天。

| checkpoint | 日均 IC | pooled IC | ICIR | IC>0 天数占比 | 十分位多空 | 配对差 / t / 胜天（17） | 严格 7 天 差 / t / 胜天 | 加权 forecast loss | 非加权 forecast loss |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| Seg0（Best@475，父本） | **0.1312** | 0.1892 | 1.013 | 76% | 1.96% | — | — | 2.3453 | — |
| 参考 Seg9（余弦 pilot） | 0.0994 | 0.1670 | 0.765 | 76% | 1.39% | −0.0318 / −4.24 / 3 | −0.0164 / −1.25 / 3 | 2.3317 | — |
| Seg4 | 0.0924 | 0.1501 | 0.721 | 71% | 1.09% | −0.0389 / −7.12 / 1 | −0.0252 / −3.83 / 1 | 2.3338 | 2.3382 |
| Seg8 | 0.1126 | 0.1820 | 0.859 | 82% | 2.01% | −0.0186 / −3.14 / 4 | −0.0025 / −0.51 / 3 | 2.3377 | 2.3423 |
| Seg12 | 0.0963 | 0.1652 | 0.795 | 82% | 1.79% | −0.0349 / −3.72 / 3 | −0.0006 / −0.09 / 3 | 2.3366 | 2.3399 |
| Seg16 | 0.1045 | 0.1758 | 0.841 | 82% | 1.65% | −0.0268 / −3.88 / 4 | −0.0135 / −1.54 / 3 | 2.3448 | 2.3488 |
| **Seg20**（当前最佳候选） | **0.1128** | 0.1754 | 0.910 | 82% | 2.38% | −0.0185 / −2.28 / 4 | +0.0029 / +0.22 / 3 | 2.3360 | 2.3400 |
| Seg24 | 0.0995 | 0.1592 | 0.829 | 82% | 1.97% | −0.0318 / −3.61 / 3 | −0.0019 / −0.17 / 3 | 2.3323 | 2.3359 |
| Seg28 | 0.0535 | 0.1041 | 0.376 | 65% | 1.38% | −0.0777 / −5.48 / 0 | −0.0434 / −3.93 / 0 | **2.3310** | **2.3350** |
| Seg32（恒定 LR 段末） | 0.0950 | 0.1537 | 0.620 | 71% | 1.48% | −0.0362 / −3.88 / 4 | −0.0458 / −3.33 / 1 | 2.3382 | 2.3419 |
| Seg36（退火第 1 个） | 0.0855 | 0.1423 | 0.643 | 65% | 1.62% | −0.0457 / −5.15 / 1 | −0.0223 / −2.45 / 1 | 2.3401 | 2.3447 |
| Seg40 | 打分中（2/17 shard） | | | | | | | 2.3451 | 2.3492 |

- 加权 forecast loss = 训练日志「Validation Weighted Forecast」（同一 8,784 窗口时间隔离 panel，driver 字段 `wfl_subsample`）；
  非加权 = 训练日志「Validation Forecast」（driver 字段 `wfl_full_val_logged`）。Seg0 / 参考 Seg9 的加权值由 driver 计算。
  全部 40 段中加权 loss 最低的是第 30 段（2.3267，非 snapshot，未打分）。

**观察（事实陈述）：**

1. 到 Seg36 为止，**没有任何 snapshot 的日均 IC 超过 Seg0**：配对差全为负（−0.018 到 −0.078），t 在 −2.28 到 −7.12，
   17 天里胜 0–4 天。
2. 当前最佳候选是 **Seg20**（0.1128，−0.0185，t −2.28，4/17），Seg8（0.1126）几乎并列；Seg8 之后 IC 在 0.095–0.113 间摆动，
   Seg28 跌到 0.054，进入退火后 Seg32 0.095、Seg36 0.086，尚未出现回升到 Seg0 水平的迹象。
3. **WFL 与 IC 再次背离**（与 C1 一轮相同）：Seg0 加权 loss 最高（2.3453）而 IC 最高；Seg28 loss 是已打分 snapshot 里最低的
   （加权 2.3310 / 非加权 2.3350），IC 却最低（0.0535）。11 个已打分 checkpoint 上 Spearman(加权 loss, IC) = +0.40（n=11，p=0.22，
   方向为 loss 越低 IC 越低，统计上不显著）。forecast loss 不能用来选 checkpoint。
4. **新 val 本身看起来可信：** 参考 Seg9 在新 val 上排在 Seg0 之后（0.0994 < 0.1312），与密封 OOS 方向一致
   （0.1455 < 0.1546），与旧泄漏 val 相反（0.3254 > 0.3141）；Best@475 在新 val 上为 0.131，接近其密封 OOS 0.155，
   而旧 val 上是 0.314。
5. **严格 7 天子集：** Seg8 / 12 / 20 / 24 与 Seg0 基本持平（|差| ≤ 0.003），Seg16 −0.013，Seg4 −0.025；
   从 Seg28 起严格子集也明显为负（Seg28 −0.043、Seg32 −0.046、Seg36 −0.022）。即 Seg24 之前的落后主要集中在 07-17..07-30
   （标签与祖先 V5/V6 训练标签部分重叠的 10 天），Seg28 之后在干净日期上也落后。
6. 剩余 Seg40 / 44 / 48（退火段）跑完后在本节之后追加完整结果；终点 Seg48 与 Seg0 的比较届时补上。

### 收尾结果（2026-10-09 00:14 CST，kernel 已结束）

> kernel `wynstonliu/kronos-beta-v21-heavy-reg-time-val` 状态 COMPLETE。`kaggle kernels logs -f` 于 2026-10-09 00:14 CST 拉全量日志（sha256 `a807ed68…b934`），末行 `phase=done`，`completed_segment=48`，`total_seconds=24448`。本节只补 Seg40 / Seg44 / Seg48 以及训练后追加的诊断点 Seg30，**不下通过或不通过结论**。第 6 节预注册规则原文未改。机读版（含上面中间表的全部行）：`finetune/reports/beta_v21_heavy_reg_time_val_final.json`。

| checkpoint | 日均 IC | pooled IC | ICIR | IC>0 天数占比 | 十分位多空 | 配对差 / t / 胜天（17） | 严格 7 天 差 / t / 胜天 | 加权 forecast loss | 非加权 forecast loss |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| Seg40 | 0.0801 | 0.1392 | 0.532 | 65% | 1.11% | −0.0511 / −4.62 / 3 | −0.0307 / −2.46 / 1 | 2.3451 | 2.3492 |
| Seg44 | 0.0850 | 0.1424 | 0.667 | 71% | 1.29% | −0.0462 / −7.49 / 0 | −0.0373 / −10.01 / 0 | 2.3381 | 2.3419 |
| Seg48（终点） | 0.0891 | 0.1454 | 0.623 | 71% | 1.10% | −0.0421 / −5.33 / 3 | −0.0339 / −3.18 / 1 | 2.3386 | 2.3426 |
| 诊断 Seg30（全程加权 loss 最低，非计划 snapshot） | 0.0841 | 0.1428 | 0.633 | 65% | 1.77% | −0.0472 / −6.08 / 0 | −0.0344 / −3.82 / 0 | **2.3267** | 2.3267 |

**补记（事实陈述）：**

1. 退火段三个 snapshot 的日均 IC 都低于 Seg0（0.1312），也低于此前最高的 Seg20（0.1128）。Seg44 在 17 天里一天都没超过 Seg0；终点 Seg48 配对差 −0.0421（t −5.33，3/17），严格 7 天同样为负（−0.0339，t −3.18，1/7）。
2. 全部已打分点里，日均 IC 最高的仍是 Seg20（0.1128，配对差 −0.0185，t −2.28，4/17）。Seg40 之后没有出现回到 Seg0 的回升。
3. 全程 48 段加权 forecast loss 最低仍是第 30 段（2.3267）。kernel 在 Seg48 之后把它当作诊断 checkpoint 打了分：日均 IC 0.0841，17 天 0 胜，严格 7 天也是 0 胜。loss 最低的点和 IC 最高的点不是同一个。
4. 驱动日志最后一条 `selection_updated` 把 c* 记在 Seg20、终点记在 Seg48，并带了预注册规则自己的布尔字段。那些字段留在 JSON 的 `raw_driver_selection_updated` 里，本文不把它当成结论。
