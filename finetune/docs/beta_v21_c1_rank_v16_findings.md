# Beta v2.1 C1 dual-T4 排序轮验证记录

只覆盖 dual-T4。没有 TPU 轮次。

预报地板是 v6 Seg1 的 weighted_forecast_loss **2.31245745**。排序的权重父本是 Seg155 best，weighted_forecast_loss **2.31236787**。Coverage seed **20261002**，same-day batch，段内不做 signal_date 排序。一整趟覆盖是 **534** 个 segment。

## 标签和 pair 规则

Utility：

`p0 * 0.046 + p1 * (-0.034) + p2 * (expm1(log_return_10) - 0.004)`

Ranking pair：同一 signal date，且 `|utility gap| >= 0.005`。

训练 loss 是

`softplus(-(score_i - score_j) * sign(utility_i - utility_j))`

对合格 pair 取平均，**不按 gap 大小加权**。

均匀 3 类交叉熵是 `ln(3) ≈ 1.0986`。Barrier loss 做了类别重加权，所以它不是准确率。

## 已关闭、不再续其权重的轮次

### v13

commit `f0c374e`。SwanLab `beta_v2_1_c1_dual_t4_rank`。学习率 2e-5，warmup 0，ranking weight 很小。Seg160–163 的 ranking loss 从 **0.68740** 变到 **0.68790**，变差。最后跑完的 Seg165：ranking **0.68779835**，weighted_forecast_loss **2.35045437**。太热，辅助头梯度是陈的。

### v15

commit `da226d30`。SwanLab `beta_v2_1_c1_dual_t4_rank_1e5`。两侧学习率都是 1e-5，warmup_constant，从 1e-6 起，warmup ratio 0.05，ranking weight 约 0.05，权重来自 Seg155。用户在大约 seg16 停掉。Ranking 平在大约 **0.6871–0.6883**。报告过的最好是 Seg3 **0.68708724**。weighted_forecast_loss 从 **2.33010** 升到 **2.35603**。输出目录被 v16 复用，best_model 被盖掉，不能再打分。

### v16

commit `49743d9`。SwanLab `beta_v2_1_c1_dual_t4_rank_warm`。ranking weight **0.5**。Trunk 1e-6 保持。Head 从 1e-6 warmup 到 1e-5，ratio 0.05。权重来自 Seg155。运行时间上限 10800s。停在 Seg13/534，不是崩溃。

最好是 Seg10：ranking_loss **0.68692991**，weighted_forecast_loss **2.33623456**。

Seg13：ranking **0.68701018**，weighted_forecast_loss **2.33270514**。

Seg1：return_loss **0.32561953**，barrier **1.22861401**，ranking **0.68866079**。

Seg13：return **0.28396697**，barrier **0.82556002**，ranking **0.68701018**。

开训前的 calibration 阈值是 ranking_loss **0.68731890**。这是 Seg155 权重的预训练校准，不是预报那个数。

## Pair 拆分（只作标签，不是正式训练 pickle）

从本地 K 线重做。48 个日期，117336 个样本，3358/4678 个训练标的，74506817 个合格 pair。

- barrier 驱动：18568706，占 pair 的 **24.92%**，占 `|gap|` 的 **24.62%**。
- return 驱动：**75.08%**。
- 以 barrier 为主的成分：pair 的 **37.0%**。
- 0 对 1 的 pair：46042871（**62%**）。标签 gap 正好是 `float32(0.046) - float32(-0.034) ≈ 0.080000`。

结论：barrier 不是污染项。方案 C（丢掉 barrier）关闭。

## v16 Seg10 best 的验证打分

数据：`/workspace/kairos/ref/val_data.pkl`，sha256 `4cce31bc3e70eab83d5b7ea05f19fce04aa57a87f3acf00b882ddfbac4219bf7`。

123836 个样本，516 个有窗口的标的，242 个交易日（2025-07-03 到 2026-07-02），symbol holdout 520 个名字。全截面合格 pair **15935132**。

Pairwise accuracy（分数打平算错）：

- 总体 **0.66304848**
- return 驱动 **0.68324952**（11819038 pair）
- barrier 驱动 **0.60504279**（4116094 pair）
- 按配方 batch 内 pair 的 accuracy **0.66335894**

日期内 Spearman rank IC：均值 **0.29427218**，中位数 **0.29250374**，242 个日期。

`|score_delta|` 的 p50 是 **0.01743279**，标签 `|gap|` 的 p50 是 **0.08000000**（4.6 倍，不是 8 倍）。**93.82%** 的 `|score_delta|` 小于 0.05。

按样本加权重算的 batch ranking loss **0.68692975**，日志是 **0.6869299138937892**，差 **1.59e-7**。重算是 CPU fp32，训练验证是 CUDA fp16 autocast。

`softplus(0) = 0.693147`，`softplus(-0.01) = 0.688160`。

## 覆盖度

验证集每天标的数：最小 477，中位数 512，最大 515。训练重建的中位数是 2349.5。`(2349.5 / 512)^2 ≈ 21.1`。观察到的每天合格 pair：65848 对 1552225，比值 **23.6**。Pair 规则相同。差距是 n 的平方，不是另一套定义。

## 教训

每个 kernel 版本要用自己的输出目录。v16 把 v15 的 best_model 写进了 `beta_v2_1_c1_dual_t4_wc`。

## 下一轮（本提交，v17）

- 最佳检查点指标改为验证 pairwise accuracy，越大越好。Pair 规则不变：同一天，`|utility gap| >= 0.005`，分数打平算错。
- 训练 loss 仍是 pairwise logistic（softplus），不改。
- 记录 rank IC（日期内 score 对 utility 的 Spearman 均值）。只记录，不拿它选最佳。
- 不加 temperature scaling。不改 utility 目标，不改 drop barrier。
- 新输出目录 `beta_v2_1_c1_dual_t4_rank_acc`。新 SwanLab id `beta_v2_1_c1_dual_t4_rank_acc`，不用 `rank_warm`。
- 预报红线只监控：相对 **2.31236787** 的 **+0.015**。代码里不因此停跑。
- 权重从 v16 Seg10 best 接着来（ranking_loss **0.68692991**），AdamW 重新初始化，不读 last_state。
- 学习率和 v16 相同：ranking weight 0.5；trunk / adaptation 1e-6 保持；head / condition 从 1e-6 warmup_constant 到 1e-5，warmup ratio 0.05。
- same-day batch，coverage seed 20261002，段内不按 signal_date 排序，offset 0。
- 2026-10-05 `kaggle quota`：GPU 已用 0.00h，剩余 30.00h，总额 30.00h，刷新 2026-10-10T00:00:00Z。GPU 会话上限 12 小时，所以本轮运行时间上限是 43200s，不是 v16 因额度不够而用的 10800s。

## v17 结果（用户停在 ~Seg17，2026-10-05 ~13:50 Asia/Shanghai）

父本是 v16 Seg10 ranking best。ranking weight **0.5**，trunk LR 1e-6 hold（`KRONOS_SPLIT_TRUNK_HEAD_LR=1`，Adaptation LR），heads / condition warmup_constant 1e-6→1e-5 ratio 0.05。输出 / SwanLab `beta_v2_1_c1_dual_t4_rank_acc`。最佳选择改为验证 pairwise accuracy。

验证 pairwise accuracy 从父本校准约 **0.66304** 升到平台 **~0.6643–0.6657**。`best_model` 里 Seg9 **0.66559216** 被 Seg15 **0.66574077** 盖掉。同时 weighted_forecast_loss 从约 **2.33** 升到 **2.34+**，并且和 accuracy **同向移动**：排序变好时预报也在变差。

结论：**trunk 漂移在用预报换排序**。继续用小 trunk LR 训练整棵树，无法同时保住 Seg155 的预报地板（weighted_forecast_loss **2.31236787**）。

## v18 方案（本提交）：冻结 trunk，只训两个 aux 头

- 父本改回 **Seg155 forecast best**（weighted_forecast_loss **2.31236787**，local segment 26）。user281434 kernel 输出里的 `beta_v2_1_c1_dual_t4_wc/best_model` 已被 v16 Seg10 盖掉，所以把本地 Seg155 副本（SHA `8b11a759e72d…`）做成数据集 `luckfu/kronos-beta-v21-c1-seg155-forecast-best` 挂进 dual-T4 kernel。
- Seg155 safetensors **没有** `return_head` / `barrier_head` 键（197 tensors；Seg10 ranking best 是 201）。在 `use_beta_v21_auxiliary=True` 下两个头会随机初始化——这对「只训头」是正确起点，不是缺陷。
- `KRONOS_TRAIN_BETA_V21_HEADS_ONLY=1`：`requires_grad=False` 冻住 trunk、预报头（`norm`/`dep_layer`/`head`）和 size/sector condition（它们进预报通路）。**只有** `return_head`、`barrier_head` 可训练。DDP `find_unused_parameters=True`。
- Head LR：warmup_constant **1e-5 → 1e-4**，warmup ratio **0.01**。头是很小的 Linear(d_model→4/3)；v17 的 1e-5 偏慢。
- Loss：与 v17 相同的 total（ranking weight **0.5** + return/barrier aux）。trunk 冻住后预报梯度无效，但 loss 形式不变。same-day ranking，coverage seed **20261002**，段内不按 signal_date 排序。
- 最佳选择：验证 pairwise accuracy。rank IC 只记。预报红线仍是 +0.015 vs **2.31236787**（只监控）。另加冻结 sanity：若 weighted_forecast_loss 相对开训校准偏离 **>1e-4**，打响 WARNING（不停跑）；期望每段都停在 ~**2.3124**。
- 新输出目录 / SwanLab id：`beta_v2_1_c1_dual_t4_rank_frozen`（绝不复用 rank_acc / rank_warm / wc）。
- 运行：43200s，MAX_SEGMENTS 30，Dual T4 only。
