# Kairos E1 -> R1 全量差异审计

日期：2026-09-30

范围：E1 小样本门控实验、R1 八 chunk 全量训练、R2 Chunk 1/2 停止现场

结论性质：代码与已保存报告的静态审计；不把无法由仓库或 Kaggle 报告证明的内容写成事实。

## 0. 执行摘要

目前不能把 R1 的 0.9182 归因于已证实的八 chunk 接力事故。当前代码显示，正常
完成的 R1 接力会恢复模型、AdamW 状态、GradScaler、固定 row-group 顺序和已处理
样本数；八个 chunk 的分区由同一个 `group_order_hash` 约束，分区边界不重叠。
但是 R1 没有 scheduler，因而不存在 scheduler 状态可恢复；同时没有保存 Python、
NumPy 或 PyTorch RNG 状态。由于模型 dropout 为 0、数据顺序由固定 seed 派生，
未保存 RNG 是严格复现缺口；没有 scheduler 是优化策略而非恢复事故。

R1 的主要问题更像训练问题与实验决策问题：全量模型实际是 hidden 768、22 层、
12 heads，和 E1 的 hidden 256、4 层、4 heads 不是同一规模；R1 只在最后做一次
全量验证，不按验证指标选 best；E1 的门控增益只在小样本和时间依赖的验证上
成立，E1 报告明确写着“不据此启动全量训练”。仓库没有找到一份记录，证明
R1 启动前完成了“稳定超过常数先验和 C”的正式门禁签字。

R2 Chunk 2 的恶化不能拿来证明 R1 接力失败。R2 四份脚本当前都能被合成控制流
复现出另一类明确 bug：跨越一个 row group 后，`group_order_pos` 前进了，但
内层循环仍使用旧的 `rows` 列表，导致旧 row group 被重复训练，后续 row group
被遗漏。这是 R2 管线缺陷，必须修复后才能谈“干净 R2”。

回答交付要求：

1. **R1 失败中接力/管线事故占多少？** **不能量化，不能写成 0% 或任意比例。**
   当前源码和合成测试没有复现 R1 八分区恢复的覆盖错误，但这不是历史运行无事故
   的证明。曾有跨 chunk 游标修复提交；本次未重新取得八次实际运行版本、全部
   checkpoint 和逐边覆盖账本，因此历史事故贡献仍未排除。可以确认的 R1 风险是
   RNG 状态未保存、前序完成检查不足，以及 row-group 内中断粒度较粗。
2. **从零用修好的 R2 接力代码跑全量是否值得？** **按当前形式不值得。**
   先修复 row-group 游标、增加独立的前缀/时间验证、明确早停和正则/容量方案，
   再做一个小型可复现回归；只有回归证明 R2 不再重复数据且能超过训练先验和
   C，才值得投入全量 GPU。单纯把 R1 或 R2 再跑一遍，不会解决 E1 门禁未满足、
   R1 容量跃迁和时间漂移问题。

## 1. 六组差异表

### 1.1 代码、模型和预算

| 项目 | E1 最后门控复验 | R1 全量 | 证据/判断 |
|---|---|---|---|
| 输入 | A_gated：Kronos s1/s2 token + size/sector；C：仅 size/sector | s1/s2 token + size/sector | 两边均有条件输入 |
| hidden | 256 | 768 | R1 实际代码值，不是 512 |
| ModernBERT 层数 | 4 | 22 | R1 实际代码值，不是 8 |
| heads | 4 | 12 | 架构同时改变 |
| intermediate size | 512 | 1152 | 架构同时改变 |
| 训练样本 | 16,384 | 9,010,965 | 约 550 倍 |
| batch | 64 | 16 | 每步统计和优化噪声不同 |
| optimizer | AdamW，lr 1e-4 | AdamW，lr 1e-4 | 名称/lr 相同，不代表轨迹相同 |
| scheduler | 无 | 无 | R1 代码没有 scheduler；不存在 scheduler state |
| AMP | CUDA float16 + GradScaler | CUDA float16 + GradScaler | 固定批次审计未发现 train/eval 直接错位 |
| token gate | `market_gate = 0` 初始 | `gate = 0` 初始 | R1 没丢掉 E1 的零初始化门控 |
| 训练预算 | 5 epochs；16,384/64 × 5 = 1,280 batch attempts/seed | 至少 563,186 个 batch attempts；准确更新数还受 row-group 尾 batch 和 AMP scaler 影响 | R1 不是 512/8、约 56 万“层/步”的同一配置 |
| 验证选择 | 每次实验训练结束验证 | 只保存最后状态 `final_model.pt` | R1 没有中途 best selection |

补充：E1 结果文档中还保留了更早的 4,096/2,048、3 epoch、192 step
实验。该 192 是早期小样本版本，不能用于描述最后的 A_gated 复验。

R1 的门控实现位于 `FullModel`：`market = fusion(...) * gate`，并且
`gate = nn.Parameter(torch.zeros(1))`。因此“R1 丢失 E1 gate”不是当前证据支持
的解释。

### 1.2 输入、tokenizer 和数据契约

两边都冻结 Kronos tokenizer，并对窗口做 120 日逐窗口标准化。E1 将训练/验证
抽样窗口先 materialize 到 token cache；R1 在训练 batch 中在线编码，每个 batch
调用 tokenizer，R2 只持久化验证 token，训练 token 仍是 segment 内存缓存。
tokenizer 本身不是 R1 失败的直接证据。

数据 manifest 的时间契约是：

```text
训练 signal end：2024-12-31
验证 signal start：2025-07-01
验证 signal end：2026-07-02
lookback/predict：120 + 10
```

训练和验证可能有相同 symbol，但日期窗口分离；这不是未来泄露。E1 代码现场
检查 symbol、as-of 日期和未来 10 日 MFE/MAE，R1 的 P0/P1 审计也通过了固定
样本标签边界检查。

输入差异：E1 行业词表从训练面板构建，R1 从训练和验证行业并集构建。
这不是标签泄露，但不是完全相同的输入管线。E1 对整个抽样集每 epoch
permutation；R1 只 shuffle 组序和组内行，不是全市场窗口全局混洗。
标签构建器按 symbol 排序、缓冲整只股票后 flush，可能保留组内股票聚集。

#### 标签组核对

| 检查 | E1 | R1 | 判定 |
|---|---|---|---|
| 窗口端点 | 历史 `start:start+120`；未来 `start+120:start+130`；基准 close 在 `start+119` | 相同 | 未发现公式迁移差异 |
| 标签 | 3/5/8/12% 上行 MFE、下行 -MAE | 相同 8 列 sidecar | 无阈值翻转证据 |
| 归一化 | 窗口 mean/std，`std+1e-5`，clip ±5 | 相同 | 不用未来数据拟合均值方差 |
| 防错强度 | 每条抽样核对 as-of/MFE/MAE、finite | 训练入口首批 MFE/MAE；P1 后补训练64/验证16条 | 不是全量逐行证明 |
| split manifest | 报告指出 `symbol_intersection=0` 不符实际 511 个交集 | 同源数据 | 应称时间切分，不是严格股票隔离 |

本地原数据 manifest 的 stats 时间范围与 cutoff 不完全一致，不能单靠
manifest 标题/统计证明全部标签区间正确。本次沿用 P0/P1 的已记录抽样证据，
不声称重新遍历了 900 万标签。E1 显式校验 tokenizer 固定 SHA；
R1 使用未锁 revision 的下载且未校验该固定 SHA，这是版本身份验证缺口。
同理 E1 的 Transformers 4.57.0 安装仅在 import 失败时触发，并非严格锁版。

### 1.3 `market_gate` 对照

| 项目 | E1 | R1 | 结论 |
|---|---|---|---|
| gate 参数 | `market_gate` | `gate` | 命名不同 |
| 初值 | `torch.zeros(1)` | `torch.zeros(1)` | 语义保留 |
| forward 位置 | market token 分支乘 gate | market token 分支乘 gate | 语义一致 |
| 是否冻结 | 否，随模型训练 | 否，随模型训练 | 可学习 |

E1 的 A_gated 五 epoch、三 seed 结果为 macro log loss `0.591066`，C 为
`0.592206`，但这是小样本时间验证体系中的结果；E1 还记录了按
2025H2/2026H1 的方向反转，不能外推成稳定全量增益。

### 1.4 验证口径

| 项目 | E1 | R1 |
|---|---|---|
| 验证规模 | 抽样 8,192 条 | 全量 123,836 条 |
| 抽样方式 | 按 Parquet row group 分配数量，再固定 seed 抽样 | 遍历全部 validation Parquet row group |
| 时间范围 | 同一时间外 sidecar 中抽样；契约 2025-07-01 至 2026-07-02，报告实际首日 2025-07-03；本次无原始分段样本量 | `<2026-01-01` 63,268；`>=2026-01-01` 60,568 |
| 选择方式 | 实验结束后计算；没有跨训练 segment 选择 | 最后 checkpoint；无中途验证选择 |
| 指标 | 8 头 macro log loss/Brier/ECE，含 per-target | 同口径；P1 另有 ROC-AUC/PR-AUC |
| 基线 | E1 报告的 validation-prevalence 常数基线 `0.594293` | 训练先验常数诊断 `0.5966`，R1 报告另有事后验证先验 `0.5964` |

“随机抽样”不等于随机划分 train/validation：最后 E1 从已经时间外的验证集
抽样。两边的后半段 mask 都没有 `<2026-07-01` 上界，因此 `2026H1` 是宽松标签，
可能包含契约允许的 7 月前两天。E1 的验证标签先验不是可部署训练先验，
不能与 P1 的 0.5966 混为同一基线。

验证精度也有差异：E1 使用 float16 autocast，sigmoid 后转 float32；
R1 最终验证没有 autocast，以 float32 运行；R2 验证再次使用 float16 autocast。
P0 的 train/eval 比较两边都使用 float16 autocast，只排除了固定批次同精度下
模式切换问题，**没有直接检验 R1 原始 FP32 验证与 FP16 训练的差异**。

训练 loss 为上/下两组四头 BCE 的和，验证为八头均值，约差两倍；
R2 日志还是 rank 0 单个本地 batch，不是全训练集损失。不能直接拿 0.2751
和验证 macro log loss 做同口径比较，也不能把批次 loss 下降称为已测得
训练集损失单调下降。

R1 最终为 macro log loss `0.9182`、Brier `0.2700`、ECE `0.2216`；
训练先验常数基线为 log loss `0.5966`、Brier `0.2052`。R1 的
`up_012` 平均预测概率 `0.0757`，正例率 `0.1971`，说明概率尺度塌陷不是
单纯“验证样本较少”可以解释。

### 1.5 checkpoint 接力与数据游标

R1 每个 chunk 的实际逻辑：

1. 用 `SHUFFLE_SEED=20260925` 生成完整 `group_order`；
2. 按 `num_row_groups * CHUNK_INDEX // 8` 和对应结束位置切分不重叠的
   `chunk_groups`；
3. 从前一 chunk 的 `last_checkpoint.pt` 加载 model、optimizer、scaler；
4. 校验 `shuffle_seed` 和 `group_order_hash`；
5. 前一 chunk 的 checkpoint 进入下一 chunk 时 `start_pos=0`，在本 chunk
   的固定分区内从第一 row group 开始；
6. 每个 row group 完整训练后保存 checkpoint，记录 `chunk_index`、
   `chunk_pos`、`row_group`、`processed_samples`。

仓库内对 1、7、8、9、181 个 row group 的合成分区测试显示，八个分区拼接后
无 overlap/gap；对“上一 chunk checkpoint”和“当前 chunk 中途 checkpoint”
两种恢复分支也验证了起点分别为 0 和保存的 `chunk_pos`。测试命令：

```bash
python3 modernbert_finance/audit_kairos_relay_control_flow.py
```

历史与限制：

- checkpoint 每个 row group 才保存一次。若任务死在 row group 中间，恢复会
  从此前完整 checkpoint 重做该组，未保存的梯度也一起回退；这不等于最终
  模型重复吸收同一段梯度。若无可读 checkpoint，无法确认恢复。
- R1 没有 scheduler，所以“scheduler 是否恢复”答案是：没有 scheduler，
  不是“scheduler 已严格恢复”。
- 没有保存 Python/NumPy/PyTorch RNG state。当前 dropout 为 0，数据顺序固定，
  所以没有证据它改变了已完成 R1 的样本覆盖；但这仍是严格复现实验的缺口。
- `dcd7f30`（09-27 13:12 +08）修复直接沿用前一 chunk 的 `order_pos`，
  改为区分 `chunk_index/chunk_pos`；`267145c`（13:31）才新增 R1 Chunk 2；
  `37babce` 后续又对缺 chunk_index 的旧文件推定为前一段。历史代码确有风险，
  但 Git 时间先后不能证明每次 Kaggle 发布版本用的就是已修复代码。
- R1 未强制要求前一 checkpoint 存在，也没验证前一分区已经完成：
  缺 checkpoint 可以从随机模型开始本分区；前序提前停止却挂作下一段可以跳组。
  这些是可达风险分支，不能把静态正常路径 PASS 写成历史运行 PASS。
- 输入链：Chunk 1 无前序，2..6 挂前一 `smmt315` 输出，7 挂
  `smmt315/...chunk-6`，8 挂 `wynstonliu/...chunk-7`。历史报告确认 8 从
  7,838,181 恢复并报最终 9,010,965；本次没拿到 1..7 实际发布版本、完整
  逐组日志、optimizer step/moment 张量，七条边的运行证据尚未闭环。
- step 精确公式是 `sum(ceil(row_group_rows/16))`，不是把总行数除以 batch
  后视作已核实值。构建器积累整只股票后才 flush，row group 不必恰好 50,000 行。
- checkpoint 直接写盘非原子，依赖挂载文件唯一；model/optimizer 成功 load
  不能替代前序版本/数据指纹和每条样本唯一覆盖校验。

### 1.6 R2 现场：已发现的管线缺陷

R2 的目标是把 row group 拆成固定 20,000 条 segment，每段全量验证。当前
`kaggle_kairos_r2_chunk1..4.py` 的外层结构读取一次 `rows`，内层循环在
`row_offset == len(rows)` 后只增加 `group_order_pos`，却没有重新读取新的
`group_id` 和 `rows`。因此一旦 segment 跨越 row-group 边界，旧 rows 会被
重复使用。

合成控制流复现（segment size 4，group sizes 5/5/3）：

```text
期望：0:0 0:1 0:2 0:3 0:4 1:0 1:1 1:2 ...
实际：0:0 0:1 0:2 0:3 0:4 0:0 0:1 0:2 ...
```

四个 R2 脚本的组首/组内偏移两类案例共 8 个，全部复现重复。合成案例中
12 次访问只有 5 个不同行；这个比例不是线上重复率。

真实日志与缺陷吻合：

| 北京时间 | segment / processed | order_pos | 实际 row_group |
|---|---|---|---|
| 09-29 22:57:49 | 2 / 40,000 | 0 | 120 |
| 09-29 23:05:50 | 3 / 60,000 | 1 | 120 |
| 09-29 23:29:39 | 6 / 120,000 | 2 | 120 |
| 09-30 00:01:25 | 10 / 200,000 | 3 | 120 |
| 09-30 00:17:24 | 12 / 240,000 | 4 | 120 |
| 09-30 00:23:36 | 13 / 250,000（半段） | 5 | 120 |

缺陷不仅在未满段的 `continue`；整段恰到组尾后重置 row_offset 也会留在
旧 rows 的内层 while。持久化游标于是指向未实际训练的后续组。
**Chunk 2 的 last checkpoint 即使可读，也不能作为干净轨迹继续，只能事故取证。**
Chunk 1 单段 smoke 和 segment 2 尚未跨第一组，不能证明跨组路径正确。

## 2. 决策门禁审计

### 2.1 E1 gate 条件是否满足

E1 报告本身给出的门禁是：A_gated 要在严格验证中稳定超过常数先验和 C，
再扩大到全量。实际证据是：

- 随机/抽样的 A_gated 五 epoch三 seed平均优于 C 和该阶段常数，但 ECE 劣于 C；
- 按时间分块时，A_gated 在 2025H2 比 C 差 `0.003786`，在 2026H1 比 C 好
  `0.006239`；
- E1 报告最后明确写着“暂不进入 E2 全量训练”。

因此，**按文档化 gate 条件，E1 没有完成稳定通过**。R1 后来启动的依据在
仓库中没有形成可核对的完整门禁通过记录，不应事后补写成“已通过”，也不能
仅凭仓库缺记录断言没有用户授权。同期 E1 的时间分块章节又写着“支持继续做带
时间验证和校准的全量训练试验”，与文末停止结论并存，是未清理的决策记录矛盾。
`85d0e63` 引入全量脚本时这些文字已存在；实际 R1 却只在最后验证一次，
未落实该段要求的训练过程中时间验证/选择。

### 2.2 R1 是否满足继续训练的证据

不满足。R1 完成后：

- 全量验证明显差于训练先验常数；
- 多头 AUC 为 `0.493..0.622`，多数上行头接近随机；
- `up_012` 概率均值系统性低估；
- P0 固定批次审计通过，未发现该批次同 AMP 条件下的 train/eval 切换差异；
- P1 仍观察到时间依赖的局部排序信号，但不足以证明稳定概率决策能力。

结论是：R1 失败不能仅用“校准坏了”解释，也没有足够证据值得直接延长同一
训练轨迹。

## 3. 归因与下一步

### 3.1 R1 失败归因

当前可分为三类，而不是强行给一个百分比：

| 类别 | 证据 | 当前归因 |
|---|---|---|
| R1 跨 chunk 样本重复/遗漏 | 当前静态代码、合成分区测试；历史有修复提交 | 当前未复现；历史七条恢复边尚未完全排除 |
| R1 恢复严格性 | RNG 未保存、缺输入/完成状态断言、row-group 内中断粒度粗 | 风险存在，未证明实际触发 |
| 模型/验证/决策问题 | 架构大幅放大、单一最终 checkpoint、时间依赖、概率塌陷、E1 gate 未稳定通过 | 有配置与现象证据，因果比例未知 |

因此，“接力事故占多少”只能严谨回答为：**不可量化，未发现不等于不存在。
缺少历史逐边状态核验和同配置不中断对照，没有依据给 R1 的 0.3216 基线差额
分配事故/方法百分比。** R2 已确认的事故也不能反向算进 R1。

### 3.2 是否值得从零跑修好的 R2

当前判断：**不值得立即跑全量；值得做一次很小的修复后回归。**

理由：

1. R2 当前存在确定的 row-group 重复 bug，必须先修；
2. Chunk 1 的 `0.6702` 仍劣于训练先验常数 `0.5966`，并不是一个已证明的
   可扩展起点；
3. 用户提供的早期验证曲线为 0.6702 -> 0.8364 -> 0.9579 -> 1.0283 ->
   1.1646；本次日志后来已到累计第 12 段完成、13 段一半。逐段验证曲线在执行
   日志中未完整记录，来源为用户看板观察，不能擅自补齐 segment 对应关系。
   已下载 best 元数据仍在第 1 段；
4. E1 的门控增益有时间依赖，不能用小样本平均值批准全量；
5. 从零重跑只有在回归同时验证“样本覆盖正确、时间验证可复现、早期校准和
   C/常数基线门禁成立”后才有信息价值。

供后续重新授权参考的前置条件（不是本次训练执行计划）：

```text
修复 R2 游标
-> 合成覆盖测试 + 真实数据前缀 hash 测试
-> 跨组、跨 chunk 的 optimizer/scaler/样本身份恢复校验
-> 明确独立开发/校准/最终评估分割与比较预算
-> 用户重新授权后，才考虑小预算 A_gated/C/训练先验比较
-> 时间稳定性和概率指标门禁成立后，才讨论全量
```

在这些条件完成前：不提交 Chunk 3/4，不重跑 Chunk 2，不调超参，不创建新训练
Kernel。

## 4. 停止现场与产出保全

2026-09-30 手工停止 `wynstonliu/kairos-r2-modernbert-chunk-2` V1；API 状态
随后显示 `CANCEL_ACKNOWLEDGED`。已成功下载并核对：

```text
artifacts/kairos_r2_stop_20260930/chunk2_v1/kairos_r2/best_metric.json
```

内容为：

```json
{
  "macro_log_loss": 0.6702645644545555,
  "chunk_index": 0,
  "segment_index": 1,
  "processed_samples": 20000
}
```

代码实际文件名是 `best_metric.json`，不是 `best_meta.json`。Chunk 2 输出下载时
大文件连接中断；本地 `best_model.pt` 当前为 0 字节，不能声称权重已经保全。
`last_checkpoint.pt`、`chunk_report.json` 和完整 `run.log` 也没有在本次下载中
完整落地。随后两个 R2 Kernel 和 R1 Chunk 1 API 均返回权限拒绝，无法继续
核对远端文件列表。Chunk 1 V6 的已发布输出也是 segment 1 best 的恢复来源；
不要删除或覆盖这两个 Kernel 版本。恢复访问后要逐文件下载、计算 SHA-256、
CPU 加载验证 best 内嵌的 segment/sample/metrics，并比较两个版本的权重。
**仅元数据证明未刷新，不能声称二进制权重已核验无污染。**

`chunk_report.json` 只在正常循环退出后生成，没有外部 cancel 的 finally；
强制停止时可能没有生成，不得伪造“原始报告”补齐。已取得的 `execution.log`
是 API 返回的执行日志快照（可解析的完整 JSON 数组），涵盖初始化至
09-30 00:23:36 +08，但没有停止收尾行，不能称最终完整停机日志。

Chunk 2 已知日志证据：

```text
checkpoint_resumed: completed_segments=1, row_offset=20000, processed_samples=20000
segment_complete: 2..12，最后完整段为 12 / 240,000
最后 training_progress: segment 13 / processed_samples=250,000，未完成
best_metric.json: best 仍为 segment 1 / 20,000 / 0.6702645644545555
```

本次尝试下载权重但未成功；没有执行模型、调整超参或提交任何 Kaggle Kernel。

## 5. 证据索引与限制

审计起点为 `58f3fc9`；开始工作前已执行 `git pull --ff-only`，返回同步。
下列路径均相对仓库根目录，行号按该提交：

| 证据 | 位置 |
|---|---|
| E1 配置/门控/优化/验证 | `finetune/kaggle_modernbert_decision_e1/kaggle_modernbert_decision_e1.py`，20、59、140、193、238、296、302、337、458、517、552、573 行 |
| R1 架构/门控 | `finetune/kaggle_modernbert_decision_full_chunk1/kaggle_modernbert_decision_full_chunk1.py`，268、272、277、289 行；2..8 相同配置 |
| R1 恢复/遍历/保存/最终验证 | 同文件 307、316、321、335、355、401、449 行；所有八份恢复分支均由离线脚本抽取测试 |
| R2 恢复与重复读取 | `finetune/kaggle_kairos_r2_chunk2/kaggle_kairos_r2_chunk2.py`，457 起、719 起、839 起 |
| P0 同精度测试范围 | `finetune/kaggle_modernbert_decision_r1_audit/kaggle_modernbert_decision_r1_audit.py`，303 起 |
| 标签定义/组写出 | `modernbert_finance/build_targets.py`，95、106、115、151、169 行 |
| 时间契约 | `data/a_share_full_market_v1_beta_temporal_symbol_validation_v1/data_manifest.json`；实际日期复核见 E1/P1 历史报告 |
| E1 历史结果及矛盾决策文字 | `modernbert_finance/modernbert_decision_e1_results_cn.md` |
| R1 P0/P1 与 Chunk 8 运行摘要 | `modernbert_finance/kairos_round1_chunk8_review_cn.md` |
| R1 恢复修改历史 | `dcd7f30`、`267145c`、`37babce` |
| 本次离线复现 | `modernbert_finance/audit_kairos_relay_control_flow.py`；只运行源代码游标片段，替换模型与 I/O |
| 本次原始日志/元数据 | `artifacts/kairos_r2_stop_20260930/chunk2_v1/`，本机保存，非已提交大模型 |

本次没有重新训练、重新评估模型，不能证明“fresh optimizer 单独修复校准”，
也不能证明“24.7M 参数使几千步必然过拟合”。R1 源码是 768/22，512/8 是方案
中的候选对照，不是本次已核实的 R1 运行配置；24.7M 参数说法没有 checkpoint
张量计数支持，不能沿用。R2 重复小片数据是确定的混杂因素，足以否定
“实现已完全通过，只剩方法问题”的前提，但不足以量化它造成的全部指标恶化。

六组核对映射：代码/架构见 §1.1、§1.3；输入见 §1.2；标签见 §1.2 子表；
验证见 §1.4；checkpoint 接力见 §1.5、§1.6；决策门禁见 §2。
优先项 1.1/1.5、1.3、4.1/4.2、5.2、6.1/6.2 按用户本次列举含义核对，
不假装取得了仓库中不存在的完整“昨晚清单”原文。
