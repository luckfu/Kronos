# Kairos 停止审计后的有限预算 LR 对照

日期：2026-09-30。用户最新授权：后续需要 Kaggle 提交的实验由代理执行，次日查看结果。
该授权用于以下有限实验，不恢复原 Chunk 2 的十小时全量接力，不自动扩展 Chunk 3/4。

## 1. 问题与假设

原 LR 为 AdamW 1e-4。降低 LR 是否改善后续验证轨迹，必须用修复后的同一数据管线
对照，不能从有重复 row group 的旧曲线直接得出“LR 太高”的结论。
本轮从干净的 Chunk 1 segment 1 接力，因此不是从零训练的 LR 搜索，也不是
验证“fresh optimizer 单独修复校准”。所有臂保留相同 AdamW 动量和 scaler。

| 臂 | Kernel | LR | 最大新增 segment | wall-time 硬上限 |
|---|---|---:|---:|---:|
| 原 LR 对照 | wynstonliu/kairos-r2-lr-probe-0-0001 | 1e-4 | 4 | 90 分钟 |
| 降低约 3 倍 | wynstonliu/kairos-r2-lr-probe-3e5 | 3e-5 | 4 | 90 分钟 |
| 降低 10 倍 | wynstonliu/kairos-r2-lr-probe-1e5 | 1e-5 | 4 | 90 分钟 |

每臂 DDP 双 T4、global batch 16、segment 20,000。最多新增 80,000 条/臂。
三臂合计最多 4.5 kernel 小时，双卡按设备计最多 9 GPU 小时；平台实际额度口径另计。
4 是本次受控比较的样本上限，不是把接力 chunk 数写死。每段仍根据实测最慢段耗时、
剩余预算和 15 分钟收尾余量决定是否开始下一段。

## 2. 固定与门禁

- 只挂原两个数据集和 Chunk 1 输出，绝不挂 Chunk 2。
- 起点必须是 chunk_index=0、completed_segments=1、processed_samples=20,000、
  group_order_pos=0、row_offset=20,000，原始 log loss=0.6702645644545555。
- 逐张量核对 model、optimizer、scaler 恢复；通过后才覆盖 optimizer param-group LR。
- 记录起点 checkpoint SHA-256，三臂必须相同。固定 shuffle seed、batch、模型和损失。
- 独立枚举真实数据前 100,000 条的 sample identity，记录 SHA；逐段与其顺序比较并
  排重，包括已经训练的前 20,000 条。任何不匹配在下一次 forward 前直接失败。
- 初始完整验证必须重现起点，容许 0.002 浮点/运行差异，否则停止，不训练。
- 修复跨组时留在旧 rows 内层循环的 bug；包含恰到组尾和最后不足一段情形。
- checkpoint 使用临时文件原子替换。best、last、逐段验证历史和日志分别保存。
- 验证概率在 NumPy 中转 float32 后再 clip/log，避免 float16 的 1-epsilon 舍入；
  三臂同样处理，可能与历史指标有微小数值差异，因此保存初始重算值。

没有新增 scheduler、dropout、weight decay 或模型容量变更。未声称做到完整 RNG
跨进程恢复；当前 dropout 为 0，三臂启动 seed 一致，数值确定性仍不作保证。

## 3. 停止与记录

每个完整段后全量验证 123,836 条，并记录 all、2025H2、2026H1 的 log loss、
Brier、ECE、AUC 和 per-head 指标，保存 validation_history.json。

任意一项成立即停止该臂：

1. 非有限验证指标；
2. log loss 比本臂重算起点恶化超过 0.20；
3. 连续两段没有至少 0.001 的新低改善；
4. 已完成 4 个新增 segment；
5. 剩余运行时间不足安全启动下一段，或平台 90 分钟硬超时。

不覆盖历史 R2 SwanLab run。新臂若无环境凭据则 offline，文件 dashboard.html、
progress.json、run.log、validation_history.json 仍完整生成；不把凭据写入生成脚本。

## 4. 解释限制和晨间交付

这里只能比较相同起点、相同已处理样本数上的 LR 轨迹。早停使最大训练量可能不同，
不能只挑各自最小值就宣布赢家。比较共同 segment 的结果和两个时间块，保留起点。
现有验证集已反复用于开发，不是独立最终测试集；单 seed 结果也不支持显著性结论。
本轮未运行 C-control，不单凭任何一臂超过 0.5966 就批准全量。

预期交付是可信的结果表与失败原因，不承诺指标必然提升。若全部变差，保留原 best，
停止追加预算。若某臂改善，仅提出下一轮有独立验证与 C 对照的建议，不自动跑全量。

## 5. 执行记录

- 本地回归：29 tests passed，覆盖四份游标、组尾、短末段、逐段恢复、生成配置、
  覆盖断言与早停规则。命令：`/opt/miniconda3/bin/python -m pytest tests/test_kairos_relay_cursor.py -q`。
- 构建：`python3 finetune/build_kairos_lr_probe.py`；生成三个独立目录，附源码 SHA。
- 北京时间 02:19：1e-4、1e-5 V1 均在 SwanLab offline URL 属性读取处退出，
  未到数据加载与训练。修复为 offline 时不访问 URL，并增加对应测试。
- 北京时间 02:21：两臂 V2 提交成功。重试 timeout 降为 5100 秒（85 分钟），
  为此前启动失败预留扣除 5 分钟；第三臂也统一采用 85 分钟。
- 3e-5 尚未提交成功：平台提示最多两个 batch GPU session。待释放后补交一次。
- 已建立本线程每 15 分钟跟进任务 `kairos-lr`：核对远端再补交，抓小日志和报告，
  不扩展全量。三臂完成交付后暂停。桌面任务能否持续执行依赖本机运行状态；
  已提交 Kaggle 作业不依赖本机在线。
- 上述为首次提交记录；当前进度如下。

### 02:38 巡检

| 臂 | 远端版本/状态 | 已核验结果 |
|---|---|---|
| 1e-4 | V2 RUNNING | 初始验证精确重现 0.6702645645；累计 segment 2 完成，40,000 条，log loss 0.7094477750；segment 3 跨组覆盖检查通过 |
| 1e-5 | V2 ERROR | 02:30 初始验证复现门禁失败，未训练；不自动重跑，不构成 LR 优劣证据 |
| 3e-5 | V1 提交成功 | 空闲位置出现后补交，timeout 5100 秒，等待运行门禁 |

1e-4 与 1e-5 的恢复核对都通过：checkpoint SHA 为
`45f4d7327d730e803fdf22c81e9baff20af5ea2e86b5e718d3d58b7467eabbcb`，
model/optimizer/scaler 精确恢复；100,000 条前缀 SHA 均为
`210cc90d37566b42872acadfa95c6f29333a5adbad8576e60bcdbd9f0133501a`。
这些校验仍不足以解释两臂初始验证行为不同。1e-5 V2 在输出初始分数前抛出异常，
因此没有实际偏差值，不能归因为低 LR（尚未 optimizer step），也不能擅自归因于
浮点误差、数据或模型。后续需检查缓存内容、运行环境和初始推理重现性。

本次仅增加诊断可见性：初始验证 JSON/日志在门禁前落盘，异常带 observed、expected
和 tolerance。门禁阈值仍为 0.002，训练逻辑不变。仅重新生成尚未提交的 3e-5，
保留另两份已发布 V2 的源码和 SHA；没有重跑失败的 1e-5。测试增至 30 项通过。

1e-4 累计 segment 2：2025H2 log loss=0.6606973242，2026H1=0.7603713796；
初始分别为 0.6288237590 和 0.7135526687，两个时间块都恶化。all Brier=0.2340588113，
ECE=0.1445018642。还没有共同进度的跨 LR 结果，不宣布赢家。

1e-5 V2 小文件及平台执行日志已保存于
`artifacts/kairos_lr_probe_20260930/1e5_v2/`，仅下载 run.log、best_metric.json、
progress.json 与执行日志，未下载全部 output。无新增训练，输出权重为复制的起点，
不当作新 best。原 Chunk 1/停止现场保全权重不受影响。

### 02:49 巡检

1e-4 V2 仍运行；02:44 完成累计 segment 3（60,000 条），更新 best，随后
segment 4 覆盖检查通过。当前同一臂轨迹如下，尚非最终结果：

| 累计 segment | 处理数 | all log loss | 2025H2 | 2026H1 | all Brier | all ECE |
|---|---:|---:|---:|---:|---:|---:|
| 1，起点重算 | 20,000 | 0.67026456 | 0.62882376 | 0.71355267 | 0.22637036 | 0.11877311 |
| 2 | 40,000 | 0.70944778 | 0.66069732 | 0.76037138 | 0.23405881 | 0.14450186 |
| 3 | 60,000 | 0.62168036 | 0.59366650 | 0.65094305 | 0.21341609 | 0.08163642 |

segment 3 的真实身份顺序哈希为
`7c987ac6435bf5818565635ad6498f5f988256ca6d07ba1720de2d8090712ec8`，
结束游标为 group_order_pos=1、row_offset=8861，跨越第一 row group 后未重复。
这说明修复后曲线不再是旧事故轨迹的单调恶化，但不能将差异全部归因于单个修复，
也不代表通过概率预测验收：all log loss 仍高于训练先验基线 0.5966。

3e-5 V1 已 ERROR，02:44:53 左右退出，**没有训练**。初始验证为：

| 范围 | log loss | Brier | ECE |
|---|---:|---:|---:|
| all，123,836 条 | 0.7002594993 | 0.2334600603 | 0.1386489632 |
| 2025H2，63,268 条 | 0.6472727489 | 0.2188340565 | 0.1329147764 |
| 2026H1，60,568 条 | 0.7556082234 | 0.2487380486 | 0.1571075514 |

相对期望 0.6702645645 偏差约 0.029995，超过 0.002 门禁。
checkpoint SHA、model/optimizer/scaler 精确恢复、训练前缀 SHA 与另两臂相同，
rank0 验证缓存身份校验通过，tokenizer SHA 相同。由于尚未任何 optimizer step，
不能称为 LR=3e-5 的训练结果，也不能认为 LR 数值导致初始预测变化。
需要进一步定位初始推理/缓存/未持久化状态的复现问题；当前证据不支持指定根因。
未放宽门禁，未重跑任一失败臂，未增加训练。

3e-5 的 `initial_validation.json`、`run.log`、`best_metric.json`、`progress.json`
及平台执行日志已选择性保全于 `artifacts/kairos_lr_probe_20260930/3e5_v1/`。
1e-4 的新 best/last 待正常退出后下载验证，当前只确认日志中的指标与覆盖校验，
不提前宣称新权重已在本机保全。两个低 LR 臂均缺训练结果，因此本轮无法完成
有效的三臂 LR 比较；继续守候唯一运行臂的预算内完成与产出保全。

## 6. 最终结果与后续授权

09-30 后续查询确认 1e-4 V2 为 COMPLETE，新增四段全部完成，累计 100,000 条。
完整轨迹为 0.67026456 -> 0.70944778 -> 0.62168036 -> 0.68731514 ->
0.74707275。报告 stop_reason=validation_patience；此时也恰好达到四个新增段上限。
总 runtime_elapsed_seconds=2413.1864（约 40.2 分钟）。
segment 4/5 的时间块 log loss 分别为 0.64981671/0.72648522 和
0.70157116/0.79460265。

用户指出四段有升有降不应要求线性下降，代理接受此纠正：两段 patience 是预算保护，
不是已证明适用于该数据分组的收敛准则。现有四段不能证明后续不会再改善。
用户同意先定位初始验证复现差异，再设计更长的有上限观察窗口。
不放宽初始复现门禁；目前不新开训练或据此断言方法无效。

### 保全进度

1e-4 完整小文件报告和日志位于 `artifacts/kairos_lr_probe_20260930/1e4_v2/`。
新 best_model.pt 已完整下载，450,461,639 字节，SHA-256：
`103c8e4ec901e62ad0b090e00a9d9eddd942fbb1cd3380c5c0fbac0e35d1a052`。
ZIP CRC 和 torch CPU weights_only/mmap 读取通过，内嵌 chunk_index=1、
segment_index=3、processed_samples=60,000、log loss=0.6216803565621376。
与 best_metric.json 一致，没有覆盖原停止现场文件。
last_checkpoint.pt 下载缓慢，本次中止下载并保留 .download 临时文件，
尚未完整落盘或验证；不可把临时文件当可恢复 checkpoint。

### 只推理诊断

新建 `finetune/build_kairos_inference_diagnostic.py` 和生成目录
`finetune/kaggle_kairos_inference_diagnostic/`。计划用 Chunk 1 同一 checkpoint、
同一缓存，连续两次完整验证，保存每 rank 的模型/全部 buffer/cache 指纹、
模型配置与 sector_ids，检查未持久化状态和重复推理。
DIAGNOSTIC_ONLY 分支无 optimizer step，在训练循环之前返回；平台上限 1200 秒。
本地 31 项测试通过，包括诊断分支在训练前退出的检查。

**尚未提交成功**：CLI 认证失败；显式读取已有 OAuth 后 SaveKernel 返回 HTTP 401。
不把准备好的脚本称作已运行诊断。需要恢复 Kaggle 登录后核对远端再提交，
不进行重复训练。复现差异根因仍未知。

Git 交付受当前权限阻塞：尝试先 pull，`.git/FETCH_HEAD` 返回 Operation not permitted，
当前环境将 `.git` 设为只读。本节与诊断修改仅在本地，尚未 commit/push；
不能将它们误记为上一提交 `24f6f1b` 已包含的内容。

### CLI 恢复后的执行记录

用户恢复 Kaggle CLI 后，已先确认原 LR 作业 COMPLETE，并查询本人诊断任务列表
未发现已有提交，再成功提交 `wynstonliu/kairos-inference-diagnostic` **V1**。
状态已核实为 RUNNING，timeout=1200 秒，仅推理、不训练；**不要重复提交**。
31 项本地测试再次通过。诊断尚无最终报告，不能提前指定复现差异根因。

权重保全脚本已增加 Range 续传、每轮180秒下载窗口及逐文件清单落盘，
继续保留并核验 best，尝试补全 last；`.download` 不作完整 checkpoint 使用。

本次再次执行 pull 仍返回 `.git/FETCH_HEAD: Operation not permitted`；
尝试更新 `kairos-lr` 自动任务也被权限拒绝，**自动任务的旧提示尚未更新**。
后续必须以本节和实际远端状态为准，不因旧提示而重新提交诊断或重开训练。
所有未提交的文档、诊断代码及测试仍仅在本地工作区。

### Git 状态更正

2026-09-30自动任务状态更正：`kairos-lr`更新失败是历史记录；接手核查时
任务已为`PAUSED`，随后按用户要求通过应用工具成功删除，返回
`deleteStatus: deleted`。旧任务已删除，无需继续检查停用，不影响Kaggle训练。

以上权限描述是当次命令失败的记录，不是仓库持久权限问题的定论。
后续用户确认 FETCH_HEAD 可写；代理重新执行 `git fetch --no-tags origin`
也已成功，随后 `git add` 成功。`git rev-list --left-right --count
HEAD...origin/master` 返回 `0 0`，表明 fetch 后双方提交一致，无需合并新提交。
同轮 `pull` 写 FETCH_HEAD 和 `merge --ff-only` 创建 ORIG_HEAD.lock 仍被拒绝，
原因未明，应按具体命令区分，不能说所有 Git 写入都受限。
工作区的 M/?? 只代表未提交内容，不是权限证据，不需要清理或丢弃。

## 7. Inference Diagnostic V1 完成：发现跨卡 RoPE 错位

2026-09-30 查询状态 COMPLETE。仅下载小 JSON 与日志，目录：
`artifacts/kairos_inference_diagnostic_20260930/v1/kairos_r2/`。
两次 full validation 的 log loss 均为 **0.6702645644545555**；
与期望值差为 0，重复差也为 0。Brier、ECE 和两个时间块指标亦一致。
这只证明该进程组合的重复推理稳定，**不能判为分布式语义验收通过**。

两卡诊断结果：

| 检查 | 结果 |
|---|---|
| state_dict 各张量哈希 | 同名全部一致 |
| config / sector_ids | 一致 |
| 软件与硬件 | 两卡均 PyTorch 2.10.0+cu128、Transformers 5.0.0、T4 capability 7.5 |
| rank0 buffer 注册顺序 | sliding_inv、sliding_original_inv、full_inv、full_original_inv |
| rank1 buffer 注册顺序 | full_inv、full_original_inv、sliding_inv、sliding_original_inv |
| 同名 RoPE buffer 哈希 | full/sliding 四个 buffer 恰好交叉互换 |

rank0 sliding SHA=`734f9e6b629f7baccc4de81b7e8d2982035ee1b81f73f158e7d33a4dcf13fa0e`，
full SHA=`adb41194debe8e6c185d47754f68cc2dd59960a03cb4ccbc71acd14b860c4604`；
rank1 同名 full/sliding 分别收到相反哈希。
配置 full rope_theta=160000，sliding rope_theta=10000。
两个 rank 的验证缓存哈希不同是预期的分片结果，不能单凭不同认定缓存损坏；
此前两个失败进程没有同等级指纹，尚不能做完整跨运行缓存对照。

### 机制与复现

核对官方 v5.0.0 源码 `ModernBertRotaryEmbedding.__init__` 第258行：
`self.layer_types = list(set(config.layer_types))`，按该顺序注册四个 buffer，
第270/271行明确 `persistent=False`，所以 state_dict 恢复检查看不到它们。
Python 进程间集合枚举顺序可能不同，原启动未固定 PYTHONHASHSEED。
DDP 按注册顺序同步 buffer，跨卡顺序不同导致相同形状的不同语义值被交换。

源码定位：
`https://raw.githubusercontent.com/huggingface/transformers/v5.0.0/src/transformers/models/modernbert/modeling_modernbert.py`

新增 CPU-only 机制复现脚本 `modernbert_finance/reproduce_rope_buffer_order.py`。
命令：
`KMP_DUPLICATE_LIB_OK=TRUE /opt/miniconda3/bin/python modernbert_finance/reproduce_rope_buffer_order.py`
本机 PyTorch 2.13.0、Gloo，两进程，无金融模型训练：
注册顺序反向时 rank1 full/sliding 值交换；统一排序后两卡同名值一致。
两种情况 state_dict 都只有相同 weight，说明仅比较 state_dict 无法覆盖此缺陷。
该本地实验不是 Kaggle 2.10 CUDA 的完全同环境复现；Kaggle 实际错位由其
diagnostic_rank0/1.json 独立确认。

### 结论边界和后续门禁

已确认：本次诊断存在真实的跨卡 RoPE buffer 语义错位，不能把初始验证偏差
解释为 LR 高低，因为那时尚未任何更新。注册顺序和 DDP 同步机制与现场吻合。
它提供了跨运行结果不同的具体机制，但此前两个失败进程没有完整 buffer 快照，
还不能量化各自差额或断言所有偏差只有这一个原因。

现有 0.67026/0.62168 是历史运行口径的指标，不应直接视为单一、语义一致模型的
健康验证指标；权重继续保全，先做一致 RoPE 的重新评估再决定使用价值。
不把这个 R2 双卡缺陷反向算进 R1 单卡失败的归因。

下一步在 DDP 包装之前固定 buffer 注册顺序，并在构造后、恢复后及推理前
校验两卡同名 buffer 值与配置的理论值一致。仅设置随机数 seed 不足以替代这些
语义断言。修复后先只推理复验，不强求健康配置重现旧的混合口径 0.67026，
但必须保存旧/新口径差异；不得静默放宽原复现门禁。
本次仅增加诊断结论和 CPU 复现脚本，没有修改训练脚本、重提诊断或新开训练。

后续授权与执行：用户同意一次修复验收后做更长的封顶观察。
现已修复并提交 `kairos-rope-fixed-bounded-trial` V1，最多12新增段或3小时，
验收不通过则不训练；第8段前不因普通波动早停。完整边界、起点、语义门禁及
交接见 `kairos_rope_fixed_trial_cn.md`，不按本节此前的“仅推理下一步”误判当前授权。
