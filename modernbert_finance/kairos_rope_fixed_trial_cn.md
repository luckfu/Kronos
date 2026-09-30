# RoPE 修复后的封顶观察

日期：2026-09-30。用户同意“修复验收一次 + 更长但封顶的训练观察”，
不要求最初几段指标单调下降，不自动扩大全量。

## 执行边界

- Kernel：`wynstonliu/kairos-rope-fixed-bounded-trial`，**V1 已提交成功**。
- 双 T4 DDP，global batch 16，LR=1e-4，沿用 AdamW/scaler，不新增 scheduler、
  dropout、weight decay 或改容量。
- 训练起点固定为 Chunk 1 segment 1 的完整 checkpoint，SHA：
  `45f4d7327d730e803fdf22c81e9baff20af5ea2e86b5e718d3d58b7467eabbcb`。
  沿用 optimizer/scaler 与20,000条游标，非从零训练；
  历史学习轨迹可能受旧 RoPE 错位影响，本轮不声称清除了这部分历史影响。
- 最多新增12个20,000条 segment，即新增240,000条、累计最多260,000条。
- 全部初始化、验收、重评和训练共用10800秒硬上限，保留900秒收尾余量。
  每段完成后按实测最慢段耗时及1.10安全系数决定是否有时间开始下一段。
  3小时指双卡运行墙钟时间，按设备时间合计上限6 GPU小时。
- 前8段之前不因普通验证回升早停；从新增第8段开始，连续4段未取得
  至少0.001的验证log loss新低才触发patience停止。非有限loss/验证值、
  数据覆盖错误、两卡状态不一致仍立即中止。
- 不自动提交下一轮，不重跑旧Chunk2/3/4，不再并行开LR扫描。

## 一次验收

1. 模型初始化后、DDP包装前，按名称排序每个模块的buffer注册表；
   保留其数值及persistent标记。torchrun子进程另固定PYTHONHASHSEED，
   但不以hash seed代替语义校验。
2. 在DDP之前/之后、checkpoint恢复后、每次验证前和每段训练前，核对：
   full/sliding RoPE频率符合各自配置公式；两卡有序同名buffer的形状、
   dtype及SHA完全一致。理论值容差rtol=1e-6、atol=1e-8，跨卡SHA要求精确相同。
3. 逐张量验证原checkpoint的model/optimizer/scaler恢复，核验tokenizer权重SHA。
   tokenizer实现固定在仓库提交`199471183f21b0b8de073bc1547f0199ac14b81c`。
4. 使用修复后的forward全量验证原起点。
5. 挂载并单独重新评估旧best segment3（旧口径0.62168），校验SHA：
   `103c8e4ec901e62ad0b090e00a9d9eddd942fbb1cd3380c5c0fbac0e35d1a052`。
   它只用于评估，不替换训练起点，不继承它的历史best分数。
6. 从原checkpoint恢复起点权重，再核验model/optimizer/scaler未被重评改变，
   然后完整验证第二次。两次起点预测SHA和全部指标必须精确一致。
   未通过即退出，无任何optimizer更新。

输出`trial_preflight.json`同时保留旧口径分数、健康口径初始/重复验证、
旧best重评结果及passed字段。由于旧分数混合了错位的RoPE语义，本轮明确采用
新的语义与重复预测门禁，**不强制等于0.67026，也不静默提高旧阈值**。
通过后以重新评估的起点作为本轮best初值，随后只比较同一健康口径下的指标。

模型和数据初始化、旧best与起点重评总共三次完整验证，全部在同一个预算内。
它们不是追加的训练实验或新的超参扫描。

## 训练覆盖与判断

预先独立枚举同一shuffle顺序的前260,000条sample identity，逐段核对顺序并排重，
包括起点已经训练的20,000条。挂载两个源kernel后，checkpoint/meta/cache均
按源kernel目录精确区分，不从模糊的全局搜索中取任意匹配。

从全部训练标签重新计算8头正例率，以同一验证样本和时间块重算常数基线；
保存all、2025H2、2026H1的log loss/Brier及模型ECE/AUC。验证集仍为反复使用的
开发验证集，不当独立测试。时间块沿用旧边界，2026H1标签可能含7月初数据。

达到停止条件后必须报告：

- 起点、每段、best和last完整曲线，不只展示最小值；
- best相对起点及训练先验基线的全量/两个时间块差距；
- 12段是否跑满、停止原因、实际运行时间和样本覆盖；
- 新旧口径差异及历史权重受旧RoPE训练影响的限制；
- 是否值得后续投入的判断，不自动追加预算。

## 复验与交接

本地35项测试通过；CPU双进程复现已改为调用实际runner中的排序函数，
原始反向注册顺序复现错位，应用修复后两卡一致。本机torch2.13/Gloo测试不冒充
Kaggle torch2.10/CUDA验收，后者由本次kernel内的强制门禁完成。

构建：`python3 finetune/build_kairos_rope_fixed_trial.py`。
发布目录：`finetune/kaggle_kairos_rope_fixed_trial/`，含构建SHA和metadata。
提交命令：
`kaggle kernels push -p finetune/kaggle_kairos_rope_fixed_trial -t 10800 --accelerator NvidiaTeslaT4`。

当前确认V1提交成功且远端状态RUNNING；尚未把本地测试写成线上验收通过或已有训练结果。
后续先查远端状态/日志，不重复提交。只选择性下载preflight、rope指纹、逐段历史、
chunk_report、best/meta/last和日志，不下载整个output。

尝试把既有`kairos-lr`自动任务更新为本轮收尾时，工具返回需要审批且当前不允许审批，
所以**自动任务提示尚未更新成功**。旧提示不代表当前授权边界，后续先读本文件并核对
远端新kernel，不重复旧诊断或旧LR实验。平台内的门禁、早停及3小时硬上限不依赖
该自动任务是否可用。
