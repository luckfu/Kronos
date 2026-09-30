# R2 正式重启交接

2026-09-30，用户明确要求合并修复后重新启动 R2，不再分叉进行短实验。

**接手请先读本文件。用户已要求由其他人负责后续监控；不要依据旧实验文档重开任务。**

## 已提交版本

- Kaggle：`wynstonliu/kairos-r2-restart-canonical`，V1 已提交成功。
- 云看板：`https://swanlab.cn/@roc_fu/finance/runs/kairos-r2-restart-20260930`。
  提交前已验证登录、创建 run 和上传完成；预检指标不代表训练已开始。
- 双 T4，global batch 16，AdamW LR=1e-4，20,000 条/segment。
- 首会话硬上限36,000秒，1,800秒收尾预留；每段后使用实测最慢段耗时乘1.10
  判断能否开始下一段。不固定接力chunk数量；451仅为全数据集segment上限。
- 不使用短实验12段上限或指标patience，也不因普通指标回升自动停止。
  全量验证、best/last分别保存，时间预算/数据错误/非有限数值仍生效。

## 初始化与修复

源只挂R1 Chunk8的final_model.pt，不挂试验best/last。R1权重严格载入，
optimizer和scaler新建、游标和已处理样本归零。它是原R2定义下的重启，
不是随机初始化模型，也不能称R1历史学习影响已被清除。

合入row-group遍历修复；每段用独立组长度与绝对偏移枚举身份，核对实际顺序和段内
唯一性，避免重复旧组或跳组。RoPE buffer在DDP之前按名排序，包装前后、恢复后、
验证前和每段训练前核对理论值与两卡同名SHA。

初始两次完整验证要求指标与预测SHA精确一致，通过后进入训练；不强求旧错位
口径的0.67026。重算训练正例率常数基线，逐段保存全量及两个时间块指标。
tokenizer权重SHA固定，代码固定于199471183f21b0b8de073bc1547f0199ac14b81c。

## 看板与凭据

跟踪目录`finetune/kaggle_kairos_r2_restart/`不含云看板密钥；
`finetune/stage_kairos_r2_restart.py`复用已授权凭据，先验证云run，再写入
Git忽略的`artifacts/kairos_r2_restart_20260930/private_staging/`供私有kernel上传。
不打印或提交密钥，不允许无云凭据静默降为offline。

## 接力边界

首个kernel是fresh-only入口，拒绝已有working checkpoint，不能原封不动重推当接力。
后续接力必须恢复本轮model/optimizer/scaler/游标/顺序哈希以及best，校验RoPE，
沿同一云run续写；根据剩余GPU额度及上一会话实测耗时计算下一会话预算。
本次尚未提交后续接力kernel，不声称已设置自动接力；不得误用旧Chunk3/4模板。
旧实验权重、诊断与报告保留，不覆盖。

本地37项测试通过。提交回执只证明平台接受V1，完整线上验收和训练进度应以
run.log中的restart_preflight_passed与后续segment_complete为准。

## 交接现场：2026-09-30

最近查询为RUNNING。已从平台日志核验：

- 云端`SwanLab ready`返回的是本文件的新run URL，不是昨日
  `modernbert-decision-full-gated-round2-v2`。旧run只留档，新训练不往旧run写入。
- 北京时间18:53:54，DDP前后RoPE理论值/同名buffer校验通过；18:54:01恢复后校验通过。
- R1 final权重SHA：`2886179355fc8c8be6d1e3dc3956200fb86c019dc81e7776e0bf10a0c8d20ef5`，
  日志明确optimizer_reset=true。
- 18:54:04开始初始完整验证，18:58:49进入第二次验证。
  本次抓取尚未看到`restart_preflight_passed`或`segment_complete`，
  **不能把RUNNING写成已完成某段训练**；接手后查询刷新。
- 代码提交`9b41986`包含正式重启实现、生成脚本、构建SHA、云端预检和37项测试。
- 预计首会话约50–60段仅为旧实测耗时的估算，不是固定配额或完成保证；
  全数据9,010,965条共451段，末段不足20,000条。

## 接手监控命令

在仓库根目录运行，确认CLI登录的是能访问wynstonliu私有kernel的账号：

```sh
kaggle kernels status wynstonliu/kairos-r2-restart-canonical
kaggle kernels logs wynstonliu/kairos-r2-restart-canonical --follow
```

`--follow`为实时流，人工使用后Ctrl-C退出；自动调用应设置20–30秒超时。
普通`logs`在运行时可能为空，不据此断言无日志。无需下载所有output。
结束后先选择小文件：

```sh
kaggle kernels output wynstonliu/kairos-r2-restart-canonical \
  -p artifacts/kairos_r2_restart_20260930/v1 \
  --file-pattern '(^|/)(run\.log|progress\.json|dashboard\.html|restart_preflight\.json|rope_.*\.json|validation_history\.json|best_metric\.json|chunk_report\.json|full_training_report\.json)$' \
  --page-size 200
```

每次监控记录平台状态、最后完整segment、processed_samples、覆盖hash、
验证log loss/Brier/ECE与时间块、best所在段、runtime_budget剩余时间及停止原因。
训练中的loss只是局部batch值，不能与八头平均验证log loss直接比较。
普通指标涨跌不触发本版本patience；用户明确希望同一路线深入运行，不得擅自
恢复短实验的两段早停或不断分叉LR实验。异常traceback、非有限数值或覆盖/RoPE
门禁失败须报告，不放宽门禁掩盖问题。

## 停机保全与接力检查

1. 确認完整段已验证并保存；外部cancel可能没有chunk_report，不能伪造。
2. 选择性下载best_model.pt、best_metric.json、last_checkpoint.pt及小报告日志。
   大文件流式下载，落盘后记录字节数/SHA/ZIP CRC，CPU weights_only方式检查。
   `.download`、`.tmp`不是可恢复checkpoint。密钥、缓存和大权重不进入Git。
3. 本轮last应带`run_purpose=r2-restart-clean-cursor-canonical-rope`、
   `rope_semantics=canonical_verified`、model/optimizer/scaler、group_order_pos、
   row_offset、processed_samples、shuffle_seed、group_order_hash和last_validation。
4. best仅有权重与metrics，不含optimizer；不得拿best配last的optimizer假装无缝接力。
5. 下一会话必须专门实现并测试resume入口，不能重推fresh-only入口，不能直接用旧
   chunk3/4脚本。当前代码仍包含短实验分支里的特定checkpoint断言，接力生成器必须
   显式选择正确模式，而不是仅修改CHUNK_INDEX。不得对接力重置optimizer或游标。
6. 本轮未保存完整Python/NumPy/CUDA RNG状态；dropout为0、数据顺序固定，但不能
   宣称逐bit训练恢复。恢复时应核对同名RoPE、optimizer/scaler和真实样本身份。
7. 先确认GPU余额与平台单会话上限，再按实测每段耗时和收尾余量确定预算。
   接力数动态确定，不承诺固定4个chunk，不自动提交重复任务。

## 历史信息与优先级

- `kairos_e1_full_diff_audit_cn.md`：E1→R1差异与原R2重复数据事故。
- `kairos_lr_probe_20260930_cn.md`：LR短实验、RoPE错位发现和诊断证据。
- `kairos_rope_fixed_trial_cn.md`：修复后12段封顶观察；已结束，不是当前训练。
- 正式重启的实现入口：`finetune/build_kairos_r2_restart.py`；
  无密钥版本：`finetune/kaggle_kairos_r2_restart/`；
  云端私有上传预检：`finetune/stage_kairos_r2_restart.py`。

旧实验的0.67026/0.62168含旧RoPE口径，不能直接与本轮分数拼成一条曲线。
修复后的短观察初始0.70026、best0.64544、last0.80891属于另一条已终止轨迹。
当前正式重启从R1开始，图表step从0计，本轮初始值以restart_preflight.json为准。

## 自动化与凭据注意

旧自动任务`kairos-lr`此前更新及删除调用曾被工具审批策略拦截，
原始错误为`MCP tool call requires approval, but approval policy is never`，
并非用户拒绝授权。2026-09-30接手核查时，本地配置已为`PAUSED`；
随后按用户要求通过应用工具成功删除，返回`deleteStatus: deleted`。
旧任务已删除，无需接手人再次停用；该操作不影响Kaggle训练。
未设置正式重启的自动接力，后续监控仍由用户指定的接手人负责。

私有上传staging在Git忽略目录，云凭据没有新增到跟踪文件；历史仓库中仍有旧脚本
内嵌凭据，属于既存风险，不能把“新提交无密钥”写成“全仓库无密钥”。不要复制进
交接文档；后续可迁移到Kaggle Secrets并轮换，注意不要影响当前运行。
