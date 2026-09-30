# R2 正式重启交接

2026-09-30，用户明确要求合并修复后重新启动 R2，不再分叉进行短实验。

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
