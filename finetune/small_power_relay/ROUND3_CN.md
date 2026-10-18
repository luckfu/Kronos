# Small Clean 第三轮接力：C05

2026-10-18，用户要求检查C04是否完成，并按12小时预算开始下一次接力，保留原看板。

## C04终态验收

`user281434/kronos-small-clean-wc-dual-t4-c04` 为COMPLETE。published progress与summary均为completed，累计720段；2026-10-13 20:51:19北京时间的output_verified记录global step225324、next_segment721。

完整Output清单包含State、Best/Last推理权重、tokenizer、指标、日志和manifest。第二轮521–720段各有一次validation_large，共200条，每次123,836个样本；采样计划400万次全部完成。完整State在云端读取，不下载权重到本地。

父State SHA-256为 `578e3bc420aa94cd7345da8ecb3637a7d2156eee2d6a4143cde1bb28e0299cd2`。第二轮Best为715段，weighted forecast选模loss为2.281827528184304；这不能替代OOS结论。

## 新计划及预算

- 提交账号`user281434`，目标`user281434/kronos-small-clean-wc-dual-t4-c05`，挂载C04完整Output。
- 从C04 Last State恢复模型、AdamW、scheduler、AMP、RNG；首段721、首global step225325，计划到920段、global step287924。
- 新采样seed20261018，200段，每段20,000次，共400万次有放回抽样。60%封顶幂律权重分支与40%自然随机分支、5%参考幅度、指数1、权重上限4均不变。
- 本地生产Dataset预检覆盖3,269,922个唯一训练窗口；A≥5%占比78.727175%。计划SHA-256为 `b1b846ed8471a3482474d5793c8d18e5b65ed017e5c09a361bf1ed1064bf4995`。
- 保持原数据、验证顺序及123,836条全量验证、模型、行业、市值、loss、horizon权重、学习率1e-5、warmup_constant及已完成的1627步warmup不变。
- 原SwanLab workspace `roc_fu`、project `finance`、experiment `small_0.1_clean_wc_dual_t4_20261011`、run id `smcl1011`；不重报已完成父段。
- 独立Output后缀为`_power_round3`，父第二轮Best元信息及manifest记录在round2_parent.json；本轮Best独立选取。
- C04相邻完整段199个间隔均值191.02秒，末20段193.99秒；200段按末段速度约10小时47分钟，准备预留10分钟，共约10小时57分钟。
- 提交硬时限43,200秒（12小时）；代码预算40,500秒（11小时15分钟），段末完整保存，预算不足则由后续chunk精确接力。当前账号查询可用GPU额度30小时。

新阶段从已完成720段的状态启动，明确切换采样计划，不绕过普通resume guard伪装成原计划未完成的chunk。历史optimizer步数直接继承225324，不再按第三轮采样游标推算前两轮的步数。

## 验证与提交

冻结计划：`finetune/small_power_relay/round3_plan.json`。
生产数据预检：`finetune/reports/small_power_round3_preflight_20261018.json`。

```bash
python finetune/small_power_relay/push_chunk.py 5 --stage-plan finetune/small_power_relay/round3_plan.json --dry-run
python finetune/small_power_relay/push_chunk.py 5 --stage-plan finetune/small_power_relay/round3_plan.json
kaggle kernels logs --follow user281434/kronos-small-clean-wc-dual-t4-c05
```

云端必须再次验证父State哈希、完整状态、原看板及新采样哈希，观察第721段真实训练后才确认启动。尚未验收第三轮终态与OOS表现。
