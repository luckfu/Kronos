# Small Clean 第二轮：有上限的幂律加权混合采样

2026-10-13。用户授权从 Clean C03 接力，沿用原 SwanLab 看板，使用12小时上限长训；不重做 C2 起点实验。

## 接力与训练合同

- 提交账号：`user281434`。
- 首个 Kernel：`user281434/kronos-small-clean-wc-dual-t4-c04`，标题 `Kronos Small Clean WC Dual T4 C04`。
- 父 Output：`luckfu/kronos-small-clean-wc-dual-t4-c03`，已核验 COMPLETE、完整轻量产物、520段终态、global step 162724；完整 State 在云端挂载读取，不下载权重到本地。
- 从父 `last_state.pt` 保留模型、AdamW 动量、scheduler、AMP scaler、RNG；不从 Best518 或 Last 推理权重重建优化器。
- 原看板：workspace `roc_fu`、project `finance`、experiment `small_0.1_clean_wc_dual_t4_20261011`、run id `smcl1011`。
- 新段为全局521–720；首个 optimizer step 162725，计划末步225324。第520段只有277步，不能用 `520 * 313` 当父步数。
- 新阶段使用独立 Output `small_0.1_clean_wc_dual_t4_20261011_power_round2`；第一轮 Best 元信息留在 `round1_parent.json`，原权重保留在父 Output；第二轮 Best 从本阶段首次验证开始。
- 每段20,000次抽样，共200段、4,000,000次抽样。全局 batch64（每张T4为32），float16 AMP。
- 模型、行业映射、tokenizer、forecast loss、history权重0.02、原horizon权重、AdamW与 `warmup_constant` 不变；两组LR均为1e-5，保留原1627步warmup的已完成状态，不重新warmup。
- 数据清单SHA-256 `b85ccc45f7cd0e895e607e6e75c7a603427bd62a353a016742a52b2943d652cd`，只使用清洗训练分区构建采样索引。
- 全量验证仍123,836条、242个信号日，保持原验证覆盖seed20261011和顺序；新采样使用独立seed20261013。OOS不参与采样、训练或选模。

## 采样定义

对每个合法训练窗口，以第120行close为P0，定义：

`A_i = max(h=1..10, abs(close[119+h] / close[119] - 1))`

权重与抽样概率：

```text
w_i = min(max(A_i / 0.05, 1) ** 1.0, 4)
p_i = 0.4 / N + 0.6 * w_i / sum(w)
```

这是有上限的幂律加权混合采样，并不声称抽样后的幅度分布严格服从Pareto分布。60%是加权分支概率，不是“大波段最终占比”。无逆概率loss修正，本阶段有意增加大波段训练暴露；loss计算保持原样。

抽样有放回。预先冻结整个400万次索引序列，每段消耗连续20,000个位置；通过 seed、参数、计划SHA-256及 `next_epoch` 恢复。段内中断整段重跑，后续chunk禁止修改计划参数或源码包。

本地生产Dataset预检实测：

| 项目 | 结果 |
|---|---:|
| 合法训练窗口 | 10,397,679 |
| 抽样次数 | 4,000,000 |
| 唯一窗口 | 3,269,159，约31.44%训练池 |
| 自然分布A≥5% | 70.6890% |
| 抽样后A≥5% | 78.6866% |
| 自然分布平均A | 9.5905% |
| 抽样后平均A | 11.8581% |
| 本地构建训练与验证Dataset | 约47秒 |

计划SHA-256：`fa746204cda4f373d9b5992f30bfa1c98eb112c604ca471fdcc700d09508d71c`。云端生成后必须匹配本地预检，后续chunk的resume guard同样校验该值。

预检归档：`finetune/reports/small_power_round2_preflight_20261013.json`。

## 长训预算与保存

重新下载C03轻量日志后，按相邻完整segment的summary时间差估算，78个间隔均值185.80秒，末20段均值184.50秒，最大192.02秒。每段包括训练、全量验证与段末保存。

- 200段按均值约37,160秒，即10小时19分20秒。
- 另预留10分钟准备与索引构建，总计约10小时29分；此为估算，不保证云端吞吐。
- API提交 `--timeout 43200 --accelerator NvidiaTeslaT4`，硬上限12小时。
- 代码预算40,500秒（11小时15分），包含准备时间；段末正常退出，给发布Output保留约45分钟。
- `max_segments_per_run=0`，由200段总计划及代码预算共同约束；如果本次未完成全部200段，下一chunk只接完整published State，不改采样计划。

## 看板与恢复验收

原图表有segment和step两个轴，都必须保持连续：

```text
chart_segment = global segment
chart_step = 162724 + (global_segment - 521) * 313 + step_in_segment
```

验证记录使用该段最后一个step。续训不重复回填父段历史记录。第521段step1为162725，第521段末为163037，第522段step1为163038。

测试覆盖真实Dataset采样、自然验证顺序、权重上限和抽样频率、反转波段幅度、计划恢复、实际AdamW状态恢复、全局步数及原最后短段。云端必须先输出原看板身份、父State边界、采样计划一致和第521段真实训练日志，才能报告“已开始训练”。提交成功或RUNNING本身不是训练完成证据。

## 提交

```bash
python finetune/small_power_relay/push_chunk.py 4 --dry-run
python finetune/small_power_relay/push_chunk.py 4
kaggle kernels logs --follow user281434/kronos-small-clean-wc-dual-t4-c04
```

使用包含工作区修改的独立源码包，记录当前Git HEAD与源码包SHA-256；Git HEAD不代表这些尚未提交的修改已进入Git历史。不自动提交Git或创建分支。

## 云端执行记录

第1版已提交，API接受12小时上限参数；服务端下载源码与本地提交入口逐字节一致，源码包SHA-256为 `23bcf7c5d5897401c899bf0fc19f17853a0d2ddec9b8e8062a53c4af78d8410f`。

云端入口于北京时间2026-10-13 10:08:16输出started；10:12:02成功接回原看板 `smcl1011`，双T4和PyTorch2.6.0+cu124已核验。10:12:21验证父520段完整State，SHA-256 `f02d051a903aa7d0be0e3e2b4a75097b862f51d6002070a52f1bffef094a137c`；10:12:22启动训练子进程。

两张卡各自构建的采样计划都与本地哈希完全一致；全量验证为123,836条。trainer日志明确记录 `Sampling stage resumed: next_segment=521, global_step=162724, optimizer/scheduler/AMP/RNG preserved`，随后真实执行到第521/720段step276/313，两组LR均为1e-5、global batch64。当前为RUNNING，启动已验收，第二轮终态及性能尚未验收。

账号GPU额度API以 `timedelta.total_seconds()`读取，总额度30小时；启动时剩余约29.89小时。注意SDK的字符串序列化可能丢失day部分，不能用其字符串里的21600秒误判为6小时额度。

启动证据归档：`finetune/reports/small_power_round2_startup_20261013.json`。本地30项相关测试通过，追加额度单位测试后共31项；语法检查与 `git diff --check`通过。
