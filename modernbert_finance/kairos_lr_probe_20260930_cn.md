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
- 尚未产生可比较的训练结果，不能报告 LR 优劣。
