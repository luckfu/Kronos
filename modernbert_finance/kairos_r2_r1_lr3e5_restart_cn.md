# R2 从 R1 降 LR 重启交接

日期：2026-09-30（北京时间）。

## 决策与边界

用户已手工停止 `wynstonliu/kairos-r2-restart-canonical`，并明确授权从 R1 权重重新开始、降低 LR。
本次选择固定 LR `3e-5`，不是从旧 R2 best/last 接力，不随机初始化，也不启动多臂实验。
LR 过高仅是待验证假设；不能把震荡直接归因于 LR，也不能声称已排除全部管线问题。

新配置：

- Kaggle：`wynstonliu/kairos-r2-r1-restart-lr-3e-5`，只提交一次，版本与启动状态见下方现场记录。
- 新云看板：https://swanlab.cn/@roc_fu/finance/runs/kairos-r2-r1-lr3e5-20260930 。不续写旧轮曲线。
- 来源仅为 `wynstonliu/modernbert-decision-full-chunk-8` 的 `final_model.pt`。
- 旧轮核验的 R1 权重 SHA256：`2886179355fc8c8be6d1e3dc3956200fb86c019dc81e7776e0bf10a0c8d20ef5`。
  新轮需用运行日志再次核对，不能仅凭挂载名称认为相同。
- AdamW LR `3e-5`；optimizer/scaler 全新，数据游标零，shuffle seed 沿用 `20260927`。
- 双 T4，global batch 16，每段 20,000 条；每段完整验证 123,836 条，分别保存 best/last。
- 继续原单会话 36,000 秒上限、1,800 秒收尾预留，按最慢段实测耗时乘 1.10 判断是否继续。
  451 是全数据集上限，不是本次承诺段数。不设置 4 段或 12 段上限，不启用指标 patience。
  不承诺平台 GPU 剩余额度；若平台拒绝额度，不自动换账号/设备或重推。
- 保持 RoPE 理论值/双卡哈希门禁、独立样本覆盖校验、初始两次全量验证一致性检查。

## 实现检查

`build_kairos_r2_restart.py --profile lr3e5` 生成独立目录 `finetune/kaggle_kairos_r2_restart_lr3e5/`。
旧 canonical 生成目录保留，不覆盖历史提交产物。

注意：原 `LEARNING_RATE_OVERRIDE` 只在 checkpoint restore 分支生效；fresh R1 的
AdamW 初始化写死 1e-4。因此生成器显式修改实际 AdamW 的 `lr`，不是只改看板字段。
测试验证实际 AST 初始化值、R1-only 来源、fresh 模式、预算和门禁配置。
38 项 `tests/test_kairos_relay_cursor.py` 测试通过。
对旧 canonical 和新生成脚本进行 AST 对照，除 LR 配置/AdamW LR 与云 run ID 外无其他变化。

凭据仅写入 Git 忽略的 `artifacts/kairos_r2_r1_lr3e5_20260930/private_staging/`。
云登录与上传已通过；这不是训练成功的证明。

## 被停旧轮

CLI 状态已确认 `CANCEL_ACKNOWLEDGED`。
小文件已选择性下载到 `artifacts/kairos_r2_restart_20260930/v1/kairos_r2/`：
`run.log`、`progress.json`、`restart_preflight.json`、`validation_history.json`、`best_metric.json`。
平台 kernel 日志另存于上一级。取消轮没有正常收尾报告时不伪造 `chunk_report.json`。

- 已完成验证 25 段，50 万条；progress 显示第 26 段训练到 52 万条，但没有其完整验证记录。
  不把 progress 的 52 万条等同于 last checkpoint 的可恢复进度。
- best：第 8 段，160,000 条，macro log loss `0.6386612914502621`。
- 第 24/25 段：`0.721216008067131` / `0.8261488676071167`。
- 初始值 `0.9181997701525688`；常数基线约 `0.5966101885`。best 仍未胜出。
- 前四个五段窗口均值：0.79695、0.78792、0.78029、0.79233。
  早期改善后没有持续下移，不应据此声称长训必然有效。

旧权重在 Kaggle output 列表中仍有 `best_model.pt` 和 `last_checkpoint.pt`。
本地大权重归档未完成：CLI 下载使用整文件内存缓冲，已停止；改用流式下载后速度约
0.6–0.8 MB/s，为不让不用于新训练的旧权重阻塞本次提交，已结束本地传输。
仅留下 `best_model.pt.download` 残片；删除 CLI 产生的 0 字节 `best_model.pt` 占位文件。
未完成权重 SHA/ZIP CRC/CPU 安全读取检查，不得称旧 checkpoint 已本地保全或可恢复。
后续如需归档，选择性下载上述两文件；本次新训练只依赖 Kaggle 挂载的 R1 final。

## 监控与判定

核对新轮 `r1_warm_start` 的来源 SHA、optimizer_reset、零游标、初始复现门禁与实际逐段 LR。
同一 R1 起点的初始验证应与旧轮 `0.9181997701525688` 对照；若不一致先调查，不能掩盖。
比较相同累计 segment 的数据 coverage hash，确认实际输入一致，再比较 LR 曲线。

重点比较连续五段/十段均值、震荡幅度、running best、两个时间块与常数先验的差值；
不要求每段下降，不因单点 best 就宣布有效。校准改善与排序信号要分开解释。
验证集已多次用于开发，不把其中最小 loss 当独立泛化证据。
本次是固定起点、相同数据顺序的单个降 LR 重启，随机数与 GPU 执行仍需考虑，不能声称逐 bit 对照。
未经进一步授权不修改数据/正则/模型、不同时试 1e-5、不恢复旧 Chunk2/3/4。
后续接力仍须专用 resume 入口；fresh-only 脚本不能原封不动重推当续训。

## 现场记录

新 kernel V1 已提交成功；云看板预检通过。Kaggle 按标题规范化了新 kernel 地址，
回执实际地址为 https://www.kaggle.com/code/wynstonliu/kairos-r2-r1-restart-lr-3e-5 。
本地 metadata 已对齐该地址，未因地址变化重复提交。
远端已确认 `RUNNING`。北京时间 22:54:15 启动、22:54:39 检测到双 T4；
22:55:22 的 DDP 前后 RoPE 校验通过，22:55:42 从 R1 加载，来源 SHA 与上文一致，
`optimizer_reset=true`；恢复后 RoPE 校验通过，22:55:52 开始初始验证。
此时尚未看到初始双验证通过或完成第 1 段，不能把 RUNNING 写成训练验收通过。
私有上传脚本与公开生成脚本的 AST 除凭据字段外完全一致，实际 AdamW 初始化 LR=3e-5。

```sh
kaggle kernels status wynstonliu/kairos-r2-r1-restart-lr-3e-5
kaggle kernels logs wynstonliu/kairos-r2-r1-restart-lr-3e-5 --follow
```

自动抓取 `--follow` 必须设置 20–30 秒超时；只下载所需日志/JSON/权重，不下载整个 output。
没有创建新的自动接力或定时监控任务；旧 `kairos-lr` 已删除，不能恢复它监控本轮。
