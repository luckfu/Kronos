# Base Main 完成后的决策：先校准对照，再有预算地延续 Small 路径

日期：2026-11-02，Asia/Singapore。
状态：本轮方案确定；本文件不代表已提交新的训练任务。

## 1. 决策

保留 Small C2 为现行基准，Base Main 为研究候选。先完成一次评估口径与同阶段对照审计；审计通过后进入 Base Extend 01，沿用 Small 的阶段顺序，不因本次 OOS 较差直接放弃 Base，也不立即换损失函数或跳到 refinement。

本文件取代此前聊天中“Base 全面失败，应停止加训”“恒定高 LR 已被证明无效”“平台是数据底”等未经充分控制实验支持的结论。旧日志和报告保留以便追溯，不应继续引用那些因果判断。

## 2. 已有完整 OOS 观测

来源：/tmp/monitor_oos.log 中 phase=finished 的完整汇总，两模型各 42 个日期、213,564 个窗口。属于训练进程日志里的完成证据，尚未在本轮重新下载核验 Kaggle 输出文件。C2 来源：small_sealed_oos_20260803_20260930.json / full_42d / scores / c2_best / official。

| 模型 | 全路径方向准确率 | 路径收益 MAE | 路径收益 RMSE | D10 日均 Rank IC |
|---|---:|---:|---:|---:|
| Base Main best@516 | 0.51704173 | 0.06296761 | 0.09828920 | 0.02644956 |
| Base Main last@520 | 0.51702862 | 0.06302191 | 0.09848001 | 0.02638074 |
| Small C2（历史报告） | 0.53563990 | 0.05455762 | 0.08495722 | 0.11161198 |

Base Best/Last 的日均 IC 差（Last−Best）=-0.00006882，报告里的配对区间 [-0.00160339, 0.00802569] 包含 0。它只比较 Base Best/Last，不是 Base/C2 显著性检验。

9/18–9/30 子窗实际只有 8 个信号日，不是字段 first_use_9d 所暗示的 9 天；对整个项目也已不是首次查看的新盲测。Base best 在该窗日均 IC=-0.11609147，值得检查后半程方向偏差，但不据此调参或反复选模型。

当前数字表明 Base Main 在这次参考评估中落后 C2；不能据此断言模型容量无效、数据清洗有害或继续训练无效。未经成本与成交约束回测，不把预测指标等同于 alpha/P&L。

## 3. 对标阶段不等价

依据 SMALL_0_1_MODEL_AND_TRAINING_TECHNICAL_REPORT_CN.md §6 表与 §6.6：

| 阶段 | Small 到 C2 候选的路径 | 当前 Base |
|---|---|---|
| Bootstrap | 534 段 | 389 段 |
| Main | 534 段 | 520 段，一遍 clean coverage |
| Extend 01 | 190 段，新 seed，warmup+cosine | 尚未执行 |
| WC first round | 534 段，warmup 后固定 1e-5 | 尚未执行 |
| WC dual T4 | 534 段，warmup 后固定 1e-5 | 尚未执行 |
| Cosine refinement | 候选在第 179 段；该阶段总计跑完 267 段 | 尚未执行 |

到 C2 选中点累计约 2,505 个 segment-equivalents；Base 主线累计 909。每段约 20k 窗口，前者数据暴露大致是后者 2.8 倍，但末段不足 20k、batch/更新次数及清洗数据不同，不能当成严格等算力比较。Small 报告还存在全阶段轨迹口径差异，精确更新数要以各父状态为准。

Small Main→C2 是后续多个阶段的成果。Base Main vs Small C2 可以回答当前候选是否替代成熟基准，不能隔离回答 Base vs Small 架构谁更好。也不能把 Base 更低的验证 CE 当成已经胜过 C2。

## 4. 执行次序及边界

### A. 先完成一次小范围评估审计

1. 固化 Base best@516、last@520 与 Small C2 的 checkpoint SHA、模型配置、tokenizer、训练时间边界、数据清洗版本及行业 vocabulary 映射。
2. 对齐完整 42 日期与窗口 identity，核验历史 C2 结果采用的归一化、反归一化、采样数、温度、top-p、精度与 batch；不要把不同日期子集均值和全窗均值直接比较。
3. 当前 evaluate_base_sealed.py 的 worker 没有显式设置随机 seed，单路径采样不可视为精确可重复。先补随机性控制及运行清单；如旧产物不满足相同协议，重新跑统一评价。多 seed 稳健性检查优先放在开发验证集，不把本次 OOS 变成调参集。
4. 对比表增加 Small Main 末点，形成 Small Main / Base Main / Small C2 三列。若 Small Main 权重不可访问，明确标记缺失，不用 C2 代替同阶段对照，也不凭历史 CE 推算 OOS。
5. 先检查 checkpoint 正确加载、条件映射、输入窗口和标签日期无漂移。若发现实现问题，先修复再出结论。

该审计不做新 LR 网格，不做大规模超参搜索，不据已看过的 OOS 选择新配置。

### B. 审计通过，进入 Base Extend 01

- 父模型：当前 Main last@520；固定 SHA。Main best@516 独立保留为回滚/对照。
- 只跨阶段加载权重，新建 optimizer/scheduler；条件输出不得重新初始化。
- 按 Small Extend：全参数峰值 1e-5，warmup 起点 1e-6，1% warmup + cosine，coverage seed=20260907；global batch=64，纯 forecast + 0.02 history，aux off。
- scheduler 按完整 520-segment coverage 建立；不能把第一棒的段数当成 cosine 总长度。
- 每段全量 123,836 窗口验证，沿用同一 selection 指标；明确区分 unweighted forecast 与 weighted selection loss。
- 新阶段新输出目录、新 run id；同阶段接力保存 optimizer/scaler/RNG 并沿用该 run id。不能续写 Main r2 伪装成同一 schedule。
- 首轮观察上限 60 段。实际单棒 N 仍按账户剩余 GPU 时间、最近完整训练+验证耗时和启动/收尾余量计算，必要时拆棒；以此前约 9–10 分钟/段估算，60 段约 9–10 小时，非性能承诺。
- 阶段累计上限约 185 段，等于 Small Extend 190/534 coverage 比例映射到 clean 数据 520 段。不默认追加完整第二遍或无限续训。

### C. 预先设定开发集检查，不用参考 OOS 决定续训

- 开跑前，在全量开发验证集记录 Main last 基线，并锁定一个未用来挑 checkpoint 的时间开发面板用于日均 Rank IC/校准辅助检查；如果没有独立开发面板，明确标记该证据弱，不从已经看过的 OOS 再切一块冒充独立集。
- 初始 warmup 允许短期波动；不凭前两段判断成败。
- 第 60 段检查全量验证 selection loss 的最后 20 段中位数是否低于阶段起点，并检查末期趋势、方向偏差和开发期日均 IC；若无改善或发生稳定退化，停止自动追加，保留 Main 候选并排查，不自动推进 WC。
- 若改善且健康，续到累计约 185 段；在此边界再决定是否进入 WC。WC 若进入，仍使用 Small 的 warmup 后恒定 1e-5，并保留独立阶段与预算；不能引用之前的 10 段探针否决 WC。
- 本轮不承诺把 Small 的所有阶段无条件烧完。只有阶段开发验证支持继续，才追加预算；明确这属于有条件沿用路径，不是严格等训练量的架构实验。

## 5. 此前 LR 探针为什么不能定论

探针不仅改变 LR，还重建 AdamW，并将数据访问起点重置到阶段 1；它不是保留 optimizer/RNG/数据位置的单因素对照。另有起点段号混淆（2.33616 属于 seg284，先前被写成 seg294），以及把 weighted selection 2.34141 与 unweighted forecast 比较的错误。不能据此得出“提高 LR 无效”“不存在重置瞬态”或“模型已到数据底”。本方案不再追加探针，也不沿用这些判断。

## 6. 终点与正式评价

C2 保留作现行基准；Base 继续作为研究路线。完成阶段训练、基于开发集固定候选后，才在预先锁定、标签完整的独立测试窗做一次同口径执行评估（含交易成本、换手、涨跌停/停牌约束）。8/03–9/30 窗口保留为已知参考窗，不再宣称盲测，不据其反馈调阈值或换 loss。
