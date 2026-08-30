# Kronos v1-beta 模型血缘与本地归档

更新日期：2026-08-30。所有模型均为 120 个交易日输入、10 个交易日路径预测。

当前正式 Beta 版本为 **Beta v1.3**，同时包含 `Best@343` 与 `Last@528` 两个产出物。稳定入口：
`models/a_share_v1_beta/releases/beta_v1.3/best_model` 和
`models/a_share_v1_beta/releases/beta_v1.3/last_model`。发布说明见
`finetune/BETA_V1_3_RELEASE_CN.md`。Beta v1.1/v1.2 保留为不可变父版本；Beta v1.3 没有自动替换旧 V6 Web/Modal 生产服务。

## 本地路径

| 本地模型目录 | 血缘 | 状态与用途 |
|---|---|---|
| `models/a_share_v1_beta/last1058` | 官方双速率分支：父权重为 Best@109；从该分支训练到 Last@1058 | 当前终点横截面排序基线。未来自然集 Day 10 Top 20% 超额收益约 1.948%。绝对方向偏空，不能直接作为个股涨跌判断。 |
| `models/a_share_v1_beta/natural_best185` | 从 Last@1058 只加载模型权重，统一 LR 自然验证阶段；Best@185 | 路径事件候选。50 路复核中多头 +5% 事件 Top 20% 命中 54.67%，仅比 Last@1058 的 54.33% 高 0.33 个百分点；优势太小，未晋级。 |
| `models/a_share_v1_beta/v6_natural_twospeed_v2_bf16_seed100/checkpoints/best_model` | 从 V6 Segment 568 权重重新初始化行业/市值条件层，fresh optimizer/scheduler，BF16 双学习率训练；Best@818 | 本轮多头首选候选。固定 24k objective 为 2.447197；严格未来集多头 Top 20% 为 55.12%、Rank IC 0.0844，但相对 Last 优势未达到可区分程度。 |
| `models/a_share_v1_beta/v6_natural_twospeed_v2_bf16_seed100/checkpoints/last_model` | 与上项同一训练，完整两轮后的 Last@1058 | 固定 24k objective 为 2.448726；严格未来集多头 Top 20% 为 54.75%、Rank IC 0.0736。保留作稳定末点对照；同目录 `last_state.pt` 可恢复 optimizer/scheduler。 |
| `models/a_share_v1_beta/beta_v1_2_full_predictor_bs64_seed100/checkpoints/best_model` | 从 Beta v1.1 Best@818 只加载权重，保留行业/连续市值条件分支，fresh AdamW + 统一 OneCycle，全 Predictor 训练；Best@871 | **Beta v1.2 推荐 checkpoint**。固定 24k objective 为 **2.416530**；尚未使用新的严格未来时间段复评。 |
| `models/a_share_v1_beta/beta_v1_2_full_predictor_bs64_seed100/checkpoints/last_model` | 与上项同一训练，完整两遍覆盖后的 Last@1056 | **Beta v1.2 完整训练终点与恢复锚点**。固定 24k objective 为 2.416884；同目录 `last_state.pt` 保存 optimizer/scheduler/RNG 状态。 |
| `models/a_share_v1_beta/beta_v1_2_best871_full_every_segment_onecycle_1pass_seed100/.../checkpoints/best_model` | 从 Beta v1.2 Best@871 只加载权重，fresh AdamW + 统一 OneCycle，全 Predictor 单遍覆盖；Best@343 | **Beta v1.3 默认 checkpoint**。固定 24k objective 为 **2.410465**。 |
| `models/a_share_v1_beta/beta_v1_2_best871_full_every_segment_onecycle_1pass_seed100/.../checkpoints/last_model` | 与上项同一训练，完整单遍覆盖后的 Last@528 | **Beta v1.3 完整训练终点与恢复锚点**。固定 24k objective 为 2.410773；`last_state.pt` 保存 optimizer/scheduler/RNG 状态。 |

其余自然 Last@528 与 late10 条件/去条件 Best/Last 已全部否决，本地保留它们的完整逐样本
评估结果和血缘名称，不重复保存约 2 GB 权重。需要审计时见
`finetune/artifacts/a800_20260828/runs/path_events_candidates_20260828/`。

机器可读血缘与精确 SHA-256：`models/a_share_v1_beta/LINEAGE.json`。

## Beta v1.2 全 Predictor 增量微调（2026-08-30）

父权重为正式 Beta v1.1 Best@818（SHA-256 `b8907713...e405673`）。本阶段不重置行业层或连续
市值层，训练整个 Predictor 的 102,437,248 个参数；fresh AdamW、统一 OneCycle 峰值 `1e-6`、
batch size 64、BF16、seed 100、两遍 coverage，共 1,056 segments。训练完成，无 NaN/Inf。

固定 24k 全量复评使用与 Beta v1.1 相同的 Natural Validation manifest，objective 口径为
`forecast_loss + 0.02 × history_loss`：

| checkpoint | Combined | 2025H2 | 2026H1 | 相对 Beta v1.1 Best@818 Combined |
|---|---:|---:|---:|---:|
| Beta v1.1 Best@818 | 2.447197 | 2.429536 | 2.458560 | - |
| Beta v1.2 Best@871 | **2.416530** | **2.403448** | **2.423210** | **-1.2532%** |
| Beta v1.2 Last@1056 | 2.416884 | 2.404488 | 2.423992 | -1.2387% |

Best@871 在 Combined、2025H2 和 2026H1 均略优于 Last@1056，因此作为 Beta v1.2 首选，Last
作为完整终点与续训恢复锚点共同发布。Large Validation 轨迹在约 Segment 780 后进入窄幅平台，
Segment 880 的轨迹最低值为 2.416632；保存的 Best@871 独立复评为 2.416530，未遗漏可用的更优
checkpoint。上述验证集已参与训练期模型观察，不能代替全新严格未来时间段评估；Beta v1.2 暂不继承
Beta v1.1 在旧 6 个未来信号日上的结果。

完整训练日志、指标、manifest 和独立复评位于
`finetune/artifacts/a800_20260828/runs/beta_v1_1_best818_full_predictor_bs64_seed100/`。

## Beta v1.3 全 Predictor 单遍续训（2026-08-30）

父权重为正式 Beta v1.2 Best@871（SHA-256 `965c3bc4...62393dd`）。本阶段只加载模型权重，
行业层和连续市值层均保留；fresh AdamW、统一 OneCycle 峰值 `1e-6`、batch size 64、BF16、
seed 100，完成 528 Segment 的一次全市场覆盖。每个 Segment 都只执行固定 24k full-only
验证，528 条 `validation_large`、0 条 quick、0 个 NaN/Inf。

| checkpoint | Combined | 2025H2 | 2026H1 | 相对 Beta v1.2 Best@871 Combined |
|---|---:|---:|---:|---:|
| Beta v1.2 Best@871 | 2.416530 | 2.403448 | 2.423210 | - |
| Beta v1.3 Best@343 | **2.410465** | 2.399127 | **2.415734** | **-0.2510%** |
| Beta v1.3 Last@528 | 2.410773 | **2.398807** | 2.416429 | -0.2382% |

Best@343 Combined 与 2026H1 更低，因此作为 Beta v1.3 默认 checkpoint；Last@528 的 2025H2
略低，但总体不替代 Best。训练退出后对两者分别重新加载权重并执行独立固定 24k 复评，结果与
训练记录逐项一致。完整模型、resume state、日志、实际源码快照和 29 项 SHA-256 manifest 位于
`models/a_share_v1_beta/beta_v1_2_best871_full_every_segment_onecycle_1pass_seed100/`。

Beta v1.3 尚未获得全新严格未来时间段评估，不能继承 Beta v1.2 的旧未来数据结论。本次定版
也不自动切换 Web、Modal 或其他线上生产服务。

## V6 Natural two-speed v2 BF16（2026-08-29）

父权重为 V6 Segment 568（SHA-256 `69999253...f99d1e0`），不继承旧 optimizer；行业层和
市值层重新初始化。单卡 GPU 0、BF16、seed 100，共训练 1,058 segments，约两轮覆盖。
训练本体完成，无 NaN/Inf。固定验证集为 2025H2 + 2026H1 的自然分布，24,000 条样本均从
训练候选中排除；manifest SHA-256 为 `f329b1ca...be1ac7`。

固定 24k 全量复评：

| checkpoint | Combined | 2025H2 | 2026H1 | 相对 Segment 0 Combined |
|---|---:|---:|---:|---:|
| Segment 0 V6 baseline | 2.461922 | 2.441398 | 2.476384 | - |
| Best@818 | **2.447197** | **2.429536** | **2.458560** | **-0.5981%** |
| Last@1058 | 2.448726 | 2.430896 | 2.460494 | -0.5360% |

Best@818 相对 Last@1058 的 Combined objective 再低 0.001529（约 0.0624%），并且 2025H2、
2026H1 两段均更低，因此本轮推荐 Best@818。训练轨迹每 10 segments 的 Large 最低点曾出现在
Segment 870，但代码没有保存该点；完整复评表明实际保存的 Best@818 比该轨迹点还低，故不存在
缺失更优可用权重的问题。

本验证集参与了本轮 checkpoint 选择，只能证明训练内泛化改善，不能替代严格未来评估。完整日志、
指标和复评结果位于
`finetune/artifacts/a800_20260828/runs/v6_natural_twospeed_v2_bf16_seed100/`。

### V6 严格未来路径事件（50 路）

使用封存未来集、固定 `T=0.6, top_p=0.9` 和相同的 3,000 个样本。因 50 路概率为离散值，
Top 20% 截止同分采用比例计入，避免任意行顺序改变结果。

| checkpoint | 多头 Top 20% | 多头基准 | 多头 IC | 空头 Top 20% | 空头基准 | 空头 IC |
|---|---:|---:|---:|---:|---:|---:|
| Best@818 | **55.12%** | 46.83% | **0.0844** | 14.19% | 16.70% | -0.0105 |
| Last@1058 | 54.75% | 46.83% | 0.0736 | **14.78%** | 16.70% | -0.0102 |

两者多头提升均在 5/6 个日期为正。Best 的多头命中仅领先 Last 0.37 个百分点，按日期聚类的
95% 区间跨零，不能判定 Best 明确胜出；结合固定验证 loss 与 IC，暂定 Best 为首选候选，同时
保留 Last。两者空头 Top 20% 在 6/6 个日期均不高于基准，当前空头分数不可用。完整结果见
`finetune/artifacts/a800_20260828/runs/path_events_v6_bf16_20260829/RESULTS_CN.md`。

## 封存评估数据

目录：`data/a_share_v1_beta_eval_20260826/package/`

- 训练目标最晚日期：2026-07-31。
- 未来信号日：2026-08-03 至 2026-08-10。
- 未来目标日：2026-08-17 至 2026-08-24。
- 该包只可用于最终评估，不可参与训练、阈值选择或解码参数调优。
- 本地包已与 A800 导出包核验一致：`evaluation_manifest.json`、`evaluation_samples.jsonl`、`evaluation_panel.pkl` 的 SHA-256 均匹配。

## 评估结论

完整原始结果：`finetune/artifacts/a800_20260828/runs/`。

### 终点 Day 10 排序

- Last@1058：Top 20% 超额收益约 1.948%，Top-Bottom spread 约 2.072%。
- late10 条件/去条件两条短训练均未超过该基线，因此不扩展。
- 历史选出的解码 `T=0.5, top_p=0.8` 在未来自然集弱于原 `T=0.6, top_p=0.9`，不采用。

### 路径事件（5 路生成，3,000 未来样本）

入场为下一交易日开盘；多头事件为未来 10 日先由 high 触及 +5%、此前 low 未触及
-3%；空头事件反向为 low 先触及 -5%、此前 high 未触及 +3%。日线同日目标/止损同触及
因无法确定盘中顺序，保守算失败。

| 模型 | 多头 Top 20% 命中 | 多头 IC | 空头 Top 20% 命中 | 空头 IC |
|---|---:|---:|---:|---:|
| Last@1058 | 51.00% | 0.0345 | 16.83% | 0.0086 |
| natural Best@185 | 51.33% | 0.0431 | 16.83% | 0.0166 |
| natural Last@528 | 51.17% | 0.0436 | 16.50% | -0.0007 |
| late10 condition Best@18 | 49.33% | 0.0332 | 17.50% | 0.0131 |
| late10 condition Last@40 | 50.50% | 0.0342 | 16.67% | 0.0064 |
| late10 no-condition Best@19 | 50.33% | 0.0334 | 16.83% | 0.0060 |
| late10 no-condition Last@40 | 50.83% | 0.0427 | 17.50% | 0.0031 |

结论：自然 Best@185 是唯一值得进行高路径数复核的候选，但证据很弱；其余 late10
分支均不应继续训练。

### 路径事件（50 路生成，3,000 未来样本）

固定使用 `T=0.6, top_p=0.9`，样本、事件定义与 5 路评估完全相同。完整逐样本结果见
`finetune/artifacts/a800_20260828/runs/path_events_50paths_20260828/`。

| 模型 | 多头 Top 20% 命中 | 多头基准率 | 多头 IC | 空头 Top 20% 命中 | 空头基准率 | 空头 IC |
|---|---:|---:|---:|---:|---:|---:|
| Last@1058 | 54.33% | 46.83% | 0.0796 | 15.17% | 16.70% | -0.0040 |
| natural Best@185 | 54.67% | 46.83% | 0.0848 | 15.50% | 16.70% | 0.0016 |

结论：提高到 50 路后，多头事件排序信号比 5 路更稳定，但 natural Best@185 相对
Last@1058 仍只高 0.33 个百分点，且仅覆盖 6 个未来信号日，不足以更换基线。两者的空头
Top 20% 命中都低于自然基准率、IC 接近零，当前生成概率不能用于筛选空头。继续工作应是
用历史样本训练一个小型、明确区分多头与空头的事件头，再等待全新时间段作最终封存评估；
不能继续围绕这 6 个日期选择模型或参数。
