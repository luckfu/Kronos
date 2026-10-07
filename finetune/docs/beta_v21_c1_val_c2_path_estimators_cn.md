# Beta v2.1 C1：验证集 C2 预测 + 路径估计量（预注册文档）

> 状态：**预注册**，在 Kaggle kernel 推送之前与代码一同提交。本文件中的规则、阈值、候选集合在看到任何本次验证结果之前固定，结果出来后不得修改。
>
> Kernel：`wynstonliu/kronos-val-c2-and-path-estimators`（零训练、仅验证集）
> 代码：`finetune/val_c2_path_estimators.py`、`finetune/kaggle_beta_v21_c1_val_c2_path_estimators.py`、
> `finetune/build_kaggle_beta_v21_c1_val_c2_path_estimators_kernel.py`、`tests/test_val_c2_path_estimators.py`
> SwanLab：`https://swanlab.cn/@roc_fu/finance/runs/beta_v2_1_c1_val_c2_path_estimators`

## 1. 要回答的问题

《为什么 C2 更好》分析（`beta_v21_c1_why_c2_better_analysis_cn.md`）留下三个只能在**验证集**上回答、且必须先于下一次封存窗口确认而冻结的问题：

1. **路径估计量**：Best@475 的分数是 N 条采样路径终值收益 `close_d10/last_close-1` 的均值。少数"爆发"路径会把均值推高，可能正是 top decile 内部排序失效（within-top-decile IC 为负）的来源。用中位数 / 去极值均值 / 去最大值均值能否改善？
2. **C2 与 Best@475 的秩融合**：两者互补程度如何，最优权重是多少？
3. **风格收缩（style shrink）**：OOS 描述性分析中"风格投影 + 0.5×残差"对 IC 有帮助；它在验证集上对 C2、Best@475 以及融合分数是否同样成立？

## 2. 数据契约（与 val-gen-ic 完全相同）

- 验证集：`temporal_symbol_validation_v1/processed_datasets/val_data.pkl`，SHA256 `4cce31bc3e70eab83d5b7ea05f19fce04aa57a87f3acf00b882ddfbac4219bf7`；`build_val_records` 得到 123,836 个窗口 / 242 个信号日。
- 子样本：`np.linspace` 取 24 个信号日（2025-07-03 … 2026-07-02），共 **12,256** 个窗口，与 `kronos-val-gen-ic`（Best@475 验证 IC 0.31447）逐条一致；kernel 启动时校验日期列表与窗口数，不一致即中止。
- 标签：`return_10d = close[asof+10]/close[asof]-1`；utility 与 val-gen-ic 相同。
- 条件输入：86 个排序后的证监会行业标签（未知 → 86），`size_percentile`（缺失 → 0.5），回看 120，预测 10，归一化后裁剪 ±5。

### 2.1 不接触封存 OOS

- `luckfu/a-share-120d-temporal-symbol-holdout` 数据集根目录同时包含封存 OOS 包（`evaluation_manifest.json` / `evaluation_panel.pkl` / `evaluation_samples.jsonl`），因此**不挂载**该数据集。
- 改为新建私有数据集 `wynstonliu/kronos-val-c2-path-inputs`，仅含：本地经 SHA 校验的 `val_data.pkl`、86 个行业标签、C2 权重（`c2_small_best_seg179`，SHA256 `4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a`，segment 179，val_forecast_ce 2.2944018841）。C2 原始来源（`user281434/...` 数据集、`smmt315/...` kernel 输出）对 wynstonliu 不可读（403），故复制。
- kernel 启动时若在 `/kaggle/input` 下发现任何 `evaluation_manifest.json` 或 `*time_oos*` 文件即中止。不读取任何 `kronos_beta_v2_time_oos_*`。

## 3. 模型与解码（镜像既有评估）

| 阶段 | 模型 | 解码 | 模型代码 | 有效 batch | 必需 |
|---|---|---|---|---|---|
| `c2_n5` | Small C2（seg179） | T0.65 / top_p 0.8 / top_k 0 / N5 / seed 20260906 | commit `e4b92bb` 的 `model/`（C2 封存 OOS kernel 所用） | 64 | 是 |
| `b475_n5` | Best@475（ModelScope `luckfu/Kronos-A-Share-Beta-V2-1`，SHA `e1bd5584…`） | 同上 | 当前 HEAD | 256 | 是 |
| `b475_n16` | Best@475 | T0.65 / p0.8 / N16 | 当前 HEAD | 256 | 否（按剩余时间） |

- 公共设置：tokenizer SHA `59d85f6a…6bee`，max_context 512，pred_len 10，lookback 120，clip 5，fp16 autocast，双 T4 各一个 worker，按日期动态认领。
- C2 的加载参数与 C2 OOS kernel 完全相同：`num_sectors=86, num_size_buckets=0, context_layer=6, use_size_percentile=True, size_mlp_hidden_dim=64`。
- C2 使用旧代码 + batch 64 是为了**逐比特镜像**其 OOS 评估（HEAD 相对 e4b92bb 只有数值层面的改动：RoPE 在 autocast 之外用 fp32 计算；nucleus 掩码重写）。Best@475 使用 HEAD + batch 256 以复现 val-gen-ic。
- 健全性检查：Best@475 N5 均值估计量的日均 IC 应在 0.3141–0.3145（val-gen-ic 为 0.31447）；偏离超过 0.003 时记录 `sanity_WARNING`（不中止）。
- 不计算 teacher-forcing 损失（不影响随机数序列）。零训练。

## 4. 输出

- 每个日期写一个分片 `shards/{label}__{arm}__{date}.csv.gz`：`symbol, asof_date, start_index, return_10d, utility, predicted_return_d10, path_ret_d10_00 … path_ret_d10_{N-1}` 等（每条路径的终值收益）。
- 每个阶段完成后立即打分：`results/{label}__{arm}_summary.json`、`_val_predictions.csv.gz`；`results/comparison.json` 随进度更新（`final: false → true`）。
- 指标：日均 return10d 秩 IC、合并（pooled）IC、ICIR、正 IC 比例、十分位/五分位多空价差、utility IC、top decile 内部 IC、top decile 超额、D10−D9、十分位单调性；配对按日 t 检验。

## 5. 预注册规则（固定，不得事后修改）

估计量：`mean`（现行）、`median`、`trimmed_mean`（每个窗口去掉一个最大、一个最小路径后取均值）、`mean_ex_max`（去掉最大路径后取均值）。

### 规则 1：路径估计量
在 **Best@475 N5** 验证结果上，非均值估计量只有同时满足以下三条才被采用，否则保持 `mean`：
1. 日均 IC − mean 的日均 IC **≥ +0.01**；
2. 按日配对 t **≥ 2**；
3. **修复 top decile 内部排序**：其 top decile 内部秩 IC 的日均值 **≥ 0 且严格大于** mean 估计量的同一指标。

若多个估计量满足，取 ΔIC 最大者。C2 N5、Best@475 N16 上的同样比较只作报告，不参与决策。

### 规则 2：秩融合
每个日期内：`blend = w·pctrank(C2) + (1−w)·pctrank(Best@475)`，`w ∈ {0, 0.25, 0.5, 0.75, 1}`，两边均用 mean 估计量、N5 prod 解码。选**验证日均 IC 最高**的 `w`（并列取较小 `w`，即网格中先出现者）。`c2_n5 × b475_n16` 的网格仅作次要报告。

### 规则 3：风格收缩
固定 λ = 0.5，逐日：对分数做 gauss rank，回归到 合并小行业后的行业哑变量 + 去均值 size_pct + mom5/mom20/mom60/vol20/liq20 的 gauss rank 上；新分数 = 拟合值 + 0.5 × 残差（与 `analysis/beta_v21_c1_why_c2_better.py::val_postproc` 相同，代码直接复用其函数，测试中验证数值一致）。应用于 C2、Best@475（N5、N16）以及规则 2 选出的融合分数（同时报告每个 w 的收缩结果），报告相对原始分数的配对差异。

## 6. 冻结与确认

- 本次运行结束后，规则 1 选出的估计量、规则 2 选出的 `w`、以及（C2 / 融合）× {原始, 风格收缩} 等候选全部**冻结**，不再在验证集上做任何调参。
- 冻结的候选只在 **2026-09-03 之后的下一个封存窗口**上做**一次性**确认；在此之前不读取任何封存 OOS 数据。
- 若规则 1 未通过，生产估计量保持 `mean`。

## 7. 运行预算与安全停止

- 硬上限 11 小时（`HARD_LIMIT_SECONDS = 39600`），worker 在截止时间前停止认领新日期。
- 可选阶段 `b475_n16` 仅在预计时长（`b475_n5` 时长 × 16/5 × 1.35 + 600 秒）不超过截止时间时运行。
- 预计：C2 N5 约 5 分钟，Best@475 N5 约 25 分钟，N16 约 75 分钟，加上环境准备约 10 分钟，总计约 2 小时。

## 8. 测试

`tests/test_val_c2_path_estimators.py`（CPU，`/workspace/.venv_kronos_test`）：估计量数值、规则 1 阈值逻辑、配对 t 与 scipy 一致、十分位约定与 val-gen-ic 评分一致、融合网格端点、风格收缩与 `val_postproc` 一致、`decode_with_paths` 与 `decode_records` 逐列一致（同种子）、C2 条件输入与 e4b92bb 的 `WindowStore` 一致、构建器元数据与内嵌归档（含 e4b92bb 模型代码）、封存包守卫，以及三阶段 CPU 端到端冒烟（小模型、合成验证面板，C2 阶段走 e4b92bb 代码）。
