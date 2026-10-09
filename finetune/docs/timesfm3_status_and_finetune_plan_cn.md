# TimesFM-3 当前状态与微调计划

更新：2026-10-08 CST

## 0. 2026-10-08 实施日志：LoRA 微调入口已开始

Preflight kernel：`wynstonliu/kronos-timesfm3-lora-preflight`

状态：`COMPLETE`。通过 Kaggle CLI 下载并核对了真实输出：

- `TimesFM3Forecaster.model` 的类型为 `TimesFM3Torch`；
- `model_is_torch_module=true`；
- 总参数量 `330,710,976`；
- Kaggle T4 上 `cuda=true`，PyTorch `2.10.0+cu128`；
- `forward(inputs, freeze_after, patch_cpm_mask, return_aux_outputs)` 可调用；
- `decode(...)` 是推理接口，不能直接用于训练，因为其路径使用 `torch.no_grad`。

第一版训练实现：

- 入口：`finetune/timesfm3_lora_finetune.py`；
- 测试：`tests/test_timesfm3_lora_finetune.py`；
- kernel 草稿目录：`finetune/timesfm3_lora_finetune_kernel/`；
- 通过 `TimesFM3Torch.forward` 手工构造 120 日 context + 32 日 padded horizon；
- 目标变量为 close，past-only covariates 为 open/high/low/volume/amount；
- LoRA 默认挂载 `query_proj/key_proj/value_proj/out_proj/ff0/ff1`；
- 主损失为 close path Smooth L1，另加低权重 return path loss；
- 选模指标为时间外验证集 normalized RMSE，另记录 normalized MAE、
  path direction accuracy 和 endpoint bias。

源码核对后的接口细节：

- `forward` 输入键确认为 `values/masks/patch_is_target`；
- 输入形状为 `(batch, variates, patches, 32)`，输出为
  `(batch, variates, patches, 64, quantiles)`；
- `patch_cpm_mask` 形状为 `(batch, patches)`，只在 horizon patch 置真；
- `patch_is_target` 对 target 和 past-only covariates 都置真；
- 120 日 context 左侧补到 128 日，10 日 horizon 在 stitching 模式下准备
  两个 32 日 patch，但本轮只取第一个 forecast patch 的前 10 日；
- `freeze_after=3` 对应 128 日 context 的最后一个 context patch，
  与官方 `decode` 的 running-stat 冻结逻辑一致。
- Kaggle 入口在只挂载 `evaluation_panel.pkl` 时，会自动按同一时间切分生成
  `/kaggle/working/timesfm3_dataset/train.npz` 和 `val.npz`，不依赖一个尚未
  发布的额外训练 dataset。

尚未宣称正式训练成功。当前仍需完成：

1. 本地 patch/loss 单元测试和编译检查；
2. 核对 TimesFM-3 `forward` 的真实输入字典键和 logits 形状；
3. 用真实数据做一次短 smoke run；
4. 完善 Kaggle 自包含打包后先提交 GitHub 审核，再提交正式 kernel。

当前本地验证：`PYTHONPATH=. pytest -q tests/test_timesfm3_lora_finetune.py`
结果为 `3 passed`；训练 kernel 尚未提交 Kaggle，也尚未宣称得到微调收益。

## 2026-10-08 Kaggle 执行记录

- 初版 kernel：`wynstonliu/kronos-timesfm3-lora-fine-tune`，version 1；
  状态为 `ERROR`。
- 初版已成功加载模型并进入 LoRA 注入，但 Kaggle 预装
  `torchao==0.10.0` 与最新 `peft` 的 dispatch 不兼容：
  `ImportError: Found an incompatible version of torchao`.
- 修复：移除 `peft` 依赖，改为本地 `LoRALinear` 注入，保持相同的
  attention/FFN target modules，避免依赖 Kaggle 的 torchao/peft 版本。
- 修复版 kernel：`wynstonliu/kronos-timesfm3-lora-finetune-v2`，version 1；
  当前最后观测状态为 `RUNNING`，尚未把 `RUNNING` 视为成功。
- 页面：https://www.kaggle.com/code/wynstonliu/kronos-timesfm3-lora-finetune-v2
- v2 暴露出执行控制问题：没有设置 CLI timeout，训练脚本也没有 step heartbeat；
  当前 Kaggle CLI 没有 stop 子命令，v2 只能等待平台结束。
- 下一版必须使用 `kaggle kernels push -t 1800`，脚本默认最多 200 steps，
  每 10 steps 打印 heartbeat，每 50 steps 保存 checkpoint。

补充纠正：TimesFM-3 的正式看板必须使用 SwanLab 云端实验，不是仓库里的
Markdown 文件。small/beta 的约定为 `project=finance`、
`workspace=roc_fu`、`mode=cloud`、`resume=allow`，并在启动日志打印
SwanLab 返回的真实 URL。TimesFM-3 V4 已按此约定修正。

V4 version 4 实测：

- 双 T4 已确认：`gpu_count=2`；
- 训练从 step 1 到 step 200 产生 checkpoint，step 200 文件约 1.3 GB；
- 稳态约 `1.2 steps/s`，step 10 到 step 200 总耗时约 127 秒；
- 下一 chunk 默认设置 `1200 steps`，预计纯训练约 14 分钟；
- 验证改为固定 256 个 batch，避免完整 87,705 样本验证吞掉 chunk 时限；
- 接力通过 Kaggle `kernel_sources` 挂载 V4 输出，自动发现 `step_200.pt`；
- 修复 SwanLab heartbeat 不再上传字符串 `event` 字段。

再次修正：V4/V5 都被手工停止，Kaggle 对取消任务的 output API 返回空列表，
不能作为有效接力 source。V4 的全量 checkpoint 约 1.3GB，也不适合人工下载。
后续版本改为 compact LoRA checkpoint，只保存 LoRA 参数、optimizer、
global step、best metric 和 history，避免保存 330M backbone。

## 1. 目标

TimesFM-3 不承担 Kronos 的横截面排序职责。本实验把它定位为独立的
**直接价格路径预测器**：

```text
过去 120 个交易日 OHLCVA
        ↓
TimesFM-3
        ↓
未来 10 个交易日的价格路径与预测区间
```

Kronos/C2 继续负责横截面候选排序；TimesFM-3 的评价只看直接价格预测质量，
不使用 Rank IC 作为主指标。

## 2. Zero-shot 实验

### 数据与协议

- Kaggle kernel：`wynstonliu/kronos-timesfm3-path-accuracy`
- 模型：`google/timesfm-3.0-pytorch`
- 股票：256 只
- 窗口：4,351 个
- 输入长度：120 个交易日
- 预测长度：10 个交易日
- 信号日期：2026-07-17 至 2026-08-10
- 模式：zero-shot，不训练
- TimesFM-3 协变量：使用 `past_only_covariates`，形状为
  `(num_covariates, context_length)`

### 四种输入方案

| 方案 | 预测对象 | 历史协变量 |
|---|---|---|
| `raw_ohlcva` | 原始 close | 原始 OHLCVA |
| `norm_ohlcva` | 原始 close | 归一化 OHLCVA |
| `return_features` | 日 log-return | 价格收益、成交量/成交额变化 |
| `return_price` | 日 log-return | 仅价格收益 |

return 方案通过累计 log-return 还原为价格路径后再计算价格误差。

## 3. 评估方法

主评价对象是未来 10 日整条价格轨迹，而不是只看第 10 天：

- **归一化 MAE**：整条路径绝对误差均值除以预测起点价格。
- **归一化 RMSE**：整条路径平方误差均值开方，再除以预测起点价格。
- **路径相关性**：10 个预测价格点与 10 个真实价格点的 Pearson 相关。
- **每日变动方向一致率**：逐日比较预测路径和真实路径的价格变动方向。
- **区间覆盖率**：真实路径落入预测区间的比例。
- **价格偏差**：预测路径相对真实路径的带符号误差。

方向准确率和收益相关性只作辅助诊断，不作为 TimesFM-3 的横截面晋级指标。

## 4. Zero-shot 结果

| 方案 | 归一化 MAE | 归一化 RMSE | 路径相关性 | 每日变动方向一致率 | 方向准确率 | 区间覆盖率 |
|---|---:|---:|---:|---:|---:|---:|
| 原始 OHLCVA | 5.53% | 6.29% | -0.036 | **47.07%** | **46.86%** | 77.55% |
| 归一化 OHLCVA | 5.53% | 6.30% | -0.044 | 46.78% | 46.27% | 77.48% |
| log-return + 全部特征 | **5.45%** | 6.23% | -0.242 | 42.58% | 33.03% | 97.07% |
| log-return + 价格特征 | **5.41%** | **6.19%** | -0.255 | 42.33% | 31.81% | 97.01% |

四种方案的共同现象：

- 第 10 日带符号价格偏差约为 `-5.9%`，存在明显系统性低估。
- 原始 OHLCVA 的价格路径行为最稳定，但整体路径相关性仍接近零。
- log-return 方案的归一化误差略低，但路径相关性和方向一致率明显恶化。
- 高区间覆盖率主要来自区间过宽，不能解释为点预测准确。

## 5. 当前结论

### Zero-shot 结论

TimesFM-3 的接口、模型加载、历史协变量输入和四种目标定义均已跑通。
但在当前 A 股数据快照上，zero-shot 预测没有形成可靠的价格路径：

1. 预测路径不能稳定跟随真实路径；
2. 价格水平存在约 `-5.9%` 的系统性偏差；
3. return 目标变换没有解决问题，反而破坏了路径方向；
4. 输入尺度调整没有带来实质改善。

因此当前不能把 TimesFM-3 zero-shot 直接用于线上价格预测。这个结果不等于
TimesFM-3 架构无效，而是说明通用预训练分布与我们的 A 股复权 OHLCVA、
10 日预测目标和价格尺度存在明显错配。

### 与 Kronos 的关系

- Kronos/C2：继续承担横截面排序。
- TimesFM-3：进入微调前的独立价格预测实验。
- 不把 TimesFM-3 的 Rank IC 与 Kronos 的 Rank IC 直接混为同一晋级标准。
- 微调后的 TimesFM-3 只需证明它能改善整条未来价格路径，再考虑与 Kronos 组合。

## 6. 下一步：TimesFM-3 微调

### 第一阶段：小预算直接价格微调

先固定模型结构和输入格式，只微调预测能力：

- 训练目标：未来 10 日 close price path。
- 主损失：按预测 horizon 加权的 Huber 或 smooth L1。
- 辅助损失：log-return path loss，权重较低，防止价格水平主导。
- 训练输入：原始 OHLCVA，先不加入行业、市值和横截面信息。
- 训练方式：冻结大部分 backbone，只训练输出层和最后若干层。
- 训练窗口：按时间切分，禁止随机打散跨时间泄漏。
- 选择指标：时间外验证集的归一化路径 RMSE、归一化 MAE 和路径方向一致率。

### 第二阶段：目标与输出校准

在第一阶段确认可学习后，再比较：

1. 直接 close path；
2. 相对起点价格的 path；
3. log-return path；
4. close path + return auxiliary loss。

同时检查：

- 预测水平偏差；
- 各 horizon 的误差曲线；
- 预测区间宽度与覆盖率；
- 不同价格区间和波动状态下的稳定性。

### 第三阶段：协变量扩展

只有直接价格微调有效后，再逐步加入：

- rolling return；
- realized volatility；
- volume/amount 的 log 变化；
- 行业和市值条件；
- 可选的市场指数协变量。

每次只增加一组协变量，并保持相同时间外验证集。

## 7. 微调晋级门槛

微调模型必须同时满足：

- 时间外归一化 RMSE 明显低于 zero-shot 原始 OHLCVA 基线 `6.29%`；
- 时间外归一化 MAE 明显低于 `5.53%`；
- 路径相关性显著高于当前基线约 `-0.036`；
- 每日变动方向一致率至少高于当前基线 `47.07%`，并在多个市场状态下稳定；
- 系统性价格偏差明显小于当前约 `-5.9%`；
- 预测区间覆盖率改善时，区间宽度不能无限扩大。

只改善单一误差指标、但路径相关性和方向一致率恶化的模型不晋级。

## 8. 当前保留结论

当前应保留：

- zero-shot 原始 OHLCVA 作为 TimesFM-3 基线；
- `return_price` 作为低误差但路径行为较差的对照；
- 本文中的 4,351 个窗口和时间切分协议；
- 现有 TimesFM-3 Kaggle 脚本作为微调评估入口。

当前不应做：

- 不把 zero-shot TimesFM-3 接入生产；
- 不用 Rank IC 选择 TimesFM-3；
- 不继续盲目尝试输入归一化；
- 不在微调前加入复杂的行业、市值和横截面融合。
