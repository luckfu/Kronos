# Stage 3 训练框架验收（2026-09-15）

## 当前结论

**DDP forward 与 causal eval 两个本地正确性 gate 已通过。** 尚未进行双 T4 / NCCL / 显存耗时验收，未提交 Kaggle、未跑任何训练 segment、未读取 OOS。小型合成模型做了内存中的两步更新来验证优化器；C2 best 只做 forward/backward，原文件未修改。

环境沿用 Conda base（Python 3.12 / PyTorch 2.13.0 / CPU），命令级设置 `DYLD_LIBRARY_PATH=/opt/miniconda3/lib`，没有修改 Conda 安装。

## 修改范围

### 完整目标由 DDP forward 管理

新增 `finetune/stage3_training_model.py::Stage3TrainingModel`，包含已有 predictor、冻结 tokenizer 和训练 EMA，不新增任何预测头或 predictor 参数。

训练入口现在是：

```python
model = DDP(Stage3TrainingModel(predictor, tokenizer), ...)
loss, metrics = model(x, stamp, sector_id=sec, size_percentile=pct)
loss.backward()
optimizer.step()
```

token CE、条件 s2 candidates、候选 decode、Path Loss 都由这一次受 DDP 管理的 forward 组织。候选算法、Top-16/16、Huber delta=0.02、lambda=0.05 没有改变；lambda 只乘一次。

EMA 先用 detached 的全局 batch 加权均值更新，再让所有 rank 使用同一分母。否则各卡的局部 EMA 会引入额外的 shard 加权，无法与单卡 global batch 对齐。该缩放并不保证梯度一定变小，仍要监控原始梯度并裁剪。

### 显式 causal eval，不改变旧模型默认行为

依赖 attention 增加可选 `is_causal=None` 参数。默认 None 仍使用历史 `self.training` 逻辑；Kronos 原始 forward/生产 inference 没有改传参。

只有 Stage3 包装器对普通 s2 CE 和 candidate s2 分支都显式传 `is_causal=True`。验证时保持 `model.eval()`，所以因果遮罩开启，而 dropout 关闭。冻结 tokenizer 即使在整个包装器 `.train()` 后仍保持 `.eval()`。

旧 C2 evaluator 与历史指标定义保留；Stage3 新的 causal teacher-forced validation loss 不应直接冒充旧 C2 历史 validation loss。

### 验证与记录

- 数据仍由原 `QlibDataset('train') / QlibDataset('val')` 读取；训练 20,000 窗口/segment，全量 validation。
- validation 各 rank 使用互不重叠的索引分片，不再由 DistributedSampler 补齐重复样本。只在验证结束汇总统计，允许 rank 的批次数不同。
- 验证不更新训练 EMA，objective 仍显式记录为 `token_ce + 0.05 * raw_path_huber`，训练使用 EMA-normalized path loss。
- step 1 及之后每 50 steps 打印并记录 token/raw path/normalized path/total loss、四组梯度范数、Top-16 joint mass、s1 熵、在 Top-K s1 内加权的条件 s2 熵。
- 每个 segment 的全量验证记录各 horizon MAE；归一化六特征 MAE 不是原始价格或收益单位。
- trainer 默认 1 segment、seed=20260915，训练 coverage seed 显式传入原数据加载配置。
- predictor 导出仍用原 `save_pretrained` 格式，不包含包装器前缀或 tokenizer 权重。

## 验收结果

### 自动测试：16 passed

原有 12 项候选/梯度测试全部保留通过，另增加 4 项框架验收：

1. 显式 causal eval 的未来扰动隔离、重复调用确定性、默认 legacy 行为不变。
2. 包装器端到端验证：未来输入行变化不影响前面 H1–H9 的 MAE，dropout 关闭，训练 EMA 不更新。
3. checkpoint save/load：predictor、AdamW、EMA、step、CPU RNG 恢复；下一次确定性更新与未中断参考逐参数完全一致；导出能由原 Kronos 类加载。拒绝把 C2 optimizer checkpoint 当新 Stage3 state 恢复。
4. 两进程 Gloo DDP：各 rank 使用不同的两个样本，与单卡四样本 global batch 对照，连续更新两步；再检查 5 个验证样本按 3/2 分片的准确计数与指标。

| 项目 | Step 1 | Step 2 |
| --- | ---: | ---: |
| DDP 与单卡 global batch 梯度最大绝对差 | 3.73e-8 | 5.96e-8 |
| 两 rank 参数最大绝对差 | 0 | 0 |
| DDP 与单卡更新后参数最大绝对差 | 1.49e-8 | 1.49e-8 |
| 全局 EMA | 0.01590196 | 0.01590548 |

精确验证分片：5/5 个样本，无遗漏、无重复，汇总指标与单卡对照通过。CPU checkpoint roundtrip 不代表 CUDA 多 rank RNG 接力已验收。

### C2 best + 真实验证窗口复核

与上一轮同一份 C2 best（SHA256 `4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a`）。完整 val 索引仍为 123,836，只取 8 个窗口验证工程实现，不作为模型指标估计。

- 四批 candidate s2 与显式因果 reference forward 的最大偏差均为 **0**。
- 扰动 context 第 129 行，对 query 119:129 的最大影响均为 **0**。
- Path-only backward 的 s1/s2/dependency/backbone 梯度均有限非零，tokenizer 无梯度。
- 用 C2 best 通过完整训练包装器做一次 backward：所有可训练 predictor 参数都有有限梯度，Tokenizer 仍处于 eval 且无梯度。
- 该 batch：Token Loss=2.523433；raw Path Huber=0.008134；首次 EMA normalized Path=1；加权 Path=0.05；总 Loss=2.573433。
- 四组总目标梯度范数分别为 s1=1.5684、s2=0.8653、dependency=1.0532、backbone=1.5658，均为裁剪前值；这不是完整训练健康性结论。

原始结果保存于 `scratch/stage3_causal_framework_verification.json`。此前非因果对照保存在 `scratch/stage3_conditional_joint_verification.json`，不混写。

## 下一道 gate（本次未执行）

在双 T4 上，先验证真实 C2 batch 的 forward/backward、显存与耗时、rank 参数一致性，以及 CUDA/NCCL 行为，再执行 1 segment + 全量 validation。禁止直接恢复为 40 segments。

Kaggle 提交器尚未在本轮发布或执行；提交前必须统一其显式参数（1 segment、新 run ID、C2 best、fresh optimizer），固定源代码版本，并再次核对实时日志和看板初始化。本次 trainer 会拒绝复用已废弃的旧 C1 run ID。

OOS 保持封存。日后比较 C2 best 与 Stage3 时仍须使用同一个生产 inference/evaluator；路径误差、排序能力和交易指标应分别报告，不把 Path Loss 改善等同于 Alpha 改善。

## 复现

```bash
DYLD_LIBRARY_PATH=/opt/miniconda3/lib PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /opt/miniconda3/bin/python -m pytest \
  tests/test_stage3_conditional_joint.py tests/test_stage3_training_framework.py -q -s

DYLD_LIBRARY_PATH=/opt/miniconda3/lib /opt/miniconda3/bin/python -u \
  -m finetune.verify_stage3_conditional_joint \
  --model-dir scratch/kronos_small_0_1_cosine_c2_best \
  --tokenizer-dir artifacts/kronos_small_0_1_stage2_cosine_refinement_c2_output/kronos_small_v21/models/Kronos-Tokenizer-base \
  --dataset-root data/a_share_full_market_v1_beta_temporal_symbol_validation_v1 \
  --samples 8 --batch 2 --output scratch/stage3_causal_framework_verification.json
```
