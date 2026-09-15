# Stage 3 条件联合候选验证（2026-09-15）

> 本文是第一轮检查的原始记录。随后 DDP 与 causal eval 已完成修复及本地验收，见 `STAGE3_TRAINING_FRAMEWORK_VERIFICATION_CN.md`。下方“尚未通过”描述的是当时状态；不要据此覆盖后续修复，也不要将本地验收等同于 GPU 验收。

## 结论

修正后的条件联合候选、冻结解码器和梯度链路通过本地验证；**当前训练器仍不具备正式开训条件**。尚有 DDP forward 绕过及评估时 dependency attention 非因果的问题。未提交 Kaggle、未训练、未改动权重、未读取 OOS。

旧 Stage3 C1 保留为 aborted implementation smoke test，不作为新阶段父模型。

## 环境与输入

- 最终所有结果使用 Conda base：`/opt/miniconda3/bin/python`，PyTorch 2.13.0、NumPy 2.4.1、pandas 2.3.3。
- 当前机器直接 import torch 会同时加载 torch 自带和 Conda 的两份 libomp。命令级设置 `DYLD_LIBRARY_PATH=/opt/miniconda3/lib` 可统一运行库；没有用 `KMP_DUPLICATE_LIB_OK` 忽略冲突，也没有改动 Conda 包或库文件。
- 起点：`scratch/kronos_small_0_1_cosine_c2_best/model.safetensors`。
- SHA256：`4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a`，与 C2 导出的 `checkpoints/best_model/model.safetensors` 一致。
- 数据：沿用 `QlibDataset('val')`，路径为 `data/a_share_full_market_v1_beta_temporal_symbol_validation_v1/processed_datasets`。
- 完整验证集索引计数 123,836；本次只取等间隔的 8 个索引作工程验收，batch=2。**不是全量指标评估，更不改变正式训练的全量验证设置。**
- CPU FP32，optimizer steps=0，Path Loss 单独 backward，权重 0.05，不启用 EMA normalizer，以免掩盖原始梯度尺度。

## 本次修复

1. 原 `predict_s2_candidates` 将每个 query 缩为长度 1，却保留完整 context，破坏了 RoPE 位置、causal mask 和 dependency residual 的对应关系。改为 `[B*K,T,D]` 完整序列计算，再提取指定 horizon；不修改已有模型参数或旧阶段注意力逻辑。
2. 原 s2 候选在四维张量上使用三维 gather 索引。改为展平 `(s1_rank,s2_rank)` 后，用 joint Top-K 索引取回匹配的 s2 ID。
3. 131 行窗口的最后十个 next-token logits 对应 `x[121:131]`，与目标 `x[120:130]` 错位一天。Stage3 改为 logits/context 的 `119:129` 对齐 target 的 `120:130`。
4. 候选 tokenizer decode 显式处于 `no_grad`，概率权重不 detach；仅记录用权重 detach。

## 单元测试

`tests/test_stage3_conditional_joint.py`：**12 passed**。

- train/eval 两种模式，普通长度和 T=130，候选 s2 与逐候选完整序列原始 dependency forward 对照；dropout 关闭以作确定性比较。
- 重构后的 teacher-forcing forward 与原运算序列一致。
- 256 个 joint combinations 的 Top-16 选择、s1/s2 ID 配对、形状及权重归一化。
- Path Loss 单独 backward：s1 head、s2 head、dependency layer、主干均有有限非零梯度，tokenizer 无梯度。
- 固定候选支持集的 log-probability 有限差分与方向扰动；完整候选流水线中 s1、s2 logits 分别做有限差分。
- next-token 日期偏移检查。

## C2 best + 真实验证窗口

| 检查 | 实测结果 |
| --- | --- |
| 输出形状 | 每批 `[2,10,6]` |
| candidate s2 对原始同条件 forward 最大差异 | 四批均为 0 |
| 更换 s1 候选引起 s2 logits 最大变化 | 0.2624–0.8407 |
| s1 head 梯度范数 | 8.62e-5–1.21e-4 |
| s2 head 梯度范数 | 5.08e-5–1.31e-4 |
| dependency layer 梯度范数 | 4.60e-5–1.64e-4 |
| shared transformer 梯度范数 | 1.74e-4–2.18e-4 |
| tokenizer 非空梯度数 | 0 |
| 有限差分对 autograd 绝对偏差 | 小于 3.2e-14 |
| 入选 Top-16 joint 概率质量（每批均值） | 50.02%、56.44%、77.54%、53.93% |

逐批原始结果：`scratch/stage3_conditional_joint_verification.json`。梯度非零只证明链路成立，不能由其绝对大小宣称训练健康或路径预测将改善。

## 尚未通过的开训条件

### 1. DDP forward 被绕过

`train_stage3_path_alignment.py` 包装 DDP 后，用 `raw.encode_context/predict_s1/predict_s2` 构造损失，没有调用 DDP 对象的 forward。Conda 环境双进程 Gloo 最小复现：输入分别为 1 和 2，线性模型 loss 为输出之和。

- 正常 DDP forward：两个 rank 的梯度都是 1.5。
- 绕过 DDP forward：两个 rank 的梯度分别为 1.0、2.0。

正式双卡训练必须将完整可训练目标放入 DDP 管理的 forward，再验证多 rank 梯度及参数一致性，不能把两张卡各自更新当作 global batch=64。

### 2. dependency attention 的 eval 非因果语义

`model/module.py` 的 `MultiHeadCrossAttentionWithRoPE.forward` 使用 `is_causal_flag = self.training`。因此训练为因果，eval 为非因果。对 teacher-forced 完整窗口验证，早期 horizon 可读取后续真实 token 的 context。

固定随机种子 71、T=130、位置 119:129，只扰动最后一行 context（位置129，晚于所有被检查 query）：

- train 模式：s2 logits 最大变化 **0**。
- eval 模式：最大变化 **4.099821**。

这也说明“与原始 forward 完全一致”不等于“评估无泄漏”。需为 Stage3 明确并实现因果验证，保持 dropout 关闭；不能用简单 `model.train()` 代替 eval，也不能静默改旧 C2 的模型行为与历史指标定义。

### 3. 目标与近似必须如实标注

- 当前 loss 是六个归一化特征（OHLCVA）的 Huber，并非反归一化收益路径 Huber；`delta=0.02` 不能解释成 2% 收益。
- 当前按每个 horizon 选 joint Top-16，将相同 rank 串成十日解码 anchor，再逐日加权。decoder 存在时间依赖，因此这是**截断候选混合代理目标**，不是十日完整自回归路径的精确概率期望。
- Top-16 joint 保留质量不等于 s1 Top-16 的概率覆盖率。8 个样本结果不能作为最优 K 的结论。
- Kaggle wrapper 仍写着 40 segments、旧 SwanLab run ID；正式提交前必须改成 **1 segment、新 run、C2 best、fresh optimizer**，并完成 GPU/双卡验收。本次未执行该 wrapper。

## 复现

仓库根目录执行（不安装包、不下载模型）：

```bash
DYLD_LIBRARY_PATH=/opt/miniconda3/lib PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /opt/miniconda3/bin/python -m pytest tests/test_stage3_conditional_joint.py -q

DYLD_LIBRARY_PATH=/opt/miniconda3/lib /opt/miniconda3/bin/python -u \
  -m finetune.verify_stage3_conditional_joint \
  --model-dir scratch/kronos_small_0_1_cosine_c2_best \
  --tokenizer-dir artifacts/kronos_small_0_1_stage2_cosine_refinement_c2_output/kronos_small_v21/models/Kronos-Tokenizer-base \
  --dataset-root data/a_share_full_market_v1_beta_temporal_symbol_validation_v1 \
  --samples 8 --batch 2 --output scratch/stage3_conditional_joint_verification.json
```
