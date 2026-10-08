# TimesFM-3 V7 训练交接

记录日期：2026-10-09，Asia/Shanghai。
本文件是运行中交接，不是最终训练报告。接手后先复核状态和实时日志。

## 当前任务

| 项目 | 值 |
|---|---|
| Kernel | `wynstonliu/kronos-timesfm3-lora-finetune-v7/1` |
| Kaggle | https://www.kaggle.com/code/wynstonliu/kronos-timesfm3-lora-finetune-v7 |
| 已核验状态 | `RUNNING` |
| 启动时间 | 2026-10-09 00:50:14 CST |
| 最近已核验训练证据 | 2026-10-09 00:58:50 CST，step 600；snapshot 已保存 |
| GPU | 两张 Tesla T4，日志确认 `gpu_count=2` |
| SwanLab project/workspace | `finance` / `roc_fu` |
| 新 run id | `timesfm3-lora-finetune-v7-20261009` |
| 新看板 | https://swanlab.cn/@roc_fu/finance/runs/timesfm3-lora-finetune-v7-20261009 |
| 起点 | 原始 `google/timesfm-3.0-pytorch`，fresh optimizer，step 0 |
| 目标上限 | 21,000 global steps |
| 硬上限 | 16,200 秒，4.5 小时；不再使用 30 分钟短跑上限 |
| 正常收尾 | 入口运行至预算剩余 600 秒时停止训练，完成验证、保存、上传 |

提交前 CLI quota 为剩余 4.90 小时，但用户指定只按 4.5 小时预算。
这是当时快照，不是现在的剩余额度；接手必须重新运行 `kaggle quota`。
V6 稳态区间均值最大约 0.6844 秒/step，按 0.685 保守估算后取 21,000 steps。
V7 初期实测约 1.58 steps/s；后续判断使用新的 timestamp/step 差分，
不能把启动均速或旧 checkpoint 当作当前速度、当前状态。

## 必须先读

所有操作先执行 `finetune/KAGGLE_RUNBOOK_CN.md`，特别是第九、十条。
运行中必须使用 SSE 实时日志，禁止无 `--follow` 的空结果驱动重启或取消：

```bash
kaggle kernels status wynstonliu/kronos-timesfm3-lora-finetune-v7/1
kaggle kernels logs --follow wynstonliu/kronos-timesfm3-lora-finetune-v7
kaggle kernels files wynstonliu/kronos-timesfm3-lora-finetune-v7/1 --format json --page-size 200
kaggle quota
```

**SSE slug 不带 `/1`**：本轮亲测，带版本路径的 `--follow` 返回 404；
去掉 `/1` 后能正常重放和持续读取。状态和结束后的产物查询则可指定 `/1`。
macOS 没有默认 GNU timeout；短快照用 Python subprocess 超时终止本地日志客户端，
不能终止 Kaggle 任务。CLI 账号必须是 owner `wynstonliu`。
每次记录检查时间、状态、最后日志时间、phase、step、Output 清单。
RUNNING 期间 Output 空是正常情况，不据此下结论；只在 COMPLETE 后验收。

本机**只取日志和轻量 JSON 指标**。模型、checkpoint、数据保留在 Kaggle；
接力挂载完整 Output，不通过本机下载或上传中转。历史 `/tmp/timesfm3-v6-output`
是断线前已下载文件，不能再次发起模型下载。

## V6 为什么重跑

V6 已 COMPLETE，step 1200；但实际训练 7,126,976 参数，compact state 只保存
4,096,000 LoRA 参数，遗漏 3,030,976 已训练的其他权重。不能精确恢复，
也不能把 V6 评估值赋给缺失权重的恢复模型。
V6 `step_1200.pt` 在验证前保存，Best=Infinity、history 为空；
`latest.pt` 含验证历史但同样缺权重。V6 只作不可变历史，不晋级、不接力。

V7 新实验、新曲线，与 V6 不拼接。V7 metadata 中挂载 V6 Output
**仅为了重用 train.npz/val.npz**；启动日志确认 `resume_from=null`、
`continuation_slug=null`、`auto_resume=false`，没有读取 V6 模型。

## 配置与验证范围

保持 V6 的核心参数：batch 2，AdamW，LR 1e-4，WD 0.01，
gradient clip 1.0，LoRA rank 8 / alpha 16 / dropout 0.05，
seed 20261008，120 日输入、10 日 close 目标，原始/归一化/return loss 不变。
原数据切分 train 2025-07-01 至 2026-07-02，
val 2026-07-17 至 2026-08-10，数据样本数 639,472 / 87,705。
行业特征等旧规划不是本轮已实现功能，不能混写进实际训练清单。

验证保留 V6 的固定前 256 batches，即最多 512 样本，
**不是全量验证，不能用于宣称完整时间外模型晋级**。
新增 step 0 baseline 与最终停止边界验证。V7 step 0 子集 baseline：
nRMSE 0.0301188010，nMAE 0.0258573916，方向一致率 0.4474826453。
后续只与同一验证子集比较，不与其他基线的不同评估范围直接比较。

## Checkpoint 与 Output

格式 `timesfm3_trainable_compact_v2`：
全部 requires_grad 参数、persistent buffers、optimizer、torch/CUDA/Python/NumPy RNG、
冻结 base SHA-256、数据 SHA-256、config、run id、epoch/batch cursor、Best、history。
恢复先核对 base/data/config、参数名字和顺序、验证完成标志；
旧 v1、缺 tensor、不同实验或未验证 snapshot 都拒绝。
采样 generator 每个 epoch 独立 seed，恢复重建同一排列并跳过已完成 batches。
这不是 V6 主训练器的 `next_epoch` schema，不能混用两种断点。

实际启动预检确认：

```text
trained_parameters = saved_trained_parameters = 7126976
gpu_count = 2
base_model_sha256 = c67cba095f970c2149fd6269dfdd9cccb1a274a2839487b7c3368d1a5c6fc372
```

Kaggle Output 根 `/kaggle/working/timesfm3_lora` 必须包含：

```text
run.log
metrics.jsonl
progress.json
summary.json
experiment_manifest.json
history.json
best_metrics.json
best.pt
last_state.pt
latest.pt
training_snapshot.pt
```

每 50 steps 原子覆盖 `training_snapshot.pt`，日志明确 `resumable=false`；
它未完成验证，不是完整续训源。正常结束后 `last_state.pt` 是验证完成的续训源，
`best.pt` 是本实验历史最优，`latest.pt` 是最近完整状态，不能互换。
导出在 Kaggle 内重读 `last_state.pt`，逐 tensor 与验证模型比较；
日志出现 `checkpoint_export_verified`，summary 的
`checkpoint_verified=true`、`validation_complete=true` 才算产物预验收通过。
COMPLETE + 文件契约 + 最终日志全部确认后，才可报告成功。

普通 V7 接力使用 `--continuation-slug` 指定上一 COMPLETE slug，
挂载其完整 Output，继承 Best、history 和日志，保持同一 run id。
新增步数按当时 quota 和日志实测时间计算，`max_steps` 是累计总步数。
不能对 active slug push；失败或强停只能恢复上一完整发布的状态。

## 代码与凭据

- 主入口：`finetune/timesfm3_lora_finetune.py`
- 测试：`tests/test_timesfm3_lora_finetune.py`
- 打包：`finetune/build_timesfm3_v7_kernel.py`
- 上传目录：`finetune/timesfm3_lora_finetune_v7_kernel/`
- `run-plan.json` 由打包器嵌入自包含入口，不能假设 Kaggle 上传旁侧文件。

用户明确要求保留有意配置的硬编码 SwanLab key，不得移除或强制改用
环境变量/Kaggle Secret。GitHub 提交保留当前运行版本的凭据路径，
交接文档和日志摘要不重复明文。**当前运行版本不可变**，不为此次 GitHub 更新重启。
实际上传源码 SHA-256：
`2ece59b52dba1c313f71ac0a84c3ba41f8b789ee47e9c7068e612d100d696540`。
本次打包后必须核对上述上传 hash，确保交接源码与当前运行版一致。

安装日志记录 TimesFM 源 commit `e51928e27119cb17bebc005be2696b75e0a9e688`，
版本 3.0.2；SwanLab 固定 0.10.1。后续提交需固定该 TimesFM commit，
避免上游 HEAD 漂移；本轮正在跑的代码不改动。

本地只使用微型模拟模型验证训练/验证/恢复流程，不下载真实模型。
验证命令：

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=. pytest -q tests/test_timesfm3_lora_finetune.py
python finetune/build_timesfm3_v7_kernel.py finetune/timesfm3_lora_finetune_v7_kernel
python -m py_compile finetune/timesfm3_lora_finetune.py finetune/build_timesfm3_v7_kernel.py finetune/timesfm3_lora_finetune_v7_kernel/timesfm3_lora_v7.py
git diff --check
```

OpenMP 环境变量仅为本机重复 runtime 的测试 workaround，不写入 Kaggle 配置。
未来 GPU 代码先完成 GitHub 审核，再获得提交许可；不要因交接推送改版正在运行任务。
